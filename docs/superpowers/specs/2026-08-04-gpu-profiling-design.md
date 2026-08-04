# GPU/CPU Performance Instrumentation and Benchmark Harness

**Date:** 2026-08-04
**Status:** Approved, ready for implementation planning

## Goal

Make every performance claim about VexEngine measurable and reproducible.

Today the engine has four working render paths and a README that describes them
qualitatively, with no numbers. The only instrumentation that exists is
`vex::GpuTimer` (`backends/vulkan/src/vk_gpu_timer.cpp`), which covers three
fixed slots (`RtDispatch`, `Bloom`, `Composite`) in one render mode on one
backend.

The deliverables are:

1. A per-pass GPU and CPU profiler that works on both backends and covers all
   four render modes.
2. A dockable Profiler window in the editor with a frame-time plot, a per-pass
   table, and a "Copy as Markdown" button.
3. RGP / RenderDoc debug markers and object names so captures are readable.
4. An automated benchmark mode driven by a JSON config that produces CSV, PNG,
   and run metadata.

## Non-goals

- **True headless rendering.** `EngineConfig::headless` currently returns early
  from `Engine::init` (`engine/src/core/engine.cpp:19`) and never creates a
  window *or* a graphics context. Making it real means a swapchain-less Vulkan
  init path and an offscreen GL context. That is a separate project. The
  benchmark creates a normal window; measurements are identical because
  rendering already targets an offscreen framebuffer.
- **Third-party profiler integration.** Tracy was considered and rejected: it
  puts the data in its own UI rather than in the editor and the CSV, and writing
  the query-pool ring is the point of the exercise.
- **Optimizing anything.** This work only measures. Acting on the numbers is
  future work.

## Constraints discovered in the existing code

These drive the design and must not be violated.

1. **`render_mode_rasterize.cpp` never touches a `VkCommandBuffer`.** It works
   entirely through the abstract `Framebuffer`/`Shader` API. The current
   `GpuTimer::begin(VkCommandBuffer, GpuTimerSlot)` signature is therefore
   unusable from shared app code. The profiler must resolve the command buffer
   internally.
2. **`GL_TIME_ELAPSED` queries cannot nest.** Only one may be active per target,
   so they cannot express a pass tree. `glQueryCounter(id, GL_TIMESTAMP)` has no
   such restriction and maps 1:1 onto `vkCmdWriteTimestamp`.
3. **`VK_EXT_debug_utils` is only enabled in `VEX_DEBUG` builds**, implicitly via
   `use_default_debug_messenger()` at `backends/vulkan/src/vk_context.cpp:74`.
   RGP captures are taken on Release builds, so it must be enabled
   unconditionally, guarded by an availability check.
4. **`vkCmdResetQueryPool` is illegal inside a render pass.** The profiler's
   per-frame reset must run before any pass begins.
5. **`tests/CMakeLists.txt` links only `vex_core`.** Any profiler logic that
   needs test coverage must live in `vex_core` and must not reference backend
   headers.
6. **Neither backend exposes the GPU device name.** Both only log it
   (`vk_context.cpp:286`, `gl_context.cpp:47`).
7. **nlohmann JSON is already vendored** at `engine/include/json.hpp` for
   tinygltf. The benchmark config parser needs no new dependency.

## Architecture

The profiler splits into a backend-agnostic core in `vex_core` and a thin
backend shim behind an interface, using the same factory pattern that
`GraphicsContext::create()` already uses. This satisfies constraint 5 and keeps
`#ifdef VEX_BACKEND_*` out of the app layer entirely.

### File layout

| File | Status | Contents |
|---|---|---|
| `engine/include/vex/core/profiler.h` | new | `Profiler`, `ProfileZoneResult`, `IProfilerBackend`, `ScopedZone`, macros |
| `engine/src/core/profiler.cpp` | new | All logic. No backend headers. |
| `backends/vulkan/src/vk_profiler_backend.cpp` | new | Query pools, `vkCmdWriteTimestamp`, debug-utils labels |
| `backends/opengl/src/gl_profiler_backend.cpp` | new | `glQueryCounter`, `glPushDebugGroup` |
| `app/src/editor_ui_profiler.cpp` | new | Dockable Profiler window |
| `app/src/benchmark.h` / `benchmark.cpp` | new | Config parsing, camera path, CSV writing |
| `tests/test_profiler.cpp` | new | Core logic against a mock backend |
| `backends/vulkan/include/vex/vulkan/vk_gpu_timer.h` | **deleted** | Superseded |
| `backends/vulkan/src/vk_gpu_timer.cpp` | **deleted** | Superseded |

### Backend interface

```cpp
class IProfilerBackend
{
public:
    virtual ~IProfilerBackend() = default;

    virtual bool  init()                     = 0;
    virtual void  destroy()                  = 0;
    virtual bool  supportsTimestamps() const = 0;
    virtual float ticksToMs()          const = 0;

    // Guaranteed to be called outside any render pass.
    virtual void resetQueries(uint32_t ringSlot, uint32_t queryCount) = 0;
    virtual void writeTimestamp(uint32_t ringSlot, uint32_t query)    = 0;

    // Fills `out` with queryCount raw tick values. Returns false if results
    // are not ready, leaving `out` untouched.
    virtual bool resolve(uint32_t ringSlot, uint32_t queryCount,
                         std::vector<uint64_t>& out) = 0;

    virtual void pushLabel(const char* name) = 0;
    virtual void popLabel()                  = 0;

    static std::unique_ptr<IProfilerBackend> create(); // defined per backend
};
```

### Public API

```cpp
Profiler& p = Profiler::get();
p.init(IProfilerBackend::create());
p.beginFrame();                    // resolve old results, reset pools
  VEX_GPU_ZONE("Shadow prepass");  // RAII: GPU + CPU timestamps + debug label
  VEX_CPU_ZONE("Frame changes");   // RAII: chrono only
p.endFrame();
p.recordOneShot("BVH build", ms);  // import-time costs, not per-frame
```

`VEX_GPU_ZONE` records a GPU timestamp pair **and** a CPU timestamp pair, so
every GPU zone also reports its CPU recording cost. This is what exposes a pass
that is cheap on the GPU but expensive to submit, which is the situation in the
rasterizer's per-submesh draw loop.

Zone identity is `(const char* literal, order-in-frame)`. No enum to maintain,
no per-frame allocation. Nesting is a depth counter.

Macros use the standard two-level concat so `__LINE__` expands:

```cpp
#define VEX_CONCAT_INNER(a, b) a##b
#define VEX_CONCAT(a, b)       VEX_CONCAT_INNER(a, b)
#define VEX_GPU_ZONE(name) ::vex::ScopedZone    VEX_CONCAT(_vexZone_,    __LINE__)(name)
#define VEX_CPU_ZONE(name) ::vex::ScopedCPUZone VEX_CONCAT(_vexCpuZone_, __LINE__)(name)
```

### Ring buffer and result latency

Three ring slots. Vulkan needs at least `MAX_FRAMES_IN_FLIGHT` (2) so the fence
guarantees queries have landed; a third slot removes the hidden coupling to that
constant and gives OpenGL, which has no fence here, a frame of slack.

Results are resolved with `VK_QUERY_RESULT_WITH_AVAILABILITY_BIT` and
`GL_QUERY_RESULT_AVAILABLE`, so a not-ready slot retains the previous value
instead of reporting garbage. Displayed numbers lag by about three frames, which
is invisible interactively and irrelevant to a benchmark average.

`k_maxZones = 64` per frame, so 128 queries per ring slot, 384 total. Overflow
logs once and drops the extra zones.

### Vulkan specifics

- Begin timestamps use `VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT`, end timestamps use
  `VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT`, matching the existing
  `vk_gpu_timer.cpp:76`. This is the conservative pairing: it measures from "the
  GPU reached this point in the stream" to "everything issued before the end has
  drained". On a deeply pipelined part it slightly over-attributes time to a zone
  whose predecessor is still draining, so adjacent zone times can sum to more
  than the frame total. Surfaced as a UI tooltip rather than hidden.
- `beginFrame()` issues `vkCmdResetQueryPool`. Vulkan offers no way to query
  whether a render pass is active, so this cannot be asserted directly. Instead
  the profiler asserts that its own zone stack is empty at `beginFrame()`, and
  the invariant is held by the call site: `beginFrame()` is the first statement
  of `SceneRenderer::renderScene()`, before any framebuffer is bound. Validation
  layers will catch a regression here immediately (constraint 4).
- `vk_context.cpp` instance creation adds `VK_EXT_debug_utils` unconditionally,
  gated on `vkb::SystemInfo::is_extension_available`. If unavailable,
  `pushLabel`/`popLabel` become no-ops via a runtime bool; timestamps still work.

### OpenGL specifics

- Two query objects per zone per ring slot, written with
  `glQueryCounter(id, GL_TIMESTAMP)` (constraint 2).
- Resolved via `glGetQueryObjectiv(id, GL_QUERY_RESULT_AVAILABLE, ...)` then
  `glGetQueryObjectui64v(id, GL_QUERY_RESULT, ...)`. Nanoseconds to
  milliseconds.
- Labels via `glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, id, -1, name)` and
  `glPopDebugGroup()` (KHR_debug, core in 4.3; the engine already requires 4.3+).

## Instrumentation map

`SceneRenderer::renderScene()` (`app/src/scene_renderer.cpp:745`) is the single
per-frame entry point. `beginFrame()` goes on its first line, `endFrame()` on its
last. All other zones nest inside.

| Zone | Kind | Location |
|---|---|---|
| `Frame` | GPU + CPU | root, spans all of `renderScene` |
| `Geometry rebuild` | one-shot | `scene_renderer.cpp:774` |
| `Material rebuild` | CPU | `scene_renderer.cpp:781` |
| `Outline mask` | GPU + CPU | `scene_renderer.cpp:794` |
| `Frame changes` | CPU | `scene_renderer.cpp:808` (does env reload) |
| `Shadow prepass` | GPU + CPU | `scene_renderer.cpp:626` |
| `Raster: skybox` | GPU | `RasterizeMode::renderWithSelection` |
| `Raster: meshes` | GPU + CPU | per-submesh draw loop, `render_mode_rasterize.cpp:~250-320` |
| `Bloom: threshold` | GPU | all four modes |
| `Bloom: blur` | GPU | all four modes, wraps the full ping-pong loop |
| `Composite` | GPU | all four modes |
| `RT dispatch` | GPU | `GPURaytraceMode`, replaces `GpuTimerSlot::RtDispatch` |
| `Compute dispatch` | GPU | `VKComputeRaytraceMode` |
| `CPU PT: trace` | CPU | `CPURaytraceMode`, `traceSample()` |
| `CPU PT: readback` | CPU | `getLinearHDR()` |
| `CPU PT: upload` | GPU + CPU | `raytraceTex->setData()` |
| `Pick pass` | GPU | `render_mode_rasterize.cpp:563`, only on click |
| `Denoise (OIDN)` | one-shot | `triggerDenoise` / `triggerDenoiseAux` |

One-shot CPU zones inside `SceneGeometryCache`: `Triangle flatten`, `BVH build`,
`Texture load`, `VK SSBO pack` in `rebuild()` (`scene_geometry_cache.cpp:56`),
and `BLAS build` / `TLAS build` in `buildAccelerationStructures()` (`:537`).

**Why one-shot is a separate channel.** A geometry rebuild costs roughly 1.6 s on
Bistro. As a normal per-frame zone it would blow out the plot's Y axis and poison
the running average for hundreds of frames. One-shot zones go into a persistent
timestamped list with their own UI section, which also gives a live view of the
import breakdown that currently only exists as `Log::info` strings.

### Removals

`GPURaytraceMode::m_gpuTimer`, `getGpuPassTimings()`, and the
`#ifdef VEX_BACKEND_VULKAN` block at `app/src/editor_ui.cpp:268-282` are all
deleted. The three call sites become `VEX_GPU_ZONE` and gain GL support.

## Editor Profiler window

New file `app/src/editor_ui_profiler.cpp`, matching the existing
`editor_ui_hierarchy/inspector/settings/viewport` split. Dockable, with a View
menu toggle.

```
Profiler                                      [x]
--------------------------------------------------
 GPU 6.41 ms   CPU 2.18 ms   156 FPS   [pause] [reset]
 (256-frame plot of root zone GPU time)      8 ms
--------------------------------------------------
 Pass              last    avg    max     %
 v Frame           6.41   6.38   9.02   100
     Shadow        0.82   0.79   1.94    13
     Outline       0.11   0.10   0.31     2
     Raster:sky    0.05   0.05   0.09     1
     Raster:mesh   3.91   3.88   5.02    61
     Bloom:thr     0.21   0.20   0.28     3
     Bloom:blur    0.73   0.75   0.91    11
     Composite     0.58   0.57   0.71     9
--------------------------------------------------
 > CPU zones
 > One-shot   (BVH build 1620.4 . TLAS 212.7)
--------------------------------------------------
 [ Copy as Markdown ]
```

- `avg` is an exponential moving average, alpha = 0.05. Cheap, no per-zone
  history buffer, settles fast enough to read while dragging a slider.
- `max` is a running max with a reset button.
- `%` is of the root `Frame` GPU time. Children will not sum to exactly 100
  because of the `TOP_OF_PIPE`/`BOTTOM_OF_PIPE` pairing; a tooltip on the header
  says so.
- The plot is a 256-frame ring of the root zone's GPU time via
  `ImGui::PlotLines`, auto-scaled to the running max.
- `Copy as Markdown` writes the current table to the clipboard as a GitHub
  table. This is what produces the README content, so it ships in v1.
- `pause` freezes resolution so a spike can be read instead of chased.
- The window renders when GPU timing is unavailable, showing CPU columns and a
  one-line explanation.

Header shows `GraphicsContext::deviceName()` (new, see below).

## Benchmark harness

### Config

Parsed with `engine/include/json.hpp`. Everything except `scene` has a default.

```jsonc
{
  "name":          "bistro-hwrt-1440p",   // output dir name
  "scene":         "assets/bistro/bistro.obj",
  "sceneFormat":   "obj",                 // or "gltf"; inferred from extension if absent
  "mode":          "gpu_raytrace",        // rasterize | cpu_raytrace | gpu_raytrace | compute_raytrace
  "width":  2560, "height": 1440,
  "warmupFrames":  30,
  "measureFrames": 300,
  "maxSamples":    0,                     // 0 = unlimited
  "vsync":         false,
  "camera": [
    { "target": [0,2,0], "distance": 12, "yaw": 0.0, "pitch": -0.2 },
    { "target": [4,2,-3], "distance":  8, "yaw": 1.6, "pitch": -0.1 }
  ]
}
```

**Camera semantics determine what is being measured.** Warmup holds keyframe 0 so
shader compilation, descriptor allocation, and geometry upload settle. Over the
measured frames the camera interpolates piecewise-linearly across keyframes.

- One keyframe: static camera, path tracers accumulate, measuring convergence
  over time.
- Two or more keyframes: accumulation resets every frame, every frame is 1 spp,
  measuring raw throughput in ms per sample.

Both are valid. Ship one of each for Bistro so the README can quote both.

`vsync` defaults to false. Leaving it on silently clamps every result to the
refresh rate.

### Runner

`BenchmarkRunner` is a state machine driven once per iteration of `App::run()`:

```
Loading -> Warmup(warmupFrames) -> Measure(measureFrames) -> Writing -> Done
```

`tick()` returns false when done; `App` then writes outputs and calls a new
`Engine::requestExit()`. The editor UI is not drawn while a benchmark is active,
so ImGui cost does not contaminate the frame time.

If `maxSamples > 0`, measurement stops at whichever comes first, the sample cap
or `measureFrames`.

### CLI and wiring

`main.cpp` gains `--bench <config.json>` and optional `--bench-out <dir>`.
`App::init` changes signature from `const vex::EngineConfig&` to a new
`AppConfig { vex::EngineConfig engine; std::string benchConfigPath; std::string benchOutDir; }`.
When a benchmark is active, `App::init` skips the default ChessSet import
(`app/src/app.cpp:60`) and loads the benchmark scene instead.

`Engine` gains `void requestExit()` setting `m_running = false`.

### Outputs, into `results/<name>/`

| File | Contents |
|---|---|
| `frames.csv` | one row per measured frame: `frame,cpuMs,gpuMs,<zone1>,<zone2>,...` where `cpuMs`/`gpuMs` are the root `Frame` zone's CPU and GPU times and each `<zoneN>` column is that zone's GPU time (CPU-only zones report their CPU time). Column set is fixed at the first measured frame; a zone appearing later is ignored, a zone disappearing writes an empty cell. |
| `summary.csv` | one row per zone: `zone,mean,min,max,p50,p95,p99,stddev` |
| `final.png` | last frame, via existing `SceneRenderer::saveImage` (`scene_renderer.cpp:304`) |
| `run.json` | device name, backend, driver version, resolution, git commit, timestamp, config echo |

Percentiles come from sorting the per-frame vector at the end. p95 and p99 expose
hitching that a mean hides.

A summary table is also printed to stdout and `Log::info` on completion.

### New accessor

`GraphicsContext` gains `virtual std::string deviceName() const`, implemented in
both backends (constraint 6). Used by `run.json` and the Profiler window header.

## Error handling

| Situation | Behaviour |
|---|---|
| `timestampPeriod == 0` or queue `timestampValidBits == 0` | GPU timing disabled, logged once. CPU columns and an explanation shown. Everything else works. |
| Query results not ready | Previous value retained. Never displays garbage. |
| More than `k_maxZones` zones in a frame | Logged once, extras dropped. Pool cannot overflow. |
| Unbalanced begin/end | `assert` in debug; release clamps depth at zero so it cannot underflow. |
| `VK_EXT_debug_utils` unavailable | `pushLabel`/`popLabel` become no-ops. Timestamps unaffected. |
| Bench config missing, unparsable, or unknown mode | `Log::error` plus stderr, exit code 1. Never falls back to interactive. |
| Bench scene fails to load | Same. |
| Bench mode unsupported on this backend (e.g. `compute_raytrace` on GL) | Error and exit 1, never silently renders something else. |

## Testing

`tests/test_profiler.cpp`, doctest, linked against `vex_core`, using a
`MockProfilerBackend` returning scripted tick values. Covers:

- Zone nesting produces the expected `(name, depth)` sequence, including siblings
  at the same depth.
- Ring slot rotation: results written on frame N surface on frame N+3; an
  unresolved slot leaves the previous value untouched.
- Zone overflow past `k_maxZones` drops extras without corrupting the list or the
  query index.
- Unbalanced `endZone()` cannot drive depth negative.
- Statistics: mean, min, max, p50, p95, p99 against hand-computed values,
  including single-sample and two-sample cases where naive percentile indexing
  goes out of bounds.
- CSV row formatting, including a zone name containing a comma.
- Benchmark camera interpolation: 1 keyframe is static; 2 keyframes hit exactly
  keyframe 0 on the first measured frame and keyframe 1 on the last; 3 keyframes
  cross the midpoint correctly.

### Manual verification checklist

GPU query paths cannot be unit tested without a device.

1. With vsync off, the root `Frame` zone tracks ImGui's frame time within about
   1 ms in all four render modes.
2. Sum of child zones is within about 10 percent of the root zone (not exact, see
   the `TOP_OF_PIPE` note).
3. An RGP capture of Bistro in `gpu_raytrace` shows named regions matching the
   zone names, and RGP's own dispatch timing agrees with the profiler's
   `RT dispatch` value.
4. A RenderDoc capture of the rasterizer shows named regions and named
   buffers/images.
5. Both backends build and produce plausible numbers.
6. Two consecutive benchmark runs of the same config agree within a few percent.

## Rough size

About 10 new files, 4 modified engine/backend files, roughly 10 modified app
files, one deleted class. Approximately 900 lines of new code, about 200 of them
tests.
