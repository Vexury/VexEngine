# GPU/CPU Profiler and Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give VexEngine a per-pass GPU/CPU profiler on both backends, a dockable Profiler window, RGP/RenderDoc markers, and a JSON-driven benchmark mode that emits CSV, PNG, and run metadata.

**Architecture:** A `vex::Profiler` singleton lives in `vex_core` and holds all logic (zone stack, three-slot query ring, statistics). It talks to the GPU through a six-method `IProfilerBackend` interface whose `create()` factory is defined once per backend, mirroring the existing `GraphicsContext::create()` pattern. This keeps `#ifdef VEX_BACKEND_*` out of the app layer and lets the core be unit-tested against a mock backend, since `tests/CMakeLists.txt` links only `vex_core`.

**Tech Stack:** C++20, CMake, Vulkan (volk + vk-bootstrap + VMA), OpenGL 4.3 (glad), Dear ImGui, doctest, nlohmann JSON (already vendored at `engine/include/json.hpp`), GLM.

**Spec:** `docs/superpowers/specs/2026-08-04-gpu-profiling-design.md`

## Global Constraints

- The engine builds one backend at a time. `VEX_BACKEND` is `OpenGL` or `Vulkan`; they are mutually exclusive. Every task must be verified on **both** presets unless the task is backend-specific.
- `vex_core` must never include a backend header. `tests/CMakeLists.txt` links only `vex_core`, so any logic needing test coverage lives there or in a backend-free app file added explicitly to the test target.
- `vkCmdResetQueryPool` is illegal inside a render pass. `Profiler::beginFrame()` must be the first statement of `SceneRenderer::renderScene()`, before any framebuffer is bound.
- `GL_TIME_ELAPSED` cannot nest. Use `glQueryCounter(id, GL_TIMESTAMP)` only.
- `k_maxZones = 64`, `k_ringSlots = 3`. These are `static constexpr` on `vex::Profiler` and both backends size their pools from them.
- Vulkan begin timestamps use `VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT`, end timestamps use `VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT`.
- No em dashes in any user-facing string, comment, or documentation text.
- No multi-line docstrings. Comments only where the reason is non-obvious.
- Follow existing file conventions: 4-space indent, `m_` member prefix, `k_` constexpr prefix, `namespace vex` for engine code, no namespace for app code.

### Build and test commands

```powershell
# Vulkan app
cmake --preset vulkan-release
cmake --build build-vk --config Release

# OpenGL app
cmake --preset opengl-release
cmake --build build-gl --config Release

# Unit tests (backend-independent; OpenGL chosen because it has no SDK requirement)
cmake -S . -B build_tests -DVEX_BUILD_TESTS=ON -DVEX_BUILD_APP=OFF -DVEX_BACKEND=OpenGL
cmake --build build_tests --config Release --target vex_tests
.\build_tests\Release\vex_tests.exe
```

The app must be run from the repository root so `VexAssetsCC0/` and `shaders/` resolve.

---

### Task 1: Profiler core in `vex_core`

Pure logic plus a mock-backed test suite. No backend code, no GPU. `IProfilerBackend::create()` is declared but deliberately left undefined here; nothing in `vex_core` or the tests calls it, so there is no link error.

**Files:**
- Create: `engine/include/vex/core/profiler.h`
- Create: `engine/src/core/profiler.cpp`
- Create: `tests/test_profiler.cpp`
- Modify: `engine/CMakeLists.txt:15` (add `src/core/profiler.cpp`)
- Modify: `tests/CMakeLists.txt:7` (add `test_profiler.cpp`)

**Interfaces:**
- Consumes: `vex::Log::warn` from `<vex/core/log.h>`.
- Produces: `vex::Profiler` (singleton via `Profiler::get()`), `vex::IProfilerBackend` (abstract, with `static std::unique_ptr<IProfilerBackend> create()`), `vex::ProfileZoneResult { const char* name; int depth; float gpuMs; float cpuMs; }`, `vex::ProfileOneShot { std::string name; float ms; double timestamp; }`, macros `VEX_GPU_ZONE(name)` and `VEX_CPU_ZONE(name)`, constants `Profiler::k_maxZones == 64` and `Profiler::k_ringSlots == 3`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_profiler.cpp`:

```cpp
#include <doctest/doctest.h>
#include <vex/core/profiler.h>

#include <memory>
#include <string>
#include <vector>

namespace
{

// Returns scripted tick values so the core can be exercised without a GPU.
// Each call to resolve() hands back m_ticks, or reports "not ready" when
// m_ready is false.
class MockProfilerBackend final : public vex::IProfilerBackend
{
public:
    bool  init() override                     { return true; }
    void  destroy() override                  {}
    bool  supportsTimestamps() const override { return m_supported; }
    float ticksToMs() const override          { return 1e-6f; } // ticks are ns

    void resetQueries(uint32_t slot, uint32_t count) override
    {
        resetCalls.push_back({slot, count});
    }

    void writeTimestamp(uint32_t slot, uint32_t query) override
    {
        writes.push_back({slot, query});
    }

    bool resolve(uint32_t slot, uint32_t queryCount,
                 std::vector<uint64_t>& out) override
    {
        if (!m_ready) return false;
        out.assign(queryCount, 0u);
        for (uint32_t i = 0; i < queryCount; ++i)
            out[i] = tickFor(slot, i);
        return true;
    }

    void pushLabel(const char* name) override { labels.push_back(name); }
    void popLabel() override                  { ++popCount; }

    // Test knobs
    bool m_supported = true;
    bool m_ready     = true;
    // Zone i in slot s spans [i*2000000, i*2000000 + 1000000] ns, so every
    // zone measures exactly 1.0 ms regardless of slot.
    static uint64_t tickFor(uint32_t, uint32_t query)
    {
        const uint32_t zone = query / 2;
        const bool     isEnd = (query & 1u) != 0u;
        return static_cast<uint64_t>(zone) * 2000000ull
             + (isEnd ? 1000000ull : 0ull);
    }

    struct Call { uint32_t slot; uint32_t count; };
    std::vector<Call>        resetCalls;
    std::vector<Call>        writes;
    std::vector<std::string> labels;
    int                      popCount = 0;
};

// The Profiler is a singleton; each test starts from a clean instance.
MockProfilerBackend* installMock()
{
    auto  mock = std::make_unique<MockProfilerBackend>();
    auto* raw  = mock.get();
    vex::Profiler::get().shutdown();
    vex::Profiler::get().init(std::move(mock));
    return raw;
}

} // namespace

TEST_CASE("profiler records nested zones with correct depth and order")
{
    installMock();
    auto& p = vex::Profiler::get();

    // Frame 0 records the zones; results only surface k_ringSlots frames later.
    for (int i = 0; i < static_cast<int>(vex::Profiler::k_ringSlots) + 1; ++i)
    {
        p.beginFrame();
        p.beginZone("Frame");
          p.beginZone("Shadow");
          p.endZone();
          p.beginZone("Raster");
            p.beginZone("Meshes");
            p.endZone();
          p.endZone();
        p.endZone();
        p.endFrame();
    }

    const auto& r = p.results();
    REQUIRE(r.size() == 4);
    CHECK(std::string(r[0].name) == "Frame");   CHECK(r[0].depth == 0);
    CHECK(std::string(r[1].name) == "Shadow");  CHECK(r[1].depth == 1);
    CHECK(std::string(r[2].name) == "Raster");  CHECK(r[2].depth == 1);
    CHECK(std::string(r[3].name) == "Meshes");  CHECK(r[3].depth == 2);
    CHECK(r[0].gpuMs == doctest::Approx(1.0f));
}

TEST_CASE("profiler results lag by exactly k_ringSlots frames")
{
    installMock();
    auto& p = vex::Profiler::get();

    p.beginFrame();
    p.beginZone("Only");
    p.endZone();
    p.endFrame();

    // Frames 1 and 2 land on the other two ring slots, which hold no data.
    for (uint32_t i = 1; i < vex::Profiler::k_ringSlots; ++i)
    {
        p.beginFrame();
        p.endFrame();
        CHECK(p.results().empty());
    }

    // Frame k_ringSlots returns to slot 0 and resolves frame 0's zones.
    p.beginFrame();
    REQUIRE(p.results().size() == 1);
    CHECK(std::string(p.results()[0].name) == "Only");
    p.endFrame();
}

TEST_CASE("unresolved query results leave the previous values untouched")
{
    auto* mock = installMock();
    auto& p    = vex::Profiler::get();

    for (uint32_t i = 0; i < vex::Profiler::k_ringSlots + 1; ++i)
    {
        p.beginFrame();
        p.beginZone("Stable");
        p.endZone();
        p.endFrame();
    }
    REQUIRE(p.results().size() == 1);
    const float before = p.results()[0].gpuMs;

    mock->m_ready = false;
    for (uint32_t i = 0; i < vex::Profiler::k_ringSlots; ++i)
    {
        p.beginFrame();
        p.beginZone("Different");
        p.endZone();
        p.endFrame();
    }

    REQUIRE(p.results().size() == 1);
    CHECK(std::string(p.results()[0].name) == "Stable");
    CHECK(p.results()[0].gpuMs == doctest::Approx(before));
}

TEST_CASE("zone overflow past k_maxZones drops extras without corruption")
{
    installMock();
    auto& p = vex::Profiler::get();

    const uint32_t over = vex::Profiler::k_maxZones + 10;
    for (uint32_t f = 0; f < vex::Profiler::k_ringSlots + 1; ++f)
    {
        p.beginFrame();
        for (uint32_t i = 0; i < over; ++i)
        {
            p.beginZone("Zone");
            p.endZone();
        }
        p.endFrame();
    }

    CHECK(p.results().size() == vex::Profiler::k_maxZones);
}

TEST_CASE("unbalanced endZone cannot drive depth negative")
{
    installMock();
    auto& p = vex::Profiler::get();

    p.beginFrame();
    p.endZone();          // stray, must be ignored
    p.endZone();          // stray, must be ignored
    p.beginZone("After");
    p.endZone();
    p.endFrame();

    for (uint32_t i = 0; i < vex::Profiler::k_ringSlots; ++i)
    {
        p.beginFrame();
        p.endFrame();
    }

    REQUIRE(p.results().size() == 1);
    CHECK(std::string(p.results()[0].name) == "After");
    CHECK(p.results()[0].depth == 0);
}

TEST_CASE("cpu-only zones report cpu time and no gpu time")
{
    installMock();
    auto& p = vex::Profiler::get();

    for (uint32_t f = 0; f < vex::Profiler::k_ringSlots + 1; ++f)
    {
        p.beginFrame();
        p.beginCPUZone("CpuOnly");
        p.endCPUZone();
        p.endFrame();
    }

    REQUIRE(p.results().size() == 1);
    CHECK(p.results()[0].gpuMs < 0.0f);
    CHECK(p.results()[0].cpuMs >= 0.0f);
}

TEST_CASE("gpu zones emit debug labels in matched pairs")
{
    auto* mock = installMock();
    auto& p    = vex::Profiler::get();

    p.beginFrame();
    p.beginZone("Outer");
      p.beginZone("Inner");
      p.endZone();
      p.beginCPUZone("NoLabel");
      p.endCPUZone();
    p.endZone();
    p.endFrame();

    REQUIRE(mock->labels.size() == 2);
    CHECK(mock->labels[0] == "Outer");
    CHECK(mock->labels[1] == "Inner");
    CHECK(mock->popCount == 2);
}

TEST_CASE("one-shot records accumulate independently of frames")
{
    installMock();
    auto& p = vex::Profiler::get();

    p.clearOneShots();
    p.recordOneShot("BVH build", 1620.4f);
    p.recordOneShot("TLAS build", 212.7f);

    REQUIRE(p.oneShots().size() == 2);
    CHECK(p.oneShots()[0].name == "BVH build");
    CHECK(p.oneShots()[0].ms == doctest::Approx(1620.4f));
    CHECK(p.oneShots()[1].name == "TLAS build");
}

TEST_CASE("gpu timing unavailable degrades to cpu-only without crashing")
{
    auto mock = std::make_unique<MockProfilerBackend>();
    mock->m_supported = false;
    vex::Profiler::get().shutdown();
    vex::Profiler::get().init(std::move(mock));

    auto& p = vex::Profiler::get();
    CHECK_FALSE(p.gpuTimingAvailable());

    for (uint32_t f = 0; f < vex::Profiler::k_ringSlots + 1; ++f)
    {
        p.beginFrame();
        p.beginZone("Frame");
        p.endZone();
        p.endFrame();
    }

    REQUIRE(p.results().size() == 1);
    CHECK(p.results()[0].gpuMs < 0.0f);
    CHECK(p.results()[0].cpuMs >= 0.0f);
}
```

- [ ] **Step 2: Add the test to the build and run it to verify it fails**

Add `test_profiler.cpp` to `tests/CMakeLists.txt` after line 7:

```cmake
add_executable(vex_tests
    test_main.cpp
    test_bvh.cpp
    test_bsdf.cpp
    test_primitives.cpp
    test_camera.cpp
    test_raytracer.cpp
    test_profiler.cpp
)
```

Run:
```powershell
cmake -S . -B build_tests -DVEX_BUILD_TESTS=ON -DVEX_BUILD_APP=OFF -DVEX_BACKEND=OpenGL
cmake --build build_tests --config Release --target vex_tests
```
Expected: FAIL to compile, `cannot open include file: 'vex/core/profiler.h'`.

- [ ] **Step 3: Write the header**

Create `engine/include/vex/core/profiler.h`:

```cpp
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vex
{

// Backend shim for the profiler. One implementation per graphics backend,
// created through the static factory the same way GraphicsContext::create()
// works, so vex_core never references backend headers.
class IProfilerBackend
{
public:
    virtual ~IProfilerBackend() = default;

    virtual bool  init()                     = 0;
    virtual void  destroy()                  = 0;
    virtual bool  supportsTimestamps() const = 0;
    virtual float ticksToMs()          const = 0;

    // Called once per frame, guaranteed to be outside any render pass.
    virtual void resetQueries(uint32_t ringSlot, uint32_t queryCount) = 0;
    virtual void writeTimestamp(uint32_t ringSlot, uint32_t query)    = 0;

    // Fills out with queryCount raw tick values. Returns false when the
    // results are not ready, leaving out untouched.
    virtual bool resolve(uint32_t ringSlot, uint32_t queryCount,
                         std::vector<uint64_t>& out) = 0;

    virtual void pushLabel(const char* name) = 0;
    virtual void popLabel()                  = 0;

    static std::unique_ptr<IProfilerBackend> create();
};

struct ProfileZoneResult
{
    const char* name  = nullptr;
    int         depth = 0;
    float       gpuMs = -1.0f; // negative means unavailable
    float       cpuMs = -1.0f;
};

struct ProfileOneShot
{
    std::string name;
    float       ms        = 0.0f;
    double      timestamp = 0.0; // seconds since program start
};

class Profiler
{
public:
    static constexpr uint32_t k_maxZones  = 64;
    static constexpr uint32_t k_ringSlots = 3;

    static Profiler& get();

    void init(std::unique_ptr<IProfilerBackend> backend);
    void shutdown();

    // beginFrame must run outside any render pass: it issues the query reset.
    void beginFrame();
    void endFrame();

    void beginZone(const char* name);
    void endZone();
    void beginCPUZone(const char* name);
    void endCPUZone();

    // For costs that happen once rather than every frame, such as a BVH build.
    void recordOneShot(const char* name, float ms);

    bool gpuTimingAvailable() const { return m_gpuAvailable; }
    void setEnabled(bool e)         { m_enabled = e; }
    bool isEnabled() const          { return m_enabled; }

    const std::vector<ProfileZoneResult>& results() const  { return m_results; }
    const std::vector<ProfileOneShot>&    oneShots() const { return m_oneShots; }
    void clearOneShots() { m_oneShots.clear(); }

    float frameGpuMs() const;
    float frameCpuMs() const;

private:
    struct Zone
    {
        const char* name       = nullptr;
        int         depth      = 0;
        bool        gpu        = false;
        uint32_t    queryBegin = 0;
        std::chrono::steady_clock::time_point cpuBegin{};
        float       cpuMs      = 0.0f;
    };

    struct FrameData
    {
        std::vector<Zone> zones;
        uint32_t          queryCount = 0;
    };

    struct StackEntry
    {
        uint32_t index; // k_invalidIndex when the zone was dropped by overflow
        bool     gpu;
    };

    static constexpr uint32_t k_invalidIndex = 0xFFFFFFFFu;

    void beginZoneInternal(const char* name, bool gpu);
    void endZoneInternal();
    void resolveSlot(uint32_t slot);

    std::unique_ptr<IProfilerBackend> m_backend;
    FrameData                      m_frames[k_ringSlots];
    std::vector<StackEntry>        m_stack;
    std::vector<ProfileZoneResult> m_results;
    std::vector<ProfileOneShot>    m_oneShots;
    std::vector<uint64_t>          m_ticks;

    uint64_t m_frameIndex     = 0;
    uint32_t m_slot           = 0;
    bool     m_enabled        = true;
    bool     m_inFrame        = false;
    bool     m_gpuAvailable   = false;
    bool     m_overflowLogged = false;
};

struct ScopedZone
{
    explicit ScopedZone(const char* name) { Profiler::get().beginZone(name); }
    ~ScopedZone()                         { Profiler::get().endZone(); }
    ScopedZone(const ScopedZone&)            = delete;
    ScopedZone& operator=(const ScopedZone&) = delete;
};

struct ScopedCPUZone
{
    explicit ScopedCPUZone(const char* name) { Profiler::get().beginCPUZone(name); }
    ~ScopedCPUZone()                         { Profiler::get().endCPUZone(); }
    ScopedCPUZone(const ScopedCPUZone&)            = delete;
    ScopedCPUZone& operator=(const ScopedCPUZone&) = delete;
};

} // namespace vex

#define VEX_PROF_CONCAT_INNER(a, b) a##b
#define VEX_PROF_CONCAT(a, b)       VEX_PROF_CONCAT_INNER(a, b)
#define VEX_GPU_ZONE(name) ::vex::ScopedZone    VEX_PROF_CONCAT(_vexZone_,    __LINE__)(name)
#define VEX_CPU_ZONE(name) ::vex::ScopedCPUZone VEX_PROF_CONCAT(_vexCpuZone_, __LINE__)(name)
```

- [ ] **Step 4: Write the implementation**

Create `engine/src/core/profiler.cpp`:

```cpp
#include <vex/core/profiler.h>
#include <vex/core/log.h>

#include <cassert>

namespace vex
{

Profiler& Profiler::get()
{
    static Profiler instance;
    return instance;
}

void Profiler::init(std::unique_ptr<IProfilerBackend> backend)
{
    m_backend = std::move(backend);
    m_gpuAvailable = false;

    if (m_backend && m_backend->init())
        m_gpuAvailable = m_backend->supportsTimestamps();

    if (!m_gpuAvailable)
        Log::warn("Profiler: GPU timestamps unavailable, reporting CPU times only");
}

void Profiler::shutdown()
{
    if (m_backend)
    {
        m_backend->destroy();
        m_backend.reset();
    }
    for (auto& f : m_frames)
    {
        f.zones.clear();
        f.queryCount = 0;
    }
    m_stack.clear();
    m_results.clear();
    m_oneShots.clear();
    m_ticks.clear();
    m_frameIndex     = 0;
    m_slot           = 0;
    m_inFrame        = false;
    m_gpuAvailable   = false;
    m_overflowLogged = false;
}

void Profiler::beginFrame()
{
    if (!m_enabled) return;

    m_slot = static_cast<uint32_t>(m_frameIndex % k_ringSlots);

    // Resolve what this slot recorded k_ringSlots frames ago, before clearing it.
    resolveSlot(m_slot);

    FrameData& f = m_frames[m_slot];
    f.zones.clear();
    f.queryCount = 0;
    m_stack.clear();

    if (m_gpuAvailable && m_backend)
        m_backend->resetQueries(m_slot, k_maxZones * 2);

    m_inFrame = true;
}

void Profiler::endFrame()
{
    if (!m_enabled) return;

    assert(m_stack.empty() && "Profiler: unbalanced zones at endFrame");
    while (!m_stack.empty())
        endZoneInternal();

    m_inFrame = false;
    ++m_frameIndex;
}

void Profiler::beginZone(const char* name)    { beginZoneInternal(name, true);  }
void Profiler::endZone()                      { endZoneInternal();              }
void Profiler::beginCPUZone(const char* name) { beginZoneInternal(name, false); }
void Profiler::endCPUZone()                   { endZoneInternal();              }

void Profiler::beginZoneInternal(const char* name, bool gpu)
{
    if (!m_enabled || !m_inFrame) return;

    if (gpu && m_backend)
        m_backend->pushLabel(name);

    FrameData& f = m_frames[m_slot];
    if (f.zones.size() >= k_maxZones)
    {
        if (!m_overflowLogged)
        {
            Log::warn("Profiler: more than 64 zones in one frame, extras dropped");
            m_overflowLogged = true;
        }
        m_stack.push_back({k_invalidIndex, gpu});
        return;
    }

    Zone z;
    z.name     = name;
    z.depth    = static_cast<int>(m_stack.size());
    z.gpu      = gpu && m_gpuAvailable;
    z.cpuBegin = std::chrono::steady_clock::now();

    if (z.gpu)
    {
        z.queryBegin = f.queryCount;
        m_backend->writeTimestamp(m_slot, f.queryCount);
        f.queryCount += 2;
    }

    m_stack.push_back({static_cast<uint32_t>(f.zones.size()), gpu});
    f.zones.push_back(z);
}

void Profiler::endZoneInternal()
{
    if (!m_enabled || !m_inFrame) return;
    if (m_stack.empty())
        return; // stray end, ignore rather than underflow

    const StackEntry entry = m_stack.back();
    m_stack.pop_back();

    if (entry.index != k_invalidIndex)
    {
        Zone& z = m_frames[m_slot].zones[entry.index];
        z.cpuMs = std::chrono::duration<float, std::milli>(
                      std::chrono::steady_clock::now() - z.cpuBegin).count();
        if (z.gpu)
            m_backend->writeTimestamp(m_slot, z.queryBegin + 1);
    }

    if (entry.gpu && m_backend)
        m_backend->popLabel();
}

void Profiler::resolveSlot(uint32_t slot)
{
    FrameData& f = m_frames[slot];
    if (f.zones.empty())
        return;

    bool haveGpu = false;
    if (m_gpuAvailable && m_backend && f.queryCount > 0)
    {
        if (!m_backend->resolve(slot, f.queryCount, m_ticks))
            return; // not ready, keep the previous results
        haveGpu = true;
    }

    const float toMs = m_backend ? m_backend->ticksToMs() : 0.0f;

    std::vector<ProfileZoneResult> out;
    out.reserve(f.zones.size());
    for (const Zone& z : f.zones)
    {
        ProfileZoneResult r;
        r.name  = z.name;
        r.depth = z.depth;
        r.cpuMs = z.cpuMs;
        if (z.gpu && haveGpu &&
            static_cast<size_t>(z.queryBegin) + 1 < m_ticks.size())
        {
            const uint64_t b = m_ticks[z.queryBegin];
            const uint64_t e = m_ticks[z.queryBegin + 1];
            if (e >= b)
                r.gpuMs = static_cast<float>(e - b) * toMs;
        }
        out.push_back(r);
    }
    m_results.swap(out);
}

void Profiler::recordOneShot(const char* name, float ms)
{
    ProfileOneShot s;
    s.name = name;
    s.ms   = ms;
    s.timestamp = std::chrono::duration<double>(
                      std::chrono::steady_clock::now().time_since_epoch()).count();
    m_oneShots.push_back(std::move(s));
}

float Profiler::frameGpuMs() const
{
    return m_results.empty() ? -1.0f : m_results.front().gpuMs;
}

float Profiler::frameCpuMs() const
{
    return m_results.empty() ? -1.0f : m_results.front().cpuMs;
}

} // namespace vex
```

Add to `engine/CMakeLists.txt` after line 6 (`src/core/camera.cpp`):

```cmake
    src/core/profiler.cpp
```

- [ ] **Step 5: Run the tests and verify they pass**

Run:
```powershell
cmake --build build_tests --config Release --target vex_tests
.\build_tests\Release\vex_tests.exe
```
Expected: PASS, all 9 profiler test cases green, existing tests still green.

- [ ] **Step 6: Verify both app builds still link**

Run:
```powershell
cmake --build build-vk --config Release
cmake --build build-gl --config Release
```
Expected: both succeed. `IProfilerBackend::create()` is undefined but nothing calls it yet.

- [ ] **Step 7: Commit**

```bash
git add engine/include/vex/core/profiler.h engine/src/core/profiler.cpp engine/CMakeLists.txt tests/test_profiler.cpp tests/CMakeLists.txt
git commit -m "feat: add backend-agnostic Profiler core with mock-backed tests"
```

---

### Task 2: Vulkan profiler backend

Implements `IProfilerBackend::create()` for Vulkan, enables `VK_EXT_debug_utils` in Release, and adds `deviceName()` to the context. Deletes the superseded `GpuTimer` class and its three call sites in the same commit so the tree never has two competing timers.

**Files:**
- Create: `backends/vulkan/src/vk_profiler_backend.cpp`
- Delete: `backends/vulkan/include/vex/vulkan/vk_gpu_timer.h`
- Delete: `backends/vulkan/src/vk_gpu_timer.cpp`
- Modify: `backends/vulkan/CMakeLists.txt:12`
- Modify: `backends/vulkan/src/vk_context.cpp:68-77` (debug utils), `:281-286` (cache device name)
- Modify: `backends/vulkan/include/vex/vulkan/vk_context.h:48` (add `deviceName` override, `m_deviceName` member)
- Modify: `engine/include/vex/graphics/graphics_context.h:36` (add virtual `deviceName`)
- Modify: `app/src/render_mode_gpu_raytrace.h:39-44,71`, `app/src/render_mode_gpu_raytrace.cpp:293-306,499-675`
- Modify: `app/src/editor_ui.cpp:268-282`
- Modify: `app/src/scene_renderer.h` (remove `getGpuPassTimings`)

**Interfaces:**
- Consumes: `vex::IProfilerBackend`, `vex::Profiler::k_maxZones`, `vex::Profiler::k_ringSlots` from Task 1.
- Produces: a linkable `vex::IProfilerBackend::create()` in the Vulkan build; `vex::GraphicsContext::deviceName() const -> std::string`.

- [ ] **Step 1: Add the `deviceName` virtual to the base context**

In `engine/include/vex/graphics/graphics_context.h`, add after line 36 (`backendName`):

```cpp
    virtual std::string deviceName() const { return "Unknown"; }
```

Add `#include <string>` to the include block at the top.

- [ ] **Step 2: Cache and expose the Vulkan device name**

In `backends/vulkan/include/vex/vulkan/vk_context.h`, add after line 48 (`backendName`):

```cpp
    std::string deviceName() const override { return m_deviceName; }
```

Add to the private member block near `m_physicalDevice`:

```cpp
    std::string m_deviceName;
```

In `backends/vulkan/src/vk_context.cpp`, at line 286 where the device name is logged, store it as well:

```cpp
    m_deviceName = props2.properties.deviceName;
    Log::info(std::string("Vulkan Device: ") + m_deviceName);
```

- [ ] **Step 3: Enable `VK_EXT_debug_utils` outside debug builds**

Replace `backends/vulkan/src/vk_context.cpp:68-77` with:

```cpp
    vkb::InstanceBuilder instanceBuilder;
    instanceBuilder
        .set_app_name("VexEngine")
        .set_engine_name("VexEngine")
        .require_api_version(1, 3, 0);
#ifdef VEX_DEBUG
    instanceBuilder.request_validation_layers().use_default_debug_messenger();
#endif

    // Needed in Release too: without it, RGP and RenderDoc captures show
    // unnamed dispatches instead of the profiler's pass names.
    auto sysInfo = vkb::SystemInfo::get_system_info();
    if (sysInfo && sysInfo->is_extension_available(VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
        instanceBuilder.enable_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    auto instRet = instanceBuilder.build();
```

- [ ] **Step 4: Write the Vulkan backend**

Create `backends/vulkan/src/vk_profiler_backend.cpp`:

```cpp
#include <vex/core/profiler.h>
#include <vex/core/log.h>
#include <vex/vulkan/vk_context.h>

#include <volk.h>

#include <vector>

namespace vex
{
namespace
{

class VKProfilerBackend final : public IProfilerBackend
{
public:
    bool init() override
    {
        auto& ctx = VKContext::get();
        m_device  = ctx.getDevice();

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(ctx.getPhysicalDevice(), &props);
        m_ticksToMs = props.limits.timestampPeriod * 1e-6f;

        uint32_t famCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(ctx.getPhysicalDevice(), &famCount, nullptr);
        std::vector<VkQueueFamilyProperties> fams(famCount);
        vkGetPhysicalDeviceQueueFamilyProperties(ctx.getPhysicalDevice(), &famCount, fams.data());

        const uint32_t gfxFam = ctx.getGraphicsQueueFamily();
        const bool validBits  = gfxFam < famCount && fams[gfxFam].timestampValidBits > 0;

        m_supported = props.limits.timestampPeriod > 0.0f && validBits;
        if (!m_supported)
            return true; // init succeeded, timestamps are simply not available

        VkQueryPoolCreateInfo ci{};
        ci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        ci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
        ci.queryCount = Profiler::k_maxZones * 2;

        for (uint32_t i = 0; i < Profiler::k_ringSlots; ++i)
        {
            if (vkCreateQueryPool(m_device, &ci, nullptr, &m_pools[i]) != VK_SUCCESS)
            {
                Log::warn("Profiler: failed to create timestamp query pool");
                m_supported = false;
                return true;
            }
        }

        m_labels = vkCmdBeginDebugUtilsLabelEXT != nullptr
                && vkCmdEndDebugUtilsLabelEXT   != nullptr;
        return true;
    }

    void destroy() override
    {
        for (auto& pool : m_pools)
        {
            if (pool != VK_NULL_HANDLE)
            {
                vkDestroyQueryPool(m_device, pool, nullptr);
                pool = VK_NULL_HANDLE;
            }
        }
    }

    bool  supportsTimestamps() const override { return m_supported; }
    float ticksToMs() const override          { return m_ticksToMs; }

    void resetQueries(uint32_t slot, uint32_t queryCount) override
    {
        if (!m_supported) return;
        vkCmdResetQueryPool(VKContext::get().getCurrentCommandBuffer(),
                            m_pools[slot], 0, queryCount);
    }

    void writeTimestamp(uint32_t slot, uint32_t query) override
    {
        if (!m_supported) return;
        // Even query index is a zone begin, odd is a zone end. TOP_OF_PIPE for
        // the begin and BOTTOM_OF_PIPE for the end is the conservative pairing:
        // it can over-attribute time to a zone whose predecessor is still
        // draining, so sibling zones may sum to more than their parent.
        const VkPipelineStageFlagBits stage = (query & 1u)
            ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        vkCmdWriteTimestamp(VKContext::get().getCurrentCommandBuffer(),
                            stage, m_pools[slot], query);
    }

    bool resolve(uint32_t slot, uint32_t queryCount,
                 std::vector<uint64_t>& out) override
    {
        if (!m_supported || queryCount == 0) return false;

        // WITH_AVAILABILITY writes [value][available] per query, stride 16.
        m_raw.resize(static_cast<size_t>(queryCount) * 2);
        const VkResult res = vkGetQueryPoolResults(
            m_device, m_pools[slot], 0, queryCount,
            m_raw.size() * sizeof(uint64_t), m_raw.data(),
            sizeof(uint64_t) * 2,
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);

        if (res != VK_SUCCESS) return false;

        for (uint32_t i = 0; i < queryCount; ++i)
            if (m_raw[i * 2 + 1] == 0)
                return false;

        out.resize(queryCount);
        for (uint32_t i = 0; i < queryCount; ++i)
            out[i] = m_raw[i * 2];
        return true;
    }

    void pushLabel(const char* name) override
    {
        if (!m_labels) return;
        VkDebugUtilsLabelEXT label{};
        label.sType      = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        label.pLabelName = name;
        label.color[0] = label.color[1] = label.color[2] = label.color[3] = 1.0f;
        vkCmdBeginDebugUtilsLabelEXT(VKContext::get().getCurrentCommandBuffer(), &label);
    }

    void popLabel() override
    {
        if (!m_labels) return;
        vkCmdEndDebugUtilsLabelEXT(VKContext::get().getCurrentCommandBuffer());
    }

private:
    VkDevice    m_device = VK_NULL_HANDLE;
    VkQueryPool m_pools[Profiler::k_ringSlots] = {};
    float       m_ticksToMs = 1.0f;
    bool        m_supported = false;
    bool        m_labels    = false;
    std::vector<uint64_t> m_raw;
};

} // namespace

std::unique_ptr<IProfilerBackend> IProfilerBackend::create()
{
    return std::make_unique<VKProfilerBackend>();
}

} // namespace vex
```

In `backends/vulkan/CMakeLists.txt`, replace line 12 (`src/vk_gpu_timer.cpp`) with:

```cmake
    src/vk_profiler_backend.cpp
```

- [ ] **Step 5: Delete `GpuTimer` and its call sites**

```powershell
git rm backends/vulkan/include/vex/vulkan/vk_gpu_timer.h backends/vulkan/src/vk_gpu_timer.cpp
```

In `app/src/render_mode_gpu_raytrace.h`, delete the `getGpuPassTimings()` accessor (lines 39-44) and the `m_gpuTimer` member (line 71), plus the `#include <vex/vulkan/vk_gpu_timer.h>` if present.

In `app/src/render_mode_gpu_raytrace.cpp`, delete the timer creation at lines 293-295, the destruction at 302-306, and replace each of the six `m_gpuTimer` calls with the new macro. The `beginFrame` call at line 499 is deleted outright, since `SceneRenderer` now owns frame boundaries (Task 4). The three bracketed regions become:

```cpp
    // was: m_gpuTimer->begin(cmd, GpuTimerSlot::RtDispatch) ... end(...)
    {
        VEX_GPU_ZONE("RT dispatch");
        if (!showDenoised && hasTlas && (shared.maxSamples == 0 || m_sampleCount < shared.maxSamples))
        {
            // ... existing body unchanged ...
        }
    }
```

Apply the same brace-scoping to the bloom region (lines 525-605, zone name `"Bloom"`) and the composite region (lines 608-675, zone name `"Composite"`). Add `#include <vex/core/profiler.h>` at the top.

In `app/src/editor_ui.cpp`, delete the whole `#ifdef VEX_BACKEND_VULKAN` block at lines 268-282 and replace with:

```cpp
        ImGui::TextDisabled("Per-pass timings: see the Profiler window");
```

In `app/src/scene_renderer.h`, delete the `getGpuPassTimings()` forwarder and any `GpuPassTimings` include or forward declaration.

- [ ] **Step 6: Build and verify**

Run:
```powershell
cmake --preset vulkan-release
cmake --build build-vk --config Release
```
Expected: builds clean, no references to `GpuTimer` remain. Verify with:
```powershell
git grep -n "GpuTimer\|GpuPassTimings\|vk_gpu_timer"
```
Expected: no matches outside the deleted files.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: add Vulkan profiler backend, replace GpuTimer, enable debug utils in release"
```

---

### Task 3: OpenGL profiler backend

**Files:**
- Create: `backends/opengl/src/gl_profiler_backend.cpp`
- Modify: `backends/opengl/CMakeLists.txt:9`
- Modify: `backends/opengl/include/vex/opengl/gl_context.h` (add `deviceName` override)

**Interfaces:**
- Consumes: `vex::IProfilerBackend`, `vex::Profiler::k_maxZones`, `vex::Profiler::k_ringSlots` from Task 1; `GraphicsContext::deviceName()` virtual from Task 2 Step 1.
- Produces: a linkable `vex::IProfilerBackend::create()` in the OpenGL build.

- [ ] **Step 1: Expose the OpenGL renderer string**

In `backends/opengl/include/vex/opengl/gl_context.h`, add next to `backendName()`:

```cpp
    std::string deviceName() const override;
```

In `backends/opengl/src/gl_context.cpp`, add:

```cpp
std::string GLContext::deviceName() const
{
    const GLubyte* r = glGetString(GL_RENDERER);
    return r ? reinterpret_cast<const char*>(r) : "Unknown";
}
```

- [ ] **Step 2: Write the OpenGL backend**

Create `backends/opengl/src/gl_profiler_backend.cpp`:

```cpp
#include <vex/core/profiler.h>
#include <vex/core/log.h>

#include <glad/glad.h>

#include <vector>

namespace vex
{
namespace
{

// GL_TIME_ELAPSED queries cannot nest, so this uses glQueryCounter with
// GL_TIMESTAMP instead. That maps 1:1 onto vkCmdWriteTimestamp and supports
// arbitrary nesting.
class GLProfilerBackend final : public IProfilerBackend
{
public:
    bool init() override
    {
        GLint bits = 0;
        glGetQueryiv(GL_TIMESTAMP, GL_QUERY_COUNTER_BITS, &bits);
        m_supported = bits > 0;
        if (!m_supported)
        {
            Log::warn("Profiler: GL_TIMESTAMP counter has zero bits");
            return true;
        }

        m_queries.resize(k_queriesPerSlot * Profiler::k_ringSlots);
        glGenQueries(static_cast<GLsizei>(m_queries.size()), m_queries.data());

        m_labels = glPushDebugGroup != nullptr && glPopDebugGroup != nullptr;
        return true;
    }

    void destroy() override
    {
        if (!m_queries.empty())
        {
            glDeleteQueries(static_cast<GLsizei>(m_queries.size()), m_queries.data());
            m_queries.clear();
        }
    }

    bool  supportsTimestamps() const override { return m_supported; }
    float ticksToMs() const override          { return 1e-6f; } // GL reports ns

    // OpenGL query objects are reused directly, so there is nothing to reset.
    void resetQueries(uint32_t, uint32_t) override {}

    void writeTimestamp(uint32_t slot, uint32_t query) override
    {
        if (!m_supported) return;
        glQueryCounter(m_queries[slot * k_queriesPerSlot + query], GL_TIMESTAMP);
    }

    bool resolve(uint32_t slot, uint32_t queryCount,
                 std::vector<uint64_t>& out) override
    {
        if (!m_supported || queryCount == 0) return false;

        const uint32_t base = slot * k_queriesPerSlot;

        // Query objects complete in submission order, so the last one being
        // available implies every earlier one in this slot is too.
        GLint available = 0;
        glGetQueryObjectiv(m_queries[base + queryCount - 1],
                           GL_QUERY_RESULT_AVAILABLE, &available);
        if (!available) return false;

        out.resize(queryCount);
        for (uint32_t i = 0; i < queryCount; ++i)
        {
            GLuint64 value = 0;
            glGetQueryObjectui64v(m_queries[base + i], GL_QUERY_RESULT, &value);
            out[i] = static_cast<uint64_t>(value);
        }
        return true;
    }

    void pushLabel(const char* name) override
    {
        if (!m_labels) return;
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, name);
    }

    void popLabel() override
    {
        if (!m_labels) return;
        glPopDebugGroup();
    }

private:
    static constexpr uint32_t k_queriesPerSlot = Profiler::k_maxZones * 2;

    std::vector<GLuint> m_queries;
    bool m_supported = false;
    bool m_labels    = false;
};

} // namespace

std::unique_ptr<IProfilerBackend> IProfilerBackend::create()
{
    return std::make_unique<GLProfilerBackend>();
}

} // namespace vex
```

Add to `backends/opengl/CMakeLists.txt` after line 9:

```cmake
    src/gl_profiler_backend.cpp
```

- [ ] **Step 3: Build and verify**

Run:
```powershell
cmake --preset opengl-release
cmake --build build-gl --config Release
```
Expected: builds clean.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: add OpenGL profiler backend using glQueryCounter"
```

---

### Task 4: Wire the profiler into the frame loop

**Files:**
- Modify: `app/src/app.cpp:54` (init), and `App::shutdown`
- Modify: `app/src/scene_renderer.cpp:745` (`renderScene` begin/end and root zone), `:626` (shadow), `:774` (geometry rebuild), `:781` (material rebuild), `:794` (outline), `:808` (frame changes)

**Interfaces:**
- Consumes: `vex::Profiler::get()`, `vex::IProfilerBackend::create()` from Tasks 1-3.
- Produces: a live root zone named `"Frame"` present in `Profiler::results()` every frame.

- [ ] **Step 1: Initialise and shut down the profiler**

In `app/src/app.cpp`, add `#include <vex/core/profiler.h>` and insert after the engine init at line 54-58:

```cpp
    if (config.headless)
        return true;

    vex::Profiler::get().init(vex::IProfilerBackend::create());
```

The profiler must be initialised after the graphics context exists, because both backends query device properties in `init()`.

In `App::shutdown`, before the engine shuts down:

```cpp
    vex::Profiler::get().shutdown();
```

- [ ] **Step 2: Add frame boundaries and the root zone**

In `app/src/scene_renderer.cpp`, add `#include <vex/core/profiler.h>`, then wrap `renderScene`. Two ordering rules apply:

1. `beginFrame()` must be the very first statement, so no framebuffer is bound when the Vulkan backend issues `vkCmdResetQueryPool`.
2. The root `"Frame"` zone must live in its own explicit `{ }` block. Declaring the RAII object directly in the function body would destroy it at the closing brace, which runs *after* `endFrame()`, and the zone would never be recorded.

```cpp
void SceneRenderer::renderScene(Scene& scene, int selectedNodeIdx, int selectedSubmesh)
{
    vex::Profiler::get().beginFrame();
    {
        VEX_GPU_ZONE("Frame");

        // ... entire existing body unchanged ...
    }
    vex::Profiler::get().endFrame();
}
```

- [ ] **Step 3: Add the shared pass zones**

In the same file, inside the `"Frame"` scope:

At line 774, wrap the geometry rebuild so its cost is reported as a one-shot rather than poisoning the frame average:

```cpp
    if (scene.geometryDirty)
    {
        vex::Log::info("Building scene geometry (geometry changed)");
        const auto t0 = std::chrono::steady_clock::now();
        rebuildRaytraceGeometry(scene, nullptr);
        vex::Profiler::get().recordOneShot("Geometry rebuild",
            std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - t0).count());
        scene.geometryDirty = false;
        scene.materialDirty = false;
        m_shadowMapDirty    = true;
    }
    else if (scene.materialDirty)
    {
        VEX_CPU_ZONE("Material rebuild");
        rebuildMaterials(scene);
        scene.materialDirty = false;
    }
```

Add `#include <chrono>` if it is not already present.

At line 787, wrap the outline mask block:

```cpp
    {
        VEX_GPU_ZONE("Outline mask");
        m_outlineActive = (selectedNodeIdx >= 0
                        && selectedNodeIdx < static_cast<int>(scene.nodes.size()));
        // ... rest of the existing block unchanged ...
    }
```

At line 808, wrap the frame-change computation:

```cpp
    FrameChanges changes;
    {
        VEX_CPU_ZONE("Frame changes");
        changes = computeFrameChanges(scene);
    }
```

At the top of `SceneRenderer::renderShadowPrePass` (line 626), add as the first statement:

```cpp
    VEX_GPU_ZONE("Shadow prepass");
```

- [ ] **Step 4: Build and run both backends**

Run:
```powershell
cmake --build build-vk --config Release
cmake --build build-gl --config Release
.\build-vk\bin\Release\vex_app.exe
```
Expected: the app runs normally with no validation errors. There is no UI for the data yet; verify by attaching a debugger and inspecting `vex::Profiler::get().results()`, or temporarily add `ImGui::Text("%zu zones", vex::Profiler::get().results().size());` to the stats panel and confirm it reports 4 or more.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: wire profiler into SceneRenderer frame loop with shared pass zones"
```

---

### Task 5: Instrument the four render modes

**Files:**
- Modify: `app/src/render_mode_rasterize.cpp` (skybox, mesh loop, bloom, composite, pick)
- Modify: `app/src/render_mode_cpu_raytrace.cpp` (trace, readback, upload, bloom, composite)
- Modify: `app/src/render_mode_gpu_raytrace.cpp` (already partly done in Task 2; add the GL compute path)
- Modify: `app/src/render_mode_compute_raytrace.cpp` (dispatch, bloom, composite)

**Interfaces:**
- Consumes: `VEX_GPU_ZONE`, `VEX_CPU_ZONE` from Task 1.
- Produces: the complete zone set listed in the spec's instrumentation map.

- [ ] **Step 1: Instrument the rasterizer**

Add `#include <vex/core/profiler.h>` to `app/src/render_mode_rasterize.cpp`. In `renderWithSelection`, wrap:

- the skybox draw with `VEX_GPU_ZONE("Raster: skybox");`
- the node/submesh draw loop (around line 250 through 320) with `VEX_GPU_ZONE("Raster: meshes");`
- the bloom threshold pass with `VEX_GPU_ZONE("Bloom: threshold");`
- the ping-pong blur loop with `VEX_GPU_ZONE("Bloom: blur");`
- the fullscreen tone-map blit with `VEX_GPU_ZONE("Composite");`

In the pick pass (around line 563), add `VEX_GPU_ZONE("Pick pass");` as the first statement of the enclosing block.

Each zone needs its own `{ }` scope where one does not already exist, so the RAII object ends where the pass ends rather than at the end of the function.

- [ ] **Step 2: Instrument the CPU path tracer**

Add the include to `app/src/render_mode_cpu_raytrace.cpp` and wrap:

```cpp
    {
        VEX_CPU_ZONE("CPU PT: trace");
        m_cpuRaytracer->traceSample();
    }
    {
        VEX_CPU_ZONE("CPU PT: readback");
        m_cpuRaytracer->getLinearHDR(m_hdrScratch);
    }
    {
        VEX_GPU_ZONE("CPU PT: upload");
        shared.cpuAccumTex->setData(/* existing arguments */);
    }
```

Use the existing local variable names rather than the placeholders above. Then wrap the bloom and composite passes with `VEX_GPU_ZONE("Bloom: threshold")`, `VEX_GPU_ZONE("Bloom: blur")`, and `VEX_GPU_ZONE("Composite")` exactly as in the rasterizer.

- [ ] **Step 3: Finish the GPU raytrace mode**

Task 2 already converted the Vulkan path. Add the same three zones to the OpenGL branch of `app/src/render_mode_gpu_raytrace.cpp` (the `GLGPURaytracer` path), and split the single `"Bloom"` zone into `"Bloom: threshold"` and `"Bloom: blur"` so all four modes report the same zone names.

- [ ] **Step 4: Instrument the Vulkan compute path tracer**

In `app/src/render_mode_compute_raytrace.cpp`, add the include and wrap:

```cpp
    {
        VEX_GPU_ZONE("Compute dispatch");
        m_vkComputeRaytracer->traceSample(cmd);
        m_vkComputeRaytracer->postTraceBarrier(cmd);
    }
```

plus `"Bloom: threshold"`, `"Bloom: blur"`, and `"Composite"`.

- [ ] **Step 5: Verify zone counts in every mode**

Build both backends, run the app, and switch through all render modes available on that backend. Temporarily log the zone count once per second:

```cpp
    vex::Log::info("zones: " + std::to_string(vex::Profiler::get().results().size()));
```

Expected: Rasterize reports at least 8 zones, GPU Raytrace at least 7, CPU Raytrace at least 8, Compute Raytrace at least 7. Remove the temporary logging before committing.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: instrument all four render modes with profiler zones"
```

---

### Task 6: One-shot zones for import and acceleration structure builds

**Files:**
- Modify: `app/src/scene_geometry_cache.cpp:56` (`rebuild`), `:537` (`buildAccelerationStructures`)
- Modify: `app/src/scene_renderer.cpp` (`triggerDenoise`, `triggerDenoiseAux`)

**Interfaces:**
- Consumes: `vex::Profiler::recordOneShot(const char*, float)` from Task 1.
- Produces: one-shot entries named `Triangle flatten`, `BVH build`, `Texture load`, `VK SSBO pack`, `BLAS build`, `TLAS build`, `Denoise (OIDN)`.

- [ ] **Step 1: Add a small scoped helper**

The existing code already measures several of these stages with `std::chrono` for its `Log::info` output. Rather than duplicating that, add a local RAII helper at the top of `app/src/scene_geometry_cache.cpp`:

```cpp
namespace
{
struct OneShotTimer
{
    explicit OneShotTimer(const char* name)
        : m_name(name), m_start(std::chrono::steady_clock::now()) {}
    ~OneShotTimer()
    {
        vex::Profiler::get().recordOneShot(m_name,
            std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - m_start).count());
    }
    const char* m_name;
    std::chrono::steady_clock::time_point m_start;
};
} // namespace
```

- [ ] **Step 2: Bracket the rebuild stages**

In `SceneGeometryCache::rebuild`, wrap each existing stage in its own scope with `OneShotTimer timer("Triangle flatten");`, `OneShotTimer timer("BVH build");`, `OneShotTimer timer("Texture load");`, and `OneShotTimer timer("VK SSBO pack");`. Use the stage boundaries that the existing progress callbacks already delimit so the names match what the loading overlay shows.

In `SceneGeometryCache::buildAccelerationStructures`, add `OneShotTimer timer("BLAS build");` and `OneShotTimer timer("TLAS build");` around the corresponding loops.

- [ ] **Step 3: Bracket the denoiser**

In `SceneRenderer::triggerDenoise` and `SceneRenderer::triggerDenoiseAux`, add the same pattern with the name `"Denoise (OIDN)"`, guarded by the existing `#ifdef VEX_HAS_OIDN`.

- [ ] **Step 4: Verify**

Build both backends, run the app, import an OBJ, and confirm `vex::Profiler::get().oneShots()` is non-empty. Verify with a temporary log line, then remove it.

Expected: at least 4 entries after an import on OpenGL, at least 6 on Vulkan.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: record import and acceleration structure build costs as one-shot zones"
```

---

### Task 7: Editor Profiler window

**Files:**
- Create: `app/src/editor_ui_profiler.cpp`
- Modify: `app/src/editor_ui.h` (declare `renderProfiler`, add state members)
- Modify: `app/src/app.cpp` (call `renderProfiler` alongside the other panels)
- Modify: `app/CMakeLists.txt:16`

**Interfaces:**
- Consumes: `vex::Profiler::results()`, `oneShots()`, `frameGpuMs()`, `frameCpuMs()`, `gpuTimingAvailable()`, `GraphicsContext::deviceName()`.
- Produces: `EditorUI::renderProfiler(vex::GraphicsContext& ctx)`.

- [ ] **Step 1: Declare the entry point and state**

In `app/src/editor_ui.h`, add to the public section after `renderStats`:

```cpp
    void renderProfiler(vex::GraphicsContext& ctx);
```

Add to the private section:

```cpp
    struct ZoneAccum { float ema = -1.0f; float peak = 0.0f; };
    std::unordered_map<std::string, ZoneAccum> m_profAccum;
    float m_profHistory[256] = {};
    int   m_profHistoryPos   = 0;
    bool  m_profPaused       = false;
    bool  m_profShowCPU      = false;
    bool  m_profShowOneShot  = false;
```

Add `#include <unordered_map>` to the header includes.

- [ ] **Step 2: Write the window**

Create `app/src/editor_ui_profiler.cpp`:

```cpp
#include "editor_ui.h"

#include <vex/core/profiler.h>
#include <vex/graphics/graphics_context.h>

#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <string>

void EditorUI::renderProfiler(vex::GraphicsContext& ctx)
{
    ImGui::Begin("Profiler");

    auto&       prof    = vex::Profiler::get();
    const auto& results = prof.results();

    const float gpuMs = prof.frameGpuMs();
    const float cpuMs = prof.frameCpuMs();

    if (!m_profPaused && gpuMs >= 0.0f)
    {
        m_profHistory[m_profHistoryPos] = gpuMs;
        m_profHistoryPos = (m_profHistoryPos + 1) % IM_ARRAYSIZE(m_profHistory);
    }

    ImGui::TextDisabled("%s", ctx.deviceName().c_str());

    if (gpuMs >= 0.0f)
        ImGui::Text("GPU %.2f ms", gpuMs);
    else
        ImGui::TextDisabled("GPU timing unavailable on this device");
    ImGui::SameLine();
    ImGui::Text("  CPU %.2f ms", cpuMs < 0.0f ? 0.0f : cpuMs);
    ImGui::SameLine();
    ImGui::Text("  %.0f FPS", ImGui::GetIO().Framerate);

    ImGui::Checkbox("Pause", &m_profPaused);
    ImGui::SameLine();
    if (ImGui::Button("Reset peaks"))
        m_profAccum.clear();

    float scaleMax = 1.0f;
    for (float v : m_profHistory) scaleMax = std::max(scaleMax, v);
    ImGui::PlotLines("##frametime", m_profHistory, IM_ARRAYSIZE(m_profHistory),
                     m_profHistoryPos, nullptr, 0.0f, scaleMax * 1.1f,
                     ImVec2(-1.0f, 60.0f));

    ImGui::Separator();

    const float frameGpu = (gpuMs > 0.0f) ? gpuMs : 1.0f;

    if (ImGui::BeginTable("passes", 5,
                          ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg))
    {
        ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthStretch, 2.0f);
        ImGui::TableSetupColumn("last");
        ImGui::TableSetupColumn("avg");
        ImGui::TableSetupColumn("max");
        ImGui::TableSetupColumn("%");
        ImGui::TableHeadersRow();

        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Child zones may sum to more than their parent: "
                              "begin timestamps use TOP_OF_PIPE and end "
                              "timestamps use BOTTOM_OF_PIPE, so a zone can "
                              "absorb time from work still draining ahead of it.");

        for (const auto& r : results)
        {
            const float value = (r.gpuMs >= 0.0f) ? r.gpuMs : r.cpuMs;
            if (value < 0.0f) continue;

            auto& acc = m_profAccum[r.name ? r.name : "?"];
            if (!m_profPaused)
            {
                acc.ema  = (acc.ema < 0.0f) ? value : (acc.ema * 0.95f + value * 0.05f);
                acc.peak = std::max(acc.peak, value);
            }

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%*s%s", r.depth * 2, "", r.name ? r.name : "?");
            ImGui::TableNextColumn(); ImGui::Text("%6.2f", value);
            ImGui::TableNextColumn(); ImGui::Text("%6.2f", acc.ema);
            ImGui::TableNextColumn(); ImGui::Text("%6.2f", acc.peak);
            ImGui::TableNextColumn(); ImGui::Text("%5.0f", 100.0f * value / frameGpu);
        }
        ImGui::EndTable();
    }

    ImGui::Separator();

    if (ImGui::CollapsingHeader("One-shot costs"))
    {
        for (const auto& s : prof.oneShots())
            ImGui::Text("  %-20s %8.1f ms", s.name.c_str(), s.ms);
        if (prof.oneShots().empty())
            ImGui::TextDisabled("  none recorded yet");
        if (ImGui::Button("Clear one-shots"))
            prof.clearOneShots();
    }

    if (ImGui::Button("Copy as Markdown"))
    {
        std::string md = "| Pass | last (ms) | avg (ms) | max (ms) | % |\n";
        md += "|---|---|---|---|---|\n";
        char line[256];
        for (const auto& r : results)
        {
            const float value = (r.gpuMs >= 0.0f) ? r.gpuMs : r.cpuMs;
            if (value < 0.0f) continue;
            const auto& acc = m_profAccum[r.name ? r.name : "?"];
            std::snprintf(line, sizeof(line), "| %*s%s | %.2f | %.2f | %.2f | %.0f |\n",
                          r.depth * 2, "", r.name ? r.name : "?",
                          value, acc.ema, acc.peak, 100.0f * value / frameGpu);
            md += line;
        }
        ImGui::SetClipboardText(md.c_str());
    }

    ImGui::End();
}
```

Add `src/editor_ui_profiler.cpp` to `app/CMakeLists.txt` after line 16.

- [ ] **Step 3: Call it from the frame loop**

In `app/src/app.cpp`, next to the other `m_ui.render*` calls, add:

```cpp
    m_ui.renderProfiler(m_engine.getGraphicsContext());
```

- [ ] **Step 4: Verify visually**

Run:
```powershell
cmake --build build-vk --config Release
.\build-vk\bin\Release\vex_app.exe
```

Expected, checked by eye:
1. The Profiler window appears, is dockable, and shows the GPU name.
2. With vsync off, the `Frame` row tracks the FPS readout (a 6 ms frame at about 160 FPS).
3. Switching render modes changes the row set.
4. `Pause` freezes the numbers and the plot.
5. `Reset peaks` zeroes the max column.
6. `Copy as Markdown` puts a valid Markdown table on the clipboard.
7. Repeat on the OpenGL build.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add dockable Profiler window with frame plot and markdown export"
```

---

### Task 8: Benchmark configuration, camera path, and statistics

Pure logic with no engine dependencies, so it is added to the test target and covered directly.

**Files:**
- Create: `app/src/benchmark_config.h`
- Create: `app/src/benchmark_config.cpp`
- Create: `tests/test_benchmark_config.cpp`
- Modify: `app/CMakeLists.txt` (add `src/benchmark_config.cpp`)
- Modify: `tests/CMakeLists.txt` (add the test and `${CMAKE_SOURCE_DIR}/app/src/benchmark_config.cpp`)

**Interfaces:**
- Consumes: nothing from earlier tasks. Uses `nlohmann::json` from `engine/include/json.hpp` and GLM, both already public on `vex_core`.
- Produces: `BenchCamKey { glm::vec3 target; float distance; float yaw; float pitch; }`, `BenchmarkConfig`, `ProfileStats { float mean, min, max, p50, p95, p99, stddev; }`, `std::optional<BenchmarkConfig> parseBenchmarkConfig(const std::string&, std::string&)`, `BenchCamKey interpolateCamera(const std::vector<BenchCamKey>&, float)`, `ProfileStats computeStats(std::vector<float>)`, `std::string csvEscape(const std::string&)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmark_config.cpp`:

```cpp
#include <doctest/doctest.h>
#include "benchmark_config.h"

TEST_CASE("parse minimal config applies defaults")
{
    std::string err;
    auto cfg = parseBenchmarkConfig(R"({"scene":"a.obj"})", err);
    REQUIRE(cfg.has_value());
    CHECK(cfg->scenePath == "a.obj");
    CHECK(cfg->mode == "rasterize");
    CHECK(cfg->width == 1920);
    CHECK(cfg->height == 1080);
    CHECK(cfg->warmupFrames == 30);
    CHECK(cfg->measureFrames == 300);
    CHECK(cfg->maxSamples == 0);
    CHECK(cfg->vsync == false);
    CHECK(cfg->orbitDegrees == doctest::Approx(0.0f));
    CHECK(cfg->camera.empty());
}

TEST_CASE("parse full config reads every field")
{
    std::string err;
    auto cfg = parseBenchmarkConfig(R"({
        "name": "run1", "scene": "b.gltf", "sceneFormat": "gltf",
        "mode": "gpu_raytrace", "width": 2560, "height": 1440,
        "warmupFrames": 5, "measureFrames": 50, "maxSamples": 64,
        "vsync": true, "orbitDegrees": 90.0,
        "camera": [
            {"target":[1,2,3],"distance":4,"yaw":0.5,"pitch":-0.25},
            {"target":[4,5,6],"distance":7,"yaw":1.5,"pitch":-0.75}
        ]
    })", err);
    REQUIRE(cfg.has_value());
    CHECK(cfg->name == "run1");
    CHECK(cfg->sceneFormat == "gltf");
    CHECK(cfg->mode == "gpu_raytrace");
    CHECK(cfg->width == 2560);
    CHECK(cfg->maxSamples == 64);
    CHECK(cfg->vsync == true);
    CHECK(cfg->orbitDegrees == doctest::Approx(90.0f));
    REQUIRE(cfg->camera.size() == 2);
    CHECK(cfg->camera[0].target.y == doctest::Approx(2.0f));
    CHECK(cfg->camera[1].distance == doctest::Approx(7.0f));
}

TEST_CASE("parse rejects missing scene and malformed json")
{
    std::string err;
    CHECK_FALSE(parseBenchmarkConfig(R"({"mode":"rasterize"})", err).has_value());
    CHECK_FALSE(err.empty());

    err.clear();
    CHECK_FALSE(parseBenchmarkConfig("{not json", err).has_value());
    CHECK_FALSE(err.empty());
}

TEST_CASE("camera interpolation handles 0, 1, 2 and 3 keyframes")
{
    CHECK(interpolateCamera({}, 0.5f).distance == doctest::Approx(10.0f));

    BenchCamKey only;
    only.distance = 42.0f;
    CHECK(interpolateCamera({only}, 0.0f).distance == doctest::Approx(42.0f));
    CHECK(interpolateCamera({only}, 1.0f).distance == doctest::Approx(42.0f));

    BenchCamKey a; a.distance = 0.0f; a.yaw = 0.0f;
    BenchCamKey b; b.distance = 10.0f; b.yaw = 2.0f;
    CHECK(interpolateCamera({a, b}, 0.0f).distance == doctest::Approx(0.0f));
    CHECK(interpolateCamera({a, b}, 1.0f).distance == doctest::Approx(10.0f));
    CHECK(interpolateCamera({a, b}, 0.5f).distance == doctest::Approx(5.0f));
    CHECK(interpolateCamera({a, b}, 0.25f).yaw == doctest::Approx(0.5f));

    BenchCamKey c; c.distance = 20.0f;
    CHECK(interpolateCamera({a, b, c}, 0.5f).distance == doctest::Approx(10.0f));
    CHECK(interpolateCamera({a, b, c}, 1.0f).distance == doctest::Approx(20.0f));
    CHECK(interpolateCamera({a, b, c}, 0.25f).distance == doctest::Approx(5.0f));
}

TEST_CASE("statistics handle empty, single and small sample sets")
{
    ProfileStats empty = computeStats({});
    CHECK(empty.mean == doctest::Approx(0.0f));
    CHECK(empty.p99 == doctest::Approx(0.0f));

    ProfileStats one = computeStats({5.0f});
    CHECK(one.mean == doctest::Approx(5.0f));
    CHECK(one.min == doctest::Approx(5.0f));
    CHECK(one.max == doctest::Approx(5.0f));
    CHECK(one.p50 == doctest::Approx(5.0f));
    CHECK(one.p99 == doctest::Approx(5.0f));
    CHECK(one.stddev == doctest::Approx(0.0f));

    ProfileStats two = computeStats({2.0f, 4.0f});
    CHECK(two.mean == doctest::Approx(3.0f));
    CHECK(two.p50 == doctest::Approx(2.0f));  // nearest-rank: ceil(0.5*2)-1 = 0
    CHECK(two.p95 == doctest::Approx(4.0f));
    CHECK(two.p99 == doctest::Approx(4.0f));
}

TEST_CASE("statistics compute percentiles by nearest rank")
{
    std::vector<float> v;
    for (int i = 1; i <= 100; ++i) v.push_back(static_cast<float>(i));

    ProfileStats s = computeStats(v);
    CHECK(s.mean == doctest::Approx(50.5f));
    CHECK(s.min == doctest::Approx(1.0f));
    CHECK(s.max == doctest::Approx(100.0f));
    CHECK(s.p50 == doctest::Approx(50.0f));
    CHECK(s.p95 == doctest::Approx(95.0f));
    CHECK(s.p99 == doctest::Approx(99.0f));
}

TEST_CASE("csv escaping quotes fields containing separators")
{
    CHECK(csvEscape("plain") == "plain");
    CHECK(csvEscape("has,comma") == "\"has,comma\"");
    CHECK(csvEscape("has\"quote") == "\"has\"\"quote\"");
    CHECK(csvEscape("has\nnewline") == "\"has\nnewline\"");
}
```

- [ ] **Step 2: Add to the test build and confirm it fails**

In `tests/CMakeLists.txt`:

```cmake
add_executable(vex_tests
    test_main.cpp
    test_bvh.cpp
    test_bsdf.cpp
    test_primitives.cpp
    test_camera.cpp
    test_raytracer.cpp
    test_profiler.cpp
    test_benchmark_config.cpp
    ${CMAKE_SOURCE_DIR}/app/src/benchmark_config.cpp
)

target_include_directories(vex_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/external/doctest
    ${CMAKE_SOURCE_DIR}/app/src
)
```

Run:
```powershell
cmake -S . -B build_tests -DVEX_BUILD_TESTS=ON -DVEX_BUILD_APP=OFF -DVEX_BACKEND=OpenGL
cmake --build build_tests --config Release --target vex_tests
```
Expected: FAIL, `cannot open include file: 'benchmark_config.h'`.

- [ ] **Step 3: Write the header**

Create `app/src/benchmark_config.h`:

```cpp
#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct BenchCamKey
{
    glm::vec3 target   = glm::vec3(0.0f);
    float     distance = 10.0f;
    float     yaw      = 0.0f;
    float     pitch    = 0.0f;
};

struct BenchmarkConfig
{
    std::string name        = "bench";
    std::string scenePath;
    std::string sceneFormat;              // "obj" or "gltf"; empty infers from extension
    std::string mode        = "rasterize";
    uint32_t    width         = 1920;
    uint32_t    height        = 1080;
    uint32_t    warmupFrames  = 30;
    uint32_t    measureFrames = 300;
    uint32_t    maxSamples    = 0;        // 0 = unlimited
    bool        vsync         = false;
    // When camera is empty the runner auto-frames the scene. A non-zero
    // orbitDegrees then generates a second keyframe rotated by that much yaw,
    // which keeps shipped configs independent of any particular scene.
    float       orbitDegrees  = 0.0f;
    std::vector<BenchCamKey> camera;
};

struct ProfileStats
{
    float mean   = 0.0f;
    float min    = 0.0f;
    float max    = 0.0f;
    float p50    = 0.0f;
    float p95    = 0.0f;
    float p99    = 0.0f;
    float stddev = 0.0f;
};

std::optional<BenchmarkConfig> parseBenchmarkConfig(const std::string& jsonText,
                                                    std::string& outError);

BenchCamKey  interpolateCamera(const std::vector<BenchCamKey>& keys, float t);
ProfileStats computeStats(std::vector<float> samples);
std::string  csvEscape(const std::string& field);
```

- [ ] **Step 4: Write the implementation**

Create `app/src/benchmark_config.cpp`:

```cpp
#include "benchmark_config.h"

#include <json.hpp>

#include <algorithm>
#include <cmath>

namespace
{

float percentile(const std::vector<float>& sorted, float p)
{
    if (sorted.empty()) return 0.0f;
    const auto n    = static_cast<float>(sorted.size());
    const auto rank = static_cast<size_t>(std::ceil(p * n));
    const size_t idx = (rank == 0) ? 0 : rank - 1;
    return sorted[std::min(idx, sorted.size() - 1)];
}

BenchCamKey readKey(const nlohmann::json& j)
{
    BenchCamKey k;
    if (j.contains("target") && j["target"].is_array() && j["target"].size() == 3)
        k.target = glm::vec3(j["target"][0].get<float>(),
                             j["target"][1].get<float>(),
                             j["target"][2].get<float>());
    if (j.contains("distance")) k.distance = j["distance"].get<float>();
    if (j.contains("yaw"))      k.yaw      = j["yaw"].get<float>();
    if (j.contains("pitch"))    k.pitch    = j["pitch"].get<float>();
    return k;
}

} // namespace

std::optional<BenchmarkConfig> parseBenchmarkConfig(const std::string& jsonText,
                                                    std::string& outError)
{
    nlohmann::json j;
    try
    {
        j = nlohmann::json::parse(jsonText);
    }
    catch (const std::exception& e)
    {
        outError = std::string("JSON parse error: ") + e.what();
        return std::nullopt;
    }

    if (!j.contains("scene") || !j["scene"].is_string() ||
        j["scene"].get<std::string>().empty())
    {
        outError = "benchmark config is missing a non-empty \"scene\" field";
        return std::nullopt;
    }

    BenchmarkConfig c;
    c.scenePath = j["scene"].get<std::string>();

    if (j.contains("name"))          c.name          = j["name"].get<std::string>();
    if (j.contains("sceneFormat"))   c.sceneFormat   = j["sceneFormat"].get<std::string>();
    if (j.contains("mode"))          c.mode          = j["mode"].get<std::string>();
    if (j.contains("width"))         c.width         = j["width"].get<uint32_t>();
    if (j.contains("height"))        c.height        = j["height"].get<uint32_t>();
    if (j.contains("warmupFrames"))  c.warmupFrames  = j["warmupFrames"].get<uint32_t>();
    if (j.contains("measureFrames")) c.measureFrames = j["measureFrames"].get<uint32_t>();
    if (j.contains("maxSamples"))    c.maxSamples    = j["maxSamples"].get<uint32_t>();
    if (j.contains("vsync"))         c.vsync         = j["vsync"].get<bool>();
    if (j.contains("orbitDegrees"))  c.orbitDegrees  = j["orbitDegrees"].get<float>();

    if (j.contains("camera") && j["camera"].is_array())
        for (const auto& k : j["camera"])
            c.camera.push_back(readKey(k));

    return c;
}

BenchCamKey interpolateCamera(const std::vector<BenchCamKey>& keys, float t)
{
    if (keys.empty())  return BenchCamKey{};
    if (keys.size() == 1) return keys[0];

    t = std::clamp(t, 0.0f, 1.0f);
    const float scaled = t * static_cast<float>(keys.size() - 1);
    const size_t i     = std::min(static_cast<size_t>(scaled), keys.size() - 2);
    const float  f     = scaled - static_cast<float>(i);

    const BenchCamKey& a = keys[i];
    const BenchCamKey& b = keys[i + 1];

    BenchCamKey out;
    out.target   = a.target   + (b.target   - a.target)   * f;
    out.distance = a.distance + (b.distance - a.distance) * f;
    out.yaw      = a.yaw      + (b.yaw      - a.yaw)      * f;
    out.pitch    = a.pitch    + (b.pitch    - a.pitch)    * f;
    return out;
}

ProfileStats computeStats(std::vector<float> samples)
{
    ProfileStats s;
    if (samples.empty()) return s;

    std::sort(samples.begin(), samples.end());

    double sum = 0.0;
    for (float v : samples) sum += v;
    s.mean = static_cast<float>(sum / static_cast<double>(samples.size()));

    s.min = samples.front();
    s.max = samples.back();
    s.p50 = percentile(samples, 0.50f);
    s.p95 = percentile(samples, 0.95f);
    s.p99 = percentile(samples, 0.99f);

    double var = 0.0;
    for (float v : samples)
    {
        const double d = static_cast<double>(v) - s.mean;
        var += d * d;
    }
    s.stddev = static_cast<float>(std::sqrt(var / static_cast<double>(samples.size())));

    return s;
}

std::string csvEscape(const std::string& field)
{
    const bool needsQuotes = field.find_first_of(",\"\n\r") != std::string::npos;
    if (!needsQuotes) return field;

    std::string out = "\"";
    for (char ch : field)
    {
        if (ch == '"') out += "\"\"";
        else           out += ch;
    }
    out += "\"";
    return out;
}
```

Add `src/benchmark_config.cpp` to `app/CMakeLists.txt` after line 19 (`src/command.cpp`).

- [ ] **Step 5: Run the tests**

Run:
```powershell
cmake --build build_tests --config Release --target vex_tests
.\build_tests\Release\vex_tests.exe
```
Expected: PASS, all 7 benchmark-config cases green plus the 9 profiler cases plus the pre-existing suites.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add benchmark config parsing, camera path, and statistics with tests"
```

---

### Task 9: Benchmark runner and CLI wiring

**Files:**
- Create: `app/src/benchmark.h`
- Create: `app/src/benchmark.cpp`
- Modify: `app/src/main.cpp:9-46` (argument parsing)
- Modify: `app/src/app.h:13` (`init` signature, runner member), `app/src/app.cpp:50-60` (scene selection), `App::run` (drive the runner)
- Modify: `engine/include/vex/core/engine.h:39`, `engine/src/core/engine.cpp` (add `requestExit`)
- Modify: `app/CMakeLists.txt`

**Interfaces:**
- Consumes: `BenchmarkConfig`, `parseBenchmarkConfig`, `interpolateCamera`, `computeStats`, `csvEscape` from Task 8; `vex::Profiler` from Task 1; `SceneRenderer::saveImage` (`scene_renderer.cpp:304`); `GraphicsContext::deviceName()` from Tasks 2-3.
- Produces: `struct AppConfig { vex::EngineConfig engine; std::string benchConfigPath; std::string benchOutDir; }`, `class BenchmarkRunner`, `vex::Engine::requestExit()`.

- [ ] **Step 1: Add `Engine::requestExit`**

In `engine/include/vex/core/engine.h`, after `bool isRunning() const;`:

```cpp
    void requestExit() { m_running = false; }
```

- [ ] **Step 2: Write the runner header**

Create `app/src/benchmark.h`:

```cpp
#pragma once

#include "benchmark_config.h"

#include <string>
#include <unordered_map>
#include <vector>

struct Scene;
class  SceneRenderer;
namespace vex { class GraphicsContext; }

class BenchmarkRunner
{
public:
    // Reads and parses the config. Logs and returns false on any failure.
    bool loadFromFile(const std::string& path, const std::string& outDirOverride);

    const BenchmarkConfig& config() const { return m_cfg; }

    // Called once the scene is loaded, to capture the auto-framed camera pose.
    void begin(const Scene& scene, int rootNodeIdx);

    // Called at the top of each main-loop iteration, before renderScene.
    // Applies the camera pose for this frame. Returns false when the run is over.
    bool tick(Scene& scene);

    // Called after renderScene, to sample the profiler for this frame.
    void sample();

    // Called once after the final tick. Writes CSV, PNG, and run.json.
    void finish(SceneRenderer& renderer, vex::GraphicsContext& ctx);

private:
    enum class Phase { Warmup, Measure, Done };

    BenchmarkConfig m_cfg;
    std::string     m_outDir;
    Phase           m_phase      = Phase::Warmup;
    uint32_t        m_frame      = 0;
    BenchCamKey     m_autoFrame;                 // pose derived from scene bounds
    std::vector<BenchCamKey> m_path;             // resolved keyframes

    std::vector<std::string>                     m_columns;   // frozen at first measured frame
    std::vector<std::vector<float>>              m_rows;      // one inner vector per frame
    std::vector<float>                           m_cpuMs;
    std::vector<float>                           m_gpuMs;
};
```

- [ ] **Step 3: Write the runner**

Create `app/src/benchmark.cpp`. The key behaviours, in order:

`loadFromFile` reads the file into a string, calls `parseBenchmarkConfig`, logs `outError` and returns false on failure, and sets `m_outDir` to `outDirOverride` when non-empty or `"results/" + m_cfg.name` otherwise.

`begin` stores the auto-framed pose from the scene's root node (mirroring `App::runImport`'s focus logic at `app.cpp:377-386`: target is the world-space node center, distance is `radius * 2.5f`) and resolves `m_path`:

```cpp
    if (!m_cfg.camera.empty())
    {
        m_path = m_cfg.camera;
    }
    else
    {
        m_path.push_back(m_autoFrame);
        if (m_cfg.orbitDegrees != 0.0f)
        {
            BenchCamKey second = m_autoFrame;
            second.yaw += glm::radians(m_cfg.orbitDegrees);
            m_path.push_back(second);
        }
    }
```

`tick` applies the camera and advances the phase:

```cpp
bool BenchmarkRunner::tick(Scene& scene)
{
    if (m_phase == Phase::Done) return false;

    float t = 0.0f;
    if (m_phase == Phase::Measure && m_cfg.measureFrames > 1)
        t = static_cast<float>(m_frame) / static_cast<float>(m_cfg.measureFrames - 1);

    const BenchCamKey key = interpolateCamera(m_path, t);
    scene.camera.setOrbit(key.target, key.distance, key.yaw, key.pitch);

    return true;
}
```

`sample` records the frame, and is where the phase actually advances so that only frames that were rendered are counted:

```cpp
void BenchmarkRunner::sample()
{
    auto& prof = vex::Profiler::get();

    if (m_phase == Phase::Warmup)
    {
        if (++m_frame >= m_cfg.warmupFrames)
        {
            m_frame = 0;
            m_phase = Phase::Measure;
        }
        return;
    }

    if (m_phase != Phase::Measure) return;

    const auto& results = prof.results();

    // Freeze the column set on the first measured frame. A zone appearing
    // later is ignored; a zone disappearing writes an empty cell.
    if (m_columns.empty())
        for (const auto& r : results)
            m_columns.push_back(r.name ? r.name : "?");

    std::vector<float> row(m_columns.size(), -1.0f);
    for (const auto& r : results)
    {
        const std::string name = r.name ? r.name : "?";
        for (size_t c = 0; c < m_columns.size(); ++c)
            if (m_columns[c] == name)
            {
                row[c] = (r.gpuMs >= 0.0f) ? r.gpuMs : r.cpuMs;
                break;
            }
    }

    m_rows.push_back(std::move(row));
    m_cpuMs.push_back(prof.frameCpuMs());
    m_gpuMs.push_back(prof.frameGpuMs());

    if (++m_frame >= m_cfg.measureFrames)
        m_phase = Phase::Done;
}
```

`finish` creates `m_outDir` with `std::filesystem::create_directories`, then writes:

- `frames.csv`: header `frame,cpuMs,gpuMs,` followed by `csvEscape(column)` for each zone; one row per entry in `m_rows`, writing an empty cell where the value is negative.
- `summary.csv`: header `zone,mean,min,max,p50,p95,p99,stddev`; a row for `frame_cpu` from `computeStats(m_cpuMs)`, a row for `frame_gpu` from `computeStats(m_gpuMs)`, then one row per column built by gathering that column's non-negative values.
- `final.png` via `renderer.saveImage(m_outDir + "/final.png")`.
- `run.json` containing `name`, `device` (`ctx.deviceName()`), `backend` (`std::string(ctx.backendName())`), `width`, `height`, `mode`, `scene`, `measureFrames`, `maxSamples`, and the `frame_gpu` summary, written with `nlohmann::json` and `dump(2)`.

It also logs a summary table through `vex::Log::info` and prints it to `stdout`, one line per zone with mean, p95, and p99.

- [ ] **Step 4: Wire up the CLI**

In `app/src/main.cpp`, replace `parseArgs` with a version returning `AppConfig` (declared in `app.h`):

```cpp
static AppConfig parseArgs(int argc, char* argv[])
{
    AppConfig config;

    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i)
    {
        if (args[i] == "--headless")
            config.engine.headless = true;
        else if (args[i] == "--width" && i + 1 < args.size())
            config.engine.windowWidth = static_cast<uint32_t>(std::stoi(args[++i]));
        else if (args[i] == "--height" && i + 1 < args.size())
            config.engine.windowHeight = static_cast<uint32_t>(std::stoi(args[++i]));
        else if (args[i] == "--bench" && i + 1 < args.size())
            config.benchConfigPath = args[++i];
        else if (args[i] == "--bench-out" && i + 1 < args.size())
            config.benchOutDir = args[++i];
        else if (args[i] == "--help")
        {
            std::cout << "Usage: vex_app [options]\n"
                      << "  --headless          Run without a window\n"
                      << "  --width <W>         Window width (default 1280)\n"
                      << "  --height <H>        Window height (default 720)\n"
                      << "  --bench <file>      Run a benchmark from a JSON config and exit\n"
                      << "  --bench-out <dir>   Override the benchmark output directory\n";
            std::exit(0);
        }
    }

    return config;
}
```

`main` becomes:

```cpp
int main(int argc, char* argv[])
{
    App app;

    if (!app.init(parseArgs(argc, argv)))
        return EXIT_FAILURE;

    app.run();
    app.shutdown();

    return app.exitCode();
}
```

In `app/src/app.h`, add above `struct App`:

```cpp
struct AppConfig
{
    vex::EngineConfig engine;
    std::string       benchConfigPath;
    std::string       benchOutDir;
};
```

Change `bool init(const vex::EngineConfig& config);` to `bool init(const AppConfig& config);`, add `int exitCode() const { return m_exitCode; }`, and add the private members `BenchmarkRunner m_bench;`, `bool m_benchActive = false;`, `int m_exitCode = 0;`.

In `App::init`, when `config.benchConfigPath` is non-empty: load the runner config, override `config.engine.windowWidth/Height` from it before `m_engine.init`, set vsync from it after init, skip the default ChessSet import at `app.cpp:60`, import the benchmark scene instead, set the render mode from `m_cfg.mode`, call `m_bench.begin(...)`, and set `m_benchActive = true`. On any failure set `m_exitCode = 1` and return false.

In `App::run`, at the top of the loop:

```cpp
        if (m_benchActive && !m_bench.tick(m_scene))
        {
            m_bench.finish(m_renderer, m_engine.getGraphicsContext());
            m_engine.requestExit();
            break;
        }
```

and immediately after the `m_renderer.renderScene(...)` call:

```cpp
        if (m_benchActive)
            m_bench.sample();
```

Guard every `m_ui.render*` call with `if (!m_benchActive)`.

Add `src/benchmark.cpp` to `app/CMakeLists.txt`.

- [ ] **Step 5: Verify a real run**

Create `bench/chessset-raster-1080p.json`:

```json
{
  "name": "chessset-raster-1080p",
  "scene": "VexAssetsCC0/Scenes/ChessSet/ChessSet.obj",
  "mode": "rasterize",
  "width": 1920,
  "height": 1080,
  "warmupFrames": 30,
  "measureFrames": 300,
  "vsync": false,
  "orbitDegrees": 90.0
}
```

Run:
```powershell
cmake --build build-vk --config Release
.\build-vk\bin\Release\vex_app.exe --bench bench/chessset-raster-1080p.json
```

Expected:
1. The app opens, loads the scene, orbits, and exits on its own with code 0.
2. `results/chessset-raster-1080p/` contains `frames.csv` with 301 lines (header plus 300 rows), `summary.csv`, `final.png`, and `run.json`.
3. `final.png` shows the chess set, not a black frame.
4. `run.json` names the correct GPU.
5. A second run produces a mean within a few percent of the first.
6. `vex_app.exe --bench does-not-exist.json` prints an error and exits with code 1.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add JSON-driven benchmark mode emitting CSV, PNG, and run metadata"
```

---

### Task 10: Ship benchmark configs and publish numbers

**Files:**
- Create: `bench/chessset-raster-1080p.json` (from Task 9 Step 5), `bench/chessset-cpurt-1080p.json`, `bench/chessset-gpurt-1080p.json`, `bench/chessset-gpurt-converge.json`
- Modify: `README.md:8` (a Performance section)
- Modify: `.gitignore` (ignore `results/`)

**Interfaces:**
- Consumes: everything above.
- Produces: committed benchmark configs and a README performance table.

- [ ] **Step 1: Write the remaining configs**

`bench/chessset-cpurt-1080p.json` and `bench/chessset-gpurt-1080p.json` are copies of the raster config with `"mode"` set to `"cpu_raytrace"` and `"gpu_raytrace"`, `"measureFrames"` reduced to `60` for the CPU tracer, and matching `"name"` fields.

`bench/chessset-gpurt-converge.json` measures convergence instead of throughput, so it omits `orbitDegrees` (static camera) and sets `"maxSamples": 512`:

```json
{
  "name": "chessset-gpurt-converge",
  "scene": "VexAssetsCC0/Scenes/ChessSet/ChessSet.obj",
  "mode": "gpu_raytrace",
  "width": 1920,
  "height": 1080,
  "warmupFrames": 10,
  "measureFrames": 512,
  "maxSamples": 512,
  "vsync": false
}
```

- [ ] **Step 2: Ignore the results directory**

Add to `.gitignore`:

```
results/
```

- [ ] **Step 3: Run every config on both backends**

```powershell
cmake --build build-vk --config Release
.\build-vk\bin\Release\vex_app.exe --bench bench/chessset-raster-1080p.json
.\build-vk\bin\Release\vex_app.exe --bench bench/chessset-cpurt-1080p.json
.\build-vk\bin\Release\vex_app.exe --bench bench/chessset-gpurt-1080p.json
.\build-vk\bin\Release\vex_app.exe --bench bench/chessset-gpurt-converge.json

cmake --build build-gl --config Release
.\build-gl\bin\Release\vex_app.exe --bench bench/chessset-raster-1080p.json --bench-out results/gl-raster
.\build-gl\bin\Release\vex_app.exe --bench bench/chessset-cpurt-1080p.json --bench-out results/gl-cpurt
.\build-gl\bin\Release\vex_app.exe --bench bench/chessset-gpurt-1080p.json --bench-out results/gl-gpurt
```

Expected: every run completes and writes a full output set. `compute_raytrace` is Vulkan-only and is deliberately not in the config set; confirm that adding `"mode":"compute_raytrace"` to a config and running it on the OpenGL build exits with code 1 and a clear error.

- [ ] **Step 4: Add the README performance section**

Insert after the feature list in `README.md`, filling the table from the generated `summary.csv` files. Do not invent numbers; copy them from the actual runs.

```markdown
## Performance

Measured with the built-in benchmark mode. Every number below is reproducible:

```sh
vex_app --bench bench/chessset-gpurt-1080p.json
```

Scene: ChessSet, 1920x1080, VSync off. GPU: <copy from run.json>.

| Mode | Backend | mean (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|
| Rasterize | Vulkan | | | |
| Rasterize | OpenGL | | | |
| Path trace (HW RT) | Vulkan | | | |
| Path trace (compute) | Vulkan | | | |
| Path trace (compute) | OpenGL | | | |
| Path trace (CPU) | both | | | |

Per-pass breakdown for the Vulkan rasterizer, exported from the Profiler
window's "Copy as Markdown" button:

<paste the exported table here>

Configs live in `bench/`. Each run writes `frames.csv`, `summary.csv`,
`final.png`, and a `run.json` with the GPU name, driver, and git commit to
`results/<name>/`.
```

- [ ] **Step 5: Capture an RGP screenshot**

Take an RGP capture of `vex_app.exe` running `bench/chessset-gpurt-1080p.json` on the Vulkan build. Confirm the pass names (`Frame`, `Shadow prepass`, `RT dispatch`, `Bloom: threshold`, `Bloom: blur`, `Composite`) appear as labelled regions. Save a screenshot to `renders/rgp_capture.png` and reference it from the README performance section.

This is the manual verification that the debug-utils work from Task 2 actually landed.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "docs: add benchmark configs and measured performance numbers to README"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Profiler core, ring buffer, zone stack | 1 |
| `IProfilerBackend` interface | 1 |
| Vulkan backend, TOP/BOTTOM_OF_PIPE, debug utils in release | 2 |
| `GpuTimer` removal | 2 |
| OpenGL backend, `glQueryCounter`, `glPushDebugGroup` | 3 |
| `GraphicsContext::deviceName()` | 2 (base + VK), 3 (GL) |
| `beginFrame` outside render pass | 4 |
| Instrumentation map, shared passes | 4 |
| Instrumentation map, per-mode passes | 5 |
| One-shot zones | 6 |
| Editor Profiler window, EMA, running max, plot, markdown export | 7 |
| Benchmark config, camera semantics, statistics, CSV escaping | 8 |
| Benchmark runner, CLI, outputs, `run.json` | 9 |
| Shipped configs, README numbers, RGP verification | 10 |
| Error handling table | 1 (overflow, unbalanced, unavailable), 2 and 3 (backend unavailable, labels unavailable), 8 (config parse), 9 (missing file, unsupported mode) |
| Test plan | 1 and 8 (unit), 4, 5, 6, 7, 9, 10 (manual checklist steps) |

No gaps found.

**Deviations from the spec, deliberate:**

- The spec's config example did not include `orbitDegrees`. It was added in Task 8 so the shipped configs do not have to hard-code camera coordinates for a scene, which would make them break the moment the scene changes. The spec's explicit `camera` array still works and takes precedence.
- The spec listed `Frame changes` and `Material rebuild` as CPU zones and `Geometry rebuild` as one-shot. Task 4 implements exactly that.

**Type consistency check:** `Profiler::k_maxZones` and `k_ringSlots` are referenced by both backends and both test files with the same spelling. `ProfileZoneResult` fields (`name`, `depth`, `gpuMs`, `cpuMs`) are used identically in Tasks 1, 7, and 9. `BenchCamKey` fields (`target`, `distance`, `yaw`, `pitch`) match between Tasks 8 and 9, and match `vex::Camera::setOrbit(target, distance, yaw, pitch)`. `ProfileStats` fields match between the test, the implementation, and the `summary.csv` header. `IProfilerBackend`'s six methods have identical signatures in the header, the mock, and both backends.
