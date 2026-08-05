#pragma once

#include "benchmark_config.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct Scene;
class  SceneRenderer;
enum class RenderMode;
namespace vex { class GraphicsContext; }

// Resolves a config mode string to a RenderMode. Logs and returns false when the
// string is unknown or the mode is not available on the compiled backend.
bool parseBenchRenderMode(const std::string& name, RenderMode& out);

class BenchmarkRunner
{
public:
    // Reads and parses the config. Logs and returns false on any failure.
    bool loadFromFile(const std::string& path, const std::string& outDirOverride);

    const BenchmarkConfig& config() const { return m_cfg; }

    // Called once the scene is loaded, to capture the auto-framed camera pose.
    void begin(const Scene& scene, int rootNodeIdx);

    // Far plane that covers every resolved keyframe. Valid after begin().
    float farPlane() const { return m_farPlane; }

    // Called at the top of each main-loop iteration, before renderScene.
    // Applies the camera pose for this frame. Returns false when the run is over.
    bool tick(Scene& scene);

    // Called after renderScene, to sample the profiler for this frame.
    void sample();

    // Called once after the final tick. Writes CSV, PNG, and run.json.
    void finish(SceneRenderer& renderer, vex::GraphicsContext& ctx);

    // True once every requested frame has been measured, i.e. finish() ran.
    bool completed() const { return m_phase == Phase::Done; }

    // True when too many measured frames repeated the previous frame's profiler
    // results verbatim, which means the statistics describe copies of one
    // sample rather than a measurement. Valid after the run.
    bool stale() const;

    // Logs that the main loop ended before the run finished, so an empty output
    // directory cannot be mistaken for a completed run.
    void reportAborted() const;

private:
    enum class Phase { Warmup, Measure, Done };

    // Fraction of frame-to-frame comparisons that may repeat before the run is
    // rejected. Every healthy run recorded so far (eight runs, 1892 frames,
    // both backends, all four modes) produced zero repeats, because an exact
    // match of a whole multi-zone row of floats does not happen by chance. A
    // genuinely stalled query slot can legitimately repeat the previous result
    // for an isolated frame, so the bar is set well above any plausible
    // transient and far below the total freeze this guard exists to catch.
    static constexpr double k_maxDuplicateFraction = 0.25;

    BenchmarkConfig m_cfg;
    std::string     m_outDir;
    Phase           m_phase      = Phase::Warmup;
    uint32_t        m_frame      = 0;
    float           m_farPlane   = 1000.0f;
    BenchCamKey     m_autoFrame;                 // pose derived from scene bounds
    std::vector<BenchCamKey> m_path;             // resolved keyframes

    std::vector<std::string>                     m_columns;   // frozen at first measured frame
    std::vector<std::vector<float>>              m_rows;      // one inner vector per frame
    std::vector<float>                           m_cpuMs;
    std::vector<float>                           m_gpuMs;
    size_t                                       m_duplicateFrames = 0;
};
