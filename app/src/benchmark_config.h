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

// Fraction of frame-to-frame comparisons that may repeat before a run is
// rejected. Every healthy run recorded so far (eight runs, 1892 frames, both
// backends, all four modes) produced zero repeats, because an exact match of a
// whole multi-zone row of floats does not happen by chance. A genuinely stalled
// query slot can legitimately repeat the previous result for an isolated frame,
// so the bar sits well above any plausible transient and far below the total
// freeze this guard exists to catch (the observed failures were 99.7 percent).
inline constexpr double k_maxDuplicateFraction = 0.25;

std::optional<BenchmarkConfig> parseBenchmarkConfig(const std::string& jsonText,
                                                    std::string& outError);

BenchCamKey  interpolateCamera(const std::vector<BenchCamKey>& keys, float t);
ProfileStats computeStats(std::vector<float> samples);
std::string  csvEscape(const std::string& field);

// True when a measured frame's entire recorded state matches the previous
// frame's bit for bit, which means the profiler handed back the same resolved
// result vector twice rather than producing a new measurement. Exact float
// comparison is deliberate: the failure being detected is literal repetition,
// not similarity, so anything approximate would flag genuinely distinct samples.
bool benchFrameIsDuplicate(const std::vector<float>& series, float frameCpuMs, float frameGpuMs,
                           const std::vector<float>& prevSeries, float prevCpuMs,
                           float prevGpuMs);

// Rejection verdict for a whole run. The denominator is comparisons, not
// frames, since N recorded frames yield N-1 comparisons. A run of fewer than
// two frames is never rejected, because there is nothing to compare.
bool benchRunIsStale(size_t measuredFrames, size_t duplicateFrames);
