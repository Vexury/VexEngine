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
