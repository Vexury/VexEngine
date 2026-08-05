#include "benchmark_config.h"

#include <json.hpp>

#include <algorithm>
#include <cmath>

namespace
{

// p is taken as double, and callers pass double literals: a float32 literal like
// 0.99f already carries representation error, and promoting that to double just
// exposes it at full precision right at the rank boundary (e.g. p99 of 100
// samples). Starting from a double literal keeps that error negligible.
float percentile(const std::vector<float>& sorted, double p)
{
    if (sorted.empty()) return 0.0f;
    const auto n    = static_cast<double>(sorted.size());
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
    // Everything from parsing through field extraction can throw (malformed JSON
    // syntax, or a well-formed document with a wrong-typed field/element access),
    // so the whole body is guarded. This function must never throw out.
    try
    {
        const nlohmann::json j = nlohmann::json::parse(jsonText);

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
    catch (const std::exception& e)
    {
        outError = std::string("benchmark config parse error: ") + e.what();
        return std::nullopt;
    }
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
    s.p50 = percentile(samples, 0.50);
    s.p95 = percentile(samples, 0.95);
    s.p99 = percentile(samples, 0.99);

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

bool benchFrameIsDuplicate(const std::vector<float>& series, float frameCpuMs, float frameGpuMs,
                           const std::vector<float>& prevSeries, float prevCpuMs,
                           float prevGpuMs)
{
    return series == prevSeries && frameCpuMs == prevCpuMs && frameGpuMs == prevGpuMs;
}

bool benchRunIsStale(size_t measuredFrames, size_t duplicateFrames)
{
    // A run that recorded no samples at all is not a measurement either. Its
    // statistics would be computed over an empty set and print as a confident
    // row of zeros, which is the same failure this guard exists to catch.
    if (measuredFrames == 0) return true;
    if (measuredFrames < 2) return false;
    const double comparisons = static_cast<double>(measuredFrames - 1);
    return static_cast<double>(duplicateFrames) >= k_maxDuplicateFraction * comparisons;
}
