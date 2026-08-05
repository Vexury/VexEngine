#include "benchmark.h"

#include "scene.h"
#include "scene_renderer.h"

#include <vex/core/log.h>
#include <vex/core/profiler.h>
#include <vex/graphics/graphics_context.h>

#include <json.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace
{

std::string fmt(float v)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4f", v);
    return buf;
}

std::string readFileText(const std::string& path, std::string& outError)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        outError = "cannot open benchmark config: " + path;
        return {};
    }

    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

nlohmann::json statsToJson(const ProfileStats& s)
{
    return nlohmann::json{
        {"mean",   s.mean},
        {"min",    s.min},
        {"max",    s.max},
        {"p50",    s.p50},
        {"p95",    s.p95},
        {"p99",    s.p99},
        {"stddev", s.stddev},
    };
}

void writeStatsRow(std::ofstream& out, const std::string& zone, const ProfileStats& s,
                   const char* status)
{
    out << csvEscape(zone) << ','
        << fmt(s.mean)   << ',' << fmt(s.min) << ',' << fmt(s.max) << ','
        << fmt(s.p50)    << ',' << fmt(s.p95) << ',' << fmt(s.p99) << ','
        << fmt(s.stddev) << ',' << status << '\n';
}

void logLine(const std::string& line)
{
    vex::Log::info(line);
    std::cout << line << '\n';
}

void errLine(const std::string& line)
{
    vex::Log::error(line);
    std::cout << line << '\n';
}

} // namespace

bool parseBenchRenderMode(const std::string& name, RenderMode& out)
{
    if (name == "rasterize")     { out = RenderMode::Rasterize;   return true; }
    if (name == "cpu_raytrace")  { out = RenderMode::CPURaytrace; return true; }
    if (name == "gpu_raytrace")  { out = RenderMode::GPURaytrace; return true; }
    if (name == "compute_raytrace")
    {
#ifdef VEX_BACKEND_VULKAN
        out = RenderMode::ComputeRaytrace;
        return true;
#else
        vex::Log::error("benchmark mode \"compute_raytrace\" requires the Vulkan backend");
        return false;
#endif
    }

    vex::Log::error("unknown benchmark mode \"" + name +
                    "\"; expected rasterize, cpu_raytrace, gpu_raytrace or compute_raytrace");
    return false;
}

bool BenchmarkRunner::loadFromFile(const std::string& path, const std::string& outDirOverride)
{
    std::string error;
    const std::string text = readFileText(path, error);
    if (!error.empty())
    {
        vex::Log::error(error);
        return false;
    }

    auto parsed = parseBenchmarkConfig(text, error);
    if (!parsed)
    {
        vex::Log::error(error);
        return false;
    }

    m_cfg = std::move(*parsed);

    if (m_cfg.measureFrames == 0)
    {
        vex::Log::warn("benchmark measureFrames is 0, clamping to 1");
        m_cfg.measureFrames = 1;
    }

    if (m_cfg.width == 0 || m_cfg.height == 0)
    {
        vex::Log::error("benchmark width and height must both be non-zero");
        return false;
    }

    m_outDir = outDirOverride.empty() ? ("results/" + m_cfg.name) : outDirOverride;

    if (m_cfg.warmupFrames == 0)
        m_phase = Phase::Measure;

    return true;
}

void BenchmarkRunner::begin(const Scene& scene, int rootNodeIdx)
{
    float radius = 1.0f;

    if (rootNodeIdx >= 0 && rootNodeIdx < static_cast<int>(scene.nodes.size()))
    {
        const auto& node = scene.nodes[rootNodeIdx];
        m_autoFrame.target =
            glm::vec3(scene.getWorldMatrix(rootNodeIdx) * glm::vec4(node.center, 1.0f));
        radius = node.radius;
    }

    m_autoFrame.distance = radius * 2.5f;
    m_autoFrame.yaw      = 0.0f;
    m_autoFrame.pitch    = 0.15f;

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

    float maxDist = 0.0f;
    for (const auto& k : m_path)
        maxDist = std::max(maxDist, k.distance);
    m_farPlane = std::max(100.0f, maxDist + radius * 2.0f);
}

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

    // A frame the profiler has no results for yet carries no measurement at all.
    // Recording it would push an all-empty row, and consecutive empty rows
    // compare equal, so with warmupFrames 0 the first frames of a run would
    // count themselves as duplicates and a short run could reject itself. Such a
    // frame is simply not a sample, so it is not recorded.
    //
    // The frame counter still advances. The profiler needs k_ringSlots frames
    // before its first resolve lands, and a backend whose resolve never succeeds
    // would otherwise keep this loop running forever. A run that ends with no
    // samples at all is rejected by benchRunIsStale instead.
    if (results.empty())
    {
        if (++m_frame >= m_cfg.measureFrames)
            m_phase = Phase::Done;
        return;
    }

    // Freeze the column set on the first measured frame that has any results.
    // The profiler resolves through a three-slot query ring, so the very first
    // frames after a mode change can still be empty. A zone appearing later is
    // ignored; a zone disappearing writes an empty cell.
    if (m_columns.empty())
        for (const auto& r : results)
            m_columns.push_back(r.name ? r.name : "?");

    // Two series per zone, gpu then cpu, each carrying only its own value. A
    // single column picking whichever of the two happens to be available hides
    // the CPU cost of a GPU zone whose GPU time is legitimately near zero (an
    // upload wrapped in VEX_GPU_ZONE), and lets a column change unit mid-run
    // whenever a query resolve is not ready.
    std::vector<float> row(m_columns.size() * 2, -1.0f);
    for (const auto& r : results)
    {
        const std::string name = r.name ? r.name : "?";
        for (size_t c = 0; c < m_columns.size(); ++c)
            if (m_columns[c] == name)
            {
                row[c * 2]     = r.gpuMs;
                row[c * 2 + 1] = r.cpuMs;
                break;
            }
    }

    const float frameCpu = prof.frameCpuMs();
    const float frameGpu = prof.frameGpuMs();

    // A frame whose entire result set matches the previous frame's bit for bit
    // carries no new measurement: the profiler handed back the same resolved
    // vector twice, which happens when a query slot never becomes ready and the
    // previous results are kept.
    if (!m_rows.empty() &&
        benchFrameIsDuplicate(row, frameCpu, frameGpu,
                              m_rows.back(), m_cpuMs.back(), m_gpuMs.back()))
        ++m_duplicateFrames;

    m_rows.push_back(std::move(row));
    m_cpuMs.push_back(frameCpu);
    m_gpuMs.push_back(frameGpu);

    if (++m_frame >= m_cfg.measureFrames)
        m_phase = Phase::Done;

    if (m_frame % 100 == 0 && m_phase == Phase::Measure)
        vex::Log::info("Benchmark: " + std::to_string(m_frame) + " / " +
                       std::to_string(m_cfg.measureFrames) + " frames");
}

bool BenchmarkRunner::stale() const
{
    return benchRunIsStale(m_rows.size(), m_duplicateFrames);
}

void BenchmarkRunner::finish(SceneRenderer& renderer, vex::GraphicsContext& ctx)
{
    std::error_code ec;
    std::filesystem::create_directories(m_outDir, ec);
    if (ec)
    {
        vex::Log::error("Benchmark: cannot create output directory " + m_outDir +
                        ": " + ec.message());
        return;
    }

    // Every row of every CSV carries the run's verdict in a trailing "status"
    // column. A rejected run's numbers are therefore marked on the same line a
    // reader would quote them from, and the marker survives copying a single row
    // out of the file, which a rejected output directory name would not. "ok" is
    // written on healthy runs too, so the column's presence proves the guard ran
    // rather than leaving its absence ambiguous.
    const bool  isStale = stale();
    const char* status  = isStale ? "REJECTED" : "ok";

    // frames.csv
    {
        std::ofstream out(m_outDir + "/frames.csv", std::ios::binary);
        if (!out)
        {
            vex::Log::error("Benchmark: cannot write " + m_outDir + "/frames.csv");
        }
        else
        {
            out << "frame,cpuMs,gpuMs";
            for (const auto& c : m_columns)
                out << ',' << csvEscape(c + " gpu") << ',' << csvEscape(c + " cpu");
            out << ",status\n";

            const size_t seriesCount = m_columns.size() * 2;
            for (size_t i = 0; i < m_rows.size(); ++i)
            {
                out << i << ',';
                if (m_cpuMs[i] >= 0.0f) out << fmt(m_cpuMs[i]);
                out << ',';
                if (m_gpuMs[i] >= 0.0f) out << fmt(m_gpuMs[i]);

                const auto& row = m_rows[i];
                for (size_t s = 0; s < seriesCount; ++s)
                {
                    out << ',';
                    if (s < row.size() && row[s] >= 0.0f)
                        out << fmt(row[s]);
                }
                out << ',' << status << '\n';
            }
        }
    }

    // Series index: zone c contributes c*2 (gpu) and c*2+1 (cpu).
    auto gather = [this](size_t series)
    {
        std::vector<float> v;
        v.reserve(m_rows.size());
        for (const auto& row : m_rows)
            if (series < row.size() && row[series] >= 0.0f)
                v.push_back(row[series]);
        return v;
    };

    auto positives = [](const std::vector<float>& src)
    {
        std::vector<float> v;
        v.reserve(src.size());
        for (float f : src)
            if (f >= 0.0f) v.push_back(f);
        return v;
    };

    const ProfileStats cpuStats = computeStats(positives(m_cpuMs));
    const ProfileStats gpuStats = computeStats(positives(m_gpuMs));

    // summary.csv
    {
        std::ofstream out(m_outDir + "/summary.csv", std::ios::binary);
        if (!out)
        {
            vex::Log::error("Benchmark: cannot write " + m_outDir + "/summary.csv");
        }
        else
        {
            out << "zone,mean,min,max,p50,p95,p99,stddev,status\n";
            writeStatsRow(out, "frame_cpu", cpuStats, status);
            writeStatsRow(out, "frame_gpu", gpuStats, status);
            // A series with no samples at all is skipped, so a CPU-only zone does
            // not emit an all-empty gpu row.
            for (size_t c = 0; c < m_columns.size(); ++c)
                for (int half = 0; half < 2; ++half)
                {
                    const std::vector<float> v = gather(c * 2 + half);
                    if (v.empty()) continue;
                    writeStatsRow(out, m_columns[c] + (half == 0 ? " gpu" : " cpu"),
                                  computeStats(v), status);
                }
        }
    }

    // final.png
    if (!renderer.saveImage(m_outDir + "/final.png"))
        vex::Log::error("Benchmark: failed to write " + m_outDir + "/final.png");

    // run.json
    {
        nlohmann::json j;
        j["name"]          = m_cfg.name;
        j["device"]        = ctx.deviceName();
        j["backend"]       = std::string(ctx.backendName());
        j["width"]         = m_cfg.width;
        j["height"]        = m_cfg.height;
        j["mode"]          = m_cfg.mode;
        j["scene"]         = m_cfg.scenePath;
        j["warmupFrames"]  = m_cfg.warmupFrames;
        j["measureFrames"] = m_cfg.measureFrames;
        j["maxSamples"]    = m_cfg.maxSamples;
        j["cameraKeys"]    = m_path.size();
        j["frame_gpu"]     = statsToJson(gpuStats);
        j["frame_cpu"]     = statsToJson(cpuStats);

        // Health signal, so the artifact says for itself whether its statistics
        // were computed over distinct samples.
        const size_t comparisons = m_rows.size() > 1 ? m_rows.size() - 1 : 0;
        j["health"] = nlohmann::json{
            {"measuredFrames",    m_rows.size()},
            {"duplicateFrames",   m_duplicateFrames},
            {"duplicateFraction", comparisons ? static_cast<double>(m_duplicateFrames) /
                                                static_cast<double>(comparisons) : 0.0},
            {"stale",             isStale},
            {"status",            std::string(status)},
        };

        std::ofstream out(m_outDir + "/run.json", std::ios::binary);
        if (!out)
            vex::Log::error("Benchmark: cannot write " + m_outDir + "/run.json");
        else
            out << j.dump(2) << '\n';
    }

    // Console summary
    char buf[256];
    logLine("Benchmark '" + m_cfg.name + "' finished: " + std::to_string(m_rows.size()) +
            " frames at " + std::to_string(m_cfg.width) + "x" + std::to_string(m_cfg.height) +
            " on " + ctx.deviceName());

    if (isStale)
    {
        errLine("========================================================================");
        errLine("BENCHMARK RESULT REJECTED: not a measurement");
        if (m_rows.empty())
        {
            errLine("No frame produced any profiler results, so nothing was measured.");
        }
        else
        {
            errLine(std::to_string(m_duplicateFrames) + " of " +
                    std::to_string(m_rows.size() - 1) +
                    " measured frames repeated the previous frame's profiler results exactly.");
            errLine("The profiler was not producing fresh data, so the statistics below are");
            errLine("computed over repeated copies of the same sample. Do not publish them.");
        }
        errLine("Every row of frames.csv and summary.csv is marked status=REJECTED,");
        errLine("run.json records \"health\": {\"stale\": true}, and the process exits 2.");
        errLine("========================================================================");
    }
    std::snprintf(buf, sizeof(buf), "%-28s %10s %10s %10s", "zone", "mean ms", "p95 ms", "p99 ms");
    logLine(buf);
    std::snprintf(buf, sizeof(buf), "%-28s %10.3f %10.3f %10.3f",
                  "frame_gpu", gpuStats.mean, gpuStats.p95, gpuStats.p99);
    logLine(buf);
    std::snprintf(buf, sizeof(buf), "%-28s %10.3f %10.3f %10.3f",
                  "frame_cpu", cpuStats.mean, cpuStats.p95, cpuStats.p99);
    logLine(buf);
    for (size_t c = 0; c < m_columns.size(); ++c)
        for (int half = 0; half < 2; ++half)
        {
            const std::vector<float> v = gather(c * 2 + half);
            if (v.empty()) continue;
            const ProfileStats s = computeStats(v);
            const std::string  n = m_columns[c] + (half == 0 ? " gpu" : " cpu");
            std::snprintf(buf, sizeof(buf), "%-28s %10.3f %10.3f %10.3f",
                          n.c_str(), s.mean, s.p95, s.p99);
            logLine(buf);
        }
    if (isStale)
        errLine("Statistics above are REJECTED, see the duplicate-frame warning.");

    logLine("Benchmark output: " + m_outDir);
}

void BenchmarkRunner::reportAborted() const
{
    errLine("Benchmark '" + m_cfg.name + "' aborted before completion, no results written (" +
            std::to_string(m_rows.size()) + " of " + std::to_string(m_cfg.measureFrames) +
            " frames measured)");
}
