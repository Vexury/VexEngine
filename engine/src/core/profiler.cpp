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
