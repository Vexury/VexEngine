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
