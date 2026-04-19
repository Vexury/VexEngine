#pragma once

#include <volk.h>
#include <cstdint>

namespace vex {

struct GpuPassTimings
{
    float rtDispatchMs = -1.f;
    float bloomMs      = -1.f;
    float compositeMs  = -1.f;
};

enum class GpuTimerSlot : uint32_t
{
    RtDispatch = 0,
    Bloom      = 1,
    Composite  = 2,
    Count      = 3,
};

// Double-buffered GPU timestamp query helper.
// One query pool per frame-in-flight slot avoids reading results that may still be
// in-flight.  Call beginFrame() at the start of the VK render() call (after the
// context fence has been waited), then bracket each pass with begin()/end().
class GpuTimer
{
public:
    void init(VkDevice device, VkPhysicalDevice physDevice);
    void destroy();

    // Reads results from the pool belonging to the current frame slot (safe
    // because the context fence for this slot was waited before render() runs),
    // then resets it via a command so it's ready for new timestamps this frame.
    void beginFrame(VkCommandBuffer cmd);

    void begin(VkCommandBuffer cmd, GpuTimerSlot slot);
    void end(VkCommandBuffer cmd,   GpuTimerSlot slot);

    const GpuPassTimings& getTimings() const { return m_timings; }

private:
    static constexpr uint32_t k_slotCount  = static_cast<uint32_t>(GpuTimerSlot::Count);
    static constexpr uint32_t k_frameCount = 2; // matches MAX_FRAMES_IN_FLIGHT

    VkDevice    m_device         = VK_NULL_HANDLE;
    VkQueryPool m_pools[k_frameCount] = {};
    float       m_ticksToMs      = 1.f; // timestampPeriod / 1e6
    GpuPassTimings m_timings;
};

} // namespace vex
