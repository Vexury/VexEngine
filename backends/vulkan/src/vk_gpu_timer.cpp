#include <vex/vulkan/vk_gpu_timer.h>
#include <vex/vulkan/vk_context.h>

#include <cstring>

namespace vex {

void GpuTimer::init(VkDevice device, VkPhysicalDevice physDevice)
{
    m_device = device;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physDevice, &props);
    m_ticksToMs = props.limits.timestampPeriod * 1e-6f;

    VkQueryPoolCreateInfo ci{};
    ci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    ci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    ci.queryCount = k_slotCount * 2; // begin + end per slot

    for (uint32_t i = 0; i < k_frameCount; ++i)
        vkCreateQueryPool(m_device, &ci, nullptr, &m_pools[i]);
}

void GpuTimer::destroy()
{
    for (uint32_t i = 0; i < k_frameCount; ++i)
    {
        if (m_pools[i] != VK_NULL_HANDLE)
        {
            vkDestroyQueryPool(m_device, m_pools[i], nullptr);
            m_pools[i] = VK_NULL_HANDLE;
        }
    }
}

void GpuTimer::beginFrame(VkCommandBuffer cmd)
{
    uint32_t frameIdx = VKContext::get().getCurrentFrameIndex();
    VkQueryPool pool  = m_pools[frameIdx];

    // Read results from this pool — the context fence for this frame slot was
    // already waited on, so GPU writes are complete.
    // Layout per query with WITH_AVAILABILITY: [value u64][avail u64], stride=16.
    constexpr uint32_t queryCount = k_slotCount * 2;
    uint64_t buf[queryCount * 2]{};
    VkResult res = vkGetQueryPoolResults(
        m_device, pool,
        0, queryCount,
        sizeof(buf), buf,
        sizeof(uint64_t) * 2,  // stride: value + availability per query
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);

    if (res == VK_SUCCESS)
    {
        float* dst = &m_timings.rtDispatchMs; // sequential floats match slot order
        for (uint32_t s = 0; s < k_slotCount; ++s)
        {
            uint64_t beginVal   = buf[s * 4 + 0];
            uint64_t beginAvail = buf[s * 4 + 1];
            uint64_t endVal     = buf[s * 4 + 2];
            uint64_t endAvail   = buf[s * 4 + 3];
            if (beginAvail && endAvail)
                dst[s] = static_cast<float>(endVal - beginVal) * m_ticksToMs;
        }
    }

    // Reset the pool so this frame's timestamps start fresh.
    vkCmdResetQueryPool(cmd, pool, 0, queryCount);
}

void GpuTimer::begin(VkCommandBuffer cmd, GpuTimerSlot slot)
{
    uint32_t frameIdx = VKContext::get().getCurrentFrameIndex();
    uint32_t query    = static_cast<uint32_t>(slot) * 2;
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_pools[frameIdx], query);
}

void GpuTimer::end(VkCommandBuffer cmd, GpuTimerSlot slot)
{
    uint32_t frameIdx = VKContext::get().getCurrentFrameIndex();
    uint32_t query    = static_cast<uint32_t>(slot) * 2 + 1;
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_pools[frameIdx], query);
}

} // namespace vex
