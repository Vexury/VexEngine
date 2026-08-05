#include <vex/core/profiler.h>
#include <vex/core/log.h>
#include <vex/vulkan/vk_context.h>

#include <volk.h>

#include <vector>

namespace vex
{
namespace
{

// The profiler resolves the query slot from k_ringSlots frames back, and the
// only thing that guarantees that slot is finished on Vulkan is the
// frames-in-flight fence: the CPU can never be more than MAX_FRAMES_IN_FLIGHT
// frames ahead of the GPU. If that count ever reached the ring depth, the slot
// being resolved would be one the GPU is still executing, resolve() would
// return false forever, and Profiler would keep handing back the previous
// frame's results. The Profiler window would then freeze at one sample while
// still presenting it as a live measurement, with nothing in the editor UI to
// notice. This is exactly the failure the OpenGL backend shipped with.
static_assert(MAX_FRAMES_IN_FLIGHT < Profiler::k_ringSlots,
              "MAX_FRAMES_IN_FLIGHT must stay below Profiler::k_ringSlots, or the "
              "timestamp slot being resolved is still in flight and the profiler "
              "silently freezes on stale results. Raise Profiler::k_ringSlots too.");

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
