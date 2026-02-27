#pragma once

#include <vex/graphics/graphics_context.h>

#include <volk.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include <vector>
#include <cstdint>

namespace vex
{

static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

struct FrameData
{
    VkCommandPool   commandPool    = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer  = VK_NULL_HANDLE;
    VkSemaphore     imageAvailable = VK_NULL_HANDLE; // signaled when swapchain image is ready to render into
    VkFence         inFlightFence  = VK_NULL_HANDLE;
};

class VKContext : public GraphicsContext
{
public:
    bool init(Window& window) override;
    void shutdown() override;
    void beginFrame() override;
    void endFrame() override;
    std::string_view backendName() const override { return "Vulkan"; }
    std::function<void()> getWindowHints() const override;
    MemoryStats getMemoryStats() const override;
    void waitIdle() override;

    void imguiInit(GLFWwindow* window) override;
    void imguiShutdown() override;
    void imguiNewFrame() override;
    void imguiRenderDrawData() override;

    // Accessors for other VK classes
    VkDevice       getDevice()       const { return m_device; }
    VkPhysicalDevice getPhysicalDevice() const { return m_physicalDevice; }
    VmaAllocator   getAllocator()     const { return m_allocator; }
    VkQueue        getGraphicsQueue() const { return m_graphicsQueue; }
    uint32_t       getGraphicsQueueFamily() const { return m_graphicsQueueFamily; }
    VkInstance     getInstance()      const { return m_instance; }

    VkCommandBuffer getCurrentCommandBuffer() const;
    VkRenderPass    getSwapchainRenderPass() const { return m_swapchainRenderPass; }
    VkExtent2D      getSwapchainExtent() const { return m_swapchainExtent; }
    VkFormat        getSwapchainFormat() const { return m_swapchainFormat; }
    uint32_t        getCurrentFrameIndex() const { return m_currentFrame; }

    // Begin/end the swapchain render pass (for ImGui)
    void beginSwapchainRenderPass();
    void endSwapchainRenderPass();

    // Submit a one-shot command buffer (for uploads)
    void immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function);

    // Batch upload: accumulate buffer copies and flush in one submit.
    // Reduces N fence waits to 1 for bulk mesh imports.
    void beginBatchUpload();
    void endBatchUpload();
    bool isBatchingUploads() const { return m_batchingUploads; }
    void deferCopy(VkBuffer src, VkBuffer dst, VkDeviceSize size, VmaAllocation stagingAlloc);

    // RT pipeline properties (shaderGroupHandleSize, alignment, maxRecursionDepth, etc.)
    const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& getRTProperties() const { return m_rtProperties; }

    // Singleton access for Vulkan resource classes
    static VKContext& get();

private:
    void createSwapchain();
    void destroySwapchain();
    void recreateSwapchain();
    void createFrameData();
    void destroyFrameData();
    void createRenderFinishedSemaphores();
    void destroyRenderFinishedSemaphores();

    Window* m_window = nullptr;

    // Core Vulkan objects
    vkb::Instance   m_vkbInstance{};
    VkInstance       m_instance       = VK_NULL_HANDLE;
    VkSurfaceKHR    m_surface        = VK_NULL_HANDLE;
    vkb::Device     m_vkbDevice{};
    VkDevice        m_device         = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VmaAllocator    m_allocator      = VK_NULL_HANDLE;

    // Queue
    VkQueue  m_graphicsQueue       = VK_NULL_HANDLE;
    uint32_t m_graphicsQueueFamily = 0;

    // Swapchain
    vkb::Swapchain          m_vkbSwapchain{};
    VkSwapchainKHR          m_swapchain       = VK_NULL_HANDLE;
    VkFormat                m_swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D              m_swapchainExtent{};
    std::vector<VkImage>     m_swapchainImages;
    std::vector<VkImageView> m_swapchainImageViews;
    std::vector<VkFramebuffer> m_swapchainFramebuffers;

    // Swapchain render pass (for ImGui)
    VkRenderPass m_swapchainRenderPass = VK_NULL_HANDLE;

    // Per-frame data
    FrameData m_frames[MAX_FRAMES_IN_FLIGHT]{};
    uint32_t  m_currentFrame = 0;
    uint32_t  m_swapchainImageIndex = 0;

    // One semaphore per swapchain image — reused only when that image is re-acquired
    std::vector<VkSemaphore> m_renderFinishedSemaphores;

    // Immediate submit resources
    VkCommandPool   m_uploadPool   = VK_NULL_HANDLE;
    VkCommandBuffer m_uploadBuffer = VK_NULL_HANDLE;
    VkFence         m_uploadFence  = VK_NULL_HANDLE;

    // Batch upload state
    struct DeferredCopy { VkBuffer src, dst; VkDeviceSize size; VmaAllocation stagingAlloc; };
    bool                     m_batchingUploads = false;
    std::vector<DeferredCopy> m_deferredCopies;

    // ImGui
    VkDescriptorPool m_imguiPool = VK_NULL_HANDLE;

    // RT pipeline properties — queried once after device creation, used for SBT layout
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{};

    static VKContext* s_instance;
};

} // namespace vex
