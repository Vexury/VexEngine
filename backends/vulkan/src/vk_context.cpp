#include <vex/vulkan/vk_context.h>
#include <vex/core/window.h>
#include <vex/core/log.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <array>

namespace vex
{

VKContext* VKContext::s_instance = nullptr;

VKContext& VKContext::get()
{
    return *s_instance;
}

// Factory
std::unique_ptr<GraphicsContext> GraphicsContext::create()
{
    return std::make_unique<VKContext>();
}

std::function<void()> VKContext::getWindowHints() const
{
    return []()
    {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    };
}

bool VKContext::init(Window& window)
{
    m_window = &window;
    s_instance = this;

    // 1. Initialize volk
    VkResult result = volkInitialize();
    if (result != VK_SUCCESS)
    {
        Log::error("Failed to initialize volk");
        return false;
    }

    // 2. Create instance via vk-bootstrap
    vkb::InstanceBuilder instanceBuilder;
    auto instRet = instanceBuilder
        .set_app_name("VexEngine")
        .set_engine_name("VexEngine")
        .require_api_version(1, 3, 0)
#ifdef VEX_DEBUG
        .request_validation_layers()
        .use_default_debug_messenger()
#endif
        .build();

    if (!instRet)
    {
        Log::error("Failed to create Vulkan instance: " + instRet.error().message());
        return false;
    }

    m_vkbInstance = instRet.value();
    m_instance = m_vkbInstance.instance;

    // 3. Load instance functions
    volkLoadInstance(m_instance);

    // 4. Create surface
    if (glfwCreateWindowSurface(m_instance, window.getNativeWindow(), nullptr, &m_surface) != VK_SUCCESS)
    {
        Log::error("Failed to create window surface");
        return false;
    }

    // 5. Select physical device
    VkPhysicalDeviceFeatures requiredFeatures{};
    requiredFeatures.fillModeNonSolid = VK_TRUE;

    vkb::PhysicalDeviceSelector pdSelector(m_vkbInstance);
    auto pdRet = pdSelector
        .set_surface(m_surface)
        .set_minimum_version(1, 3)
        .set_required_features(requiredFeatures)
        .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
        .add_required_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
        .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
        .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
        .select();

    if (!pdRet)
    {
        Log::error("Failed to select physical device: " + pdRet.error().message());
        return false;
    }

    // 6. Create logical device — chain RT feature structs
    VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bdaFeatures{};
    bdaFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR;
    bdaFeatures.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
    asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asFeatures.accelerationStructure = VK_TRUE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures{};
    rtFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtFeatures.rayTracingPipeline = VK_TRUE;

    vkb::DeviceBuilder deviceBuilder(pdRet.value());
    auto devRet = deviceBuilder
        .add_pNext(&bdaFeatures)
        .add_pNext(&asFeatures)
        .add_pNext(&rtFeatures)
        .build();

    if (!devRet)
    {
        Log::error("Failed to create logical device: " + devRet.error().message());
        return false;
    }

    m_vkbDevice = devRet.value();
    m_device = m_vkbDevice.device;
    m_physicalDevice = pdRet.value().physical_device;

    // 7. Load device functions
    volkLoadDevice(m_device);

    // 8. Get graphics queue
    auto queueRet = m_vkbDevice.get_queue(vkb::QueueType::graphics);
    if (!queueRet)
    {
        Log::error("Failed to get graphics queue");
        return false;
    }
    m_graphicsQueue = queueRet.value();
    m_graphicsQueueFamily = m_vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // 9. Create VMA allocator
    VmaVulkanFunctions vmaFunctions{};
    vmaFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vmaFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocInfo{};
    allocInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    allocInfo.physicalDevice = m_physicalDevice;
    allocInfo.device = m_device;
    allocInfo.instance = m_instance;
    allocInfo.pVulkanFunctions = &vmaFunctions;
    allocInfo.vulkanApiVersion = VK_API_VERSION_1_3;

    if (vmaCreateAllocator(&allocInfo, &m_allocator) != VK_SUCCESS)
    {
        Log::error("Failed to create VMA allocator");
        return false;
    }

    // 10. Create swapchain + per-image semaphores
    createSwapchain();
    createRenderFinishedSemaphores();

    // 11. Create swapchain render pass
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = m_swapchainFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments = &colorAttachment;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(m_device, &rpInfo, nullptr, &m_swapchainRenderPass) != VK_SUCCESS)
    {
        Log::error("Failed to create swapchain render pass");
        return false;
    }

    // 12. Create swapchain framebuffers
    m_swapchainFramebuffers.resize(m_swapchainImageViews.size());
    for (size_t i = 0; i < m_swapchainImageViews.size(); ++i)
    {
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = m_swapchainRenderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = &m_swapchainImageViews[i];
        fbInfo.width = m_swapchainExtent.width;
        fbInfo.height = m_swapchainExtent.height;
        fbInfo.layers = 1;

        if (vkCreateFramebuffer(m_device, &fbInfo, nullptr, &m_swapchainFramebuffers[i]) != VK_SUCCESS)
        {
            Log::error("Failed to create swapchain framebuffer");
            return false;
        }
    }

    // 13. Create per-frame data
    createFrameData();

    // 14. Create immediate submit resources
    VkCommandPoolCreateInfo uploadPoolInfo{};
    uploadPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    uploadPoolInfo.queueFamilyIndex = m_graphicsQueueFamily;
    uploadPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(m_device, &uploadPoolInfo, nullptr, &m_uploadPool);

    VkCommandBufferAllocateInfo uploadAllocInfo{};
    uploadAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    uploadAllocInfo.commandPool = m_uploadPool;
    uploadAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    uploadAllocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(m_device, &uploadAllocInfo, &m_uploadBuffer);

    VkFenceCreateInfo uploadFenceInfo{};
    uploadFenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(m_device, &uploadFenceInfo, nullptr, &m_uploadFence);

    // Query RT pipeline properties (needed for SBT alignment in later steps)
    m_rtProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2);

    Log::info(std::string("Vulkan Device: ") + props2.properties.deviceName);
    Log::info("RT shaderGroupHandleSize:    " + std::to_string(m_rtProperties.shaderGroupHandleSize));
    Log::info("RT shaderGroupBaseAlignment: " + std::to_string(m_rtProperties.shaderGroupBaseAlignment));
    Log::info("RT maxRayRecursionDepth:     " + std::to_string(m_rtProperties.maxRayRecursionDepth));
    Log::info("Vulkan context initialized");

    return true;
}

void VKContext::createSwapchain()
{
    vkb::SwapchainBuilder swapBuilder(m_vkbDevice);
    auto swapRet = swapBuilder
        .set_desired_format({ VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_old_swapchain(m_swapchain)
        .build();

    if (!swapRet)
    {
        Log::error("Failed to create swapchain: " + swapRet.error().message());
        return;
    }

    // Destroy old swapchain if present
    if (m_swapchain != VK_NULL_HANDLE)
    {
        vkb::destroy_swapchain(m_vkbSwapchain);
    }

    m_vkbSwapchain = swapRet.value();
    m_swapchain = m_vkbSwapchain.swapchain;
    m_swapchainFormat = m_vkbSwapchain.image_format;
    m_swapchainExtent = m_vkbSwapchain.extent;
    m_swapchainImages = m_vkbSwapchain.get_images().value();
    m_swapchainImageViews = m_vkbSwapchain.get_image_views().value();
}

void VKContext::destroySwapchain()
{
    for (auto fb : m_swapchainFramebuffers)
        vkDestroyFramebuffer(m_device, fb, nullptr);
    m_swapchainFramebuffers.clear();

    for (auto iv : m_swapchainImageViews)
        vkDestroyImageView(m_device, iv, nullptr);
    m_swapchainImageViews.clear();

    vkb::destroy_swapchain(m_vkbSwapchain);
    m_swapchain = VK_NULL_HANDLE;
}

void VKContext::recreateSwapchain()
{
    // Wait for minimization
    int width = 0, height = 0;
    glfwGetFramebufferSize(m_window->getNativeWindow(), &width, &height);
    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(m_window->getNativeWindow(), &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(m_device);

    destroySwapchain();

    // Recreate swapchain + per-image semaphores
    destroyRenderFinishedSemaphores();
    createSwapchain();
    createRenderFinishedSemaphores();

    // Recreate framebuffers
    m_swapchainFramebuffers.resize(m_swapchainImageViews.size());
    for (size_t i = 0; i < m_swapchainImageViews.size(); ++i)
    {
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = m_swapchainRenderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = &m_swapchainImageViews[i];
        fbInfo.width = m_swapchainExtent.width;
        fbInfo.height = m_swapchainExtent.height;
        fbInfo.layers = 1;

        vkCreateFramebuffer(m_device, &fbInfo, nullptr, &m_swapchainFramebuffers[i]);
    }
}

void VKContext::createFrameData()
{
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = m_graphicsQueueFamily;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_frames[i].commandPool);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_frames[i].commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        vkAllocateCommandBuffers(m_device, &allocInfo, &m_frames[i].commandBuffer);

        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        vkCreateSemaphore(m_device, &semInfo, nullptr, &m_frames[i].imageAvailable);

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(m_device, &fenceInfo, nullptr, &m_frames[i].inFlightFence);
    }
}

void VKContext::destroyFrameData()
{
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vkDestroyFence(m_device, m_frames[i].inFlightFence, nullptr);
        vkDestroySemaphore(m_device, m_frames[i].imageAvailable, nullptr);
        vkDestroyCommandPool(m_device, m_frames[i].commandPool, nullptr);
    }
}

void VKContext::createRenderFinishedSemaphores()
{
    m_renderFinishedSemaphores.resize(m_swapchainImages.size());
    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (auto& sem : m_renderFinishedSemaphores)
        vkCreateSemaphore(m_device, &semInfo, nullptr, &sem);
}

void VKContext::destroyRenderFinishedSemaphores()
{
    for (auto& sem : m_renderFinishedSemaphores)
        vkDestroySemaphore(m_device, sem, nullptr);
    m_renderFinishedSemaphores.clear();
}

VkCommandBuffer VKContext::getCurrentCommandBuffer() const
{
    return m_frames[m_currentFrame].commandBuffer;
}

void VKContext::beginFrame()
{
    auto& frame = m_frames[m_currentFrame];

    // Wait for this frame's fence
    vkWaitForFences(m_device, 1, &frame.inFlightFence, VK_TRUE, UINT64_MAX);

    // Acquire next swapchain image
    VkResult result = vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX,
                                             frame.imageAvailable, VK_NULL_HANDLE,
                                             &m_swapchainImageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateSwapchain();
        return;
    }

    vkResetFences(m_device, 1, &frame.inFlightFence);

    // Reset and begin command buffer
    vkResetCommandBuffer(frame.commandBuffer, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(frame.commandBuffer, &beginInfo);
}

void VKContext::beginSwapchainRenderPass()
{
    auto cmd = getCurrentCommandBuffer();

    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = m_swapchainRenderPass;
    rpBegin.framebuffer = m_swapchainFramebuffers[m_swapchainImageIndex];
    rpBegin.renderArea.offset = { 0, 0 };
    rpBegin.renderArea.extent = m_swapchainExtent;

    VkClearValue clearColor = { { { 0.0f, 0.0f, 0.0f, 1.0f } } };
    rpBegin.clearValueCount = 1;
    rpBegin.pClearValues = &clearColor;

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    // Set viewport and scissor
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_swapchainExtent.width);
    viewport.height = static_cast<float>(m_swapchainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = m_swapchainExtent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void VKContext::endSwapchainRenderPass()
{
    vkCmdEndRenderPass(getCurrentCommandBuffer());
}

void VKContext::endFrame()
{
    auto& frame = m_frames[m_currentFrame];

    // End command buffer
    vkEndCommandBuffer(frame.commandBuffer);

    // Submit
    // renderFinished is indexed by swapchain image, not frame, to avoid reuse while
    // the presentation engine still holds the semaphore for a previously presented image
    VkSemaphore renderFinished = m_renderFinishedSemaphores[m_swapchainImageIndex];

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &frame.imageAvailable;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &frame.commandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &renderFinished;

    vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, frame.inFlightFence);

    // Present
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinished;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &m_swapchain;
    presentInfo.pImageIndices = &m_swapchainImageIndex;

    VkResult result = vkQueuePresentKHR(m_graphicsQueue, &presentInfo);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
    {
        recreateSwapchain();
    }

    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    m_window->pollEvents();
}

void VKContext::beginBatchUpload()
{
    m_batchingUploads = true;
}

void VKContext::deferCopy(VkBuffer src, VkBuffer dst, VkDeviceSize size, VmaAllocation stagingAlloc)
{
    m_deferredCopies.push_back({ src, dst, size, stagingAlloc });
}

void VKContext::endBatchUpload()
{
    m_batchingUploads = false;
    if (m_deferredCopies.empty()) return;

    immediateSubmit([&](VkCommandBuffer cmd)
    {
        for (auto& dc : m_deferredCopies)
        {
            VkBufferCopy copy{ 0, 0, dc.size };
            vkCmdCopyBuffer(cmd, dc.src, dc.dst, 1, &copy);
        }
    });

    for (auto& dc : m_deferredCopies)
        vmaDestroyBuffer(m_allocator, dc.src, dc.stagingAlloc);
    m_deferredCopies.clear();
}

void VKContext::immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    vkResetCommandBuffer(m_uploadBuffer, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_uploadBuffer, &beginInfo);

    function(m_uploadBuffer);

    vkEndCommandBuffer(m_uploadBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_uploadBuffer;

    vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_uploadFence);
    vkWaitForFences(m_device, 1, &m_uploadFence, VK_TRUE, UINT64_MAX);
    vkResetFences(m_device, 1, &m_uploadFence);
}

void VKContext::imguiInit(GLFWwindow* window)
{
    // Create descriptor pool for ImGui
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 1000;
    poolInfo.poolSizeCount = 11;
    poolInfo.pPoolSizes = poolSizes;

    vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_imguiPool);

    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = m_instance;
    initInfo.PhysicalDevice = m_physicalDevice;
    initInfo.Device = m_device;
    initInfo.QueueFamily = m_graphicsQueueFamily;
    initInfo.Queue = m_graphicsQueue;
    initInfo.DescriptorPool = m_imguiPool;
    initInfo.MinImageCount = 2;
    initInfo.ImageCount = 2;
    initInfo.PipelineInfoMain.RenderPass = m_swapchainRenderPass;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.UseDynamicRendering = false;

    ImGui_ImplVulkan_Init(&initInfo);
}

void VKContext::imguiShutdown()
{
    vkDeviceWaitIdle(m_device);
    ImGui_ImplVulkan_Shutdown();
    if (m_imguiPool)
    {
        vkDestroyDescriptorPool(m_device, m_imguiPool, nullptr);
        m_imguiPool = VK_NULL_HANDLE;
    }
}

void VKContext::imguiNewFrame()
{
    ImGui_ImplVulkan_NewFrame();
}

void VKContext::imguiRenderDrawData()
{
    beginSwapchainRenderPass();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), getCurrentCommandBuffer());
    endSwapchainRenderPass();
}

MemoryStats VKContext::getMemoryStats() const
{
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(m_allocator, budgets);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProps);

    VkDeviceSize totalUsage  = 0;
    VkDeviceSize totalBudget = 0;
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i)
    {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
        {
            totalUsage  += budgets[i].usage;
            totalBudget += budgets[i].budget;
        }
    }

    MemoryStats stats;
    stats.usedMB   = static_cast<float>(totalUsage)  / (1024.0f * 1024.0f);
    stats.budgetMB = static_cast<float>(totalBudget) / (1024.0f * 1024.0f);
    stats.available = true;
    return stats;
}

void VKContext::waitIdle()
{
    vkDeviceWaitIdle(m_device);
}

void VKContext::shutdown()
{
    if (m_device == VK_NULL_HANDLE)
        return;

    vkDeviceWaitIdle(m_device);

    // Destroy immediate submit resources
    vkDestroyFence(m_device, m_uploadFence, nullptr);
    vkDestroyCommandPool(m_device, m_uploadPool, nullptr);

    // Destroy per-frame data
    destroyFrameData();

    // Destroy swapchain and its per-image semaphores
    destroyRenderFinishedSemaphores();
    destroySwapchain();

    // Destroy render pass
    vkDestroyRenderPass(m_device, m_swapchainRenderPass, nullptr);

    // Destroy VMA
    vmaDestroyAllocator(m_allocator);

    // Destroy device and surface
    vkb::destroy_device(m_vkbDevice);
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    vkb::destroy_instance(m_vkbInstance);

    m_device = VK_NULL_HANDLE;
    m_instance = VK_NULL_HANDLE;
    m_window = nullptr;
    s_instance = nullptr;
}

} // namespace vex
