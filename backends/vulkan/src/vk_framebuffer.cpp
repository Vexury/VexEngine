#include <vex/vulkan/vk_framebuffer.h>
#include <vex/vulkan/vk_context.h>
#include <vex/core/log.h>

#include <imgui_impl_vulkan.h>

#include <array>

namespace vex
{

// Factory
std::unique_ptr<Framebuffer> Framebuffer::create(const FramebufferSpec& spec)
{
    return std::make_unique<VKFramebuffer>(spec);
}

VKFramebuffer::VKFramebuffer(const FramebufferSpec& spec)
    : m_spec(spec)
{
    createRenderPass();
    createImages();
}

VKFramebuffer::~VKFramebuffer()
{
    auto device = VKContext::get().getDevice();

    destroyImages();

    if (m_renderPass)
        vkDestroyRenderPass(device, m_renderPass, nullptr);
}

void VKFramebuffer::createRenderPass()
{
    auto device = VKContext::get().getDevice();

    VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

    std::array<VkAttachmentDescription, 2> attachments{};
    // Color
    attachments[0].format = colorFormat;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthRef{};
    depthRef.attachment = 1;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    uint32_t attachmentCount = 1;
    if (m_spec.hasDepth)
    {
        attachments[1].format = depthFormat;
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        subpass.pDepthStencilAttachment = &depthRef;
        attachmentCount = 2;
    }

    std::array<VkSubpassDependency, 2> dependencies{};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = attachmentCount;
    rpInfo.pAttachments = attachments.data();
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 2;
    rpInfo.pDependencies = dependencies.data();

    if (vkCreateRenderPass(device, &rpInfo, nullptr, &m_renderPass) != VK_SUCCESS)
    {
        Log::error("Failed to create offscreen render pass");
    }
}

void VKFramebuffer::createImages()
{
    auto& ctx = VKContext::get();
    auto device = ctx.getDevice();
    auto allocator = ctx.getAllocator();

    VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

    // --- Color image ---
    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.format = colorFormat;
        imgInfo.extent = { m_spec.width, m_spec.height, 1 };
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = 1;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        vmaCreateImage(allocator, &imgInfo, &allocInfo,
                       &m_colorImage, &m_colorAllocation, nullptr);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_colorImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = colorFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;

        vkCreateImageView(device, &viewInfo, nullptr, &m_colorImageView);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

        vkCreateSampler(device, &samplerInfo, nullptr, &m_colorSampler);
    }

    // --- Depth image ---
    if (m_spec.hasDepth)
    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.format = depthFormat;
        imgInfo.extent = { m_spec.width, m_spec.height, 1 };
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = 1;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        vmaCreateImage(allocator, &imgInfo, &allocInfo,
                       &m_depthImage, &m_depthAllocation, nullptr);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_depthImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = depthFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;

        vkCreateImageView(device, &viewInfo, nullptr, &m_depthImageView);
    }

    // --- VkFramebuffer ---
    uint32_t attachmentCount = m_spec.hasDepth ? 2 : 1;
    std::array<VkImageView, 2> fbAttachments = { m_colorImageView, m_depthImageView };

    VkFramebufferCreateInfo fbInfo{};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = m_renderPass;
    fbInfo.attachmentCount = attachmentCount;
    fbInfo.pAttachments = fbAttachments.data();
    fbInfo.width = m_spec.width;
    fbInfo.height = m_spec.height;
    fbInfo.layers = 1;

    if (vkCreateFramebuffer(device, &fbInfo, nullptr, &m_framebuffer) != VK_SUCCESS)
    {
        Log::error("Failed to create offscreen framebuffer");
    }
}

void VKFramebuffer::destroyImages()
{
    auto device = VKContext::get().getDevice();
    auto allocator = VKContext::get().getAllocator();

    if (m_imguiDescriptor != VK_NULL_HANDLE)
    {
        ImGui_ImplVulkan_RemoveTexture(m_imguiDescriptor);
        m_imguiDescriptor = VK_NULL_HANDLE;
    }

    if (m_framebuffer)    vkDestroyFramebuffer(device, m_framebuffer, nullptr);
    if (m_colorSampler)   vkDestroySampler(device, m_colorSampler, nullptr);
    if (m_colorImageView) vkDestroyImageView(device, m_colorImageView, nullptr);
    if (m_colorImage)     vmaDestroyImage(allocator, m_colorImage, m_colorAllocation);
    if (m_depthImageView) vkDestroyImageView(device, m_depthImageView, nullptr);
    if (m_depthImage)     vmaDestroyImage(allocator, m_depthImage, m_depthAllocation);

    m_framebuffer    = VK_NULL_HANDLE;
    m_colorSampler   = VK_NULL_HANDLE;
    m_colorImageView = VK_NULL_HANDLE;
    m_colorImage     = VK_NULL_HANDLE;
    m_depthImageView = VK_NULL_HANDLE;
    m_depthImage     = VK_NULL_HANDLE;
}

void VKFramebuffer::bind()
{
    // Apply any pending resize first (deferred from renderViewport)
    if (m_pendingResize)
    {
        m_pendingResize = false;
        vkDeviceWaitIdle(VKContext::get().getDevice());
        destroyImages();
        createImages();
        m_imguiDescriptor = ImGui_ImplVulkan_AddTexture(
            m_colorSampler, m_colorImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    auto cmd = VKContext::get().getCurrentCommandBuffer();

    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = m_renderPass;
    rpBegin.framebuffer = m_framebuffer;
    rpBegin.renderArea.extent = { m_spec.width, m_spec.height };

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = { { m_clearColor[0], m_clearColor[1], m_clearColor[2], m_clearColor[3] } };
    clearValues[1].depthStencil = { 1.0f, 0 };

    rpBegin.clearValueCount = m_spec.hasDepth ? 2 : 1;
    rpBegin.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = static_cast<float>(m_spec.height);
    viewport.width = static_cast<float>(m_spec.width);
    viewport.height = -static_cast<float>(m_spec.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = { m_spec.width, m_spec.height };
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void VKFramebuffer::unbind()
{
    vkCmdEndRenderPass(VKContext::get().getCurrentCommandBuffer());
}

void VKFramebuffer::resize(uint32_t width, uint32_t height)
{
    if (width == 0 || height == 0) return;
    if (width == m_spec.width && height == m_spec.height) return;

    m_spec.width = width;
    m_spec.height = height;
    m_pendingResize = true;
}

void VKFramebuffer::setClearColor(float r, float g, float b, float a)
{
    m_clearColor[0] = r;
    m_clearColor[1] = g;
    m_clearColor[2] = b;
    m_clearColor[3] = a;
}

void VKFramebuffer::clear(float r, float g, float b, float a)
{
    setClearColor(r, g, b, a);
}

uintptr_t VKFramebuffer::getColorAttachmentHandle() const
{
    if (m_imguiDescriptor == VK_NULL_HANDLE)
    {
        const_cast<VKFramebuffer*>(this)->m_imguiDescriptor = ImGui_ImplVulkan_AddTexture(
            m_colorSampler, m_colorImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    return reinterpret_cast<uintptr_t>(m_imguiDescriptor);
}

} // namespace vex
