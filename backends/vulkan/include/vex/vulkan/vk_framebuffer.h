#pragma once

#include <vex/graphics/framebuffer.h>
#include <volk.h>
#include <vk_mem_alloc.h>

namespace vex
{

class VKFramebuffer : public Framebuffer
{
public:
    explicit VKFramebuffer(const FramebufferSpec& spec);
    ~VKFramebuffer() override;

    void bind() override;
    void unbind() override;
    void resize(uint32_t width, uint32_t height) override;
    void setClearColor(float r, float g, float b, float a = 1.0f) override;
    void clear(float r, float g, float b, float a = 1.0f) override;

    uintptr_t getColorAttachmentHandle() const override;
    const FramebufferSpec& getSpec() const override { return m_spec; }
    std::vector<uint8_t> readPixels() const override;

    VkRenderPass getRenderPass() const { return m_renderPass; }
    VkImageView  getColorImageView() const { return m_colorImageView; }
    VkSampler    getColorSampler()   const { return m_colorSampler; }

private:
    void createRenderPass();
    void createImages();
    void destroyImages();

    FramebufferSpec m_spec;
    bool m_pendingResize = false;
    float m_clearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

    // Color attachment
    VkImage       m_colorImage      = VK_NULL_HANDLE;
    VmaAllocation m_colorAllocation = VK_NULL_HANDLE;
    VkImageView   m_colorImageView  = VK_NULL_HANDLE;
    VkSampler     m_colorSampler    = VK_NULL_HANDLE;

    // Depth attachment
    VkImage       m_depthImage      = VK_NULL_HANDLE;
    VmaAllocation m_depthAllocation = VK_NULL_HANDLE;
    VkImageView   m_depthImageView  = VK_NULL_HANDLE;

    VkRenderPass  m_renderPass  = VK_NULL_HANDLE;
    VkFramebuffer m_framebuffer = VK_NULL_HANDLE;

    // ImGui descriptor for displaying the color attachment
    VkDescriptorSet m_imguiDescriptor = VK_NULL_HANDLE;
};

} // namespace vex
