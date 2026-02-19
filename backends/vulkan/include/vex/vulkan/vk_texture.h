#pragma once

#include <vex/graphics/texture.h>
#include <volk.h>
#include <vk_mem_alloc.h>

namespace vex
{

class VKTexture2D : public Texture2D
{
public:
    VKTexture2D(uint32_t width, uint32_t height, uint32_t channels);
    ~VKTexture2D() override;

    void bind(uint32_t slot) override;
    void unbind() override;

    uint32_t getWidth() const override { return m_width; }
    uint32_t getHeight() const override { return m_height; }
    uintptr_t getNativeHandle() const override;

    void setData(const void* data, uint32_t width, uint32_t height, uint32_t channels) override;

    VkImageView getImageView() const { return m_imageView; }
    VkSampler   getSampler()   const { return m_sampler; }

    // Create a descriptor set for use with ImGui
    VkDescriptorSet getImGuiDescriptorSet() const { return m_imguiDescriptorSet; }
    void createImGuiDescriptorSet();

private:
    void createImage(uint32_t width, uint32_t height, VkFormat format);
    void destroyImage();
    void transitionLayout(VkCommandBuffer cmd, VkImage image,
                          VkImageLayout oldLayout, VkImageLayout newLayout);

    uint32_t m_width    = 0;
    uint32_t m_height   = 0;
    uint32_t m_channels = 4;

    VkImage       m_image      = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    VkImageView   m_imageView  = VK_NULL_HANDLE;
    VkSampler     m_sampler    = VK_NULL_HANDLE;

    VkDescriptorSet m_imguiDescriptorSet = VK_NULL_HANDLE;
};

} // namespace vex
