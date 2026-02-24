#pragma once

#include <vex/graphics/skybox.h>
#include <vex/graphics/framebuffer.h>
#include <vex/vulkan/vk_context.h>
#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <array>

namespace vex
{

class VKSkybox : public Skybox
{
public:
    VKSkybox() = default;
    ~VKSkybox() override;

    VKSkybox(const VKSkybox&) = delete;
    VKSkybox& operator=(const VKSkybox&) = delete;

    bool load(const std::string& equirectPath) override;
    void draw(const glm::mat4& inverseVP) const override;
    void preparePipeline(const Framebuffer& fb) override;

    // Must be called after the offscreen render pass is known
    void createPipeline(VkRenderPass renderPass);

private:
    void createQuad();
    void destroyResources();

    // Fullscreen triangle
    VkBuffer      m_vertexBuffer     = VK_NULL_HANDLE;
    VmaAllocation m_vertexAllocation = VK_NULL_HANDLE;

    // Envmap texture
    VkImage       m_textureImage      = VK_NULL_HANDLE;
    VmaAllocation m_textureAllocation = VK_NULL_HANDLE;
    VkImageView   m_textureImageView  = VK_NULL_HANDLE;
    VkSampler     m_textureSampler    = VK_NULL_HANDLE;
    VkFormat      m_textureFormat     = VK_FORMAT_R8G8B8A8_UNORM;

    // Pipeline
    VkShaderModule        m_vertModule         = VK_NULL_HANDLE;
    VkShaderModule        m_fragModule         = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout      = VK_NULL_HANDLE;
    VkPipeline            m_pipeline             = VK_NULL_HANDLE;
    VkDescriptorPool      m_descriptorPool       = VK_NULL_HANDLE;

    // Per-frame UBO + descriptor sets
    struct FrameData
    {
        VkBuffer       uboBuffer     = VK_NULL_HANDLE;
        VmaAllocation  uboAllocation = VK_NULL_HANDLE;
        void*          uboMapped     = nullptr;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    };
    std::array<FrameData, MAX_FRAMES_IN_FLIGHT> m_frames{};

    bool m_loaded = false;
};

} // namespace vex
