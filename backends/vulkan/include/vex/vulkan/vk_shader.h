#pragma once

#include <vex/graphics/shader.h>
#include <vex/graphics/framebuffer.h>
#include <volk.h>
#include <vk_mem_alloc.h>

#include <vex/vulkan/vk_context.h>

#include <unordered_map>
#include <array>
#include <cstdint>

namespace vex
{

class VKTexture2D;

// UBO layout matching mesh shaders
struct MeshUBO
{
    glm::mat4 view;
    glm::mat4 projection;
    glm::vec3 cameraPos;
    float _pad0 = 0;
    glm::vec3 lightPos;
    float _pad1 = 0;
    glm::vec3 lightColor;
    float _pad2 = 0;
    glm::vec3 sunDirection;
    float _pad3 = 0;
    glm::vec3 sunColor;
    float _pad4 = 0;
};

class VKShader : public Shader
{
public:
    VKShader() = default;
    ~VKShader() override;

    bool loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath) override;
    void bind() override;
    void unbind() override;

    void setInt(const std::string& name, int value) override;
    void setFloat(const std::string& name, float value) override;
    void setBool(const std::string& name, bool value) override;
    void setVec3(const std::string& name, const glm::vec3& value) override;
    void setVec4(const std::string& name, const glm::vec4& value) override;
    void setMat4(const std::string& name, const glm::mat4& value) override;

    void setTexture(uint32_t slot, Texture2D* tex) override;
    void setExternalTextureVK(uint32_t slot, VkImageView view, VkSampler sampler, VkImageLayout layout);
    void clearExternalTextureCache();

    void setWireframe(bool enabled) override;
    void preparePipeline(const Framebuffer& fb) override;

    VkPipeline       getPipeline()       const { return m_pipeline; }
    VkPipelineLayout getPipelineLayout() const { return m_pipelineLayout; }
    VkDescriptorSetLayout getDescriptorSetLayout() const { return m_descriptorSetLayout; }

    // Allow external render pass (for offscreen rendering)
    void createPipeline(VkRenderPass renderPass, bool depthTest = true,
                       bool depthWrite = true, bool hasVertexInput = true,
                       VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL);

private:
    VkShaderModule loadShaderModule(const std::string& path);
    void buildUniformMap();

    VkShaderModule m_vertModule = VK_NULL_HANDLE;
    VkShaderModule m_fragModule = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_textureSetLayout    = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout      = VK_NULL_HANDLE;
    VkPipeline            m_pipeline             = VK_NULL_HANDLE;
    VkDescriptorPool      m_descriptorPool       = VK_NULL_HANDLE;
    VkDescriptorPool      m_textureDescriptorPool = VK_NULL_HANDLE;

    // Per-frame UBO
    struct FrameUBO
    {
        VkBuffer       buffer     = VK_NULL_HANDLE;
        VmaAllocation  allocation = VK_NULL_HANDLE;
        void*          mapped     = nullptr;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    };
    std::array<FrameUBO, MAX_FRAMES_IN_FLIGHT> m_frameUBOs{};

    // Texture descriptor set cache (per VKTexture2D*)
    std::unordered_map<VKTexture2D*, VkDescriptorSet> m_textureDescriptorSets;
    // Separate pool for external image views (e.g. RT output) so it can be
    // reset independently without touching material texture sets.
    VkDescriptorPool m_externalTexturePool = VK_NULL_HANDLE;
    std::unordered_map<VkImageView, VkDescriptorSet> m_externalTextureDescSets;

    MeshUBO m_uboData{};
    std::unordered_map<std::string, size_t> m_uniformOffsets;
    size_t m_uboSize = sizeof(MeshUBO);

    struct MeshPushConstant {
        uint32_t alphaClip      = 0;   // offset  0
        int32_t  debugMode      = 0;   // offset  4
        float    nearPlane      = 0.01f; // offset 8
        float    farPlane       = 1000.0f; // offset 12
        int32_t  materialType   = 0;   // offset 16
        float    roughness      = 0.5f; // offset 20
        float    metallic       = 0.0f; // offset 24
        uint32_t hasNormalMap   = 0;   // offset 28
        uint32_t hasRoughnessMap = 0;  // offset 32
        uint32_t hasMetallicMap = 0;   // offset 36
        uint32_t flipV          = 0;   // offset 40
        float    sampleCount    = 1.0f; // offset 44
        float    exposure       = 0.0f; // offset 48
        float    gamma          = 2.2f; // offset 52
        uint32_t enableACES     = 1u;  // offset 56
    };
    MeshPushConstant m_pushData{};

    // Wireframe pipeline
    VkPipeline m_wireframePipeline = VK_NULL_HANDLE;
    bool m_wireframeActive = false;
    VkRenderPass m_cachedRenderPass = VK_NULL_HANDLE;
};

} // namespace vex
