#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

#include <vex/raytracing/bvh.h>
#include <vex/raytracing/cpu_raytracer.h>

#include <string>
#include <vector>
#include <cstdint>

namespace vex
{

// ── VKComputeUniforms ─────────────────────────────────────────────────────────
// Must exactly match the 'Uniforms' block in pathtracer.comp (std140, 304 bytes).
// Fields 0-284 are identical to RTUniforms; 288-300 are compute-only additions.
struct VKComputeUniforms
{
    float    inverseVP[16];           // mat4,  offset   0
    float    cameraOrigin[3];         // vec3,  offset  64
    float    aperture;                //        offset  76
    float    cameraRight[3];          // vec3,  offset  80
    float    focusDistance;           //        offset  92
    float    cameraUp[3];             // vec3,  offset  96
    uint32_t sampleCount;             //        offset 108
    uint32_t width;                   //        offset 112
    uint32_t height;                  //        offset 116
    int32_t  maxDepth;                //        offset 120
    float    rayEps;                  //        offset 124
    uint32_t enableNEE;               //        offset 128
    uint32_t enableAA;                //        offset 132
    uint32_t enableFireflyClamping;   //        offset 136
    uint32_t enableEnvLighting;       //        offset 140
    float    envLightMultiplier;      //        offset 144
    uint32_t flatShading;             //        offset 148
    uint32_t enableNormalMapping;     //        offset 152
    uint32_t enableEmissive;          //        offset 156
    uint32_t enableRR;                //        offset 160
    uint32_t _pad0a;                  //        offset 164
    uint32_t _pad0b;                  //        offset 168
    uint32_t _pad0c;                  //        offset 172
    float    pointLightPos[3];        // vec3,  offset 176
    uint32_t pointLightEnabled;       //        offset 188
    float    pointLightColor[3];      // vec3,  offset 192
    float    _pad1;                   //        offset 204
    float    sunDir[3];               // vec3,  offset 208
    float    sunAngularRadius;        //        offset 220
    float    sunColor[3];             // vec3,  offset 224
    uint32_t sunEnabled;              //        offset 236
    float    envColor[3];             // vec3,  offset 240
    uint32_t hasEnvMap;               //        offset 252
    int32_t  envMapWidth;             //        offset 256
    int32_t  envMapHeight;            //        offset 260
    uint32_t hasEnvCDF;               //        offset 264
    float    totalLightArea;          //        offset 268
    uint32_t lightCount;              //        offset 272
    uint32_t bilinearFiltering;       //        offset 276
    uint32_t samplerType;             //        offset 280
    uint32_t _pad2c;                  //        offset 284
    // Compute-only extensions:
    uint32_t triangleCount;           //        offset 288
    uint32_t bvhNodeCount;            //        offset 292
    uint32_t _pad3a;                  //        offset 296
    uint32_t _pad3b;                  //        offset 300
                                      // total: 304
};
static_assert(sizeof(VKComputeUniforms) == 304, "VKComputeUniforms layout mismatch");

// Convenience setters (shared with RTUniforms helpers)
inline void vkComputeUniformsSetMat4(float (&dst)[16], const glm::mat4& m)
{
    const float* p = &m[0][0];
    for (int i = 0; i < 16; ++i) dst[i] = p[i];
}
inline void vkComputeUniformsSetVec3(float (&dst)[3], const glm::vec3& v)
{
    dst[0] = v.x; dst[1] = v.y; dst[2] = v.z;
}

// ── VKComputeRaytracer ────────────────────────────────────────────────────────
// Software BVH path tracer running as a Vulkan compute shader.
// Mirrors the GL GPURaytrace path but uses Vulkan idioms.
// Does NOT require ray tracing hardware extensions.

class VKComputeRaytracer
{
public:
    bool init();
    void shutdown();

    // Upload BVH, triangle data, lights and textures.
    // Called when geometry changes. Internally builds the GPU SSBOs for bindings 1-4 and 7.
    void uploadGeometry(const std::vector<CPURaytracer::Triangle>& triangles,
                        const BVH& bvh,
                        const std::vector<uint32_t>& lightIndices,
                        const std::vector<float>& lightCDF,
                        float totalLightArea,
                        const std::vector<CPURaytracer::TextureData>& textures);

    // Upload / clear the environment map (bindings 5-6).
    void uploadEnvironmentMap(const std::vector<float>& data, int w, int h,
                              const std::vector<float>& cdf);
    void clearEnvironmentMap();

    // Copy uniforms to the always-mapped UBO. Call before traceSample() each frame.
    void setUniforms(const VKComputeUniforms& u);

    // Dispatch the compute shader (assumes setUniforms and createOutputImage were called).
    void traceSample(VkCommandBuffer cmd);

    // Insert a COMPUTE→FRAGMENT pipeline barrier on the output image.
    void postTraceBarrier(VkCommandBuffer cmd);

    // Create (or recreate) the rgba32f accumulation image and write all 9 descriptors.
    // Must be called after uploadGeometry() and (optionally) uploadEnvironmentMap().
    bool createOutputImage(uint32_t w, uint32_t h);

    // Clear the accumulation image (call when camera/scene changes).
    void reset();

    // Read back the accumulation image as linear HDR float RGB (3 floats per pixel).
    void readbackLinearHDR(std::vector<float>& outRGB);

    // Accessors
    VkImage     getOutputImage()     const { return m_outputImage; }
    VkImageView getOutputImageView() const { return m_outputImageView; }
    VkSampler   getDisplaySampler()  const { return m_displaySampler; }
    uint32_t    getTriangleCount()   const { return m_triangleCount; }
    uint32_t    getBvhNodeCount()    const { return m_bvhNodeCount; }

private:
    static VkShaderModule loadShader(const std::string& path);

    // Upload arbitrary data to a new GPU-only storage buffer via staging
    static void createAndUploadBuffer(const void* data, VkDeviceSize size,
                                      VkBufferUsageFlags extraUsage,
                                      VkBuffer& outBuf, VmaAllocation& outAlloc);

    // Destroy a VMA buffer if non-null
    static void destroyBuffer(VkBuffer& buf, VmaAllocation& alloc);

    bool createPipeline();

    // Write all 9 descriptor bindings. Skips null buffers/images.
    void writeAllDescriptors();

    // Write only the environment bindings (5 and 6) for fast env-only updates.
    void writeEnvDescriptors();

    // ── Pipeline ─────────────────────────────────────────────────────────────
    VkPipeline            m_pipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descSetLayout  = VK_NULL_HANDLE;
    VkDescriptorPool      m_descPool       = VK_NULL_HANDLE;
    VkDescriptorSet       m_descSet        = VK_NULL_HANDLE;

    // ── UBO (binding 0) — always-mapped ──────────────────────────────────────
    VkBuffer              m_uboBuffer = VK_NULL_HANDLE;
    VmaAllocation         m_uboAlloc  = VK_NULL_HANDLE;
    VKComputeUniforms*    m_uboMapped = nullptr;

    // ── Scene geometry SSBOs (bindings 1-4, 7) ───────────────────────────────
    VkBuffer      m_bvhBuffer      = VK_NULL_HANDLE;
    VmaAllocation m_bvhAlloc       = VK_NULL_HANDLE;
    VkBuffer      m_triVertsBuffer = VK_NULL_HANDLE;
    VmaAllocation m_triVertsAlloc  = VK_NULL_HANDLE;
    VkBuffer      m_lightsBuffer   = VK_NULL_HANDLE;
    VmaAllocation m_lightsAlloc    = VK_NULL_HANDLE;
    VkBuffer      m_texDataBuffer  = VK_NULL_HANDLE;
    VmaAllocation m_texDataAlloc   = VK_NULL_HANDLE;
    VkBuffer      m_triShadingBuffer = VK_NULL_HANDLE;
    VmaAllocation m_triShadingAlloc  = VK_NULL_HANDLE;

    // ── Environment SSBOs (bindings 5-6) ─────────────────────────────────────
    VkBuffer      m_envMapBuffer = VK_NULL_HANDLE;
    VmaAllocation m_envMapAlloc  = VK_NULL_HANDLE;
    VkBuffer      m_envCdfBuffer = VK_NULL_HANDLE;
    VmaAllocation m_envCdfAlloc  = VK_NULL_HANDLE;

    // ── Accumulation image (binding 8, rgba32f, GENERAL layout) ─────────────
    VkImage       m_outputImage     = VK_NULL_HANDLE;
    VkImageView   m_outputImageView = VK_NULL_HANDLE;
    VmaAllocation m_outputAlloc     = VK_NULL_HANDLE;
    uint32_t      m_width           = 0;
    uint32_t      m_height          = 0;

    // ── Readback staging buffer ──────────────────────────────────────────────
    VkBuffer      m_readbackBuffer = VK_NULL_HANDLE;
    VmaAllocation m_readbackAlloc  = VK_NULL_HANDLE;

    // ── Display sampler (linear, clamp) ─────────────────────────────────────
    VkSampler m_displaySampler = VK_NULL_HANDLE;

    // ── Geometry counts (set after uploadGeometry) ───────────────────────────
    uint32_t m_triangleCount = 0;
    uint32_t m_bvhNodeCount  = 0;
};

} // namespace vex
