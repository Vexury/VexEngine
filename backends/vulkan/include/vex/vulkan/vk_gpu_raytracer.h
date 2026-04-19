#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <vex/raytracing/cpu_raytracer.h>

#include <string>
#include <vector>
#include <cstdint>

namespace vex
{

// ── RTUniforms ────────────────────────────────────────────────────────────────
// Must exactly match the 'Uniforms' block in rt.common.glsl (std140, 292 bytes).
// Use plain float/uint fields so the layout is independent of glm alignment flags.
struct RTUniforms
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
    float    fireflyClampThreshold;   //        offset 172
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
    uint32_t samplerType;             //        offset 280 — 0=PCG  1=Halton  2=BlueNoise(IGN)
    uint32_t useLuminanceCDF;         //        offset 284
    float    envRotation;             //        offset 288
                                      // total: 292
};
static_assert(sizeof(RTUniforms) == 292, "RTUniforms layout mismatch");

// ── Convenience setters ───────────────────────────────────────────────────────
// These avoid pulling a large glm dependency into every call site.
inline void rtUniformsSetMat4(float (&dst)[16], const glm::mat4& m)
{
    const float* p = &m[0][0];
    for (int i = 0; i < 16; ++i) dst[i] = p[i];
}
inline void rtUniformsSetVec3(float (&dst)[3], const glm::vec3& v)
{
    dst[0] = v.x; dst[1] = v.y; dst[2] = v.z;
}

// ── Acceleration structures ───────────────────────────────────────────────────

struct VKBlas
{
    VkAccelerationStructureKHR handle        = VK_NULL_HANDLE;
    VkBuffer                   buffer        = VK_NULL_HANDLE;
    VmaAllocation              allocation    = VK_NULL_HANDLE;
    VkDeviceAddress            deviceAddress = 0;

    void destroy();
};

struct VKTlas
{
    VkAccelerationStructureKHR handle     = VK_NULL_HANDLE;
    VkBuffer                   buffer     = VK_NULL_HANDLE;
    VmaAllocation              allocation = VK_NULL_HANDLE;

    void destroy();
};

// ── VKGpuRaytracer ────────────────────────────────────────────────────────────

class VKGpuRaytracer
{
public:
    bool init();
    void shutdown();

    // ── Acceleration structures ──────────────────────────────────────────────

    // Build one BLAS per submesh. position must be at offset 0 in the vertex struct.
    void addBlas(VkBuffer vertexBuffer, uint32_t vertexCount, VkDeviceSize vertexStride,
                 VkBuffer indexBuffer,  uint32_t indexCount);

    // Submit all pending BLAS builds in a single GPU command. Call after all addBlas() calls.
    void commitBlasBuild();

    // Build the TLAS over all BLASes. Call after commitBlasBuild().
    // instanceTransforms: one mat4 per BLAS (same order as addBlas calls).
    // instanceOpaque: true = force opaque (skip any-hit); false = allow any-hit.
    // Pass empty vectors for defaults (identity transforms, all non-opaque).
    void buildTlas(const std::vector<glm::mat4>& instanceTransforms = {},
                   const std::vector<bool>&       instanceOpaque     = {});

    // Destroy all acceleration structures (call before rebuilding geometry)
    void clearAccelerationStructures();

    // ── Scene data ───────────────────────────────────────────────────────────

    // Upload only the volumes SSBO (binding 9). Does NOT touch triangle/light/tex data.
    // Call this when only volume parameters changed — avoids re-uploading all geometry.
    void uploadVolumes(const std::vector<float>& volumesData);

    // Upload all scene SSBOs and textures. Must be called before createOutputImage().
    // triShading: 13 vec4s (52 floats) per triangle, in per-submesh order
    // lightsData: [lightCount u32][totalLightArea f32][pad pad][indices...][CDF as float-bits...]
    // textures:   one TextureData (RGBA8 pixels + w/h) per scene texture (up to kMaxTextures)
    // envMapData: flat float RGB triples (3 floats per pixel)
    // envCdfData: [marginalCDF: H floats][condCDF: W*H floats][totalIntegral: 1 float]
    // instanceOffsets: first global tri index per BLAS (size == blas count)
    // volumesData: [count:uint,pad,pad,pad as floats][3 vec4s per volume]
    void uploadSceneData(
        const std::vector<float>&                          triShading,
        const std::vector<uint32_t>&                       lightsData,
        const std::vector<vex::CPURaytracer::TextureData>& textures,
        const std::vector<float>&                          envMapData,
        int                                                envMapWidth,
        int                                                envMapHeight,
        const std::vector<float>&                          envCdfData,
        const std::vector<uint32_t>&                       instanceOffsets,
        const std::vector<float>&                          volumesData);

    // ── Rendering ────────────────────────────────────────────────────────────

    // Create the output storage image and write all 9 descriptors.
    // Must be called after buildTlas() and uploadSceneData().
    // Safe to call again on viewport resize.
    bool createOutputImage(uint32_t width, uint32_t height);

    // Copy uniforms to the GPU-mapped UBO. Call before trace() each frame.
    void setUniforms(const RTUniforms& u);

    // Dispatch the ray generation shader (assumes setUniforms was called).
    void trace(VkCommandBuffer cmd);

    // Free all scene SSBOs and output images (call when deactivating the mode).
    // Keeps BLAS/TLAS, pipeline, SBT, UBO, and readback buffer intact.
    void freeSceneData();

    // Clear the accumulation image (call when camera moves or settings change).
    void reset();

    // Readback the accumulation image, tone-map (Reinhard + gamma), return RGBA8.
    void readbackRGBA8(std::vector<uint8_t>& out);

    // Readback the accumulation image as linear HDR float RGB (3 floats per pixel, no tone-mapping).
    void readbackLinearHDR(std::vector<float>& outRGB);

    // Readback first-hit albedo and world-space normal aux buffers (3 floats per pixel each).
    void readbackAuxBuffers(std::vector<float>& outAlbedo, std::vector<float>& outNormal);

    // ── Accessors ────────────────────────────────────────────────────────────

    VkImage     getOutputImage()     const { return m_outputImage; }
    VkImageView getOutputImageView() const { return m_outputImageView; }
    VkSampler   getDisplaySampler()  const { return m_displaySampler; }
    uint32_t    getWidth()           const { return m_width; }
    uint32_t    getHeight()          const { return m_height; }

    // Insert a pipeline barrier transitioning the output image from
    // ray-tracing write (SHADER_WRITE) to fragment shader read (SHADER_READ).
    // Call this between trace() and the graphics render pass that displays the result.
    void postTraceBarrier(VkCommandBuffer cmd);

    const std::vector<VKBlas>& getBlases() const { return m_blases; }
    const VKTlas&              getTlas()   const { return m_tlas; }

private:
    static VkShaderModule loadShader(const std::string& path);

    // Upload arbitrary data to a new GPU-only storage buffer
    static void createAndUploadBuffer(const void* data, VkDeviceSize size,
                                      VkBufferUsageFlags extraUsage,
                                      VkBuffer& outBuf, VmaAllocation& outAlloc);

    // Destroy a VMA buffer if non-null
    static void destroyBuffer(VkBuffer& buf, VmaAllocation& alloc);

    bool createPipeline();
    void buildSBT();
    void writeDescriptors();

    // ── Pending BLAS builds (batched) ────────────────────────────────────────
    struct PendingBlas
    {
        VKBlas                                      blas{};
        VkAccelerationStructureGeometryKHR          geometry{};
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        VkAccelerationStructureBuildRangeInfoKHR    rangeInfo{};
        uint32_t      primitiveCount  = 0;
        VkBuffer      scratchBuffer   = VK_NULL_HANDLE;
        VmaAllocation scratchAlloc    = VK_NULL_HANDLE;
    };
    std::vector<PendingBlas> m_pendingBlases;

    // ── Acceleration structures ──────────────────────────────────────────────
    std::vector<VKBlas> m_blases;
    VKTlas              m_tlas{};

    // ── Pipeline ─────────────────────────────────────────────────────────────
    VkPipeline            m_pipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descSetLayout  = VK_NULL_HANDLE;
    VkDescriptorPool      m_descPool       = VK_NULL_HANDLE;
    VkDescriptorSet       m_descSet        = VK_NULL_HANDLE;

    // ── Shader binding table ─────────────────────────────────────────────────
    VkBuffer                        m_sbtBuffer     = VK_NULL_HANDLE;
    VmaAllocation                   m_sbtAllocation = VK_NULL_HANDLE;
    VkStridedDeviceAddressRegionKHR m_rgenRegion{};
    VkStridedDeviceAddressRegionKHR m_missRegion{};
    VkStridedDeviceAddressRegionKHR m_hitRegion{};
    VkStridedDeviceAddressRegionKHR m_callableRegion{};

    // ── UBO (binding 2) — always-mapped, created in createPipeline ───────────
    VkBuffer      m_uboBuffer = VK_NULL_HANDLE;
    VmaAllocation m_uboAlloc  = VK_NULL_HANDLE;
    RTUniforms*   m_uboMapped = nullptr;

    // ── Scene SSBOs (bindings 3–4, 7–9) ─────────────────────────────────────
    VkBuffer      m_triShadingBuffer      = VK_NULL_HANDLE;
    VmaAllocation m_triShadingAlloc       = VK_NULL_HANDLE;
    VkBuffer      m_lightsBuffer          = VK_NULL_HANDLE;
    VmaAllocation m_lightsAlloc           = VK_NULL_HANDLE;
    VkBuffer      m_envCdfBuffer          = VK_NULL_HANDLE;
    VmaAllocation m_envCdfAlloc           = VK_NULL_HANDLE;
    VkBuffer      m_instanceOffsetsBuffer = VK_NULL_HANDLE;
    VmaAllocation m_instanceOffsetsAlloc  = VK_NULL_HANDLE;
    VkBuffer      m_volumesBuffer         = VK_NULL_HANDLE;
    VmaAllocation m_volumesAlloc          = VK_NULL_HANDLE;

    // ── Env map VkImage (binding 6) ──────────────────────────────────────────
    VkImage       m_envMapImage      = VK_NULL_HANDLE;
    VkImageView   m_envMapImageView  = VK_NULL_HANDLE;
    VmaAllocation m_envMapImageAlloc = nullptr;

    // ── Bindless texture array (binding 5) ───────────────────────────────────
    static constexpr uint32_t  kMaxTextures  = 1024;
    VkSampler                  m_textureSampler = VK_NULL_HANDLE;
    std::vector<VkImage>       m_texImages;
    std::vector<VkImageView>   m_texImageViews;
    std::vector<VmaAllocation> m_texAllocs;
    uint32_t                   m_texCount = 0;

    // ── Output storage image (rgba32f, GENERAL layout) ───────────────────────
    VkImage       m_outputImage     = VK_NULL_HANDLE;
    VkImageView   m_outputImageView = VK_NULL_HANDLE;
    VmaAllocation m_outputAlloc     = VK_NULL_HANDLE;
    uint32_t      m_width           = 0;
    uint32_t      m_height          = 0;

    // ── Aux images (bindings 10-11, rgba32f, first-hit albedo + normal) ──────
    VkImage       m_albedoImage     = VK_NULL_HANDLE;
    VkImageView   m_albedoImageView = VK_NULL_HANDLE;
    VmaAllocation m_albedoAlloc     = VK_NULL_HANDLE;
    VkImage       m_normalImage     = VK_NULL_HANDLE;
    VkImageView   m_normalImageView = VK_NULL_HANDLE;
    VmaAllocation m_normalAlloc     = VK_NULL_HANDLE;

    // ── Readback staging buffer ──────────────────────────────────────────────
    VkBuffer      m_readbackBuffer = VK_NULL_HANDLE;
    VmaAllocation m_readbackAlloc  = VK_NULL_HANDLE;

    // ── Display sampler (linear, clamp) for GPU-side display path ────────────
    VkSampler m_displaySampler = VK_NULL_HANDLE;
};

} // namespace vex
