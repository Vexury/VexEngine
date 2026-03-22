#pragma once

#include <vex/raytracing/cpu_raytracer.h>
#include <vex/raytracing/bvh.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#ifdef VEX_BACKEND_VULKAN
namespace vex { class VKGpuRaytracer; }
#endif

struct Scene;

// Owns all packed geometry data (CPU triangles, BVH, VK SSBOs) and the logic
// to build/update them. SceneRenderer holds this as a value member and passes
// a pointer into SharedRenderData so render modes can read from it.
class SceneGeometryCache
{
public:
    using ProgressFn = std::function<void(const std::string& stage, float progress)>;

    // Full rebuild — packs all CPU + VK SSBO data. Call buildAccelerationStructures()
    // afterwards on Vulkan to commit BLAS/TLAS to the GPU.
    void rebuild(const Scene& scene, vex::CPURaytracer& cpuRT, bool luminanceCDF,
                 ProgressFn progress = nullptr);

#ifdef VEX_BACKEND_VULKAN
    // Builds BLAS per submesh and TLAS. Must be called after rebuild() on Vulkan.
    void buildAccelerationStructures(const Scene& scene, vex::VKGpuRaytracer* vkRaytracer,
                                     ProgressFn progress = nullptr);
#endif

    // Patch material properties (baseColor, emissive, emissiveStrength) for changed
    // submeshes, then rebuild the light CDF. Much cheaper than full rebuild.
    void rebuildMaterials(const Scene& scene, vex::CPURaytracer* cpuRT, bool luminanceCDF);

    // Rebuild only the light CDF (called when luminanceCDF flag toggles).
    void rebuildLightCDF(bool luminanceCDF);

    bool isReady()      const { return m_ready; }
    bool isAccelReady() const { return m_blasTlasReady; }
    bool useLuminanceCDF() const { return m_luminanceCDF; }

    // --- Read-only accessors (used by render modes via SharedRenderData) ---
    const std::vector<vex::CPURaytracer::Triangle>&    triangles()      const { return m_rtTriangles; }
    const vex::BVH&                                    bvh()            const { return m_rtBVH; }
    const std::vector<uint32_t>&                       lightIndices()   const { return m_rtLightIndices; }
    const std::vector<float>&                          lightCDF()       const { return m_rtLightCDF; }
    float                                              totalLightArea() const { return m_rtTotalLightArea; }
    const std::vector<vex::CPURaytracer::TextureData>& textures()       const { return m_rtTextures; }
    const std::vector<vex::AABB>&                      nodeLocalAABBs() const { return m_nodeLocalAABBs; }

    // Mutable AABB access — rasterizer rebuilds these lazily when in rasterize mode
    std::vector<vex::AABB>& nodeLocalAABBsMut() { return m_nodeLocalAABBs; }

    // Source mapping: BVH-reordered index → {nodeIdx, submeshIdx} and tri-within-submesh index
    const std::vector<std::pair<int,int>>& triSrcSubmesh() const { return m_rtTriangleSrcSubmesh; }
    const std::vector<int>&                triSrcTriIdx()  const { return m_rtTriangleSrcTriIdx; }

    // Mutable triangle access needed by rebuildMaterials (patching existing entries)
    std::vector<vex::CPURaytracer::Triangle>& trianglesMut() { return m_rtTriangles; }

#ifdef VEX_BACKEND_VULKAN
    const std::vector<float>&    vkTriShading()      const { return m_vkTriShading; }
    const std::vector<uint32_t>& vkLights()          const { return m_vkLights; }
    const std::vector<uint32_t>& vkTexData()         const { return m_vkTexData; }
    const std::vector<uint32_t>& vkInstanceOffsets() const { return m_vkInstanceOffsets; }
    std::vector<float>&    vkTriShadingMut() { return m_vkTriShading; }
    std::vector<uint32_t>& vkLightsMut()    { return m_vkLights; }
#endif

private:
    bool m_ready        = false;
    bool m_blasTlasReady = false;
    bool m_luminanceCDF = false;

    std::vector<vex::CPURaytracer::Triangle>    m_rtTriangles;
    std::vector<std::pair<int,int>>             m_rtTriangleSrcSubmesh;
    std::vector<int>                            m_rtTriangleSrcTriIdx;
    std::vector<vex::CPURaytracer::TextureData> m_rtTextures;
    vex::BVH                                    m_rtBVH;
    std::vector<uint32_t>                       m_rtLightIndices;
    std::vector<float>                          m_rtLightCDF;
    float                                       m_rtTotalLightArea = 0.0f;
    std::vector<vex::AABB>                      m_nodeLocalAABBs;

#ifdef VEX_BACKEND_VULKAN
    std::vector<float>    m_vkTriShading;
    std::vector<uint32_t> m_vkLights;
    std::vector<uint32_t> m_vkTexData;
    std::vector<uint32_t> m_vkInstanceOffsets;
#endif
};
