#pragma once

#include <vex/graphics/shader.h>
#include <vex/graphics/framebuffer.h>
#include <vex/graphics/mesh.h>
#include <vex/graphics/texture.h>
#include <vex/raytracing/cpu_raytracer.h>
#include <vex/raytracing/bvh.h>

#ifdef VEX_BACKEND_OPENGL
#include <vex/opengl/gl_gpu_raytracer.h>
#endif

#ifdef VEX_BACKEND_VULKAN
#include <vex/vulkan/vk_gpu_raytracer.h>
#endif

#include <glm/glm.hpp>

#include <memory>
#include <string>
#include <cstdint>
#include <utility>

struct Scene;

enum class RenderMode { Rasterize, CPURaytrace, GPURaytrace };

enum class DebugMode : int {
    None       = 0,  // Normal Blinn-Phong shading
    Wireframe  = 1,  // White wireframe on dark background
    Depth      = 2,  // Linearized depth as grayscale
    Normals    = 3,  // World-space normals as RGB
    UVs        = 4,  // UV coordinates as RG
    Albedo     = 5,  // Unlit base color (vertex color * texture)
    Emission   = 6,  // Emissive channel only
    MaterialID = 7,  // Material type as distinct flat colors
};

class SceneRenderer
{
public:
    bool init(Scene& scene);
    void shutdown();

    void renderScene(Scene& scene, int selectedGroup, int selectedSubmesh = -1);
    std::pair<int,int> pick(Scene& scene, int pixelX, int pixelY);

    vex::Framebuffer* getFramebuffer() { return m_framebuffer.get(); }
    int getDrawCalls() const { return m_drawCalls; }

    // Shadow map debug display
    // Returns an ImTextureID-compatible handle (0 if shadow map not yet rendered).
    // For GL the depth texture compare mode is temporarily disabled; it is restored
    // automatically at the start of the next renderRasterize() shadow pass.
    uintptr_t getShadowMapDisplayHandle();
    bool      shadowMapFlipsUV() const; // true on GL (origin bottom-left)

    float getShadowNormalBiasTexels() const { return m_shadowNormalBiasTexels; }
    void  setShadowNormalBiasTexels(float v) { m_shadowNormalBiasTexels = v; }

    bool saveImage(const std::string& path) const;

    void setRenderMode(RenderMode mode);
    RenderMode getRenderMode() const { return m_renderMode; }

    void setDebugMode(DebugMode mode) { m_debugMode = mode; }
    DebugMode getDebugMode() const { return m_debugMode; }

    uint32_t getRaytraceSampleCount() const;

    void setMaxDepth(int depth);
    int  getMaxDepth() const;

    void setEnableNEE(bool v);
    bool getEnableNEE() const;

    void setEnableFireflyClamping(bool v);
    bool getEnableFireflyClamping() const;

    void setEnableAA(bool v);
    bool getEnableAA() const;

    void setEnableEnvironment(bool v);
    bool getEnableEnvironment() const;

    void  setEnvLightMultiplier(float v);
    float getEnvLightMultiplier() const;

    void setFlatShading(bool v);
    bool getFlatShading() const;

    void setEnableNormalMapping(bool v);
    bool getEnableNormalMapping() const;

    void setEnableEmissive(bool v);
    bool getEnableEmissive() const;

    void setExposure(float v);
    float getExposure() const;

    void setGamma(float v);
    float getGamma() const;

    void setEnableACES(bool v);
    bool getEnableACES() const;

    void setRayEps(float v);
    float getRayEps() const;

    void setEnableRR(bool v);
    bool getEnableRR() const;

    uint32_t getBVHNodeCount() const;
    size_t   getBVHMemoryBytes() const;
    vex::AABB getBVHRootAABB() const;
    float    getBVHSAHCost() const;

    // GPU raytracing settings (separate from CPU to allow independent control)
    void setGPUMaxDepth(int d);
    int  getGPUMaxDepth() const;
    void setGPUEnableNEE(bool v);
    bool getGPUEnableNEE() const;
    void setGPUEnableAA(bool v);
    bool getGPUEnableAA() const;
    void setGPUEnableFireflyClamping(bool v);
    bool getGPUEnableFireflyClamping() const;
    void setGPUEnableEnvironment(bool v);
    bool getGPUEnableEnvironment() const;
    void  setGPUEnvLightMultiplier(float v);
    float getGPUEnvLightMultiplier() const;

    void  setRasterEnableEnvLighting(bool v);
    bool  getRasterEnableEnvLighting() const;
    void  setRasterEnvLightMultiplier(float v);
    float getRasterEnvLightMultiplier() const;
    void  setRasterExposure(float v);
    float getRasterExposure() const;
    void  setRasterGamma(float v);
    float getRasterGamma() const;
    void  setRasterEnableACES(bool v);
    bool  getRasterEnableACES() const;
    void setGPUFlatShading(bool v);
    bool getGPUFlatShading() const;
    void setGPUEnableNormalMapping(bool v);
    bool getGPUEnableNormalMapping() const;
    void setGPUEnableEmissive(bool v);
    bool getGPUEnableEmissive() const;
    void setGPUExposure(float v);
    float getGPUExposure() const;
    void setGPUGamma(float v);
    float getGPUGamma() const;
    void setGPUEnableACES(bool v);
    bool getGPUEnableACES() const;

    void setGPURayEps(float v);
    float getGPURayEps() const;

    void setGPUEnableRR(bool v);
    bool getGPUEnableRR() const;

    bool reloadGPUShader();

private:
    void renderRasterize(Scene& scene, int selectedGroup, int selectedSubmesh);
    void renderCPURaytrace(Scene& scene);
    void rebuildMaterials(Scene& scene);

    static constexpr uint32_t SHADOW_MAP_SIZE = 4096;
#ifdef VEX_BACKEND_OPENGL
    void renderGPURaytrace(Scene& scene);
#endif
#ifdef VEX_BACKEND_VULKAN
    void renderVKRaytrace(Scene& scene);
#endif
    void rebuildRaytraceGeometry(Scene& scene);

    std::unique_ptr<vex::Shader> m_meshShader;
    std::unique_ptr<vex::Framebuffer> m_framebuffer;
    std::unique_ptr<vex::Texture2D> m_whiteTexture;
    std::unique_ptr<vex::Texture2D> m_flatNormalTexture;

    // Shadow map (directional / sun light)
    std::unique_ptr<vex::Framebuffer> m_shadowFB;
    std::unique_ptr<vex::Shader>      m_shadowShader;
    bool      m_shadowMapEverRendered  = false;
    float     m_shadowNormalBiasTexels = 1.5f;
    vex::AABB m_sceneAABB;             // rebuilt in rebuildRaytraceGeometry

    // Screen-space outline (both backends)
    std::unique_ptr<vex::Framebuffer> m_outlineMaskFB;
    std::unique_ptr<vex::Shader>      m_outlineMaskShader;

    // OpenGL-only: picking shader/framebuffer
    std::unique_ptr<vex::Shader>      m_pickShader;
    std::unique_ptr<vex::Framebuffer> m_pickFB;

    int m_drawCalls = 0;

    // Render mode
    RenderMode m_renderMode = RenderMode::Rasterize;
    DebugMode  m_debugMode  = DebugMode::None;
    bool m_enableNormalMapping = true;

    // CPU raytracing
    bool m_cpuBVHDirty = false; // CPU BVH not yet built for current geometry
    std::unique_ptr<vex::CPURaytracer> m_cpuRaytracer;
    std::unique_ptr<vex::Texture2D> m_raytraceTexture;
    std::unique_ptr<vex::Shader> m_fullscreenShader;
    std::unique_ptr<vex::Mesh> m_fullscreenQuad;
    uint32_t m_raytraceTexW = 0;
    uint32_t m_raytraceTexH = 0;

    // Rasterizer env + post-process state
#ifdef VEX_BACKEND_OPENGL
    uint32_t m_rasterEnvMapTex = 0;
#endif
    std::unique_ptr<vex::Framebuffer> m_rasterHDRFB;
    float m_rasterExposure   = 0.0f;
    float m_rasterGamma      = 2.2f;
    bool  m_rasterEnableACES = true;
    glm::vec3 m_rasterEnvColor { 0.5f };
    bool  m_rasterEnableEnvLighting  = false;
    float m_rasterEnvLightMultiplier = 1.0f;

    // GPU raytracing (OpenGL only)
#ifdef VEX_BACKEND_OPENGL
    std::unique_ptr<vex::GLGPURaytracer> m_gpuRaytracer;
    std::unique_ptr<vex::Shader> m_fullscreenRTShader;
    uint32_t m_gpuRTTexW = 0;
    uint32_t m_gpuRTTexH = 0;

    // GPU raytrace settings (separate from CPU)
    float m_gpuExposure = 0.0f;
    float m_gpuGamma = 2.2f;
    bool  m_gpuEnableACES = true;
#endif

#ifdef VEX_BACKEND_VULKAN
    std::unique_ptr<vex::VKGpuRaytracer> m_vkRaytracer;
    std::unique_ptr<vex::Shader> m_vkFullscreenRTShader;

    // VK RT scene data SSBOs (built in rebuildRaytraceGeometry)
    std::vector<float>    m_vkTriShading;      // 13 vec4s per tri, per-submesh order
    std::vector<uint32_t> m_vkTexData;         // texCount header + packed RGBA8 pixels
    std::vector<uint32_t> m_vkLights;          // lightCount/area header + indices + CDF
    std::vector<uint32_t> m_vkInstanceOffsets; // first global tri index per BLAS

    // VK rasterizer env map texture (RGBA8, created from float env data on env change)
    std::unique_ptr<vex::Texture2D> m_vkRasterEnvTex;

    // VK RT env map data (updated in renderVKRaytrace on env change)
    std::vector<float>    m_vkEnvMapData;
    std::vector<float>    m_vkEnvCdfData;
    int m_vkEnvMapW = 0;
    int m_vkEnvMapH = 0;

    bool     m_vkGeomDirty   = false; // triShading/lights/tex/offsets need re-upload
    uint32_t m_vkSampleCount = 0;     // accumulated sample counter (RNG seed)
    uint32_t m_vkRTTexW      = 0;     // last created output image width
    uint32_t m_vkRTTexH      = 0;     // last created output image height

    // VK RT rendering settings (mirrors GPU raytracer settings for the Vulkan build)
    int   m_vkMaxDepth              = 8;
    bool  m_vkEnableNEE             = true;
    bool  m_vkEnableAA              = true;
    bool  m_vkEnableFireflyClamping = true;
    bool  m_vkEnableEnvLighting     = false;
    float m_vkEnvLightMultiplier    = 1.0f;
    bool  m_vkFlatShading           = false;
    bool  m_vkEnableNormalMapping   = true;
    bool  m_vkEnableEmissive        = true;
    float m_vkRayEps                = 1e-4f;
    bool  m_vkEnableRR              = true;

    // VK GPU RT display tone-mapping settings
    float m_vkExposure   = 0.0f;
    float m_vkGamma      = 2.2f;
    bool  m_vkEnableACES = true;
#endif

    // Camera change detection
    glm::vec3 m_prevCameraPos{0.0f};
    glm::mat4 m_prevViewMatrix{1.0f};
    float     m_prevAperture      = 0.0f;
    float     m_prevFocusDistance = 10.0f;

    // Environment change detection
    int m_prevEnvmapIndex = -1;
    glm::vec3 m_prevSkyboxColor{-1.0f};

    // Point light change detection
    glm::vec3 m_prevLightPos{0.0f};
    glm::vec3 m_prevLightColor{0.0f};
    float m_prevLightIntensity = 0.0f;
    bool m_prevShowLight = false;

    // Sun light change detection
    glm::vec3 m_prevSunDirection{0.0f};
    glm::vec3 m_prevSunColor{0.0f};
    float m_prevSunIntensity = 0.0f;
    float m_prevSunAngularRadius = 0.0f;
    bool m_prevShowSun = false;

    // Custom env map path change detection
    std::string m_prevCustomEnvmapPath;

    // Shared geometry data (for GPU upload after CPU raytracer reorders)
    std::vector<vex::CPURaytracer::Triangle> m_rtTriangles;
    std::vector<std::pair<int,int>> m_rtTriangleSrcSubmesh; // {groupIdx, submeshIdx} per m_rtTriangles entry
    std::vector<vex::CPURaytracer::TextureData> m_rtTextures;
    vex::BVH m_rtBVH;
    std::vector<uint32_t> m_rtLightIndices;
    std::vector<float> m_rtLightCDF;
    float m_rtTotalLightArea = 0.0f;
    bool m_gpuGeometryDirty = false;
};
