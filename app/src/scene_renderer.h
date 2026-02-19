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

    bool reloadGPUShader();

private:
    void renderRasterize(Scene& scene, int selectedGroup, int selectedSubmesh);
    void renderCPURaytrace(Scene& scene);
    void rebuildMaterials(Scene& scene);
#ifdef VEX_BACKEND_OPENGL
    void renderGPURaytrace(Scene& scene);
#endif
    void rebuildRaytraceGeometry(Scene& scene);

    std::unique_ptr<vex::Shader> m_meshShader;
    std::unique_ptr<vex::Framebuffer> m_framebuffer;
    std::unique_ptr<vex::Texture2D> m_whiteTexture;
    std::unique_ptr<vex::Texture2D> m_flatNormalTexture;

    // OpenGL-only: picking and outline shaders/framebuffer
    std::unique_ptr<vex::Shader> m_pickShader;
    std::unique_ptr<vex::Shader> m_outlineShader;
    std::unique_ptr<vex::Framebuffer> m_pickFB;

    int m_drawCalls = 0;

    // Render mode
    RenderMode m_renderMode = RenderMode::Rasterize;
    DebugMode  m_debugMode  = DebugMode::None;
    bool m_enableNormalMapping = true;

    // CPU raytracing
    std::unique_ptr<vex::CPURaytracer> m_cpuRaytracer;
    std::unique_ptr<vex::Texture2D> m_raytraceTexture;
    std::unique_ptr<vex::Shader> m_fullscreenShader;
    std::unique_ptr<vex::Mesh> m_fullscreenQuad;
    uint32_t m_raytraceTexW = 0;
    uint32_t m_raytraceTexH = 0;

    // Rasterizer env + post-process state
#ifdef VEX_BACKEND_OPENGL
    uint32_t m_rasterEnvMapTex = 0;
    std::unique_ptr<vex::Framebuffer> m_rasterHDRFB;
    float m_rasterExposure   = 0.0f;
    float m_rasterGamma      = 2.2f;
    bool  m_rasterEnableACES = true;

#endif
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

    // Camera change detection
    glm::vec3 m_prevCameraPos{0.0f};
    glm::mat4 m_prevViewMatrix{1.0f};

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
