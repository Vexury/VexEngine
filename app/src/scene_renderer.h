#pragma once

#include <vex/graphics/shader.h>
#include <vex/graphics/framebuffer.h>
#include <vex/graphics/mesh.h>
#include <vex/graphics/texture.h>
#include <vex/raytracing/cpu_raytracer.h>
#include <vex/raytracing/bvh.h>

#include "denoiser.h"
#include "render_mode.h"
#include "scene_geometry_cache.h"
#include "render_mode_rasterize.h"
#include "render_mode_cpu_raytrace.h"
#include "render_mode_gpu_raytrace.h"
#ifdef VEX_BACKEND_VULKAN
#include "render_mode_compute_raytrace.h"
#endif

#include <glm/glm.hpp>

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <cstdint>
#include <utility>

struct Scene;

enum class RenderMode { Rasterize, CPURaytrace, GPURaytrace, ComputeRaytrace };

enum class DebugMode : int {
    None       = 0,  // Normal Blinn-Phong shading
    Wireframe  = 1,  // White wireframe on dark background
    Depth      = 2,  // Linearized depth as grayscale
    Normals    = 3,  // World-space normals as RGB
    UVs        = 4,  // UV coordinates as RG
    Albedo     = 5,  // Unlit base color (vertex color * texture)
    Emission   = 6,  // Emissive channel only
    MaterialType = 7,  // Material type as distinct flat colors (Microfacet/Mirror/Dielectric)
    Roughness  = 8,  // Roughness value (texture .g or scalar)
    Metallic   = 9,  // Metallic value (texture .b or scalar)
    AO         = 10, // Ambient occlusion value (texture .r or 1.0)
    MappedNormals = 11, // World-space normals after normal-map perturbation
};

class SceneRenderer
{
public:
    bool init(Scene& scene);
    void shutdown();

    void renderScene(Scene& scene, int selectedNodeIdx, int selectedSubmesh = -1);
    std::pair<int,int> pick(Scene& scene, int pixelX, int pixelY);
    void waitIdle(); // call before destroying GPU-referenced scene resources

    vex::Framebuffer* getFramebuffer() { return m_framebuffer.get(); }
    int getDrawCalls() const { return m_drawCalls; }
    float getSamplesPerSec() const;

    // Shadow map debug display
    // Returns an ImTextureID-compatible handle (0 if shadow map not yet rendered).
    // For GL the depth texture compare mode is temporarily disabled; it is restored
    // automatically at the start of the next renderRasterize() shadow pass.
    uintptr_t getShadowMapDisplayHandle();
    bool      shadowMapFlipsUV() const; // true on GL (origin bottom-left)

    uint32_t getMaxSamples() const { return m_maxSamples; }
    void     setMaxSamples(uint32_t v) { m_maxSamples = v; }

    bool saveImage(const std::string& path) const;
    bool saveShadowMap(const std::string& path) const;

    void setRenderMode(RenderMode mode);
    RenderMode getRenderMode() const { return m_renderMode; }

    void setDebugMode(DebugMode mode) { m_debugMode = mode; }
    DebugMode getDebugMode() const { return m_debugMode; }

    uint32_t getRaytraceSampleCount() const;
    void     resetAccumulation();

    void setUseLuminanceCDF(bool v);
    bool getUseLuminanceCDF() const { return m_luminanceCDF; }

    uint32_t getBVHNodeCount() const;
    size_t   getBVHMemoryBytes() const;
    vex::AABB getBVHRootAABB() const;
    float    getBVHSAHCost() const;
    size_t   getLightTriangleCount() const;
    float    getTotalLightArea() const;

    bool reloadGPUShader();

    // Denoising
    void triggerDenoise();
    void triggerDenoiseAux(); // Denoise+ with albedo/normal feature inputs
    bool isDenoiserReady() const { return m_denoiser && m_denoiser->isReady(); }
    bool getShowDenoisedResult() const { return m_showDenoisedResult; }


#ifdef VEX_BACKEND_VULKAN
    const vex::GpuPassTimings* getGpuPassTimings() const;
#endif

    // Struct-based settings access (replaces ~50 individual getters/setters)
    CPURTSettings&  getCPURTSettings()  { return m_cpuRTSettings; }
    VKRTSettings&   getGPURTSettings()  { return m_gpuMode ? m_gpuMode->getSettings() : m_gpuRTFallback; }
    RasterSettings& getRasterSettings() { return m_rasterSettings; }
    BloomSettings&  getBloomSettings()  { return m_bloomSettings; }

    // Explicitly rebuild acceleration structures / BVH with optional progress callbacks.
    // Called from App::runImport between frames so the overlay can update at each stage.
    // Clears scene.geometryDirty so renderScene won't rebuild again on the next frame.
    using ProgressFn = std::function<void(const std::string& stage, float progress)>;
    void buildGeometry(Scene& scene, ProgressFn progress = nullptr);

    // Executes any pending geometry rebuild (e.g. from a mode switch) with a progress
    // callback so the caller can pump loading overlay frames between stages.
    // Clears both m_pendingGeomRebuild and scene.geometryDirty so renderScene skips it.
    void flushPendingGeomRebuild(Scene& scene, ProgressFn progress);

private:
    // Helpers
    void renderOutlineMask(Scene& scene, int selectedNodeIdx,
                           const glm::mat4& view, const glm::mat4& proj);
    void renderShadowPrePass(Scene& scene);
    void rebuildMaterials(Scene& scene);
    void rebuildRaytraceGeometry(Scene& scene, ProgressFn progress = nullptr);

    SharedRenderData buildSharedRenderData();
    FrameChanges     computeFrameChanges(Scene& scene);

    // Env loading helpers (update m_vkEnvMapData/CdfData + m_vkRasterEnvTex + m_rasterEnvMapTex)
    void loadEnvData(Scene& scene); // called from computeFrameChanges on env change

    // Lazy-apply settings to the underlying render mode objects (called each renderScene())
    void applyCPURTSettings();
    void applyRasterSettings();
#ifdef VEX_BACKEND_OPENGL
    void applyGPURTSettingsGL();
#endif

    // --- Render mode objects ---
    IRenderMode*                        m_activeMode  = nullptr; // non-owning, points into one of the unique_ptrs below
    std::unique_ptr<RasterizeMode>      m_rasterMode;
    std::unique_ptr<CPURaytraceMode>    m_cpuMode;
    std::unique_ptr<GPURaytraceMode>    m_gpuMode;
#ifdef VEX_BACKEND_VULKAN
    std::unique_ptr<VKComputeRaytraceMode> m_computeMode;
#endif

    // --- Core shared resources (never moved to modes) ---
    std::unique_ptr<vex::Shader>      m_meshShader;
    std::unique_ptr<vex::Framebuffer> m_framebuffer;
    std::unique_ptr<vex::Texture2D>   m_whiteTexture;
    std::unique_ptr<vex::Texture2D>   m_flatNormalTexture;

    // Screen-space outline
    std::unique_ptr<vex::Framebuffer> m_outlineMaskFB;
    std::unique_ptr<vex::Shader>      m_outlineMaskShader;
    bool m_outlineActive = false;

    // Fullscreen display shaders/quad (shared by all modes)
    std::unique_ptr<vex::Mesh>   m_fullscreenQuad;
    std::unique_ptr<vex::Shader> m_fullscreenRTShader;

    // Bloom post-processing (shared across all render modes, FBs owned here)
    std::unique_ptr<vex::Framebuffer> m_bloomFB[2];
    std::unique_ptr<vex::Shader>      m_bloomThresholdShader;
    std::unique_ptr<vex::Shader>      m_bloomBlurShader;

    // Shadow map pre-pass (shared across all render modes)
    static constexpr uint32_t SHADOW_MAP_SIZE = 4096;
    std::unique_ptr<vex::Framebuffer> m_shadowFB;
    std::unique_ptr<vex::Shader>      m_shadowShader;
    bool      m_shadowMapDirty        = true;
    bool      m_shadowMapEverRendered = false;
    glm::mat4 m_shadowLightVP         {1.0f};
    float     m_shadowOrthoScale      = 0.0f; // 2*orthoSize/SHADOW_MAP_SIZE, reused each frame

    // Sample limits
    uint32_t m_maxSamples = 0;

    int m_drawCalls = 0;

    // Render mode
    RenderMode m_renderMode = RenderMode::Rasterize;
    DebugMode  m_debugMode  = DebugMode::None;

    // Settings structs (single source of truth for each domain)
    CPURTSettings  m_cpuRTSettings;
    RasterSettings m_rasterSettings;
    BloomSettings  m_bloomSettings;
    VKRTSettings   m_gpuRTFallback;  // returned by getGPURTSettings() when m_gpuMode is null

    // CPU raytracing
    bool m_pendingGeomRebuild = false;
    std::unique_ptr<vex::CPURaytracer> m_cpuRaytracer;
    std::unique_ptr<vex::Texture2D>    m_raytraceTexture; // CPU/denoised display texture
    uint32_t m_raytraceTexW      = 0;
    uint32_t m_raytraceTexH      = 0;
    bool     m_raytraceTexIsFloat = false;

    // Rasterizer env + env color (updated by env-loading path, read by RasterizeMode via shared)
#ifdef VEX_BACKEND_OPENGL
    uint32_t           m_rasterEnvMapTex = 0;
    std::vector<float> m_glEnvMapData;   // raw HDR float data for GL GPU raytracer
    int                m_glEnvMapW = 0;
    int                m_glEnvMapH = 0;
#endif
#ifdef VEX_BACKEND_VULKAN
    std::unique_ptr<vex::Texture2D> m_vkRasterEnvTex;
#endif
    glm::vec3 m_rasterEnvColor { 0.5f };

#ifdef VEX_BACKEND_VULKAN
    std::vector<float>    m_vkVolumesData;

    // VK RT env map data (reloaded by loadEnvData() on env change)
    std::vector<float> m_vkEnvMapData;
    std::vector<float> m_vkEnvCdfData;
    int m_vkEnvMapW = 0;
    int m_vkEnvMapH = 0;
#endif

    // Camera change detection
    glm::vec3 m_prevCameraPos{0.0f};
    glm::mat4 m_prevViewMatrix{1.0f};
    float     m_prevAperture      = 0.0f;
    float     m_prevFocusDistance = 10.0f;

    // Environment change detection
    int   m_prevEnvmapIndex  = -1;
    glm::vec3 m_prevSkyboxColor{-1.0f};
    float m_prevEnvRotation  = 0.0f;

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

    // Volume change detection
    std::vector<float> m_prevVolumesData;

    // Denoising
    std::unique_ptr<vex::Denoiser> m_denoiser;
    std::vector<float>   m_denoiseLinearHDR;
    std::vector<float>   m_denoiseAlbedo;
    std::vector<float>   m_denoiseNormal;
    std::vector<float>   m_denoisedHDR;   // RGBA32F, pre-normalized HDR for shader tone mapping
    bool m_showDenoisedResult = false;

    // Geometry cache (all packed data + rebuild logic)
    SceneGeometryCache m_geomCache;

    bool  m_luminanceCDF = false;
};
