#pragma once

#include <vex/graphics/shader.h>
#include <vex/graphics/framebuffer.h>
#include <vex/graphics/mesh.h>
#include <vex/graphics/texture.h>
#include <vex/raytracing/cpu_raytracer.h>
#include <vex/raytracing/bvh.h>

#include "scene_geometry_cache.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct Scene;

// ---- VK GPU RT / Compute RT settings (POD, shared by GPURaytraceMode + ComputeRaytraceMode) ----
struct VKRTSettings
{
    int   maxDepth              = 8;
    bool  enableNEE             = true;
    bool  enableAA              = true;
    bool  enableFireflyClamping = true;
    bool  enableEnvLighting     = true;
    float envLightMultiplier    = 0.3f;
    bool  flatShading           = false;
    bool  enableNormalMapping   = true;
    bool  enableEmissive        = true;
    bool  bilinearFiltering     = true;
    int   samplerType           = 1; // 0=PCG  1=Halton  2=BlueNoise(IGN)
    float rayEps                = 1e-4f;
    bool  enableRR              = true;
    float exposure              = 0.f;
    float gamma                 = 2.2f;
    bool  enableACES            = true;

    bool operator==(const VKRTSettings& o) const
    {
        // exposure/gamma/enableACES are display-only — applied in fullscreen shader,
        // do NOT reset the accumulator when they change
        return maxDepth == o.maxDepth && enableNEE == o.enableNEE &&
               enableAA == o.enableAA && enableFireflyClamping == o.enableFireflyClamping &&
               enableEnvLighting == o.enableEnvLighting && envLightMultiplier == o.envLightMultiplier &&
               flatShading == o.flatShading && enableNormalMapping == o.enableNormalMapping &&
               enableEmissive == o.enableEmissive && bilinearFiltering == o.bilinearFiltering &&
               samplerType == o.samplerType && rayEps == o.rayEps && enableRR == o.enableRR;
    }
    bool operator!=(const VKRTSettings& o) const { return !(*this == o); }
};

// ---- CPU path-tracer settings ----
struct CPURTSettings
{
    int   maxDepth              = 5;
    bool  enableNEE             = true;
    bool  enableAA              = true;
    bool  enableFireflyClamping = true;
    bool  enableEnvLighting     = true;
    float envLightMultiplier    = 0.3f;
    bool  flatShading           = false;
    bool  enableNormalMapping   = true;
    bool  enableEmissive        = true;
    float exposure              = 0.f;
    float gamma                 = 2.2f;
    bool  enableACES            = true;
    float rayEps                = 1e-4f;
    bool  enableRR              = true;
};

// ---- Rasterizer settings ----
struct RasterSettings
{
    float     exposure              = 0.f;
    float     gamma                 = 2.2f;
    bool      enableACES            = true;
    bool      enableEnvLighting     = true;
    float     envLightMultiplier    = 0.3f;
    bool      enableNormalMapping   = true;
    bool      enableShadows         = true;
    float     shadowBiasTexels      = 1.5f;
    float     shadowStrength        = 1.0f;
    glm::vec3 shadowColor           = {0.f, 0.f, 0.f};
};

// ---- Bloom settings (shared across all render modes) ----
struct BloomSettings
{
    bool  enabled    = false;
    float intensity  = 0.05f;
    float threshold  = 0.8f;
    int   blurPasses = 5;
};

// ---- Stable data injected once into each render mode during init() ----
// Populated by SceneRenderer::init(); each mode copies what it needs into members.
// Never passed to render() — use SharedRenderData for per-frame data.
struct RenderModeInitData
{
    vex::Mesh*        fullscreenQuad       = nullptr;
    vex::Texture2D*   whiteTexture         = nullptr;
    vex::Texture2D*   flatNormalTexture    = nullptr;
    vex::Shader*      meshShader           = nullptr;
    vex::Shader*      fullscreenRTShader   = nullptr;
    vex::Framebuffer* bloomFB[2]           = {nullptr, nullptr};
    vex::Shader*      bloomThresholdShader = nullptr;
    vex::Shader*      bloomBlurShader      = nullptr;
    SceneGeometryCache* geomCache          = nullptr;
    vex::CPURaytracer*  cpuRaytracer       = nullptr;
    std::function<vex::Texture2D*(uint32_t w, uint32_t h)> resizeCPUAccumTex;
#ifdef VEX_BACKEND_VULKAN
    const VKRTSettings* vkRTSettings = nullptr;
#endif
};

// ---- Per-frame data passed to every render() call ----
struct SharedRenderData
{
    vex::Framebuffer* outputFB           = nullptr;
    vex::Framebuffer* outlineMaskFB      = nullptr;
    vex::Texture2D*   cpuAccumTex        = nullptr; // CPU RT accumulation texture
    bool              outlineActive      = false;
    bool              enableNormalMapping = true;
    bool*             showDenoisedResult  = nullptr;
    int*              drawCalls           = nullptr; // write-back into SceneRenderer::m_drawCalls
    uint32_t          maxSamples          = 0;
    int               debugMode           = 0;       // cast of DebugMode enum
    int               selectedNodeIdx     = -1;
    int               selectedSubmesh     = -1;

    // ── Bloom settings (user-controlled, vary at runtime) ─────────────────────
    bool  bloomEnabled    = false;
    float bloomIntensity  = 0.05f;
    float bloomThreshold  = 0.8f;
    int   bloomBlurPasses = 5;

    // ── Rasterizer env (changes on env switch) ────────────────────────────────
    glm::vec3 rasterEnvColor {0.5f};
#ifdef VEX_BACKEND_OPENGL
    uint32_t     rasterEnvMapTex = 0;
#endif
#ifdef VEX_BACKEND_VULKAN
    vex::Texture2D* vkRasterEnvTex = nullptr;
#endif

    // ── VK-only per-frame data ────────────────────────────────────────────────
#ifdef VEX_BACKEND_VULKAN
    const std::vector<float>* vkVolumesData = nullptr;
#endif
};

// ---- Per-frame change flags computed by SceneRenderer ----
struct FrameChanges
{
    bool cameraChanged      = false;
    bool dofChanged         = false;
    bool lightChanged       = false;
    bool sunChanged         = false;
    bool envChanged         = false;       // envmap index or custom path changed
    bool envDataChanged     = false;       // HDR env data was reloaded this frame
    bool skyboxColorChanged = false;
    bool volumesChanged     = false;

    glm::vec3 camPos    {0.0f};
    glm::mat4 viewMatrix{1.0f};
    glm::mat4 projMatrix{1.0f};
    glm::vec3 sunDir    {0.0f};

#ifdef VEX_BACKEND_OPENGL
    // GL env data (pointer into SceneRenderer's m_glEnvMapData; valid this frame)
    const float* glEnvMapData = nullptr; // nullptr = no HDR map
    int glEnvMapW = 0;
    int glEnvMapH = 0;
#endif

#ifdef VEX_BACKEND_VULKAN
    // VK env data (pointers into SceneRenderer's m_vkEnvMap* vectors; valid this frame)
    const std::vector<float>* vkEnvMapData = nullptr; // nullptr = no HDR map
    const std::vector<float>* vkEnvCdfData = nullptr;
    int vkEnvMapW = 0;
    int vkEnvMapH = 0;
#endif
};

// ---- Common render mode interface ----
class IRenderMode
{
public:
    virtual ~IRenderMode() = default;

    virtual bool init(const RenderModeInitData& init)           = 0;
    virtual void shutdown()                                      = 0;
    virtual void activate()                                      = 0;  // called on mode switch-in
    virtual void deactivate()                                    = 0;  // called on mode switch-out
    virtual void render(Scene& scene,
                        const SharedRenderData& shared,
                        const FrameChanges& changes)             = 0;

    virtual void     onGeometryRebuilt()       {}
    virtual uint32_t getSampleCount()   const { return 0; }
    virtual float    getSamplesPerSec() const { return 0.f; }
    virtual bool     reloadShader()           { return false; }
};
