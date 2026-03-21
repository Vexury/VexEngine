#pragma once
#include "render_mode.h"

#include <chrono>
#include <cstdint>
#include <memory>

struct Scene;

#ifdef VEX_BACKEND_OPENGL
#include <vex/opengl/gl_gpu_raytracer.h>
#else
#include <vex/vulkan/vk_gpu_raytracer.h>
#endif

// GPU ray-tracing render mode.
// OpenGL build: wraps GLGPURaytracer (compute-shader path tracer).
// Vulkan build:  wraps VKGpuRaytracer (hardware RT pipeline).
class GPURaytraceMode : public IRenderMode
{
public:
    bool init(const RenderModeInitData& init) override;
    void shutdown() override;
    void activate() override;
    void deactivate() override;
    void render(Scene& scene, const SharedRenderData& shared, const FrameChanges& changes) override;

    uint32_t getSampleCount()   const override;
    float    getSamplesPerSec() const override { return m_samplesPerSec; }
    bool     reloadShader()           override;

    VKRTSettings&       getSettings()       { return m_settings; }
    const VKRTSettings& getSettings() const { return m_settings; }

    void onGeometryRebuilt() override;

#ifdef VEX_BACKEND_OPENGL
    vex::GLGPURaytracer* getRaytracer() { return m_raytracer.get(); }
#else
    vex::VKGpuRaytracer* getRaytracer() { return m_raytracer.get(); }
#endif

private:
#ifdef VEX_BACKEND_OPENGL
    std::unique_ptr<vex::GLGPURaytracer> m_raytracer;
#else
    std::unique_ptr<vex::VKGpuRaytracer> m_raytracer;
#endif

    // Stable resources injected at init() — never change after that
    vex::Mesh*          m_fullscreenQuad       = nullptr;
    vex::Texture2D*     m_whiteTexture         = nullptr;
    vex::Shader*        m_fullscreenRTShader   = nullptr;
    vex::Framebuffer*   m_bloomFB[2]           = {nullptr, nullptr};
    vex::Shader*        m_bloomThresholdShader = nullptr;
    vex::Shader*        m_bloomBlurShader      = nullptr;
    SceneGeometryCache* m_geomCache            = nullptr;

    VKRTSettings m_settings;
#ifdef VEX_BACKEND_VULKAN
    VKRTSettings m_prevSettings; // snapshot for accumulator-reset on settings change
#endif
    bool     m_geomDirty   = false;
    uint32_t m_sampleCount = 0;   // incremented per frame on VK; unused on GL (GL raytracer tracks it)
    uint32_t m_rtTexW      = 0;
    uint32_t m_rtTexH      = 0;

    float                                 m_samplesPerSec  = 0.f;
    std::chrono::steady_clock::time_point m_lastSampleTime = {};
};
