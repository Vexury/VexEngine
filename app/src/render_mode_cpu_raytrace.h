#pragma once
#include "render_mode.h"

#include <chrono>
#include <cstdint>

struct Scene;

// CPU path-tracing render mode.
// Owns timing state; the CPURaytracer itself stays in SceneRenderer (shared
// across triggerDenoise etc.) and is read from SharedRenderData::cpuRaytracer
// during init().
class CPURaytraceMode : public IRenderMode
{
public:
    bool init(const RenderModeInitData& init) override;
    void shutdown() override {}
    void activate() override;
    void deactivate() override {}
    void render(Scene& scene, const SharedRenderData& shared, const FrameChanges& changes) override;

    float    getSamplesPerSec() const override { return m_samplesPerSec; }
    uint32_t getSampleCount()   const override { return m_cpuRaytracer ? m_cpuRaytracer->getSampleCount() : 0; }
    void     resetAccumulation()      override { if (m_cpuRaytracer) m_cpuRaytracer->reset(); }

private:
    // Stable resources injected at init() — never change after that
    vex::Mesh*        m_fullscreenQuad       = nullptr;
    vex::Texture2D*   m_whiteTexture         = nullptr;
    vex::Shader*      m_fullscreenRTShader   = nullptr;
    vex::Framebuffer* m_bloomFB[2]           = {nullptr, nullptr};
    vex::Shader*      m_bloomThresholdShader = nullptr;
    vex::Shader*      m_bloomBlurShader      = nullptr;
    std::function<vex::Texture2D*(uint32_t w, uint32_t h)> m_resizeCPUAccumTex;

    vex::CPURaytracer* m_cpuRaytracer = nullptr;

    std::vector<float>                     m_cpuHDRScratch;   // RGB scratch from getLinearHDR
    std::vector<float>                     m_cpuRGBAScratch;  // RGBA32F scratch for texture upload
    float                                  m_samplesPerSec  = 0.0f;
    std::chrono::steady_clock::time_point  m_lastSampleTime = {};
};
