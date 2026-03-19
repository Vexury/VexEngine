#pragma once

#ifdef VEX_BACKEND_VULKAN

#include "render_mode.h"

#include <vex/vulkan/vk_compute_raytracer.h>

#include <chrono>
#include <cstdint>
#include <memory>

struct Scene;

// Vulkan compute (software BVH) path-tracing render mode.
class VKComputeRaytraceMode : public IRenderMode
{
public:
    bool init(const RenderModeInitData& init) override;
    void shutdown() override;
    void activate() override;
    void deactivate() override {}
    void render(Scene& scene, const SharedRenderData& shared, const FrameChanges& changes) override;

    uint32_t getSampleCount()   const override { return m_vkComputeSampleCount; }
    float    getSamplesPerSec() const override { return m_vkComputeSamplesPerSec; }

    void onGeometryRebuilt() override;

    vex::VKComputeRaytracer* getRaytracer() { return m_vkComputeRaytracer.get(); }

private:
    // Stable resources injected at init() — never change after that
    vex::Mesh*          m_fullscreenQuad       = nullptr;
    vex::Shader*        m_fullscreenRTShader   = nullptr;
    vex::Framebuffer*   m_bloomFB[2]           = {nullptr, nullptr};
    vex::Shader*        m_bloomThresholdShader = nullptr;
    vex::Shader*        m_bloomBlurShader      = nullptr;
    SceneGeometryCache* m_geomCache            = nullptr;
    const VKRTSettings* m_vkRTSettings         = nullptr;

    std::unique_ptr<vex::VKComputeRaytracer> m_vkComputeRaytracer;

    bool     m_vkComputeGeomDirty    = false;
    uint32_t m_vkComputeSampleCount  = 0;
    uint32_t m_vkComputeRTTexW       = 0;
    uint32_t m_vkComputeRTTexH       = 0;

    float                                  m_vkComputeSamplesPerSec = 0.0f;
    std::chrono::steady_clock::time_point  m_vkComputeLastSampleTime = {};
};

#endif // VEX_BACKEND_VULKAN
