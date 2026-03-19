#pragma once
#include "render_mode.h"

#include <glm/glm.hpp>
#include <string>
#include <utility>
#include <vector>

struct Scene;

class RasterizeMode : public IRenderMode
{
public:
    bool init(const RenderModeInitData& init) override;
    void shutdown() override;
    void activate() override  {}
    void deactivate() override {}
    void render(Scene& scene, const SharedRenderData& shared, const FrameChanges& changes) override;

    // Extended render that includes selection indices (called by SceneRenderer)
    void renderWithSelection(Scene& scene, const SharedRenderData& shared,
                             int selectedNodeIdx, int selectedSubmesh);

    // Called from SceneRenderer::init() after the shared HDR FB is ready (VK only)
    void lateInitVK(const RenderModeInitData& init);

    // GL picking
    std::pair<int,int> pick(Scene& scene, const SharedRenderData& shared,
                             int pixelX, int pixelY);

    // Shadow map debug display
    uintptr_t getShadowMapDisplayHandle();
    bool      shadowMapFlipsUV() const;
    bool      saveShadowMap(const std::string& path) const;

    // Settings getters/setters forwarded from SceneRenderer public API
    float     getShadowNormalBiasTexels() const   { return m_shadowNormalBiasTexels; }
    void      setShadowNormalBiasTexels(float v)  { m_shadowNormalBiasTexels = v; }
    bool      getEnableShadows() const            { return m_rasterEnableShadows; }
    void      setEnableShadows(bool v)            { m_rasterEnableShadows = v; }
    float     getShadowStrength() const           { return m_shadowStrength; }
    void      setShadowStrength(float v)          { m_shadowStrength = v; }
    glm::vec3 getShadowColor() const              { return m_shadowColor; }
    void      setShadowColor(glm::vec3 v)         { m_shadowColor = v; }
    bool      getEnableEnvLighting() const        { return m_rasterEnableEnvLighting; }
    void      setEnableEnvLighting(bool v)        { m_rasterEnableEnvLighting = v; }
    float     getEnvLightMultiplier() const       { return m_rasterEnvLightMultiplier; }
    void      setEnvLightMultiplier(float v)      { m_rasterEnvLightMultiplier = v; }
    float     getExposure() const                 { return m_rasterExposure; }
    void      setExposure(float v)                { m_rasterExposure = v; }
    float     getGamma() const                    { return m_rasterGamma; }
    void      setGamma(float v)                   { m_rasterGamma = v; }
    bool      getEnableACES() const               { return m_rasterEnableACES; }
    void      setEnableACES(bool v)               { m_rasterEnableACES = v; }

    static constexpr uint32_t SHADOW_MAP_SIZE = 4096;

private:
    // Stable resources injected at init() — never change after that
    vex::Mesh*          m_fullscreenQuad       = nullptr;
    vex::Texture2D*     m_whiteTexture         = nullptr;
    vex::Texture2D*     m_flatNormalTexture    = nullptr;
    vex::Shader*        m_meshShader           = nullptr;
    vex::Shader*        m_fullscreenRTShader   = nullptr;
    vex::Framebuffer*   m_bloomFB[2]           = {nullptr, nullptr};
    vex::Shader*        m_bloomThresholdShader = nullptr;
    vex::Shader*        m_bloomBlurShader      = nullptr;
    SceneGeometryCache* m_geomCache            = nullptr;

    // Shadow map (directional / sun light)
    std::unique_ptr<vex::Framebuffer> m_shadowFB;
    std::unique_ptr<vex::Shader>      m_shadowShader;
    bool      m_shadowMapEverRendered  = false;
    float     m_shadowNormalBiasTexels = 1.5f;
    bool      m_rasterEnableShadows    = true;
    float     m_shadowStrength         = 1.0f;
    glm::vec3 m_shadowColor            = {0.0f, 0.0f, 0.0f};

    // HDR intermediate framebuffer
    std::unique_ptr<vex::Framebuffer> m_rasterHDRFB;

    // Tone-mapping settings
    float m_rasterExposure   = 0.0f;
    float m_rasterGamma      = 2.2f;
    bool  m_rasterEnableACES = true;

    // Env lighting settings
    bool      m_rasterEnableEnvLighting  = true;
    float     m_rasterEnvLightMultiplier = 0.3f;

#ifdef VEX_BACKEND_OPENGL
    std::unique_ptr<vex::Shader>      m_pickShader;
    std::unique_ptr<vex::Framebuffer> m_pickFB;
    uint32_t m_rasterEnvMapTex = 0;   // GL env texture (owned here after move from SceneRenderer)

    // VK bloom dimension cache (kept here to detect resize)
    uint32_t m_bloomFBW = 0, m_bloomFBH = 0;
#endif

#ifdef VEX_BACKEND_VULKAN
    std::unique_ptr<vex::Texture2D> m_vkRasterEnvTex;

    uint32_t m_vkBloomFBW = 0, m_vkBloomFBH = 0;
#endif
};
