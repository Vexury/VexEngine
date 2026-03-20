#include "scene_renderer.h"
#include "scene.h"

#include <vex/graphics/mesh.h>
#include <vex/graphics/skybox.h>
#include <vex/scene/mesh_data.h>
#include <vex/core/log.h>

#include <stb_image.h>
#include <stb_image_write.h>
#include <tinyexr.h>

#ifdef VEX_BACKEND_OPENGL
#include <glad/glad.h>
#include <vex/opengl/gl_framebuffer.h>
#endif

#ifdef VEX_BACKEND_VULKAN
#include <vex/vulkan/vk_mesh.h>
#include <vex/vulkan/vk_context.h>
#include <vex/vulkan/vk_shader.h>
#include <vex/vulkan/vk_framebuffer.h>
#include <vex/vulkan/vk_texture.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <unordered_map>
#include <vector>


static vex::MeshData buildFullscreenQuadData()
{
    vex::MeshData data;

    vex::Vertex v{};
    v.normal   = { 0.0f, 0.0f, 1.0f };
    v.color    = { 1.0f, 1.0f, 1.0f };
    v.emissive = { 0.0f, 0.0f, 0.0f };

    // Natural OpenGL UVs: bottom-left = (0,0), top-right = (1,1).
    // Sources that store pixels top-to-bottom (raytracers) set u_flipV=true in their shader.
    v.position = { -1.0f, -1.0f, 0.0f }; v.uv = { 0.0f, 0.0f }; data.vertices.push_back(v);
    v.position = {  1.0f, -1.0f, 0.0f }; v.uv = { 1.0f, 0.0f }; data.vertices.push_back(v);
    v.position = { -1.0f,  1.0f, 0.0f }; v.uv = { 0.0f, 1.0f }; data.vertices.push_back(v);
    v.position = {  1.0f,  1.0f, 0.0f }; v.uv = { 1.0f, 1.0f }; data.vertices.push_back(v);

    data.indices = { 0, 1, 2, 1, 3, 2 };
    return data;
}

bool SceneRenderer::init([[maybe_unused]] Scene& scene)
{
    // Create 1x1 white fallback texture for untextured meshes
    m_whiteTexture = vex::Texture2D::create(1, 1, 4);
    uint8_t white[] = { 255, 255, 255, 255 };
    m_whiteTexture->setData(white, 1, 1, 4);

    // Create 1x1 flat-normal fallback (tangent-space up = (0,0,1))
    m_flatNormalTexture = vex::Texture2D::create(1, 1, 4);
    uint8_t flatNormal[] = { 128, 128, 255, 255 };
    m_flatNormalTexture->setData(flatNormal, 1, 1, 4);

    std::string dir = vex::Shader::shaderDir();
    std::string ext = vex::Shader::shaderExt();

    m_meshShader = vex::Shader::create();
    if (!m_meshShader->loadFromFiles(dir + "mesh.vert" + ext, dir + "mesh.frag" + ext))
        return false;

    m_fullscreenQuad = vex::Mesh::create();
    m_fullscreenQuad->upload(buildFullscreenQuadData());

    m_framebuffer = vex::Framebuffer::create({ .width = 1280, .height = 720, .hasDepth = true });

    // Screen-space outline mask framebuffer (no depth)
    m_outlineMaskFB = vex::Framebuffer::create({ .width = 1280, .height = 720 });
    m_outlineMaskShader = vex::Shader::create();
    if (!m_outlineMaskShader->loadFromFiles(dir + "outline_mask.vert" + ext, dir + "outline_mask.frag" + ext))
        return false;

    // Create backend-specific pipelines for the offscreen framebuffer
    m_meshShader->preparePipeline(*m_framebuffer);
    if (scene.skybox)
        scene.skybox->preparePipeline(*m_framebuffer);

#ifdef VEX_BACKEND_VULKAN
    {
        auto* vkMaskShader = static_cast<vex::VKShader*>(m_outlineMaskShader.get());
        auto* vkMaskFB     = static_cast<vex::VKFramebuffer*>(m_outlineMaskFB.get());
        vkMaskShader->createPipeline(vkMaskFB->getRenderPass(),
                                     false, false, 5, VK_POLYGON_MODE_FILL);
    }
#endif

    // Initialize CPU raytracer
    m_cpuRaytracer = std::make_unique<vex::CPURaytracer>();

    // Initialize denoiser (no-op if OIDN not compiled in)
    m_denoiser = std::make_unique<vex::Denoiser>();
    m_denoiser->init();

#ifdef VEX_BACKEND_OPENGL
    // Load fullscreen RT shader (tone mapping - shared by rasterizer and GPU RT)
    m_fullscreenRTShader = vex::Shader::create();
    if (!m_fullscreenRTShader->loadFromFiles(dir + "fullscreen.vert" + ext, dir + "fullscreen_rt.frag" + ext))
    {
        vex::Log::error("Failed to load fullscreen_rt shader");
        m_fullscreenRTShader.reset();
    }

    // Bloom shaders (fullscreen.vert is shared)
    m_bloomThresholdShader = vex::Shader::create();
    if (!m_bloomThresholdShader->loadFromFiles(dir + "fullscreen.vert" + ext,
                                               dir + "bloom_threshold.frag" + ext))
    {
        vex::Log::error("Failed to load bloom_threshold shader");
        m_bloomThresholdShader.reset();
    }
    m_bloomBlurShader = vex::Shader::create();
    if (!m_bloomBlurShader->loadFromFiles(dir + "fullscreen.vert" + ext,
                                          dir + "bloom_blur.frag" + ext))
    {
        vex::Log::error("Failed to load bloom_blur shader");
        m_bloomBlurShader.reset();
    }
    m_bloomFB[0] = vex::Framebuffer::create({ .width = 640, .height = 360 });
    m_bloomFB[1] = vex::Framebuffer::create({ .width = 640, .height = 360 });
#endif

#ifdef VEX_BACKEND_VULKAN
    m_fullscreenRTShader = vex::Shader::create();
    static_cast<vex::VKShader*>(m_fullscreenRTShader.get())->setVertexAttrCount(5);
    if (!m_fullscreenRTShader->loadFromFiles(dir + "fullscreen.vert" + ext, dir + "fullscreen_rt.frag" + ext))
    {
        vex::Log::error("Failed to load Vulkan fullscreen_rt shader");
        m_fullscreenRTShader.reset();
    }
    else
    {
        m_fullscreenRTShader->preparePipeline(*m_framebuffer);
    }

    m_bloomFB[0] = vex::Framebuffer::create({ .width = 640, .height = 360, .hdrColor = true });
    m_bloomFB[1] = vex::Framebuffer::create({ .width = 640, .height = 360, .hdrColor = true });

    m_bloomThresholdShader = vex::Shader::create();
    static_cast<vex::VKShader*>(m_bloomThresholdShader.get())->setVertexAttrCount(5);
    if (!m_bloomThresholdShader->loadFromFiles(dir + "fullscreen.vert" + ext,
                                               dir + "bloom_threshold.frag" + ext))
    {
        vex::Log::error("Failed to load Vulkan bloom_threshold shader");
        m_bloomThresholdShader.reset();
    }
    else
    {
        m_bloomThresholdShader->preparePipeline(*m_bloomFB[0]);
    }

    m_bloomBlurShader = vex::Shader::create();
    static_cast<vex::VKShader*>(m_bloomBlurShader.get())->setVertexAttrCount(5);
    if (!m_bloomBlurShader->loadFromFiles(dir + "fullscreen.vert" + ext,
                                          dir + "bloom_blur.frag" + ext))
    {
        vex::Log::error("Failed to load Vulkan bloom_blur shader");
        m_bloomBlurShader.reset();
    }
    else
    {
        m_bloomBlurShader->preparePipeline(*m_bloomFB[0]);
    }
#endif

    // --- Initialize render mode objects ---
    // Populate init-time data (stable for the lifetime of each mode)
    RenderModeInitData initData;
    initData.fullscreenQuad       = m_fullscreenQuad.get();
    initData.whiteTexture         = m_whiteTexture.get();
    initData.flatNormalTexture    = m_flatNormalTexture.get();
    initData.meshShader           = m_meshShader.get();
    initData.fullscreenRTShader   = m_fullscreenRTShader.get();
    initData.bloomFB[0]           = m_bloomFB[0].get();
    initData.bloomFB[1]           = m_bloomFB[1].get();
    initData.bloomThresholdShader = m_bloomThresholdShader.get();
    initData.bloomBlurShader      = m_bloomBlurShader.get();
    initData.geomCache            = &m_geomCache;
    initData.cpuRaytracer         = m_cpuRaytracer.get();
    initData.resizeCPUAccumTex    = [this](uint32_t w, uint32_t h) -> vex::Texture2D*
    {
        if (w != m_raytraceTexW || h != m_raytraceTexH || !m_raytraceTexIsFloat)
        {
#ifdef VEX_BACKEND_VULKAN
            vkDeviceWaitIdle(vex::VKContext::get().getDevice());
#endif
            m_raytraceTexture    = vex::Texture2D::create(w, h, 4, true);
            m_raytraceTexW       = w;
            m_raytraceTexH       = h;
            m_raytraceTexIsFloat = true;
#ifdef VEX_BACKEND_VULKAN
            if (m_fullscreenRTShader)
                static_cast<vex::VKShader*>(m_fullscreenRTShader.get())->clearExternalTextureCache();
#endif
        }
        return m_raytraceTexture.get();
    };
#ifdef VEX_BACKEND_VULKAN
    initData.vkRTSettings = &m_gpuRTFallback;
#endif

    m_rasterMode = std::make_unique<RasterizeMode>();
    if (!m_rasterMode->init(initData))
        return false;

#ifdef VEX_BACKEND_VULKAN
    m_rasterMode->lateInitVK(initData);
#endif

    m_cpuMode = std::make_unique<CPURaytraceMode>();
    m_cpuMode->init(initData);

    m_gpuMode = std::make_unique<GPURaytraceMode>();
    if (!m_gpuMode->init(initData))
        return false;

#ifdef VEX_BACKEND_VULKAN
    initData.vkRTSettings = &m_gpuMode->getSettings();
    m_computeMode = std::make_unique<VKComputeRaytraceMode>();
    if (!m_computeMode->init(initData))
        return false;
#endif

    m_activeMode = m_rasterMode.get(); // default mode
    return true;
}

void SceneRenderer::waitIdle()
{
#ifdef VEX_BACKEND_VULKAN
    vkDeviceWaitIdle(vex::VKContext::get().getDevice());
#endif
}

void SceneRenderer::shutdown()
{
    m_activeMode = nullptr;
    // Shut down mode objects first (they may reference shared resources)
#ifdef VEX_BACKEND_VULKAN
    if (m_computeMode) { m_computeMode->shutdown(); m_computeMode.reset(); }
#endif
    if (m_gpuMode)     { m_gpuMode->shutdown();     m_gpuMode.reset(); }
    if (m_cpuMode)     { m_cpuMode->shutdown();     m_cpuMode.reset(); }
    if (m_rasterMode)  { m_rasterMode->shutdown();  m_rasterMode.reset(); }

    // Shared resources owned by SceneRenderer
#ifdef VEX_BACKEND_OPENGL
    if (m_rasterEnvMapTex) { glDeleteTextures(1, &m_rasterEnvMapTex); m_rasterEnvMapTex = 0; }
    m_fullscreenRTShader.reset();
    m_bloomThresholdShader.reset();
    m_bloomBlurShader.reset();
    m_bloomFB[0].reset();
    m_bloomFB[1].reset();
#endif

#ifdef VEX_BACKEND_VULKAN
    m_vkRasterEnvTex.reset();
    m_fullscreenRTShader.reset();
    m_bloomThresholdShader.reset();
    m_bloomBlurShader.reset();
    m_bloomFB[0].reset();
    m_bloomFB[1].reset();
#endif
    m_outlineMaskShader.reset();
    m_outlineMaskFB.reset();
    m_cpuRaytracer.reset();
    m_raytraceTexture.reset();
    m_fullscreenQuad.reset();

    m_meshShader.reset();
    m_whiteTexture.reset();
    m_flatNormalTexture.reset();
    m_framebuffer.reset();
}

bool SceneRenderer::saveImage(const std::string& path) const
{
    if (!m_framebuffer)
        return false;

    const auto& spec = m_framebuffer->getSpec();
    std::vector<uint8_t> pixels = m_framebuffer->readPixels();
    if (pixels.empty())
        return false;

    int w = static_cast<int>(spec.width);
    int h = static_cast<int>(spec.height);
    return stbi_write_png(path.c_str(), w, h, 4, pixels.data(), w * 4) != 0;
}

bool SceneRenderer::saveShadowMap(const std::string& path) const
{
    return m_rasterMode && m_rasterMode->saveShadowMap(path);
}

void SceneRenderer::setRenderMode(RenderMode mode)
{
    if (m_renderMode == mode)
        return;

    const RenderMode prevMode = m_renderMode;
    m_renderMode = mode;

    const char* modeName = (mode == RenderMode::Rasterize)       ? "Rasterize"
                         : (mode == RenderMode::CPURaytrace)     ? "CPU Raytrace"
                         : (mode == RenderMode::GPURaytrace)     ? "GPU Raytrace"
                         : (mode == RenderMode::ComputeRaytrace) ? "Compute Raytrace"
                                                                  : "Unknown";
    vex::Log::info("Render mode: " + std::string(modeName));

    // Invalidate all change-detection state so the new mode fully
    // re-initialises its camera, lights, environment, etc.
    m_prevCameraPos        = glm::vec3(std::numeric_limits<float>::quiet_NaN());
    m_prevViewMatrix       = glm::mat4(std::numeric_limits<float>::quiet_NaN());
    m_prevEnvmapIndex      = -1;
    m_prevSkyboxColor      = glm::vec3(-1.0f);
    m_prevShowLight        = !m_prevShowLight;
    m_prevLightPos         = glm::vec3(std::numeric_limits<float>::quiet_NaN());
    m_prevLightColor       = glm::vec3(-1.0f);
    m_prevLightIntensity   = -1.0f;
    m_prevShowSun          = !m_prevShowSun;
    m_prevSunDirection     = glm::vec3(std::numeric_limits<float>::quiet_NaN());
    m_prevSunColor         = glm::vec3(-1.0f);
    m_prevSunIntensity     = -1.0f;
    m_prevSunAngularRadius = -1.0f;
    m_prevCustomEnvmapPath.clear();
    m_prevAperture      = -1.0f;
    m_prevFocusDistance = -1.0f;

    m_showDenoisedResult = false;

    // Force a geometry rebuild only when leaving rasterizer mode
    if (mode != RenderMode::Rasterize && prevMode == RenderMode::Rasterize)
        m_pendingGeomRebuild = true;

    // Resolve the mode pointer — the only authoritative switch in the renderer
    switch (mode)
    {
    case RenderMode::Rasterize:       m_activeMode = m_rasterMode.get();  break;
    case RenderMode::CPURaytrace:     m_activeMode = m_cpuMode.get();     break;
    case RenderMode::GPURaytrace:     m_activeMode = m_gpuMode.get();     break;
#ifdef VEX_BACKEND_VULKAN
    case RenderMode::ComputeRaytrace: m_activeMode = m_computeMode.get(); break;
#endif
    default:                          m_activeMode = nullptr;              break;
    }

    if (m_activeMode) m_activeMode->activate();
}

float SceneRenderer::getSamplesPerSec() const
{
    return m_activeMode ? m_activeMode->getSamplesPerSec() : 0.0f;
}

uint32_t SceneRenderer::getRaytraceSampleCount() const
{
    return m_activeMode ? m_activeMode->getSampleCount() : 0;
}

void SceneRenderer::setUseLuminanceCDF(bool v)
{
    if (m_luminanceCDF == v) return;
    m_luminanceCDF = v;

    if (m_geomCache.isReady())
        m_geomCache.rebuildLightCDF(v);

    if (m_cpuRaytracer) m_cpuRaytracer->setUseLuminanceCDF(v);
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuMode && m_gpuMode->getRaytracer())
        m_gpuMode->getRaytracer()->setUseLuminanceCDF(v);
#endif
    if (m_gpuMode) m_gpuMode->onGeometryRebuilt();
#ifdef VEX_BACKEND_VULKAN
    if (m_computeMode) m_computeMode->onGeometryRebuilt();
#endif
}

uint32_t  SceneRenderer::getBVHNodeCount()  const { return m_geomCache.isReady() ? m_geomCache.bvh().nodeCount()   : 0; }
size_t    SceneRenderer::getBVHMemoryBytes() const { return m_geomCache.isReady() ? m_geomCache.bvh().memoryBytes() : 0; }
vex::AABB SceneRenderer::getBVHRootAABB()   const { return m_geomCache.isReady() ? m_geomCache.bvh().rootAABB()    : vex::AABB{}; }
float     SceneRenderer::getBVHSAHCost()    const { return m_geomCache.isReady() ? m_geomCache.bvh().sahCost()     : 0.0f; }

// --- Lazy-apply helpers ---
// Settings structs (m_cpuRTSettings / m_rasterSettings) are the single source of truth.
// These helpers push current struct values into the underlying render objects once per frame.


void SceneRenderer::applyCPURTSettings()
{
    if (!m_cpuRaytracer) return;
    const auto& s = m_cpuRTSettings;
    m_cpuRaytracer->setMaxDepth(s.maxDepth);
    m_cpuRaytracer->setEnableNEE(s.enableNEE);
    m_cpuRaytracer->setEnableAA(s.enableAA);
    m_cpuRaytracer->setEnableFireflyClamping(s.enableFireflyClamping);
    m_cpuRaytracer->setEnableEnvironment(s.enableEnvLighting);
    m_cpuRaytracer->setEnvLightMultiplier(s.envLightMultiplier);
    m_cpuRaytracer->setFlatShading(s.flatShading);
    m_cpuRaytracer->setEnableNormalMapping(s.enableNormalMapping);
    m_cpuRaytracer->setEnableEmissive(s.enableEmissive);
    m_cpuRaytracer->setExposure(s.exposure);
    m_cpuRaytracer->setGamma(s.gamma);
    m_cpuRaytracer->setEnableACES(s.enableACES);
    m_cpuRaytracer->setRayEps(s.rayEps);
    m_cpuRaytracer->setEnableRR(s.enableRR);
}

void SceneRenderer::applyRasterSettings()
{
    if (!m_rasterMode) return;
    const auto& s = m_rasterSettings;
    m_rasterMode->setExposure(s.exposure);
    m_rasterMode->setGamma(s.gamma);
    m_rasterMode->setEnableACES(s.enableACES);
    m_rasterMode->setEnableEnvLighting(s.enableEnvLighting);
    m_rasterMode->setEnvLightMultiplier(s.envLightMultiplier);
    m_rasterMode->setEnableShadows(s.enableShadows);
    m_rasterMode->setShadowNormalBiasTexels(s.shadowBiasTexels);
    m_rasterMode->setShadowStrength(s.shadowStrength);
    m_rasterMode->setShadowColor(s.shadowColor);
}

#ifdef VEX_BACKEND_OPENGL
void SceneRenderer::applyGPURTSettingsGL()
{
    if (!m_gpuMode || !m_gpuMode->getRaytracer()) return;
    auto* rt = m_gpuMode->getRaytracer();
    const auto& s = m_gpuMode->getSettings();
    rt->setMaxDepth(s.maxDepth);
    rt->setEnableNEE(s.enableNEE);
    rt->setEnableAA(s.enableAA);
    rt->setEnableFireflyClamping(s.enableFireflyClamping);
    rt->setEnableEnvironment(s.enableEnvLighting);
    rt->setEnvLightMultiplier(s.envLightMultiplier);
    rt->setFlatShading(s.flatShading);
    rt->setEnableNormalMapping(s.enableNormalMapping);
    rt->setEnableEmissive(s.enableEmissive);
    rt->setBilinearFiltering(s.bilinearFiltering);
    rt->setRayEps(s.rayEps);
    rt->setEnableRR(s.enableRR);
}
#endif

bool SceneRenderer::reloadGPUShader()
{
    return m_gpuMode ? m_gpuMode->reloadShader() : false;
}

void SceneRenderer::buildGeometry(Scene& scene, ProgressFn progress)
{
    vex::Log::info("Building scene geometry (scene load)");
    rebuildRaytraceGeometry(scene, std::move(progress));
    scene.geometryDirty = false;
    scene.materialDirty = false;
}

void SceneRenderer::rebuildRaytraceGeometry(Scene& scene, ProgressFn progress)
{
    m_geomCache.rebuild(scene, *m_cpuRaytracer, m_luminanceCDF, progress);
#ifdef VEX_BACKEND_VULKAN
    m_geomCache.buildAccelerationStructures(scene, m_gpuMode ? m_gpuMode->getRaytracer() : nullptr, progress);
#endif
    if (m_gpuMode) m_gpuMode->onGeometryRebuilt();
#ifdef VEX_BACKEND_VULKAN
    if (m_computeMode) m_computeMode->onGeometryRebuilt();
#endif
}

void SceneRenderer::rebuildMaterials(Scene& scene)
{
    m_geomCache.rebuildMaterials(scene, m_cpuRaytracer.get(), m_luminanceCDF);
    if (m_gpuMode) m_gpuMode->onGeometryRebuilt();
#ifdef VEX_BACKEND_VULKAN
    if (m_computeMode) m_computeMode->onGeometryRebuilt();
#endif
}

void SceneRenderer::renderScene(Scene& scene, int selectedNodeIdx, int selectedSubmesh)
{
    // If we just switched to a path tracing mode, force a geometry rebuild so that
    // any gizmo model matrix changes from rasterization mode are applied.
    if (m_pendingGeomRebuild)
    {
        scene.geometryDirty   = true;
        m_pendingGeomRebuild  = false;
    }

    // Full geometry rebuild (new mesh loaded, transform changed in RT mode, etc.)
    // In rasterizer mode, defer the expensive rebuild: just mark m_pendingGeomRebuild
    // so it fires the moment the user switches to a RT mode.
    if (scene.geometryDirty && m_renderMode == RenderMode::Rasterize)
    {
        m_pendingGeomRebuild = true;
        scene.geometryDirty  = false;
        // material-only changes still need to propagate for the rasterizer
        if (scene.materialDirty)
        {
            rebuildMaterials(scene);
            scene.materialDirty = false;
        }
    }

    if (scene.geometryDirty)
    {
        vex::Log::info("Building scene geometry (geometry changed)");
        rebuildRaytraceGeometry(scene, nullptr);
        scene.geometryDirty = false;
        scene.materialDirty = false; // geometry rebuild includes material bake
    }
    else if (scene.materialDirty)
    {
        rebuildMaterials(scene);
        scene.materialDirty = false;
    }

    // Outline mask pass — runs unconditionally for all render modes so path tracers
    // can sample it in their display pass. Must happen before the mode dispatch.
    {
        m_outlineActive = (selectedNodeIdx >= 0
                        && selectedNodeIdx < static_cast<int>(scene.nodes.size()));
        const auto& spec = m_framebuffer->getSpec();
        float aspect = static_cast<float>(spec.width) / static_cast<float>(spec.height);
        glm::mat4 maskView = scene.camera.getViewMatrix();
        glm::mat4 maskProj = scene.camera.getProjectionMatrix(aspect);
        renderOutlineMask(scene, selectedNodeIdx, maskView, maskProj);
    }

    // Apply settings structs to underlying render mode objects (each setter has internal no-ops)
    applyCPURTSettings();
    applyRasterSettings();
#ifdef VEX_BACKEND_OPENGL
    applyGPURTSettingsGL();
#endif

    // Build shared data and dispatch to the active render mode
    SharedRenderData shared = buildSharedRenderData();
    shared.selectedNodeIdx = selectedNodeIdx;
    shared.selectedSubmesh = selectedSubmesh;
    FrameChanges changes = computeFrameChanges(scene);

    if (m_activeMode)
        m_activeMode->render(scene, shared, changes);
}

// ---------------------------------------------------------------------------
// SceneRenderer: helpers to build per-frame shared data and change detection
// ---------------------------------------------------------------------------

SharedRenderData SceneRenderer::buildSharedRenderData()
{
    SharedRenderData shared;
    shared.outputFB            = m_framebuffer.get();
    shared.outlineMaskFB       = m_outlineMaskFB.get();
    shared.cpuAccumTex         = m_raytraceTexture.get();
    shared.outlineActive       = m_outlineActive;
    shared.enableNormalMapping = m_rasterSettings.enableNormalMapping;
    shared.showDenoisedResult  = &m_showDenoisedResult;
    shared.maxSamples          = m_maxSamples;
    shared.debugMode           = static_cast<int>(m_debugMode);
    shared.drawCalls           = &m_drawCalls;

    shared.bloomEnabled    = m_bloomSettings.enabled;
    shared.bloomIntensity  = m_bloomSettings.intensity;
    shared.bloomThreshold  = m_bloomSettings.threshold;
    shared.bloomBlurPasses = m_bloomSettings.blurPasses;

    shared.rasterEnvColor = m_rasterEnvColor;
#ifdef VEX_BACKEND_OPENGL
    shared.rasterEnvMapTex = m_rasterEnvMapTex;
#endif
#ifdef VEX_BACKEND_VULKAN
    shared.vkRasterEnvTex = m_vkRasterEnvTex.get();
    shared.vkVolumesData  = &m_vkVolumesData;
#endif

    return shared;
}

// Load/update env map data into SceneRenderer-owned buffers.
// Called from computeFrameChanges() when the env index or custom path changes.
void SceneRenderer::loadEnvData(Scene& scene)
{
    if (scene.currentEnvmap > Scene::SolidColor)
    {
        std::string envPath = (scene.currentEnvmap == Scene::CustomHDR)
            ? scene.customEnvmapPath
            : std::string(Scene::envmapPaths[scene.currentEnvmap]);

        int ew = 0, eh = 0, ech = 0;
        stbi_set_flip_vertically_on_load(false);
        float* envData = stbi_loadf(envPath.c_str(), &ew, &eh, &ech, 3);
        if (envData)
        {
            // Compute average env color for rasterizer ambient diffuse
            {
                float rSum = 0, gSum = 0, bSum = 0;
                int n = ew * eh;
                for (int i = 0; i < n; ++i) { rSum += envData[3*i]; gSum += envData[3*i+1]; bSum += envData[3*i+2]; }
                m_rasterEnvColor = glm::clamp(glm::vec3(rSum, gSum, bSum) / float(n), 0.0f, 1.0f);
            }

#ifdef VEX_BACKEND_OPENGL
            if (m_rasterEnvMapTex) glDeleteTextures(1, &m_rasterEnvMapTex);
            glGenTextures(1, &m_rasterEnvMapTex);
            glBindTexture(GL_TEXTURE_2D, m_rasterEnvMapTex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, ew, eh, 0, GL_RGB, GL_FLOAT, envData);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);
            m_glEnvMapData.assign(envData, envData + static_cast<size_t>(ew) * eh * 3);
            m_glEnvMapW = ew;
            m_glEnvMapH = eh;
#endif

#ifdef VEX_BACKEND_VULKAN
            // Build VK env map SSBO data
            m_vkEnvMapW = ew;
            m_vkEnvMapH = eh;
            m_vkEnvMapData.assign(envData, envData + static_cast<size_t>(ew) * eh * 3);

            // Build CDF for env importance sampling
            {
                int npix = ew * eh;
                std::vector<float> lum(npix);
                for (int i = 0; i < npix; ++i)
                    lum[i] = 0.2126f * envData[3*i] + 0.7152f * envData[3*i+1] + 0.0722f * envData[3*i+2];

                // Marginal CDF (rows)
                // Weight each row by sin(theta) for correct solid-angle distribution
                // of a latitude-longitude (equirectangular) map.
                constexpr float PI = 3.14159265f;
                std::vector<float> rowSums(eh);
                for (int y = 0; y < eh; ++y)
                {
                    float sinTheta = std::sin((y + 0.5f) / float(eh) * PI);
                    for (int x = 0; x < ew; ++x)
                        rowSums[y] += lum[y * ew + x] * sinTheta;
                }

                std::vector<float> margCDF(eh);
                float runMarg = 0.0f;
                for (int y = 0; y < eh; ++y) { runMarg += rowSums[y]; margCDF[y] = runMarg; }
                float totalIntegral = (runMarg > 0.0f) ? runMarg : 1.0f;
                for (float& v : margCDF) v /= totalIntegral;

                // Conditional CDF (per-row), also weighted by sinTheta
                std::vector<float> condCDF(static_cast<size_t>(ew) * eh);
                for (int y = 0; y < eh; ++y)
                {
                    float sinTheta = std::sin((y + 0.5f) / float(eh) * PI);
                    float run = 0.0f;
                    float rowSum = (rowSums[y] > 0.0f) ? rowSums[y] : 1.0f;
                    for (int x = 0; x < ew; ++x)
                    { run += lum[y * ew + x] * sinTheta; condCDF[y * ew + x] = run / rowSum; }
                }

                m_vkEnvCdfData.resize(static_cast<size_t>(eh) + static_cast<size_t>(ew) * eh + 1);
                std::copy(margCDF.begin(), margCDF.end(), m_vkEnvCdfData.begin());
                std::copy(condCDF.begin(), condCDF.end(), m_vkEnvCdfData.begin() + eh);
                m_vkEnvCdfData.back() = totalIntegral;
            }

            // RGBA8 env texture for VK rasterizer
            {
                int npix = ew * eh;
                std::vector<uint8_t> rgba8(npix * 4);
                for (int i = 0; i < npix; ++i)
                {
                    float r = envData[3*i+0]; r = r / (1.0f + r);
                    float g = envData[3*i+1]; g = g / (1.0f + g);
                    float b = envData[3*i+2]; b = b / (1.0f + b);
                    rgba8[4*i+0] = static_cast<uint8_t>(std::min(r, 1.0f) * 255.0f);
                    rgba8[4*i+1] = static_cast<uint8_t>(std::min(g, 1.0f) * 255.0f);
                    rgba8[4*i+2] = static_cast<uint8_t>(std::min(b, 1.0f) * 255.0f);
                    rgba8[4*i+3] = 255;
                }
                m_vkRasterEnvTex = vex::Texture2D::create(static_cast<uint32_t>(ew),
                                                           static_cast<uint32_t>(eh), 4);
                m_vkRasterEnvTex->setData(rgba8.data(), static_cast<uint32_t>(ew),
                                          static_cast<uint32_t>(eh), 4);
            }
#endif
            // CPU raytracer: upload env map
            if (m_cpuRaytracer)
                m_cpuRaytracer->setEnvironmentMap(envData, ew, eh);

            stbi_image_free(envData);
        }
    }
    else
    {
        // No HDR env map — clear env data and set solid sky color
        m_rasterEnvColor = scene.skyboxColor;
        if (m_cpuRaytracer)
        {
            m_cpuRaytracer->clearEnvironmentMap();
            m_cpuRaytracer->setEnvironmentColor(scene.skyboxColor);
        }
#ifdef VEX_BACKEND_OPENGL
        if (m_rasterEnvMapTex) { glDeleteTextures(1, &m_rasterEnvMapTex); m_rasterEnvMapTex = 0; }
        m_glEnvMapData.clear();
        m_glEnvMapW = 0;
        m_glEnvMapH = 0;
#endif
#ifdef VEX_BACKEND_VULKAN
        m_vkRasterEnvTex.reset();
        m_vkEnvMapData.clear();
        m_vkEnvCdfData.clear();
        m_vkEnvMapW = 0;
        m_vkEnvMapH = 0;
#endif
    }
}

FrameChanges SceneRenderer::computeFrameChanges(Scene& scene)
{
    FrameChanges changes;

    const auto& spec = m_framebuffer->getSpec();
    float aspect = static_cast<float>(spec.width) / static_cast<float>(spec.height);
    changes.camPos     = scene.camera.getPosition();
    changes.viewMatrix = scene.camera.getViewMatrix();
    changes.projMatrix = scene.camera.getProjectionMatrix(aspect);
    changes.sunDir     = scene.getSunDirection();

    // Camera change detection
    changes.cameraChanged = (changes.camPos != m_prevCameraPos || changes.viewMatrix != m_prevViewMatrix);
    if (changes.cameraChanged)
    {
        m_prevCameraPos  = changes.camPos;
        m_prevViewMatrix = changes.viewMatrix;
    }

    // DoF change detection
    changes.dofChanged = (scene.camera.aperture != m_prevAperture ||
                          scene.camera.focusDistance != m_prevFocusDistance);
    if (changes.dofChanged)
    {
        m_prevAperture      = scene.camera.aperture;
        m_prevFocusDistance = scene.camera.focusDistance;
    }

    // Env map change detection (index or custom path)
    bool customPathChanged = (scene.customEnvmapPath != m_prevCustomEnvmapPath);
    changes.envChanged = (scene.currentEnvmap != m_prevEnvmapIndex || customPathChanged);
    if (changes.envChanged)
    {
        m_prevEnvmapIndex      = scene.currentEnvmap;
        m_prevCustomEnvmapPath = scene.customEnvmapPath;
        loadEnvData(scene);
        changes.envDataChanged = true;
    }

    // Skybox color change (only relevant when using solid color env)
    changes.skyboxColorChanged = (scene.currentEnvmap == Scene::SolidColor
                                  && scene.skyboxColor != m_prevSkyboxColor);
    if (changes.skyboxColorChanged)
    {
        m_prevSkyboxColor = scene.skyboxColor;
        m_rasterEnvColor  = scene.skyboxColor;
        if (m_cpuRaytracer)
            m_cpuRaytracer->setEnvironmentColor(scene.skyboxColor);
        // Treat as env change so all RT modes reset their accumulators
        changes.envChanged = true;
    }

    // Point light change detection
    changes.lightChanged = (scene.showLight      != m_prevShowLight
                         || scene.lightPos       != m_prevLightPos
                         || scene.lightColor     != m_prevLightColor
                         || scene.lightIntensity != m_prevLightIntensity);
    if (changes.lightChanged)
    {
        m_prevShowLight      = scene.showLight;
        m_prevLightPos       = scene.lightPos;
        m_prevLightColor     = scene.lightColor;
        m_prevLightIntensity = scene.lightIntensity;
    }

    // Sun change detection
    changes.sunChanged = (scene.showSun           != m_prevShowSun
                       || changes.sunDir           != m_prevSunDirection
                       || scene.sunColor           != m_prevSunColor
                       || scene.sunIntensity       != m_prevSunIntensity
                       || scene.sunAngularRadius   != m_prevSunAngularRadius);
    if (changes.sunChanged)
    {
        m_prevShowSun          = scene.showSun;
        m_prevSunDirection     = changes.sunDir;
        m_prevSunColor         = scene.sunColor;
        m_prevSunIntensity     = scene.sunIntensity;
        m_prevSunAngularRadius = scene.sunAngularRadius;
    }

    // Volume change detection (packed into m_vkVolumesData for VK RT modes)
#ifdef VEX_BACKEND_VULKAN
    {
        auto fBF = [](float f) -> float { return f; };
        (void)fBF;
        std::vector<float> packed;
        const auto& vols = scene.volumes;
        uint32_t activeCount = 0;
        for (const auto& v : vols) if (v.enabled) ++activeCount;
        float countBits = 0.0f;
        std::memcpy(&countBits, &activeCount, sizeof(float));
        packed.push_back(countBits);
        packed.push_back(0.0f); packed.push_back(0.0f); packed.push_back(0.0f);
        for (const auto& v : vols)
        {
            if (!v.enabled) continue;
            packed.push_back(v.center.x); packed.push_back(v.center.y); packed.push_back(v.center.z);
            packed.push_back(v.density);
            packed.push_back(v.halfSize.x); packed.push_back(v.halfSize.y); packed.push_back(v.halfSize.z);
            packed.push_back(v.aniso);
            packed.push_back(v.albedo.r * v.density);
            packed.push_back(v.albedo.g * v.density);
            packed.push_back(v.albedo.b * v.density);
            packed.push_back(v.infinite ? 1.0f : 0.0f);
        }
        changes.volumesChanged = (packed != m_prevVolumesData);
        if (changes.volumesChanged)
        {
            m_vkVolumesData   = packed;
            m_prevVolumesData = packed;
        }
    }

    // Populate VK env data pointers in changes (valid this frame after loadEnvData)
    changes.vkEnvMapData = m_vkEnvMapData.empty()  ? nullptr : &m_vkEnvMapData;
    changes.vkEnvCdfData = m_vkEnvCdfData.empty()  ? nullptr : &m_vkEnvCdfData;
    changes.vkEnvMapW    = m_vkEnvMapW;
    changes.vkEnvMapH    = m_vkEnvMapH;
#endif

#ifdef VEX_BACKEND_OPENGL
    // Populate GL env data pointer in changes (valid this frame after loadEnvData)
    changes.glEnvMapData = m_glEnvMapData.empty() ? nullptr : m_glEnvMapData.data();
    changes.glEnvMapW    = m_glEnvMapW;
    changes.glEnvMapH    = m_glEnvMapH;
#endif

    return changes;
}

// ---------------------------------------------------------------------------
// Denoising
// ---------------------------------------------------------------------------

void SceneRenderer::triggerDenoise()
{
    if (!m_denoiser || !m_denoiser->isReady()) return;

    const auto& spec = m_framebuffer->getSpec();
    uint32_t w = spec.width;
    uint32_t h = spec.height;

    // Ensure accumulation texture exists at the right size and as float (HDR)
    auto ensureAccumTex = [&]()
    {
        if (!m_raytraceTexture || w != m_raytraceTexW || h != m_raytraceTexH || !m_raytraceTexIsFloat)
        {
            m_raytraceTexture    = vex::Texture2D::create(w, h, 4, true);
            m_raytraceTexW       = w;
            m_raytraceTexH       = h;
            m_raytraceTexIsFloat = true;
#ifdef VEX_BACKEND_VULKAN
            if (m_fullscreenRTShader)
                static_cast<vex::VKShader*>(m_fullscreenRTShader.get())->clearExternalTextureCache();
#endif
        }
    };

#ifdef VEX_BACKEND_VULKAN
    if (m_renderMode == RenderMode::GPURaytrace && m_gpuMode && m_gpuMode->getRaytracer())
        m_gpuMode->getRaytracer()->readbackLinearHDR(m_denoiseLinearHDR);
    else if (m_renderMode == RenderMode::ComputeRaytrace && m_computeMode && m_computeMode->getRaytracer())
        m_computeMode->getRaytracer()->readbackLinearHDR(m_denoiseLinearHDR);
    else
#endif
    if (m_renderMode == RenderMode::CPURaytrace && m_cpuRaytracer)
        m_cpuRaytracer->getLinearHDR(m_denoiseLinearHDR);
    else { return; }

    if (m_denoiseLinearHDR.empty()) return;

    uint32_t sampleCount = getRaytraceSampleCount();

    auto t0 = std::chrono::steady_clock::now();
    m_denoiser->denoise(m_denoiseLinearHDR.data(), w, h);
    float ms = std::chrono::duration<float, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();

    // Pack RGB HDR → RGBA32F; tone mapping applied in fullscreen shader
    uint32_t pixelCount = w * h;
    m_denoisedHDR.resize(pixelCount * 4);
    for (uint32_t i = 0; i < pixelCount; ++i)
    {
        m_denoisedHDR[i * 4 + 0] = m_denoiseLinearHDR[i * 3 + 0];
        m_denoisedHDR[i * 4 + 1] = m_denoiseLinearHDR[i * 3 + 1];
        m_denoisedHDR[i * 4 + 2] = m_denoiseLinearHDR[i * 3 + 2];
        m_denoisedHDR[i * 4 + 3] = 1.0f;
    }
    ensureAccumTex();
    m_raytraceTexture->setData(m_denoisedHDR.data(), w, h, 4);
    m_showDenoisedResult = true;

    char buf[128];
    std::snprintf(buf, sizeof(buf), "Denoiser: %ux%u on %u samples took %.0f ms", w, h, sampleCount, ms);
    vex::Log::info(buf);
}

// ---------------------------------------------------------------------------

void SceneRenderer::triggerDenoiseAux()
{
    if (!m_denoiser || !m_denoiser->isReady()) return;

    const auto& spec = m_framebuffer->getSpec();
    uint32_t w = spec.width;
    uint32_t h = spec.height;

    // Ensure accumulation texture exists at the right size and as RGBA32F float (HDR)
    auto ensureAccumTex = [&]()
    {
        if (!m_raytraceTexture || w != m_raytraceTexW || h != m_raytraceTexH || !m_raytraceTexIsFloat)
        {
            m_raytraceTexture    = vex::Texture2D::create(w, h, 4, true);
            m_raytraceTexW       = w;
            m_raytraceTexH       = h;
            m_raytraceTexIsFloat = true;
#ifdef VEX_BACKEND_VULKAN
            if (m_fullscreenRTShader)
                static_cast<vex::VKShader*>(m_fullscreenRTShader.get())->clearExternalTextureCache();
#endif
        }
    };

#ifdef VEX_BACKEND_VULKAN
    if (m_renderMode == RenderMode::GPURaytrace && m_gpuMode && m_gpuMode->getRaytracer())
    {
        m_gpuMode->getRaytracer()->readbackLinearHDR(m_denoiseLinearHDR);
        m_gpuMode->getRaytracer()->readbackAuxBuffers(m_denoiseAlbedo, m_denoiseNormal);
    }
    else if (m_renderMode == RenderMode::ComputeRaytrace && m_computeMode && m_computeMode->getRaytracer())
    {
        m_computeMode->getRaytracer()->readbackLinearHDR(m_denoiseLinearHDR);
        m_computeMode->getRaytracer()->readbackAuxBuffers(m_denoiseAlbedo, m_denoiseNormal);
    }
    else
#endif
    if (m_renderMode == RenderMode::CPURaytrace && m_cpuRaytracer)
    {
        m_cpuRaytracer->getLinearHDR(m_denoiseLinearHDR);
        m_cpuRaytracer->getAuxBuffers(m_denoiseAlbedo, m_denoiseNormal);
    }
    else { return; }

    if (m_denoiseLinearHDR.empty() || m_denoiseAlbedo.empty() || m_denoiseNormal.empty()) return;

    uint32_t sampleCount = getRaytraceSampleCount();

    auto t0 = std::chrono::steady_clock::now();
    m_denoiser->denoiseAux(m_denoiseLinearHDR.data(), m_denoiseAlbedo.data(), m_denoiseNormal.data(), w, h);
    float ms = std::chrono::duration<float, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();

    // Pack RGB HDR → RGBA32F; tone mapping applied in fullscreen shader
    uint32_t pixelCount = w * h;
    m_denoisedHDR.resize(pixelCount * 4);
    for (uint32_t i = 0; i < pixelCount; ++i)
    {
        m_denoisedHDR[i * 4 + 0] = m_denoiseLinearHDR[i * 3 + 0];
        m_denoisedHDR[i * 4 + 1] = m_denoiseLinearHDR[i * 3 + 1];
        m_denoisedHDR[i * 4 + 2] = m_denoiseLinearHDR[i * 3 + 2];
        m_denoisedHDR[i * 4 + 3] = 1.0f;
    }
    ensureAccumTex();
    m_raytraceTexture->setData(m_denoisedHDR.data(), w, h, 4);
    m_showDenoisedResult = true;

    char buf[128];
    std::snprintf(buf, sizeof(buf), "Denoiser+: %ux%u on %u samples took %.0f ms", w, h, sampleCount, ms);
    vex::Log::info(buf);
}

// ---------------------------------------------------------------------------
// Outline mask (selection highlight — rendered every frame before mode dispatch)
// ---------------------------------------------------------------------------

void SceneRenderer::renderOutlineMask(Scene& scene, int selectedNodeIdx,
                                      const glm::mat4& view, const glm::mat4& proj)
{
    m_outlineMaskFB->bind();
    m_outlineMaskFB->clear(0.0f, 0.0f, 0.0f, 1.0f);

    if (m_outlineActive)
    {
#ifdef VEX_BACKEND_OPENGL
        glDisable(GL_DEPTH_TEST);
#endif
#ifdef VEX_BACKEND_VULKAN
        // VK: bind() flushes UBO to GPU, so uniforms must be written first.
        m_outlineMaskShader->setMat4("u_view",        view);
        m_outlineMaskShader->setMat4("u_projection",  proj);
        m_outlineMaskShader->bind();
#else
        // GL: glUniform* requires the shader to be bound first.
        m_outlineMaskShader->bind();
        m_outlineMaskShader->setMat4("u_view",       view);
        m_outlineMaskShader->setMat4("u_projection", proj);
#endif

        const glm::mat4 nodeWorld = scene.getWorldMatrix(selectedNodeIdx);
        for (auto& sm : scene.nodes[selectedNodeIdx].submeshes)
        {
            m_outlineMaskShader->setMat4("u_model", nodeWorld * sm.modelMatrix);
            sm.mesh->draw();
        }

        m_outlineMaskShader->unbind();
#ifdef VEX_BACKEND_OPENGL
        glEnable(GL_DEPTH_TEST);
#endif
    }

    m_outlineMaskFB->unbind();
}

// ---------------------------------------------------------------------------
// Shadow map debug display
// ---------------------------------------------------------------------------

uintptr_t SceneRenderer::getShadowMapDisplayHandle()
{
    return m_rasterMode ? m_rasterMode->getShadowMapDisplayHandle() : 0;
}

bool SceneRenderer::shadowMapFlipsUV() const
{
    return m_rasterMode ? m_rasterMode->shadowMapFlipsUV() : false;
}

std::pair<int,int> SceneRenderer::pick(Scene& scene, int pixelX, int pixelY)
{
#ifdef VEX_BACKEND_VULKAN
    // CPU ray-triangle intersection (Möller–Trumbore) — no GPU readback needed.
    const auto& spec = m_framebuffer->getSpec();
    float aspect = static_cast<float>(spec.width) / static_cast<float>(spec.height);

    glm::mat4 view  = scene.camera.getViewMatrix();
    glm::mat4 proj  = scene.camera.getProjectionMatrix(aspect);
    glm::mat4 invVP = glm::inverse(proj * view);

    float ndcX = (pixelX + 0.5f) / static_cast<float>(spec.width)  * 2.0f - 1.0f;
    float ndcY = 1.0f - (pixelY + 0.5f) / static_cast<float>(spec.height) * 2.0f;

    glm::vec4 nearH = invVP * glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
    glm::vec4 farH  = invVP * glm::vec4(ndcX, ndcY,  1.0f, 1.0f);
    nearH /= nearH.w;
    farH  /= farH.w;

    glm::vec3 rayOrigin = glm::vec3(nearH);
    glm::vec3 rayDir    = glm::normalize(glm::vec3(farH) - rayOrigin);

    float bestT       = std::numeric_limits<float>::max();
    int   bestGroup   = -1;
    int   bestSubmesh = -1;

    constexpr float EPS = 1e-7f;

    for (int ni = 0; ni < static_cast<int>(scene.nodes.size()); ++ni)
    {
        const glm::mat4 nodeWorld = scene.getWorldMatrix(ni);
        for (int si = 0; si < static_cast<int>(scene.nodes[ni].submeshes.size()); ++si)
        {
            const auto& sm      = scene.nodes[ni].submeshes[si];
            const glm::mat4 M   = nodeWorld * sm.modelMatrix;
            const auto& md      = sm.meshData;
            const auto& verts   = md.vertices;
            const auto& indices = md.indices;

            for (size_t i = 0; i + 2 < indices.size(); i += 3)
            {
                glm::vec3 v0 = glm::vec3(M * glm::vec4(verts[indices[i + 0]].position, 1.0f));
                glm::vec3 v1 = glm::vec3(M * glm::vec4(verts[indices[i + 1]].position, 1.0f));
                glm::vec3 v2 = glm::vec3(M * glm::vec4(verts[indices[i + 2]].position, 1.0f));

                glm::vec3 e1 = v1 - v0;
                glm::vec3 e2 = v2 - v0;
                glm::vec3 h  = glm::cross(rayDir, e2);
                float     a  = glm::dot(e1, h);
                if (a > -EPS && a < EPS) continue;

                float     f = 1.0f / a;
                glm::vec3 s = rayOrigin - v0;
                float     u = f * glm::dot(s, h);
                if (u < 0.0f || u > 1.0f) continue;

                glm::vec3 q = glm::cross(s, e1);
                float     v = f * glm::dot(rayDir, q);
                if (v < 0.0f || u + v > 1.0f) continue;

                float t = f * glm::dot(e2, q);
                if (t > EPS && t < bestT)
                {
                    bestT       = t;
                    bestGroup   = ni;
                    bestSubmesh = si;
                }
            }
        }
    }
    return { bestGroup, bestSubmesh };
#else
    if (!m_rasterMode)
        return { -1, -1 };
    SharedRenderData shared = buildSharedRenderData();
    return m_rasterMode->pick(scene, shared, pixelX, pixelY);
#endif
}

#ifdef VEX_BACKEND_VULKAN
float SceneRenderer::getVKSamplesPerSec() const
{
    return m_gpuMode ? m_gpuMode->getSamplesPerSec() : 0.0f;
}

float SceneRenderer::getVKComputeSamplesPerSec() const
{
    return m_computeMode ? m_computeMode->getSamplesPerSec() : 0.0f;
}
#endif
