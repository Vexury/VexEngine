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
#include <cstdio>
#include <limits>
#include <unordered_map>
#include <vector>

static constexpr float GEOMETRY_EPSILON = 1e-8f;

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

#ifdef VEX_BACKEND_OPENGL
    m_pickShader = vex::Shader::create();
    if (!m_pickShader->loadFromFiles(dir + "pick.vert" + ext, dir + "pick.frag" + ext))
        return false;

#endif

    // Fullscreen shader and quad for raytracing display
    m_fullscreenShader = vex::Shader::create();
    if (!m_fullscreenShader->loadFromFiles(dir + "fullscreen.vert" + ext, dir + "fullscreen.frag" + ext))
        return false;
#ifdef VEX_BACKEND_VULKAN
    static_cast<vex::VKShader*>(m_fullscreenShader.get())->setVertexAttrCount(5);
#endif

    m_fullscreenQuad = vex::Mesh::create();
    m_fullscreenQuad->upload(buildFullscreenQuadData());

    m_framebuffer = vex::Framebuffer::create({ .width = 1280, .height = 720, .hasDepth = true });
#ifdef VEX_BACKEND_OPENGL
    m_pickFB = vex::Framebuffer::create({ .width = 1280, .height = 720, .hasDepth = true });
#endif

    // Shadow map framebuffer (depth-only, fixed resolution)
    m_shadowFB = vex::Framebuffer::create({ .width = SHADOW_MAP_SIZE, .height = SHADOW_MAP_SIZE,
                                            .hasDepth = true, .depthOnly = true });
    m_shadowShader = vex::Shader::create();
    if (!m_shadowShader->loadFromFiles(dir + "shadow.vert" + ext, dir + "shadow.frag" + ext))
        return false;

    // Screen-space outline mask framebuffer (no depth — captures full silhouette of selected objects)
    m_outlineMaskFB = vex::Framebuffer::create({ .width = 1280, .height = 720 });
    m_outlineMaskShader = vex::Shader::create();
    if (!m_outlineMaskShader->loadFromFiles(dir + "outline_mask.vert" + ext, dir + "outline_mask.frag" + ext))
        return false;

    // Create backend-specific pipelines for the offscreen framebuffer
    m_meshShader->preparePipeline(*m_framebuffer);
    m_fullscreenShader->preparePipeline(*m_framebuffer);
    if (scene.skybox)
        scene.skybox->preparePipeline(*m_framebuffer);

#ifdef VEX_BACKEND_VULKAN
    {
        auto* vkShadowShader = static_cast<vex::VKShader*>(m_shadowShader.get());
        auto* vkShadowFB     = static_cast<vex::VKFramebuffer*>(m_shadowFB.get());
        vkShadowShader->createPipeline(vkShadowFB->getRenderPass(),
                                       true, true, 1, VK_POLYGON_MODE_FILL, true);

        auto* vkMaskShader = static_cast<vex::VKShader*>(m_outlineMaskShader.get());
        auto* vkMaskFB     = static_cast<vex::VKFramebuffer*>(m_outlineMaskFB.get());
        vkMaskShader->createPipeline(vkMaskFB->getRenderPass(),
                                     false, false, 5, VK_POLYGON_MODE_FILL);
    }
#endif

    // Initialize CPU raytracer
    m_cpuRaytracer = std::make_unique<vex::CPURaytracer>();

#ifdef VEX_BACKEND_OPENGL
    // Initialize GPU raytracer
    m_gpuRaytracer = std::make_unique<vex::GLGPURaytracer>();
    if (!m_gpuRaytracer->init())
    {
        vex::Log::error("Failed to initialize GPU raytracer");
        m_gpuRaytracer.reset();
    }

    // Load fullscreen RT shader (tone mapping)
    m_fullscreenRTShader = vex::Shader::create();
    if (!m_fullscreenRTShader->loadFromFiles(dir + "fullscreen.vert" + ext, dir + "fullscreen_rt.frag" + ext))
    {
        vex::Log::error("Failed to load fullscreen_rt shader");
        m_fullscreenRTShader.reset();
    }

    // Intermediate HDR framebuffer for rasterizer (geometry renders here, then tone-mapped to m_framebuffer)
    m_rasterHDRFB = vex::Framebuffer::create({ .width = 1280, .height = 720, .hasDepth = true });

#endif

#ifdef VEX_BACKEND_VULKAN
    m_rasterHDRFB = vex::Framebuffer::create({ .width = 1280, .height = 720, .hasDepth = true });

    m_vkRaytracer = std::make_unique<vex::VKGpuRaytracer>();
    if (!m_vkRaytracer->init())
    {
        vex::Log::error("Failed to initialize Vulkan RT raytracer");
        m_vkRaytracer.reset();
    }

    m_vkFullscreenRTShader = vex::Shader::create();
    static_cast<vex::VKShader*>(m_vkFullscreenRTShader.get())->setVertexAttrCount(5);
    if (!m_vkFullscreenRTShader->loadFromFiles(dir + "fullscreen.vert" + ext, dir + "fullscreen_rt.frag" + ext))
    {
        vex::Log::error("Failed to load Vulkan fullscreen_rt shader");
        m_vkFullscreenRTShader.reset();
    }
    else
    {
        m_vkFullscreenRTShader->preparePipeline(*m_framebuffer);
    }
#endif

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
#ifdef VEX_BACKEND_OPENGL
    if (m_rasterEnvMapTex) { glDeleteTextures(1, &m_rasterEnvMapTex); m_rasterEnvMapTex = 0; }
    if (m_gpuRaytracer)
        m_gpuRaytracer->shutdown();
    m_gpuRaytracer.reset();
    m_fullscreenRTShader.reset();
#endif

#ifdef VEX_BACKEND_VULKAN
    m_vkRasterEnvTex.reset();
    m_vkFullscreenRTShader.reset();
    if (m_vkRaytracer)
        m_vkRaytracer->shutdown();
    m_vkRaytracer.reset();
#endif
    m_outlineMaskShader.reset();
    m_outlineMaskFB.reset();
    m_rasterHDRFB.reset();
    m_shadowShader.reset();
    m_shadowFB.reset();
    m_cpuRaytracer.reset();
    m_raytraceTexture.reset();
    m_fullscreenQuad.reset();
    m_fullscreenShader.reset();
    m_meshShader.reset();
    m_whiteTexture.reset();
    m_flatNormalTexture.reset();
    m_pickShader.reset();
    m_pickFB.reset();
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
    if (!m_shadowFB || !m_shadowMapEverRendered)
        return false;

    constexpr uint32_t OUT  = 1024;
    constexpr uint32_t SRC  = SHADOW_MAP_SIZE; // 4096
    constexpr uint32_t STEP = SRC / OUT;       // 4

    std::vector<float>   srcDepth;
    std::vector<uint8_t> outPixels(OUT * OUT);

#ifdef VEX_BACKEND_OPENGL
    {
        auto* fb = static_cast<vex::GLFramebuffer*>(m_shadowFB.get());
        fb->prepareDepthForDisplay(); // disable compare mode for raw float read
        srcDepth.resize(static_cast<size_t>(SRC) * SRC);
        glBindTexture(GL_TEXTURE_2D, fb->getDepthAttachment());
        glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, srcDepth.data());
        glBindTexture(GL_TEXTURE_2D, 0);
        // compare mode stays disabled until the next shadow pass calls restoreDepthForSampling()

        // Box-filter downsample; flip vertically (GL stores bottom-to-top)
        for (uint32_t oy = 0; oy < OUT; ++oy)
        {
            uint32_t sy0 = (OUT - 1 - oy) * STEP;
            for (uint32_t ox = 0; ox < OUT; ++ox)
            {
                float sum = 0.0f;
                uint32_t sx0 = ox * STEP;
                for (uint32_t dy = 0; dy < STEP; ++dy)
                    for (uint32_t dx = 0; dx < STEP; ++dx)
                        sum += srcDepth[(sy0 + dy) * SRC + sx0 + dx];
                outPixels[oy * OUT + ox] = static_cast<uint8_t>(
                    std::clamp(sum * (1.0f / float(STEP * STEP)), 0.0f, 1.0f) * 255.0f + 0.5f);
            }
        }
    }
#endif

#ifdef VEX_BACKEND_VULKAN
    {
        auto* fb = static_cast<vex::VKFramebuffer*>(m_shadowFB.get());
        srcDepth = fb->readDepthPixels();
        if (srcDepth.empty())
            return false;

        // Box-filter downsample; flip vertically (shadow map row 0 = scene bottom in VK)
        for (uint32_t oy = 0; oy < OUT; ++oy)
        {
            uint32_t sy0 = (OUT - 1 - oy) * STEP;
            for (uint32_t ox = 0; ox < OUT; ++ox)
            {
                float sum = 0.0f;
                uint32_t sx0 = ox * STEP;
                for (uint32_t dy = 0; dy < STEP; ++dy)
                    for (uint32_t dx = 0; dx < STEP; ++dx)
                        sum += srcDepth[(sy0 + dy) * SRC + sx0 + dx];
                outPixels[oy * OUT + ox] = static_cast<uint8_t>(
                    std::clamp(sum * (1.0f / float(STEP * STEP)), 0.0f, 1.0f) * 255.0f + 0.5f);
            }
        }
    }
#endif

    return stbi_write_png(path.c_str(),
        static_cast<int>(OUT), static_cast<int>(OUT),
        1, outPixels.data(), static_cast<int>(OUT)) != 0;
}

void SceneRenderer::setRenderMode(RenderMode mode)
{
    if (m_renderMode == mode)
        return;

    m_renderMode = mode;

    // Invalidate all change-detection state so the new mode fully
    // re-initialises its camera, lights, environment, etc.
    m_prevCameraPos        = glm::vec3(std::numeric_limits<float>::quiet_NaN());
    m_prevViewMatrix       = glm::mat4(std::numeric_limits<float>::quiet_NaN());
    m_prevEnvmapIndex      = -1;
    m_prevSkyboxColor      = glm::vec3(-1.0f);
    m_prevShowLight        = !m_prevShowLight;   // toggle to guarantee mismatch
    m_prevLightPos         = glm::vec3(std::numeric_limits<float>::quiet_NaN());
    m_prevLightColor       = glm::vec3(-1.0f);
    m_prevLightIntensity   = -1.0f;
    m_prevShowSun          = !m_prevShowSun;
    m_prevSunDirection     = glm::vec3(std::numeric_limits<float>::quiet_NaN());
    m_prevSunColor         = glm::vec3(-1.0f);
    m_prevSunIntensity     = -1.0f;
    m_prevSunAngularRadius = -1.0f;
    m_prevCustomEnvmapPath.clear();
    m_prevAperture      = -1.0f;  // force DoF re-upload on next frame
    m_prevFocusDistance = -1.0f;

    // When entering any path tracing mode, force a full geometry rebuild so that
    // any model matrix changes made via gizmos during rasterization are applied.
    if (mode == RenderMode::CPURaytrace || mode == RenderMode::GPURaytrace)
        m_pendingGeomRebuild = true;

    if (mode == RenderMode::CPURaytrace && m_cpuRaytracer)
        m_cpuRaytracer->reset();

#ifdef VEX_BACKEND_OPENGL
    if (mode == RenderMode::GPURaytrace && m_gpuRaytracer)
    {
        m_gpuRaytracer->reset();
        m_gpuGeometryDirty = true;
    }
#endif
#ifdef VEX_BACKEND_VULKAN
    if (mode == RenderMode::GPURaytrace && m_vkRaytracer)
    {
        m_vkRaytracer->reset();
        m_vkSampleCount = 0;
        m_vkGeomDirty = true;
    }
#endif
}

uint32_t SceneRenderer::getRaytraceSampleCount() const
{
#ifdef VEX_BACKEND_OPENGL
    if (m_renderMode == RenderMode::GPURaytrace && m_gpuRaytracer)
        return m_gpuRaytracer->getSampleCount();
#endif
#ifdef VEX_BACKEND_VULKAN
    if (m_renderMode == RenderMode::GPURaytrace)
        return m_vkSampleCount;
#endif
    return m_cpuRaytracer ? m_cpuRaytracer->getSampleCount() : 0;
}

void SceneRenderer::setMaxDepth(int depth)
{
    if (m_cpuRaytracer)
        m_cpuRaytracer->setMaxDepth(depth);
}

int SceneRenderer::getMaxDepth() const
{
    return m_cpuRaytracer ? m_cpuRaytracer->getMaxDepth() : 5;
}

void SceneRenderer::setEnableNEE(bool v)             { if (m_cpuRaytracer) m_cpuRaytracer->setEnableNEE(v); }
bool SceneRenderer::getEnableNEE() const              { return m_cpuRaytracer ? m_cpuRaytracer->getEnableNEE() : true; }

void SceneRenderer::setEnableFireflyClamping(bool v)  { if (m_cpuRaytracer) m_cpuRaytracer->setEnableFireflyClamping(v); }
bool SceneRenderer::getEnableFireflyClamping() const   { return m_cpuRaytracer ? m_cpuRaytracer->getEnableFireflyClamping() : true; }

void SceneRenderer::setEnableAA(bool v)               { if (m_cpuRaytracer) m_cpuRaytracer->setEnableAA(v); }
bool SceneRenderer::getEnableAA() const                { return m_cpuRaytracer ? m_cpuRaytracer->getEnableAA() : true; }

void SceneRenderer::setEnableEnvironment(bool v)      { if (m_cpuRaytracer) m_cpuRaytracer->setEnableEnvironment(v); }
bool SceneRenderer::getEnableEnvironment() const       { return m_cpuRaytracer ? m_cpuRaytracer->getEnableEnvironment() : false; }

void SceneRenderer::setEnvLightMultiplier(float v)    { if (m_cpuRaytracer) m_cpuRaytracer->setEnvLightMultiplier(v); }
float SceneRenderer::getEnvLightMultiplier() const    { return m_cpuRaytracer ? m_cpuRaytracer->getEnvLightMultiplier() : 1.0f; }

void SceneRenderer::setFlatShading(bool v)             { if (m_cpuRaytracer) m_cpuRaytracer->setFlatShading(v); }
bool SceneRenderer::getFlatShading() const              { return m_cpuRaytracer ? m_cpuRaytracer->getFlatShading() : false; }

void SceneRenderer::setEnableNormalMapping(bool v)      { m_enableNormalMapping = v; if (m_cpuRaytracer) m_cpuRaytracer->setEnableNormalMapping(v); }
bool SceneRenderer::getEnableNormalMapping() const       { return m_enableNormalMapping; }

void SceneRenderer::setEnableEmissive(bool v)           { if (m_cpuRaytracer) m_cpuRaytracer->setEnableEmissive(v); }
bool SceneRenderer::getEnableEmissive() const            { return m_cpuRaytracer ? m_cpuRaytracer->getEnableEmissive() : true; }

void SceneRenderer::setExposure(float v)               { if (m_cpuRaytracer) m_cpuRaytracer->setExposure(v); }
float SceneRenderer::getExposure() const                { return m_cpuRaytracer ? m_cpuRaytracer->getExposure() : 0.0f; }

void SceneRenderer::setGamma(float v)                  { if (m_cpuRaytracer) m_cpuRaytracer->setGamma(v); }
float SceneRenderer::getGamma() const                   { return m_cpuRaytracer ? m_cpuRaytracer->getGamma() : 2.2f; }

void SceneRenderer::setEnableACES(bool v)              { if (m_cpuRaytracer) m_cpuRaytracer->setEnableACES(v); }
bool SceneRenderer::getEnableACES() const               { return m_cpuRaytracer ? m_cpuRaytracer->getEnableACES() : true; }

void SceneRenderer::setRayEps(float v)                 { if (m_cpuRaytracer) m_cpuRaytracer->setRayEps(v); }
float SceneRenderer::getRayEps() const                  { return m_cpuRaytracer ? m_cpuRaytracer->getRayEps() : 1e-4f; }

void SceneRenderer::setEnableRR(bool v)                { if (m_cpuRaytracer) m_cpuRaytracer->setEnableRR(v); }
bool SceneRenderer::getEnableRR() const                 { return m_cpuRaytracer ? m_cpuRaytracer->getEnableRR() : true; }

uint32_t SceneRenderer::getBVHNodeCount() const    { return m_cpuRaytracer ? m_cpuRaytracer->getBVHNodeCount() : 0; }
size_t   SceneRenderer::getBVHMemoryBytes() const  { return m_cpuRaytracer ? m_cpuRaytracer->getBVHMemoryBytes() : 0; }
vex::AABB SceneRenderer::getBVHRootAABB() const    { return m_cpuRaytracer ? m_cpuRaytracer->getBVHRootAABB() : vex::AABB{}; }
float     SceneRenderer::getBVHSAHCost() const     { return m_cpuRaytracer ? m_cpuRaytracer->getBVHSAHCost() : 0.0f; }

// --- GPU Raytracing settings ---

void SceneRenderer::setGPUMaxDepth([[maybe_unused]] int d)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setMaxDepth(d);
#endif
#ifdef VEX_BACKEND_VULKAN
    if (d != m_vkMaxDepth) { m_vkMaxDepth = d; if (m_vkRaytracer) { m_vkRaytracer->reset(); m_vkSampleCount = 0; } }
#endif
}

int SceneRenderer::getGPUMaxDepth() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getMaxDepth() : 5;
#elif defined(VEX_BACKEND_VULKAN)
    return m_vkMaxDepth;
#else
    return 5;
#endif
}

void SceneRenderer::setGPUEnableNEE([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setEnableNEE(v);
#endif
#ifdef VEX_BACKEND_VULKAN
    if (v != m_vkEnableNEE) { m_vkEnableNEE = v; if (m_vkRaytracer) { m_vkRaytracer->reset(); m_vkSampleCount = 0; } }
#endif
}

bool SceneRenderer::getGPUEnableNEE() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getEnableNEE() : true;
#elif defined(VEX_BACKEND_VULKAN)
    return m_vkEnableNEE;
#else
    return true;
#endif
}

void SceneRenderer::setGPUEnableAA([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setEnableAA(v);
#endif
#ifdef VEX_BACKEND_VULKAN
    if (v != m_vkEnableAA) { m_vkEnableAA = v; if (m_vkRaytracer) { m_vkRaytracer->reset(); m_vkSampleCount = 0; } }
#endif
}

bool SceneRenderer::getGPUEnableAA() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getEnableAA() : true;
#elif defined(VEX_BACKEND_VULKAN)
    return m_vkEnableAA;
#else
    return true;
#endif
}

void SceneRenderer::setGPUEnableFireflyClamping([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setEnableFireflyClamping(v);
#endif
#ifdef VEX_BACKEND_VULKAN
    if (v != m_vkEnableFireflyClamping) { m_vkEnableFireflyClamping = v; if (m_vkRaytracer) { m_vkRaytracer->reset(); m_vkSampleCount = 0; } }
#endif
}

bool SceneRenderer::getGPUEnableFireflyClamping() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getEnableFireflyClamping() : true;
#elif defined(VEX_BACKEND_VULKAN)
    return m_vkEnableFireflyClamping;
#else
    return true;
#endif
}

void SceneRenderer::setGPUEnableEnvironment([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setEnableEnvironment(v);
#endif
#ifdef VEX_BACKEND_VULKAN
    if (v != m_vkEnableEnvLighting) { m_vkEnableEnvLighting = v; if (m_vkRaytracer) { m_vkRaytracer->reset(); m_vkSampleCount = 0; } }
#endif
}

bool SceneRenderer::getGPUEnableEnvironment() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getEnableEnvironment() : false;
#elif defined(VEX_BACKEND_VULKAN)
    return m_vkEnableEnvLighting;
#else
    return false;
#endif
}

void SceneRenderer::setGPUEnvLightMultiplier([[maybe_unused]] float v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setEnvLightMultiplier(v);
#endif
#ifdef VEX_BACKEND_VULKAN
    if (v != m_vkEnvLightMultiplier) { m_vkEnvLightMultiplier = v; if (m_vkRaytracer) m_vkRaytracer->reset(); m_vkSampleCount = 0; }
#endif
}

float SceneRenderer::getGPUEnvLightMultiplier() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getEnvLightMultiplier() : 1.0f;
#elif defined(VEX_BACKEND_VULKAN)
    return m_vkEnvLightMultiplier;
#else
    return 1.0f;
#endif
}

void  SceneRenderer::setRasterEnableEnvLighting(bool v)   { m_rasterEnableEnvLighting = v; }
bool  SceneRenderer::getRasterEnableEnvLighting() const    { return m_rasterEnableEnvLighting; }
void  SceneRenderer::setRasterEnvLightMultiplier(float v)  { m_rasterEnvLightMultiplier = v; }
float SceneRenderer::getRasterEnvLightMultiplier() const   { return m_rasterEnvLightMultiplier; }

void  SceneRenderer::setRasterExposure(float v)   { m_rasterExposure = v; }
float SceneRenderer::getRasterExposure() const     { return m_rasterExposure; }

void  SceneRenderer::setRasterGamma(float v)       { m_rasterGamma = v; }
float SceneRenderer::getRasterGamma() const        { return m_rasterGamma; }

void  SceneRenderer::setRasterEnableACES(bool v)   { m_rasterEnableACES = v; }
bool  SceneRenderer::getRasterEnableACES() const   { return m_rasterEnableACES; }

void SceneRenderer::setGPUFlatShading([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setFlatShading(v);
#endif
#ifdef VEX_BACKEND_VULKAN
    if (v != m_vkFlatShading) { m_vkFlatShading = v; if (m_vkRaytracer) { m_vkRaytracer->reset(); m_vkSampleCount = 0; } }
#endif
}

bool SceneRenderer::getGPUFlatShading() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getFlatShading() : false;
#elif defined(VEX_BACKEND_VULKAN)
    return m_vkFlatShading;
#else
    return false;
#endif
}

void SceneRenderer::setGPUEnableNormalMapping([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setEnableNormalMapping(v);
#endif
#ifdef VEX_BACKEND_VULKAN
    if (v != m_vkEnableNormalMapping) { m_vkEnableNormalMapping = v; if (m_vkRaytracer) { m_vkRaytracer->reset(); m_vkSampleCount = 0; } }
#endif
}

bool SceneRenderer::getGPUEnableNormalMapping() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getEnableNormalMapping() : true;
#elif defined(VEX_BACKEND_VULKAN)
    return m_vkEnableNormalMapping;
#else
    return true;
#endif
}

void SceneRenderer::setGPUEnableEmissive([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setEnableEmissive(v);
#endif
#ifdef VEX_BACKEND_VULKAN
    if (v != m_vkEnableEmissive) { m_vkEnableEmissive = v; if (m_vkRaytracer) { m_vkRaytracer->reset(); m_vkSampleCount = 0; } }
#endif
}

bool SceneRenderer::getGPUEnableEmissive() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getEnableEmissive() : true;
#elif defined(VEX_BACKEND_VULKAN)
    return m_vkEnableEmissive;
#else
    return true;
#endif
}

void SceneRenderer::setGPUBilinearFiltering([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setBilinearFiltering(v);
#endif
#ifdef VEX_BACKEND_VULKAN
    if (v != m_vkBilinearFiltering) { m_vkBilinearFiltering = v; if (m_vkRaytracer) { m_vkRaytracer->reset(); m_vkSampleCount = 0; } }
#endif
}

bool SceneRenderer::getGPUBilinearFiltering() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getBilinearFiltering() : true;
#elif defined(VEX_BACKEND_VULKAN)
    return m_vkBilinearFiltering;
#else
    return true;
#endif
}

void SceneRenderer::setGPUExposure(float v)
{
#ifdef VEX_BACKEND_OPENGL
    m_gpuExposure = v;
#elif defined(VEX_BACKEND_VULKAN)
    m_vkExposure = v;
#endif
}

float SceneRenderer::getGPUExposure() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuExposure;
#elif defined(VEX_BACKEND_VULKAN)
    return m_vkExposure;
#else
    return 0.0f;
#endif
}

void SceneRenderer::setGPUGamma(float v)
{
#ifdef VEX_BACKEND_OPENGL
    m_gpuGamma = v;
#elif defined(VEX_BACKEND_VULKAN)
    m_vkGamma = v;
#endif
}

float SceneRenderer::getGPUGamma() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuGamma;
#elif defined(VEX_BACKEND_VULKAN)
    return m_vkGamma;
#else
    return 2.2f;
#endif
}

void SceneRenderer::setGPUEnableACES(bool v)
{
#ifdef VEX_BACKEND_OPENGL
    m_gpuEnableACES = v;
#elif defined(VEX_BACKEND_VULKAN)
    m_vkEnableACES = v;
#endif
}

bool SceneRenderer::getGPUEnableACES() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuEnableACES;
#elif defined(VEX_BACKEND_VULKAN)
    return m_vkEnableACES;
#else
    return true;
#endif
}

void SceneRenderer::setGPURayEps([[maybe_unused]] float v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setRayEps(v);
#endif
#ifdef VEX_BACKEND_VULKAN
    if (v != m_vkRayEps) { m_vkRayEps = v; if (m_vkRaytracer) { m_vkRaytracer->reset(); m_vkSampleCount = 0; } }
#endif
}

float SceneRenderer::getGPURayEps() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getRayEps() : 1e-4f;
#elif defined(VEX_BACKEND_VULKAN)
    return m_vkRayEps;
#else
    return 1e-4f;
#endif
}

void SceneRenderer::setGPUEnableRR([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setEnableRR(v);
#endif
#ifdef VEX_BACKEND_VULKAN
    if (v != m_vkEnableRR) { m_vkEnableRR = v; if (m_vkRaytracer) { m_vkRaytracer->reset(); m_vkSampleCount = 0; } }
#endif
}

bool SceneRenderer::getGPUEnableRR() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getEnableRR() : true;
#elif defined(VEX_BACKEND_VULKAN)
    return m_vkEnableRR;
#else
    return true;
#endif
}

bool SceneRenderer::reloadGPUShader()
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer && m_gpuRaytracer->reloadShader();
#else
    return false;
#endif
}

void SceneRenderer::buildGeometry(Scene& scene, ProgressFn progress)
{
    rebuildRaytraceGeometry(scene, std::move(progress));
    scene.geometryDirty = false;
    scene.materialDirty = false;
}

void SceneRenderer::rebuildRaytraceGeometry(Scene& scene, ProgressFn progress)
{
    vex::Log::info("Building raytrace geometry...");

    std::vector<vex::CPURaytracer::Triangle> triangles;
    std::vector<vex::CPURaytracer::TextureData> textures;

    // Deduplicate textures by path; each unique path is loaded from disk at most once.
    std::unordered_map<std::string, int> textureMap;

    auto resolveTexture = [&](const std::string& path) -> int
    {
        if (path.empty()) return -1;
        auto it = textureMap.find(path);
        if (it != textureMap.end()) return it->second;
        int idx = -1;

        // EXR path (no vertical flip — raytracer UV space matches stbi no-flip convention)
        if (path.size() >= 4 &&
            (path.compare(path.size() - 4, 4, ".exr") == 0 ||
             path.compare(path.size() - 4, 4, ".EXR") == 0))
        {
            float* exrRGBA = nullptr;
            int tw = 0, th = 0;
            const char* err = nullptr;
            if (LoadEXR(&exrRGBA, &tw, &th, path.c_str(), &err) == TINYEXR_SUCCESS)
            {
                idx = static_cast<int>(textures.size());
                vex::CPURaytracer::TextureData td;
                td.width  = tw;
                td.height = th;
                td.pixels.resize(static_cast<size_t>(tw) * th * 4);
                for (size_t i = 0; i < td.pixels.size(); ++i)
                    td.pixels[i] = static_cast<unsigned char>(
                        std::clamp(exrRGBA[i], 0.0f, 1.0f) * 255.0f + 0.5f);
                textures.push_back(std::move(td));
                free(exrRGBA);
            }
            else
            {
                std::string errMsg = err ? err : "";
                if (errMsg == "Unknown compression type.")
                    errMsg += " (DWAA/DWAB not supported, re-export with ZIP or PIZ compression)";
                vex::Log::error("Failed to load EXR texture: " + path +
                                (errMsg.empty() ? "" : " (" + errMsg + ")"));
                FreeEXRErrorMessage(err);
            }
        }
        else
        {
            int tw, th, tch;
            stbi_set_flip_vertically_on_load(false);
            unsigned char* texData = stbi_load(path.c_str(), &tw, &th, &tch, 4);
            if (texData)
            {
                idx = static_cast<int>(textures.size());
                vex::CPURaytracer::TextureData td;
                td.width  = tw;
                td.height = th;
                td.pixels.assign(texData, texData + tw * th * 4);
                textures.push_back(std::move(td));
                stbi_image_free(texData);
            }
        }

        textureMap[path] = idx;
        return idx;
    };

    // Look up a path already loaded by resolveTexture (no disk I/O).
    auto lookupTexture = [&](const std::string& path) -> int
    {
        if (path.empty()) return -1;
        auto it = textureMap.find(path);
        return it != textureMap.end() ? it->second : -1;
    };

    // Rebuild per-group local-space AABBs (no model matrix — raw mesh positions).
    // The shadow pass transforms these corners by group.modelMatrix each frame.
    m_groupLocalAABBs.clear();
    m_groupLocalAABBs.resize(scene.meshGroups.size());
    for (size_t gi = 0; gi < scene.meshGroups.size(); ++gi)
        for (const auto& sm : scene.meshGroups[gi].submeshes)
            for (const auto& v : sm.meshData.vertices)
                m_groupLocalAABBs[gi].grow(v.position);

    for (const auto& group : scene.meshGroups)
    {
        glm::mat3 normalMat = glm::mat3(glm::transpose(glm::inverse(group.modelMatrix)));

        for (const auto& sm : group.submeshes)
        {
            const auto& verts = sm.meshData.vertices;
            const auto& indices = sm.meshData.indices;

            int texIdx          = resolveTexture(sm.meshData.diffuseTexturePath);
            int emissiveTexIdx  = resolveTexture(sm.meshData.emissiveTexturePath);
            int normalTexIdx    = resolveTexture(sm.meshData.normalTexturePath);
            int roughnessTexIdx = resolveTexture(sm.meshData.roughnessTexturePath);
            int metallicTexIdx  = resolveTexture(sm.meshData.metallicTexturePath);

            for (size_t i = 0; i + 2 < indices.size(); i += 3)
            {
                const auto& v0 = verts[indices[i + 0]];
                const auto& v1 = verts[indices[i + 1]];
                const auto& v2 = verts[indices[i + 2]];

                glm::vec3 p0 = glm::vec3(group.modelMatrix * glm::vec4(v0.position, 1.0f));
                glm::vec3 p1 = glm::vec3(group.modelMatrix * glm::vec4(v1.position, 1.0f));
                glm::vec3 p2 = glm::vec3(group.modelMatrix * glm::vec4(v2.position, 1.0f));

                glm::vec3 edge1 = p1 - p0;
                glm::vec3 edge2 = p2 - p0;
                glm::vec3 cross = glm::cross(edge1, edge2);
                float len = glm::length(cross);

                vex::CPURaytracer::Triangle tri;
                tri.v0 = p0;
                tri.v1 = p1;
                tri.v2 = p2;
                tri.n0 = glm::normalize(normalMat * v0.normal);
                tri.n1 = glm::normalize(normalMat * v1.normal);
                tri.n2 = glm::normalize(normalMat * v2.normal);
                tri.uv0 = v0.uv;
                tri.uv1 = v1.uv;
                tri.uv2 = v2.uv;
                tri.color = v0.color;
                tri.emissive = v0.emissive;
                tri.geometricNormal = (len > GEOMETRY_EPSILON) ? (cross / len) : glm::vec3(0, 1, 0);
                tri.area = len * 0.5f;
                tri.textureIndex = texIdx;
                tri.emissiveTextureIndex = emissiveTexIdx;
                tri.normalMapTextureIndex = normalTexIdx;
                tri.roughnessTextureIndex = roughnessTexIdx;
                tri.metallicTextureIndex = metallicTexIdx;
                tri.alphaClip = sm.meshData.alphaClip;
                tri.materialType = sm.meshData.materialType;
                tri.ior = sm.meshData.ior;
                tri.roughness = sm.meshData.roughness;
                tri.metallic = sm.meshData.metallic;

                // Compute tangent from UV gradients
                glm::vec2 dUV1 = v1.uv - v0.uv;
                glm::vec2 dUV2 = v2.uv - v0.uv;
                float det = dUV1.x * dUV2.y - dUV2.x * dUV1.y;
                if (std::abs(det) > GEOMETRY_EPSILON)
                {
                    float f = 1.0f / det;
                    tri.tangent = glm::normalize(f * (dUV2.y * edge1 - dUV1.y * edge2));
                    glm::vec3 B = f * (-dUV2.x * edge1 + dUV1.x * edge2);
                    tri.bitangentSign = (glm::dot(glm::cross(tri.geometricNormal, tri.tangent), B) < 0.0f) ? -1.0f : 1.0f;
                }

                triangles.push_back(tri);
            }
        }
    }

    // Always keep a CPU-side copy of textures — needed for the VK RT texture atlas SSBO.
    m_rtTextures = textures;

    // CPU BVH build is expensive (~10s for large scenes).
    // In the Vulkan build, GPU RT uses BLAS/TLAS so we can skip the CPU BVH when not in CPU RT mode.
    // In the OpenGL build, GPU RT also uses m_rtTriangles + m_rtBVH, so always build it.
#ifdef VEX_BACKEND_VULKAN
    const bool needsCpuBvh = (m_renderMode == RenderMode::CPURaytrace);
#else
    const bool needsCpuBvh = true;
#endif
    if (needsCpuBvh)
    {
        m_cpuRaytracer->setGeometry(std::move(triangles), std::move(textures));

        // Build ordering BVH for CPU RT triangle data
        {
            std::vector<vex::CPURaytracer::Triangle> gpuTriangles;
            std::vector<std::pair<int,int>> gpuTriangleSrc;
            for (int gi = 0; gi < static_cast<int>(scene.meshGroups.size()); ++gi)
            {
                for (int si = 0; si < static_cast<int>(scene.meshGroups[gi].submeshes.size()); ++si)
                {
                const auto& sm = scene.meshGroups[gi].submeshes[si];
                    const auto& verts = sm.meshData.vertices;
                    const auto& idx   = sm.meshData.indices;

                    int texIdx           = lookupTexture(sm.meshData.diffuseTexturePath);
                    int emissiveTexIdx   = lookupTexture(sm.meshData.emissiveTexturePath);
                    int normalTexIdx2    = lookupTexture(sm.meshData.normalTexturePath);
                    int roughnessTexIdx2 = lookupTexture(sm.meshData.roughnessTexturePath);
                    int metallicTexIdx2  = lookupTexture(sm.meshData.metallicTexturePath);

                    for (size_t i = 0; i + 2 < idx.size(); i += 3)
                    {
                        const auto& v0 = verts[idx[i + 0]];
                        const auto& v1 = verts[idx[i + 1]];
                        const auto& v2 = verts[idx[i + 2]];

                        glm::vec3 edge1 = v1.position - v0.position;
                        glm::vec3 edge2 = v2.position - v0.position;
                        glm::vec3 cr = glm::cross(edge1, edge2);
                        float len = glm::length(cr);

                        vex::CPURaytracer::Triangle tri;
                        tri.v0 = v0.position; tri.v1 = v1.position; tri.v2 = v2.position;
                        tri.n0 = v0.normal;   tri.n1 = v1.normal;   tri.n2 = v2.normal;
                        tri.uv0 = v0.uv;      tri.uv1 = v1.uv;      tri.uv2 = v2.uv;
                        tri.color = v0.color;
                        tri.emissive = v0.emissive;
                        tri.geometricNormal = (len > GEOMETRY_EPSILON) ? (cr / len) : glm::vec3(0, 1, 0);
                        tri.area = len * 0.5f;
                        tri.textureIndex = texIdx;
                        tri.emissiveTextureIndex = emissiveTexIdx;
                        tri.normalMapTextureIndex = normalTexIdx2;
                        tri.roughnessTextureIndex = roughnessTexIdx2;
                        tri.metallicTextureIndex = metallicTexIdx2;
                        tri.alphaClip = sm.meshData.alphaClip;
                        tri.materialType = sm.meshData.materialType;
                        tri.ior = sm.meshData.ior;
                        tri.roughness = sm.meshData.roughness;
                        tri.metallic = sm.meshData.metallic;

                        glm::vec2 dUV1 = v1.uv - v0.uv;
                        glm::vec2 dUV2 = v2.uv - v0.uv;
                        float det = dUV1.x * dUV2.y - dUV2.x * dUV1.y;
                        if (std::abs(det) > GEOMETRY_EPSILON)
                        {
                            float f = 1.0f / det;
                            tri.tangent = glm::normalize(f * (dUV2.y * edge1 - dUV1.y * edge2));
                            glm::vec3 B = f * (-dUV2.x * edge1 + dUV1.x * edge2);
                            tri.bitangentSign = (glm::dot(glm::cross(tri.geometricNormal, tri.tangent), B) < 0.0f) ? -1.0f : 1.0f;
                        }

                        gpuTriangles.push_back(tri);
                        gpuTriangleSrc.push_back({gi, si});
                    }
                }
            }

            uint32_t count = static_cast<uint32_t>(gpuTriangles.size());
            std::vector<vex::AABB> triBounds(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                triBounds[i].grow(gpuTriangles[i].v0);
                triBounds[i].grow(gpuTriangles[i].v1);
                triBounds[i].grow(gpuTriangles[i].v2);
            }
            m_rtBVH.build(triBounds);

            const auto& bvhIndices = m_rtBVH.indices();
            std::vector<vex::CPURaytracer::Triangle> reordered(count);
            std::vector<std::pair<int,int>> reorderedSrc(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                reordered[i]    = gpuTriangles[bvhIndices[i]];
                reorderedSrc[i] = gpuTriangleSrc[bvhIndices[i]];
            }
            m_rtTriangles          = std::move(reordered);
            m_rtTriangleSrcSubmesh = std::move(reorderedSrc);

            m_rtLightIndices.clear();
            m_rtLightCDF.clear();
            m_rtTotalLightArea = 0.0f;
            for (uint32_t i = 0; i < static_cast<uint32_t>(m_rtTriangles.size()); ++i)
            {
                if (glm::length(m_rtTriangles[i].emissive) > 0.001f)
                {
                    m_rtLightIndices.push_back(i);
                    m_rtTotalLightArea += m_rtTriangles[i].area;
                    m_rtLightCDF.push_back(m_rtTotalLightArea);
                }
            }
            if (m_rtTotalLightArea > 0.0f)
                for (float& c : m_rtLightCDF) c /= m_rtTotalLightArea;
        }

        char sahBuf[32];
        std::snprintf(sahBuf, sizeof(sahBuf), "%.1f", m_cpuRaytracer->getBVHSAHCost());
        vex::Log::info("  BVH built: " + std::to_string(m_cpuRaytracer->getBVHNodeCount()) + " nodes, "
                      + std::to_string(m_rtTriangles.size()) + " triangles, SAH cost " + sahBuf);
        if (!m_rtLightIndices.empty())
            vex::Log::info("  " + std::to_string(m_rtLightIndices.size()) + " emissive triangle(s)");

        m_cpuBVHDirty = false;
    }
    else
    {
        vex::Log::info("  Skipping CPU BVH (not in CPU RT mode)");
        m_cpuBVHDirty = true;
    }

#ifdef VEX_BACKEND_VULKAN
    if (m_vkRaytracer)
    {
        m_vkRaytracer->clearAccelerationStructures();

        // Helper: store int/float bit patterns for SSBO fields read via floatBitsToInt/uintBitsToFloat in GLSL
        auto iBF = [](int   v) -> float    { float    f; std::memcpy(&f, &v, sizeof(f)); return f; };
        auto fBU = [](float v) -> uint32_t { uint32_t u; std::memcpy(&u, &v, sizeof(u)); return u; };

        m_vkTriShading.clear();
        m_vkInstanceOffsets.clear();

        std::vector<uint32_t>  vkLightIndices;
        std::vector<float>     vkLightCDF;
        std::vector<glm::mat4> blasTransforms;
        float vkTotalLightArea = 0.0f;
        uint32_t globalTriOffset = 0;

        for (const auto& group : scene.meshGroups)
        {
            glm::mat3 vkNormalMat = glm::mat3(glm::transpose(glm::inverse(group.modelMatrix)));

            for (const auto& sm : group.submeshes)
            {
                auto* vkMesh = static_cast<vex::VKMesh*>(sm.mesh.get());
                m_vkRaytracer->addBlas(
                    vkMesh->getVertexBuffer(), vkMesh->getVertexCount(), sizeof(vex::Vertex),
                    vkMesh->getIndexBuffer(),  vkMesh->getIndexCount());
                blasTransforms.push_back(group.modelMatrix);

                m_vkInstanceOffsets.push_back(globalTriOffset);

                const auto& verts   = sm.meshData.vertices;
                const auto& indices = sm.meshData.indices;

                int texIdx           = lookupTexture(sm.meshData.diffuseTexturePath);
                int emissiveTexIdx   = lookupTexture(sm.meshData.emissiveTexturePath);
                int normalTexIdx     = lookupTexture(sm.meshData.normalTexturePath);
                int roughnessTexIdx  = lookupTexture(sm.meshData.roughnessTexturePath);
                int metallicTexIdx   = lookupTexture(sm.meshData.metallicTexturePath);

                for (size_t i = 0; i + 2 < indices.size(); i += 3)
                {
                    const auto& v0 = verts[indices[i + 0]];
                    const auto& v1 = verts[indices[i + 1]];
                    const auto& v2 = verts[indices[i + 2]];

                    glm::vec3 p0 = glm::vec3(group.modelMatrix * glm::vec4(v0.position, 1.0f));
                    glm::vec3 p1 = glm::vec3(group.modelMatrix * glm::vec4(v1.position, 1.0f));
                    glm::vec3 p2 = glm::vec3(group.modelMatrix * glm::vec4(v2.position, 1.0f));
                    glm::vec3 n0 = glm::normalize(vkNormalMat * v0.normal);
                    glm::vec3 n1 = glm::normalize(vkNormalMat * v1.normal);
                    glm::vec3 n2 = glm::normalize(vkNormalMat * v2.normal);

                    glm::vec3 e1 = p1 - p0;
                    glm::vec3 e2 = p2 - p0;
                    glm::vec3 cr = glm::cross(e1, e2);
                    float len = glm::length(cr);
                    glm::vec3 geoN = (len > GEOMETRY_EPSILON) ? (cr / len) : glm::vec3(0, 1, 0);
                    float area = len * 0.5f;

                    // Tangent from UV gradients (using world-space edges)
                    glm::vec2 dUV1 = v1.uv - v0.uv;
                    glm::vec2 dUV2 = v2.uv - v0.uv;
                    float det = dUV1.x * dUV2.y - dUV2.x * dUV1.y;
                    glm::vec3 tangent(1, 0, 0);
                    float bitangentSign = 1.0f;
                    if (std::abs(det) > GEOMETRY_EPSILON)
                    {
                        float f = 1.0f / det;
                        tangent = glm::normalize(f * (dUV2.y * e1 - dUV1.y * e2));
                        glm::vec3 B = f * (-dUV2.x * e1 + dUV1.x * e2);
                        bitangentSign = (glm::dot(glm::cross(geoN, tangent), B) < 0.0f) ? -1.0f : 1.0f;
                    }

                    // [0] n0.xyz + roughnessTexIdx (as int bits)
                    m_vkTriShading.push_back(n0.x); m_vkTriShading.push_back(n0.y);
                    m_vkTriShading.push_back(n0.z); m_vkTriShading.push_back(iBF(roughnessTexIdx));
                    // [1] n1.xyz + metallicTexIdx (as int bits)
                    m_vkTriShading.push_back(n1.x); m_vkTriShading.push_back(n1.y);
                    m_vkTriShading.push_back(n1.z); m_vkTriShading.push_back(iBF(metallicTexIdx));
                    // [2] n2.xyz + pad
                    m_vkTriShading.push_back(n2.x); m_vkTriShading.push_back(n2.y);
                    m_vkTriShading.push_back(n2.z); m_vkTriShading.push_back(0.0f);
                    // [3] uv0.xy + uv1.zw
                    m_vkTriShading.push_back(v0.uv.x); m_vkTriShading.push_back(v0.uv.y);
                    m_vkTriShading.push_back(v1.uv.x); m_vkTriShading.push_back(v1.uv.y);
                    // [4] uv2.xy + roughness + metallic
                    m_vkTriShading.push_back(v2.uv.x); m_vkTriShading.push_back(v2.uv.y);
                    m_vkTriShading.push_back(sm.meshData.roughness); m_vkTriShading.push_back(sm.meshData.metallic);
                    // [5] color.xyz + texIdx (as int bits)
                    m_vkTriShading.push_back(v0.color.x); m_vkTriShading.push_back(v0.color.y);
                    m_vkTriShading.push_back(v0.color.z); m_vkTriShading.push_back(iBF(texIdx));
                    // [6] emissive.xyz + area
                    m_vkTriShading.push_back(v0.emissive.x); m_vkTriShading.push_back(v0.emissive.y);
                    m_vkTriShading.push_back(v0.emissive.z); m_vkTriShading.push_back(area);
                    // [7] geoNormal.xyz + normalMapTexIdx (as int bits)
                    m_vkTriShading.push_back(geoN.x); m_vkTriShading.push_back(geoN.y);
                    m_vkTriShading.push_back(geoN.z); m_vkTriShading.push_back(iBF(normalTexIdx));
                    // [8] alphaClip + materialType (float) + ior + emissiveTexIdx (as int bits)
                    m_vkTriShading.push_back(sm.meshData.alphaClip ? 1.0f : 0.0f);
                    m_vkTriShading.push_back(static_cast<float>(sm.meshData.materialType));
                    m_vkTriShading.push_back(sm.meshData.ior);
                    m_vkTriShading.push_back(iBF(emissiveTexIdx));
                    // [9] tangent.xyz + bitangentSign
                    m_vkTriShading.push_back(tangent.x); m_vkTriShading.push_back(tangent.y);
                    m_vkTriShading.push_back(tangent.z); m_vkTriShading.push_back(bitangentSign);
                    // [10] v0.xyz + pad  (world space)
                    m_vkTriShading.push_back(p0.x); m_vkTriShading.push_back(p0.y);
                    m_vkTriShading.push_back(p0.z); m_vkTriShading.push_back(0.0f);
                    // [11] v1.xyz + pad  (world space)
                    m_vkTriShading.push_back(p1.x); m_vkTriShading.push_back(p1.y);
                    m_vkTriShading.push_back(p1.z); m_vkTriShading.push_back(0.0f);
                    // [12] v2.xyz + pad  (world space)
                    m_vkTriShading.push_back(p2.x); m_vkTriShading.push_back(p2.y);
                    m_vkTriShading.push_back(p2.z); m_vkTriShading.push_back(0.0f);

                    // Track emissive triangles for lights SSBO (per-submesh global index)
                    if (glm::length(v0.emissive) > 0.001f)
                    {
                        vkLightIndices.push_back(globalTriOffset + static_cast<uint32_t>(i / 3));
                        vkTotalLightArea += area;
                        vkLightCDF.push_back(vkTotalLightArea);
                    }
                }
                globalTriOffset += static_cast<uint32_t>(indices.size() / 3);
            }
        }

        // Normalize light CDF to [0, 1]
        if (vkTotalLightArea > 0.0f)
            for (float& c : vkLightCDF) c /= vkTotalLightArea;

        // ── Build lights SSBO ─────────────────────────────────────────────────
        // std430 layout: [lightCount uint][totalLightArea float][pad][pad]
        //   + lightRawData[]: [0..N-1]=tri indices, [N..2N-1]=CDF (float bits via uintBitsToFloat in GLSL)
        m_vkLights.clear();
        uint32_t lightCount = static_cast<uint32_t>(vkLightIndices.size());
        m_vkLights.push_back(lightCount);
        m_vkLights.push_back(fBU(vkTotalLightArea));
        m_vkLights.push_back(0); m_vkLights.push_back(0); // pad
        for (uint32_t idx : vkLightIndices) m_vkLights.push_back(idx);
        for (float c : vkLightCDF) m_vkLights.push_back(fBU(c));

        // ── Build texData SSBO ────────────────────────────────────────────────
        // Layout: [texCount][{offset, w, h, pad} per tex...][packed RGBA8 pixels as uint...]
        m_vkTexData.clear();
        uint32_t texCount = static_cast<uint32_t>(m_rtTextures.size());
        m_vkTexData.push_back(texCount);
        uint32_t headerSize = 1u + texCount * 4u;
        m_vkTexData.resize(headerSize, 0u);
        uint32_t pixelBase = headerSize;
        for (uint32_t ti = 0; ti < texCount; ++ti)
        {
            const auto& td = m_rtTextures[ti];
            uint32_t hBase = 1u + ti * 4u;
            m_vkTexData[hBase + 0] = pixelBase;
            m_vkTexData[hBase + 1] = static_cast<uint32_t>(td.width);
            m_vkTexData[hBase + 2] = static_cast<uint32_t>(td.height);
            m_vkTexData[hBase + 3] = 0u;
            uint32_t pixelCount = static_cast<uint32_t>(td.width * td.height);
            for (uint32_t pi = 0; pi < pixelCount; ++pi)
            {
                uint8_t r = td.pixels[pi * 4 + 0], g = td.pixels[pi * 4 + 1];
                uint8_t b = td.pixels[pi * 4 + 2], a = td.pixels[pi * 4 + 3];
                m_vkTexData.push_back(r | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24));
            }
            pixelBase += pixelCount;
        }

        if (progress) progress("Building BLASes...", 0.7f);
        m_vkRaytracer->commitBlasBuild(); // single GPU submission for all BLASes
        if (progress) progress("Building TLAS...", 0.9f);
        m_vkRaytracer->buildTlas(blasTransforms);
        m_vkGeomDirty = true;  // uploadSceneData + createOutputImage deferred to renderVKRaytrace
        m_vkSampleCount = 0;

        vex::Log::info("  VK RT geometry built: "
                      + std::to_string(m_vkInstanceOffsets.size()) + " BLASes, "
                      + std::to_string(m_vkTriShading.size() / 52) + " triangles, "
                      + std::to_string(lightCount) + " emissive triangle(s)");
    }
#endif

    m_gpuGeometryDirty = true;
}

void SceneRenderer::rebuildMaterials(Scene& scene)
{
    // Patch material scalars into the already-reordered triangle array
    for (size_t i = 0; i < m_rtTriangles.size(); ++i)
    {
        auto [gi, si] = m_rtTriangleSrcSubmesh[i];
        const auto& md = scene.meshGroups[gi].submeshes[si].meshData;
        m_rtTriangles[i].materialType = md.materialType;
        m_rtTriangles[i].ior          = md.ior;
        m_rtTriangles[i].roughness    = md.roughness;
        m_rtTriangles[i].metallic     = md.metallic;
    }

    // CPU raytracer: patch m_triData in-place, reset accumulation (no BVH rebuild)
    if (m_cpuRaytracer)
        m_cpuRaytracer->updateMaterials(m_rtTriangles);

#ifdef VEX_BACKEND_OPENGL
    // GPU raytracer: re-upload triangle buffer (BVH already built), reset accumulation
    if (m_gpuRaytracer)
    {
        m_gpuGeometryDirty = true;
        m_gpuRaytracer->reset();
    }
#endif

#ifdef VEX_BACKEND_VULKAN
    // VK raytracer: patch material scalars directly in m_vkTriShading (per-submesh order,
    // one entry per BLAS). BLAS/TLAS geometry doesn't change so no GPU rebuild needed —
    // just re-upload the shading data and reset the accumulator.
    if (m_vkRaytracer && !m_vkTriShading.empty())
    {
        static constexpr size_t FLOATS_PER_TRI = 52;
        size_t blasIdx = 0;
        for (const auto& group : scene.meshGroups)
        {
            for (const auto& sm : group.submeshes)
            {
                uint32_t triStart = m_vkInstanceOffsets[blasIdx];
                size_t   triCount = sm.meshData.indices.size() / 3;
                for (size_t t = 0; t < triCount; ++t)
                {
                    size_t base = (triStart + t) * FLOATS_PER_TRI;
                    m_vkTriShading[base + 18] = sm.meshData.roughness;
                    m_vkTriShading[base + 19] = sm.meshData.metallic;
                    m_vkTriShading[base + 32] = sm.meshData.alphaClip ? 1.0f : 0.0f;
                    m_vkTriShading[base + 33] = static_cast<float>(sm.meshData.materialType);
                    m_vkTriShading[base + 34] = sm.meshData.ior;
                }
                ++blasIdx;
            }
        }
        m_vkGeomDirty   = true;
        m_vkSampleCount = 0;
        m_vkRaytracer->reset();
    }
#endif
}

void SceneRenderer::renderScene(Scene& scene, int selectedGroup, int selectedSubmesh,
                                 const std::string& selectedObjectName)
{
    // If we just switched to a path tracing mode, force a geometry rebuild so that
    // any gizmo model matrix changes from rasterization mode are applied.
    if (m_pendingGeomRebuild)
    {
        scene.geometryDirty   = true;
        m_pendingGeomRebuild  = false;
    }

    // Full geometry rebuild (new mesh loaded, etc.)
    // VK: also trigger when switching to CPU RT with a stale BVH (GPU RT uses BLAS/TLAS, not the CPU BVH).
    // GL: trigger whenever BVH is dirty — GPU RT also needs m_rtTriangles/m_rtBVH.
#ifdef VEX_BACKEND_VULKAN
    if (scene.geometryDirty || (m_renderMode == RenderMode::CPURaytrace && m_cpuBVHDirty))
#else
    if (scene.geometryDirty || m_cpuBVHDirty)
#endif
    {
        rebuildRaytraceGeometry(scene);
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
        m_outlineActive = (selectedGroup >= 0
                        && selectedGroup < static_cast<int>(scene.meshGroups.size()));
        const auto& spec = m_framebuffer->getSpec();
        float aspect = static_cast<float>(spec.width) / static_cast<float>(spec.height);
        glm::mat4 maskView = scene.camera.getViewMatrix();
        glm::mat4 maskProj = scene.camera.getProjectionMatrix(aspect);
        renderOutlineMask(scene, selectedGroup, selectedObjectName, maskView, maskProj);
    }

    switch (m_renderMode)
    {
        case RenderMode::CPURaytrace:
            renderCPURaytrace(scene);
            break;
        case RenderMode::GPURaytrace:
#ifdef VEX_BACKEND_OPENGL
            renderGPURaytrace(scene);
            break;
#elif defined(VEX_BACKEND_VULKAN)
            renderVKRaytrace(scene);
            break;
#else
            // Fall through to rasterize — no backend supports GPU raytracing
            [[fallthrough]];
#endif
        case RenderMode::Rasterize:
            renderRasterize(scene, selectedGroup, selectedSubmesh, selectedObjectName);
            break;
    }
}

void SceneRenderer::renderRasterize(Scene& scene, int selectedGroup, [[maybe_unused]] int selectedSubmesh,
                                     const std::string& selectedObjectName)
{
    // Keep the intermediate HDR framebuffer in sync with the output framebuffer size
    bool hdrFBResized = false;
    {
        const auto& outSpec = m_framebuffer->getSpec();
        const auto& hdrSpec = m_rasterHDRFB->getSpec();
        if (hdrSpec.width != outSpec.width || hdrSpec.height != outSpec.height)
        {
            m_rasterHDRFB->resize(outSpec.width, outSpec.height);
            hdrFBResized = true;
        }
    }
    vex::Framebuffer* renderFB = m_rasterHDRFB.get();

    // --- Compute light view-projection for shadow mapping ---
    glm::mat4 lightVP         = glm::mat4(1.0f);
    float     shadowNormalBias = 0.0f;
    if (scene.showSun && m_rasterEnableShadows && m_shadowFB && m_shadowShader)
    {
        glm::vec3 sunDir = scene.getSunDirection();
        glm::vec3 up = (std::abs(sunDir.y) < 0.99f) ? glm::vec3(0.0f, 1.0f, 0.0f)
                                                     : glm::vec3(1.0f, 0.0f, 0.0f);

        // Fit the shadow frustum to the world-space AABB of all scene geometry.
        // Computed per-frame by transforming each group's cached local-space AABB
        // corners through the current group.modelMatrix, so gizmo transforms are
        // reflected immediately without a full geometry rebuild.
        // Fall back to a unit cube around the camera target if no geometry is loaded yet.
        vex::AABB worldAABB;
        for (size_t gi = 0; gi < scene.meshGroups.size() && gi < m_groupLocalAABBs.size(); ++gi)
        {
            const vex::AABB& local = m_groupLocalAABBs[gi];
            const glm::mat4& M     = scene.meshGroups[gi].modelMatrix;
            for (int c = 0; c < 8; ++c)
            {
                glm::vec3 corner(
                    (c & 1) ? local.max.x : local.min.x,
                    (c & 2) ? local.max.y : local.min.y,
                    (c & 4) ? local.max.z : local.min.z);
                worldAABB.grow(glm::vec3(M * glm::vec4(corner, 1.0f)));
            }
        }
        glm::vec3 aabbMin = worldAABB.min;
        glm::vec3 aabbMax = worldAABB.max;
        if (aabbMin.x > aabbMax.x)
        {
            aabbMin = scene.camera.getTarget() - glm::vec3(1.0f);
            aabbMax = scene.camera.getTarget() + glm::vec3(1.0f);
        }
        glm::vec3 sceneCenter = (aabbMin + aabbMax) * 0.5f;
        glm::vec3 halfExtent  = (aabbMax - aabbMin) * (0.5f * 1.02f);
        aabbMin = sceneCenter - halfExtent;
        aabbMax = sceneCenter + halfExtent;

        // Eye position along the light direction doesn't matter for an ortho projection;
        // any point behind the scene along -sunDir gives the correct view orientation.
        // Place the eye far enough back that ALL AABB corners are in front of it —
        // the half-diagonal of the AABB is the worst-case projection along any direction.
        float eyeDist = glm::length(halfExtent) + 1.0f;
        glm::mat4 lightView = glm::lookAt(sceneCenter - sunDir * eyeDist, sceneCenter, up);

        // Project all 8 AABB corners into light view space and read exact extents
        float lMin = std::numeric_limits<float>::max(), lMax = -std::numeric_limits<float>::max();
        float bMin = std::numeric_limits<float>::max(), bMax = -std::numeric_limits<float>::max();
        float zNear = std::numeric_limits<float>::max(), zFar = -std::numeric_limits<float>::max();
        for (int i = 0; i < 8; ++i)
        {
            glm::vec3 corner(
                (i & 1) ? aabbMax.x : aabbMin.x,
                (i & 2) ? aabbMax.y : aabbMin.y,
                (i & 4) ? aabbMax.z : aabbMin.z
            );
            glm::vec4 lv = lightView * glm::vec4(corner, 1.0f);
            lMin  = std::min(lMin,  lv.x);  lMax  = std::max(lMax,  lv.x);
            bMin  = std::min(bMin,  lv.y);  bMax  = std::max(bMax,  lv.y);
            zNear = std::min(zNear, -lv.z); zFar  = std::max(zFar,  -lv.z);
        }

        // Small margin so shadow casters exactly at the boundary don't get clipped
        float margin    = (zFar - zNear) * 0.05f + 0.1f;
        float lightNear = std::max(0.01f, zNear - margin);
        float lightFar  = zFar + margin;

#ifdef VEX_BACKEND_VULKAN
        glm::mat4 lightProj = glm::orthoRH_ZO(lMin, lMax, bMin, bMax, lightNear, lightFar);
#else
        glm::mat4 lightProj = glm::ortho(lMin, lMax, bMin, bMax, lightNear, lightFar);
#endif
        lightVP = lightProj * lightView;

        // Normal offset bias scaled to the actual world-space texel size.
        // Use the larger of the two ortho extents to stay conservative.
        float orthoSize = std::max(lMax - lMin, bMax - bMin) * 0.5f;
        shadowNormalBias = m_shadowNormalBiasTexels * (2.0f * orthoSize / float(SHADOW_MAP_SIZE));

        // --- Shadow pass (depth-only) ---
        // Restore GL depth compare mode in case it was disabled for ImGui display last frame
#ifdef VEX_BACKEND_OPENGL
        static_cast<vex::GLFramebuffer*>(m_shadowFB.get())->restoreDepthForSampling();
#endif
        // Set VP into shadow shader's UBO before bind() so Vulkan sees it this frame
        m_shadowShader->setMat4("u_shadowViewProj", lightVP);

        m_shadowFB->bind();
        m_shadowFB->clear(0.0f, 0.0f, 0.0f, 1.0f);
        m_shadowShader->bind();

        // OpenGL: set the actual GL uniform (after bind, program is active)
        // Vulkan: "u_lightViewProj" not in uniform map → no-op
        m_shadowShader->setMat4("u_lightViewProj", lightVP);

        // Small constant hardware depth bias during shadow rendering.
        // Normal offset (in mesh.frag) handles grazing angles; this tiny constant
        // covers surfaces that directly face the light (where normal offset → 0).
        // Slope-scale is intentionally zero: at grazing angles tan(theta) → infinity,
        // which creates the exact peter-panning gap we are trying to avoid.
#ifdef VEX_BACKEND_OPENGL
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(0.0f, 4.0f); // constant only, no slope component
#endif
#ifdef VEX_BACKEND_VULKAN
        vkCmdSetDepthBias(vex::VKContext::get().getCurrentCommandBuffer(), 1.25f, 0.0f, 0.0f);
#endif

        for (auto& group : scene.meshGroups)
        {
            m_shadowShader->setMat4("u_model", group.modelMatrix);
            for (auto& sm : group.submeshes)
                sm.mesh->draw();
        }

#ifdef VEX_BACKEND_OPENGL
        glDisable(GL_POLYGON_OFFSET_FILL);
#endif
#ifdef VEX_BACKEND_VULKAN
        vkCmdSetDepthBias(vex::VKContext::get().getCurrentCommandBuffer(), 0.0f, 0.0f, 0.0f);
#endif

        m_shadowShader->unbind();
        m_shadowFB->unbind();
        m_shadowMapEverRendered = true;
    }

    renderFB->bind();

#ifdef VEX_BACKEND_VULKAN
    // If the HDR framebuffer was recreated this frame, the cached sampler descriptor
    // in the fullscreen shader is stale — clear it so it gets rebuilt on the next draw.
    if (hdrFBResized && m_vkFullscreenRTShader)
        static_cast<vex::VKShader*>(m_vkFullscreenRTShader.get())->clearExternalTextureCache();
#endif

    bool useSolidColor = (scene.currentEnvmap == Scene::SolidColor);

    if (useSolidColor)
        renderFB->clear(scene.skyboxColor.r, scene.skyboxColor.g, scene.skyboxColor.b, 1.0f);
    else
        renderFB->clear(0.0f, 0.0f, 0.0f, 1.0f);

    float aspect = static_cast<float>(m_framebuffer->getSpec().width)
                 / static_cast<float>(m_framebuffer->getSpec().height);

    glm::mat4 view = scene.camera.getViewMatrix();
    glm::mat4 proj = scene.camera.getProjectionMatrix(aspect);

    m_drawCalls = 0;

    if (scene.showSkybox && scene.skybox && !useSolidColor)
    {
        scene.skybox->draw(glm::inverse(proj * view));
        ++m_drawCalls;
    }

    [[maybe_unused]] bool hasSelection = (selectedGroup >= 0
                                          && selectedGroup < static_cast<int>(scene.meshGroups.size()));

    // --- Main mesh pass ---
    m_meshShader->bind();
    m_meshShader->setMat4("u_view",       view);
    m_meshShader->setMat4("u_projection", proj);
    m_meshShader->setVec3("u_cameraPos", scene.camera.getPosition());
    m_meshShader->setVec3("u_lightPos", scene.lightPos);
    m_meshShader->setVec3("u_lightColor", scene.showLight ? scene.lightColor * scene.lightIntensity : glm::vec3(0.0f));
    m_meshShader->setVec3("u_sunDirection", scene.getSunDirection());
    m_meshShader->setVec3("u_sunColor", scene.showSun ? scene.sunColor * scene.sunIntensity : glm::vec3(0.0f));

    // Debug mode uniforms
    int dm = static_cast<int>(m_debugMode);
    m_meshShader->setInt("u_debugMode", dm);
    m_meshShader->setFloat("u_nearPlane", scene.camera.nearPlane);
    m_meshShader->setFloat("u_farPlane", scene.camera.farPlane);

    if (m_debugMode == DebugMode::Wireframe)
        m_meshShader->setWireframe(true);

#ifdef VEX_BACKEND_OPENGL
    {
        // Bind env map for rasterizer (slot 5, set once per frame)
        bool hasEnvMap = (m_rasterEnvMapTex != 0);
        glActiveTexture(GL_TEXTURE5);
        if (hasEnvMap)
            glBindTexture(GL_TEXTURE_2D, m_rasterEnvMapTex);
        else
            glBindTexture(GL_TEXTURE_2D, 0);
        m_meshShader->setInt("u_envMap", 5);
        m_meshShader->setBool("u_hasEnvMap", hasEnvMap);
        m_meshShader->setBool("u_enableEnvLighting", m_rasterEnableEnvLighting);
        glm::vec3 envCol = useSolidColor ? scene.skyboxColor : m_rasterEnvColor;
        m_meshShader->setVec3("u_envColor", envCol);
        m_meshShader->setFloat("u_envLightMultiplier", m_rasterEnvLightMultiplier);

        // Bind shadow map (slot 6) and set shadow uniforms
        if (m_shadowFB)
        {
            auto* glShadowFB = static_cast<vex::GLFramebuffer*>(m_shadowFB.get());
            glActiveTexture(GL_TEXTURE6);
            glBindTexture(GL_TEXTURE_2D, glShadowFB->getDepthAttachment());
            m_meshShader->setInt("u_shadowMap", 6);
            m_meshShader->setMat4("u_shadowViewProj", lightVP);
            m_meshShader->setBool("u_enableShadows", scene.showSun && m_rasterEnableShadows);
            m_meshShader->setFloat("u_shadowNormalBias", shadowNormalBias);
        }
    }
#endif

#ifdef VEX_BACKEND_VULKAN
    {
        bool hasEnvMap = (m_vkRasterEnvTex != nullptr);
        glm::vec3 envCol = useSolidColor ? scene.skyboxColor : m_rasterEnvColor;
        m_meshShader->setVec3("u_envColor", envCol);
        m_meshShader->setFloat("u_envLightMultiplier", m_rasterEnvLightMultiplier);
        m_meshShader->setBool("u_enableEnvLighting", m_rasterEnableEnvLighting);
        m_meshShader->setBool("u_hasEnvMap", hasEnvMap && m_rasterEnableEnvLighting);
        // Bind env map at slot 5 (→ descriptor set 6)
        m_meshShader->setTexture(5, hasEnvMap ? m_vkRasterEnvTex.get() : m_whiteTexture.get());

        // Shadow map and shadow uniforms
        m_meshShader->setMat4("u_shadowViewProj", lightVP);
        m_meshShader->setBool("u_enableShadows", scene.showSun && m_rasterEnableShadows);
        m_meshShader->setFloat("u_shadowNormalBias", shadowNormalBias);
        if (m_shadowFB)
        {
            auto* vkShadowFB    = static_cast<vex::VKFramebuffer*>(m_shadowFB.get());
            auto* vkMeshShader  = static_cast<vex::VKShader*>(m_meshShader.get());
            // Bind shadow depth at slot 6 (→ descriptor set 7)
            vkMeshShader->setExternalTextureVK(6,
                vkShadowFB->getDepthImageView(),
                vkShadowFB->getDepthCompSampler(),
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        }
    }
#endif

    for (int gi = 0; gi < static_cast<int>(scene.meshGroups.size()); ++gi)
    {
        m_meshShader->setMat4("u_model", scene.meshGroups[gi].modelMatrix);
        for (int si = 0; si < static_cast<int>(scene.meshGroups[gi].submeshes.size()); ++si)
        {
            auto& sm = scene.meshGroups[gi].submeshes[si];

#ifdef VEX_BACKEND_OPENGL
            bool isSelectedGroup = hasSelection && gi == selectedGroup;
            bool writeStencil = isSelectedGroup && (selectedSubmesh < 0 || si == selectedSubmesh);
            if (writeStencil)
            {
                glEnable(GL_STENCIL_TEST);
                glStencilFunc(GL_ALWAYS, 1, 0xFF);
                glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
                glStencilMask(0xFF);
            }
#endif

            vex::Texture2D* tex = sm.diffuseTexture
                ? sm.diffuseTexture.get()
                : m_whiteTexture.get();
            m_meshShader->setTexture(0, tex);

            bool hasNorm = m_enableNormalMapping && sm.normalTexture != nullptr;
            vex::Texture2D* normTex = hasNorm
                ? sm.normalTexture.get()
                : m_flatNormalTexture.get();
            m_meshShader->setTexture(1, normTex);
            m_meshShader->setBool("u_hasNormalMap", hasNorm);

            bool hasRoughMap = sm.roughnessTexture != nullptr;
            m_meshShader->setTexture(2, hasRoughMap ? sm.roughnessTexture.get() : m_whiteTexture.get());
            m_meshShader->setBool("u_hasRoughnessMap", hasRoughMap);

            bool hasMetalMap = sm.metallicTexture != nullptr;
            m_meshShader->setTexture(3, hasMetalMap ? sm.metallicTexture.get() : m_whiteTexture.get());
            m_meshShader->setBool("u_hasMetallicMap", hasMetalMap);

            bool hasEmissive = sm.emissiveTexture != nullptr;
            m_meshShader->setTexture(4, hasEmissive ? sm.emissiveTexture.get() : m_whiteTexture.get());
            m_meshShader->setBool("u_hasEmissiveMap", hasEmissive);

            m_meshShader->setInt("u_materialType", sm.meshData.materialType);
            m_meshShader->setFloat("u_roughness", sm.meshData.roughness);
            m_meshShader->setFloat("u_metallic", sm.meshData.metallic);
            m_meshShader->setBool("u_alphaClip", sm.meshData.alphaClip);
            sm.mesh->draw();
            ++m_drawCalls;

#ifdef VEX_BACKEND_OPENGL
            if (writeStencil)
            {
                glStencilMask(0x00);
                glDisable(GL_STENCIL_TEST);
            }
#endif
        }
    }

    if (m_debugMode == DebugMode::Wireframe)
        m_meshShader->setWireframe(false);

    m_meshShader->unbind();

    renderFB->unbind();

#ifdef VEX_BACKEND_OPENGL
    // --- Tone-map blit: HDR intermediate buffer → output framebuffer ---
    if (m_fullscreenRTShader)
    {
        m_framebuffer->bind();
        m_framebuffer->clear(0.0f, 0.0f, 0.0f, 1.0f);

        glDisable(GL_DEPTH_TEST);

        m_fullscreenRTShader->bind();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(m_rasterHDRFB->getColorAttachmentHandle()));
        m_fullscreenRTShader->setInt("u_accumMap", 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(m_outlineMaskFB->getColorAttachmentHandle()));
        m_fullscreenRTShader->setInt("u_outlineMask", 1);
        m_fullscreenRTShader->setFloat("u_sampleCount", 1.0f);
        m_fullscreenRTShader->setFloat("u_exposure", m_rasterExposure);
        m_fullscreenRTShader->setFloat("u_gamma", m_rasterGamma);
        m_fullscreenRTShader->setBool("u_enableACES", m_rasterEnableACES);
        m_fullscreenRTShader->setBool("u_flipV", false); // GL framebuffer: natural bottom-left origin, no flip needed
        m_fullscreenRTShader->setBool("u_enableOutline", m_outlineActive);
        m_fullscreenQuad->draw();
        m_fullscreenRTShader->unbind();

        glEnable(GL_DEPTH_TEST);

        m_framebuffer->unbind();
    }
#endif

#ifdef VEX_BACKEND_VULKAN
    // --- Tone-map blit: HDR intermediate buffer → output framebuffer ---
    if (m_vkFullscreenRTShader)
    {
        auto* vkHDRFB    = static_cast<vex::VKFramebuffer*>(m_rasterHDRFB.get());
        auto* rtShaderVK = static_cast<vex::VKShader*>(m_vkFullscreenRTShader.get());

        m_framebuffer->bind();
        m_framebuffer->clear(0.0f, 0.0f, 0.0f, 1.0f);

        m_vkFullscreenRTShader->setFloat("u_sampleCount", 1.0f);
        m_vkFullscreenRTShader->setFloat("u_exposure",    m_rasterExposure);
        m_vkFullscreenRTShader->setFloat("u_gamma",       m_rasterGamma);
        m_vkFullscreenRTShader->bind();
        m_vkFullscreenRTShader->setBool("u_enableACES", m_rasterEnableACES);
        m_vkFullscreenRTShader->setBool("u_flipV",      true);

        rtShaderVK->setExternalTextureVK(0,
            vkHDRFB->getColorImageView(),
            vkHDRFB->getColorSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        auto* vkMaskFB = static_cast<vex::VKFramebuffer*>(m_outlineMaskFB.get());
        rtShaderVK->setExternalTextureVK(1,
            vkMaskFB->getColorImageView(),
            vkMaskFB->getColorSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        m_vkFullscreenRTShader->setBool("u_enableOutline", m_outlineActive);

        m_fullscreenQuad->draw();
        m_vkFullscreenRTShader->unbind();

        m_framebuffer->unbind();
    }
#endif
}

void SceneRenderer::renderOutlineMask(Scene& scene, int selectedGroup,
                                      const std::string& selectedObjectName,
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
        m_outlineMaskShader->setMat4("u_model", scene.meshGroups[selectedGroup].modelMatrix);
        m_outlineMaskShader->bind();
#else
        // GL: glUniform* requires the shader to be bound first.
        m_outlineMaskShader->bind();
        m_outlineMaskShader->setMat4("u_view",       view);
        m_outlineMaskShader->setMat4("u_projection", proj);
        m_outlineMaskShader->setMat4("u_model", scene.meshGroups[selectedGroup].modelMatrix);
#endif

        auto& maskSubmeshes = scene.meshGroups[selectedGroup].submeshes;
        for (auto& sm : maskSubmeshes)
        {
            if (selectedObjectName.empty() ||
                sm.meshData.objectName == selectedObjectName)
                sm.mesh->draw();
        }

        m_outlineMaskShader->unbind();
#ifdef VEX_BACKEND_OPENGL
        glEnable(GL_DEPTH_TEST);
#endif
    }

    m_outlineMaskFB->unbind();
}

void SceneRenderer::renderCPURaytrace(Scene& scene)
{
    if (!m_cpuRaytracer)
        return;

    const auto& spec = m_framebuffer->getSpec();
    uint32_t w = spec.width;
    uint32_t h = spec.height;

    // Resize raytracer if viewport changed
    m_cpuRaytracer->resize(w, h);

    // Recreate texture if size changed
    if (w != m_raytraceTexW || h != m_raytraceTexH)
    {
        m_raytraceTexture = vex::Texture2D::create(w, h, 4);
        m_raytraceTexW = w;
        m_raytraceTexH = h;
    }

    // Update environment
    bool envChanged = false;

    // Check for custom env map path changes
    bool customPathChanged = (scene.customEnvmapPath != m_prevCustomEnvmapPath);
    if (customPathChanged)
        m_prevCustomEnvmapPath = scene.customEnvmapPath;

    if (scene.currentEnvmap != m_prevEnvmapIndex || customPathChanged)
    {
        m_prevEnvmapIndex = scene.currentEnvmap;
        envChanged = true;

        if (scene.currentEnvmap > Scene::SolidColor)
        {
            std::string envPath = (scene.currentEnvmap == Scene::CustomHDR)
                ? scene.customEnvmapPath
                : std::string(Scene::envmapPaths[scene.currentEnvmap]);

            int ew, eh, ech;
            stbi_set_flip_vertically_on_load(false);
            float* envData = stbi_loadf(envPath.c_str(), &ew, &eh, &ech, 3);
            if (envData)
            {
                m_cpuRaytracer->setEnvironmentMap(envData, ew, eh);

                // Compute average env color for ambient diffuse (used by both backends)
                {
                    float rSum = 0, gSum = 0, bSum = 0;
                    int n = ew * eh;
                    for (int i = 0; i < n; ++i) { rSum += envData[3*i]; gSum += envData[3*i+1]; bSum += envData[3*i+2]; }
                    m_rasterEnvColor = glm::clamp(glm::vec3(rSum, gSum, bSum) / float(n), 0.0f, 1.0f);
                }

#ifdef VEX_BACKEND_OPENGL
                // Create rasterizer GL env texture from the float data
                if (m_rasterEnvMapTex) glDeleteTextures(1, &m_rasterEnvMapTex);
                glGenTextures(1, &m_rasterEnvMapTex);
                glBindTexture(GL_TEXTURE_2D, m_rasterEnvMapTex);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, ew, eh, 0, GL_RGB, GL_FLOAT, envData);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glBindTexture(GL_TEXTURE_2D, 0);
#endif

#ifdef VEX_BACKEND_VULKAN
                // Convert float RGB → RGBA8 (Reinhard tonemap) for Vulkan rasterizer env texture
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

                stbi_image_free(envData);
            }
        }
        else
        {
            m_cpuRaytracer->clearEnvironmentMap();
#ifdef VEX_BACKEND_OPENGL
            if (m_rasterEnvMapTex) { glDeleteTextures(1, &m_rasterEnvMapTex); m_rasterEnvMapTex = 0; }
#endif
#ifdef VEX_BACKEND_VULKAN
            m_vkRasterEnvTex.reset();
#endif
            m_rasterEnvColor = scene.skyboxColor;
        }
    }

    if (scene.currentEnvmap == Scene::SolidColor && scene.skyboxColor != m_prevSkyboxColor)
    {
        m_prevSkyboxColor = scene.skyboxColor;
        m_cpuRaytracer->setEnvironmentColor(scene.skyboxColor);
        m_rasterEnvColor = scene.skyboxColor;
        envChanged = true;
    }

    if (envChanged)
        m_cpuRaytracer->reset();

    // Update point light and detect changes
    bool lightChanged = (scene.showLight    != m_prevShowLight
                      || scene.lightPos     != m_prevLightPos
                      || scene.lightColor   != m_prevLightColor
                      || scene.lightIntensity != m_prevLightIntensity);
    if (lightChanged)
    {
        m_cpuRaytracer->setPointLight(scene.lightPos, scene.lightColor * scene.lightIntensity, scene.showLight);
        m_cpuRaytracer->reset();
        m_prevShowLight      = scene.showLight;
        m_prevLightPos       = scene.lightPos;
        m_prevLightColor     = scene.lightColor;
        m_prevLightIntensity = scene.lightIntensity;
    }

    // Update sun light and detect changes
    glm::vec3 sunDir = scene.getSunDirection();
    bool sunChanged = (scene.showSun          != m_prevShowSun
                    || sunDir                  != m_prevSunDirection
                    || scene.sunColor          != m_prevSunColor
                    || scene.sunIntensity      != m_prevSunIntensity
                    || scene.sunAngularRadius  != m_prevSunAngularRadius);
    if (sunChanged)
    {
        m_cpuRaytracer->setDirectionalLight(
            sunDir,
            scene.sunColor * scene.sunIntensity,
            scene.sunAngularRadius,
            scene.showSun);
        m_cpuRaytracer->reset();
        m_prevShowSun          = scene.showSun;
        m_prevSunDirection     = sunDir;
        m_prevSunColor         = scene.sunColor;
        m_prevSunIntensity     = scene.sunIntensity;
        m_prevSunAngularRadius = scene.sunAngularRadius;
    }

    // Update camera and detect changes
    float aspect = static_cast<float>(w) / static_cast<float>(h);
    glm::mat4 view = scene.camera.getViewMatrix();
    glm::mat4 proj = scene.camera.getProjectionMatrix(aspect);
    glm::vec3 camPos = scene.camera.getPosition();

    if (camPos != m_prevCameraPos || view != m_prevViewMatrix)
    {
        m_cpuRaytracer->reset();
        m_prevCameraPos = camPos;
        m_prevViewMatrix = view;
    }

    glm::mat4 vp = proj * view;
    m_cpuRaytracer->setCamera(camPos, glm::inverse(vp));

    // Depth of field — extract camera basis from view matrix and pass to raytracer
    {
        glm::vec3 right = glm::vec3(view[0][0], view[1][0], view[2][0]);
        glm::vec3 up    = glm::vec3(view[0][1], view[1][1], view[2][1]);
        m_cpuRaytracer->setDoF(scene.camera.aperture, scene.camera.focusDistance, right, up);
    }

    // Trace one sample (skip if sample limit reached)
    if (m_cpuMaxSamples == 0 || m_cpuRaytracer->getSampleCount() < m_cpuMaxSamples)
        m_cpuRaytracer->traceSample();

    // Upload result to texture
    const auto& pixels = m_cpuRaytracer->getPixelBuffer();
    m_raytraceTexture->setData(pixels.data(), w, h, 4);

    // Render fullscreen quad to framebuffer (with optional outline overlay)
    m_framebuffer->bind();
    m_framebuffer->clear(0.0f, 0.0f, 0.0f, 1.0f);

#ifdef VEX_BACKEND_OPENGL
    glDisable(GL_DEPTH_TEST);

    // Use the RT fullscreen shader so we get the outline composite for free.
    // CPU RT output is already display-ready (RGBA8), so we pass through with
    // sampleCount=1, exposure=0, gamma=1, ACES=false (all identity operations).
    m_fullscreenRTShader->bind();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(m_raytraceTexture->getNativeHandle()));
    m_fullscreenRTShader->setInt("u_accumMap", 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(m_outlineMaskFB->getColorAttachmentHandle()));
    m_fullscreenRTShader->setInt("u_outlineMask", 1);
    m_fullscreenRTShader->setFloat("u_sampleCount", 1.0f);
    m_fullscreenRTShader->setFloat("u_exposure", 0.0f);   // pow(2,0)=1 — no change
    m_fullscreenRTShader->setFloat("u_gamma", 1.0f);      // pow(c,1)=c — no change
    m_fullscreenRTShader->setBool("u_enableACES", false);  // clamp only — no change
    m_fullscreenRTShader->setBool("u_flipV", true);
    m_fullscreenRTShader->setBool("u_enableOutline", m_outlineActive);
    m_fullscreenQuad->draw();
    m_fullscreenRTShader->unbind();

    glEnable(GL_DEPTH_TEST);
#endif

#ifdef VEX_BACKEND_VULKAN
    if (m_vkFullscreenRTShader)
    {
        auto* rtShaderVK = static_cast<vex::VKShader*>(m_vkFullscreenRTShader.get());
        auto* vkTex      = static_cast<vex::VKTexture2D*>(m_raytraceTexture.get());
        auto* vkMaskFB   = static_cast<vex::VKFramebuffer*>(m_outlineMaskFB.get());

        m_vkFullscreenRTShader->setFloat("u_sampleCount", 1.0f);
        m_vkFullscreenRTShader->setFloat("u_exposure",    0.0f);
        m_vkFullscreenRTShader->setFloat("u_gamma",       1.0f);
        m_vkFullscreenRTShader->bind();
        m_vkFullscreenRTShader->setBool("u_enableACES",    false);
        m_vkFullscreenRTShader->setBool("u_flipV",         true);
        m_vkFullscreenRTShader->setBool("u_enableOutline", m_outlineActive);

        rtShaderVK->setExternalTextureVK(0,
            vkTex->getImageView(),
            vkTex->getSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        rtShaderVK->setExternalTextureVK(1,
            vkMaskFB->getColorImageView(),
            vkMaskFB->getColorSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        m_fullscreenQuad->draw();
        m_vkFullscreenRTShader->unbind();
    }
#endif

    m_framebuffer->unbind();
    m_drawCalls = 1;
}

#ifdef VEX_BACKEND_OPENGL
void SceneRenderer::renderGPURaytrace(Scene& scene)
{
    if (!m_gpuRaytracer || !m_fullscreenRTShader)
        return;

    const auto& spec = m_framebuffer->getSpec();
    uint32_t w = spec.width;
    uint32_t h = spec.height;

    // Resize GPU raytracer if viewport changed
    m_gpuRaytracer->resize(w, h);

    // Upload geometry if dirty
    if (m_gpuGeometryDirty)
    {
        m_gpuRaytracer->uploadGeometry(m_rtTriangles, m_rtBVH,
                                        m_rtLightIndices, m_rtLightCDF,
                                        m_rtTotalLightArea, m_rtTextures);
        m_gpuGeometryDirty = false;
    }

    // Update environment
    bool envChanged = false;

    bool customPathChanged = (scene.customEnvmapPath != m_prevCustomEnvmapPath);
    if (customPathChanged)
        m_prevCustomEnvmapPath = scene.customEnvmapPath;

    if (scene.currentEnvmap != m_prevEnvmapIndex || customPathChanged)
    {
        m_prevEnvmapIndex = scene.currentEnvmap;
        envChanged = true;

        if (scene.currentEnvmap > Scene::SolidColor)
        {
            std::string envPath = (scene.currentEnvmap == Scene::CustomHDR)
                ? scene.customEnvmapPath
                : std::string(Scene::envmapPaths[scene.currentEnvmap]);

            int ew, eh, ech;
            stbi_set_flip_vertically_on_load(false);
            float* envData = stbi_loadf(envPath.c_str(), &ew, &eh, &ech, 3);
            if (envData)
            {
                m_gpuRaytracer->setEnvironmentMap(envData, ew, eh);

                // Create rasterizer GL env texture from the float data
                if (m_rasterEnvMapTex) glDeleteTextures(1, &m_rasterEnvMapTex);
                glGenTextures(1, &m_rasterEnvMapTex);
                glBindTexture(GL_TEXTURE_2D, m_rasterEnvMapTex);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, ew, eh, 0, GL_RGB, GL_FLOAT, envData);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glBindTexture(GL_TEXTURE_2D, 0);
                // Compute average env color for ambient diffuse
                float rSum = 0, gSum = 0, bSum = 0;
                int n = ew * eh;
                for (int i = 0; i < n; ++i) { rSum += envData[3*i]; gSum += envData[3*i+1]; bSum += envData[3*i+2]; }
                m_rasterEnvColor = glm::clamp(glm::vec3(rSum, gSum, bSum) / float(n), 0.0f, 1.0f);

                stbi_image_free(envData);
            }
        }
        else
        {
            m_gpuRaytracer->clearEnvironmentMap();
            if (m_rasterEnvMapTex) { glDeleteTextures(1, &m_rasterEnvMapTex); m_rasterEnvMapTex = 0; }
            m_rasterEnvColor = scene.skyboxColor;
        }
    }

    if (scene.currentEnvmap == Scene::SolidColor && scene.skyboxColor != m_prevSkyboxColor)
    {
        m_prevSkyboxColor = scene.skyboxColor;
        m_gpuRaytracer->setEnvironmentColor(scene.skyboxColor);
        m_rasterEnvColor = scene.skyboxColor;
        envChanged = true;
    }

    if (envChanged)
        m_gpuRaytracer->reset();

    // Update point light
    bool lightChanged = (scene.showLight    != m_prevShowLight
                      || scene.lightPos     != m_prevLightPos
                      || scene.lightColor   != m_prevLightColor
                      || scene.lightIntensity != m_prevLightIntensity);
    if (lightChanged)
    {
        m_gpuRaytracer->setPointLight(scene.lightPos, scene.lightColor * scene.lightIntensity, scene.showLight);
        m_gpuRaytracer->reset();
        m_prevShowLight      = scene.showLight;
        m_prevLightPos       = scene.lightPos;
        m_prevLightColor     = scene.lightColor;
        m_prevLightIntensity = scene.lightIntensity;
    }

    // Update sun light
    glm::vec3 sunDir = scene.getSunDirection();
    bool sunChanged = (scene.showSun          != m_prevShowSun
                    || sunDir                  != m_prevSunDirection
                    || scene.sunColor          != m_prevSunColor
                    || scene.sunIntensity      != m_prevSunIntensity
                    || scene.sunAngularRadius  != m_prevSunAngularRadius);
    if (sunChanged)
    {
        m_gpuRaytracer->setDirectionalLight(
            sunDir,
            scene.sunColor * scene.sunIntensity,
            scene.sunAngularRadius,
            scene.showSun);
        m_gpuRaytracer->reset();
        m_prevShowSun          = scene.showSun;
        m_prevSunDirection     = sunDir;
        m_prevSunColor         = scene.sunColor;
        m_prevSunIntensity     = scene.sunIntensity;
        m_prevSunAngularRadius = scene.sunAngularRadius;
    }

    // Update camera
    float aspect = static_cast<float>(w) / static_cast<float>(h);
    glm::mat4 view = scene.camera.getViewMatrix();
    glm::mat4 proj = scene.camera.getProjectionMatrix(aspect);
    glm::vec3 camPos = scene.camera.getPosition();

    if (camPos != m_prevCameraPos || view != m_prevViewMatrix)
    {
        m_gpuRaytracer->reset();
        m_prevCameraPos = camPos;
        m_prevViewMatrix = view;
    }

    glm::mat4 vp = proj * view;
    m_gpuRaytracer->setCamera(camPos, glm::inverse(vp));

    // Depth of field — extract camera basis from view matrix and pass to raytracer
    {
        glm::vec3 right = glm::vec3(view[0][0], view[1][0], view[2][0]);
        glm::vec3 up    = glm::vec3(view[0][1], view[1][1], view[2][1]);

        if (scene.camera.aperture != m_prevAperture || scene.camera.focusDistance != m_prevFocusDistance)
        {
            m_gpuRaytracer->reset();
            m_prevAperture      = scene.camera.aperture;
            m_prevFocusDistance = scene.camera.focusDistance;
        }

        m_gpuRaytracer->setDoF(scene.camera.aperture, scene.camera.focusDistance, right, up);
    }

    // Dispatch compute shader (skip if sample limit reached)
    if (m_gpuMaxSamples == 0 || m_gpuRaytracer->getSampleCount() < m_gpuMaxSamples)
        m_gpuRaytracer->traceSample();

    // Display result with tone mapping
    m_framebuffer->bind();
    m_framebuffer->clear(0.0f, 0.0f, 0.0f, 1.0f);

    glDisable(GL_DEPTH_TEST);

    m_fullscreenRTShader->bind();

    // Bind accumulation texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_gpuRaytracer->getAccumTexture());
    m_fullscreenRTShader->setInt("u_accumMap", 0);

    // Bind outline mask
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(m_outlineMaskFB->getColorAttachmentHandle()));
    m_fullscreenRTShader->setInt("u_outlineMask", 1);

    m_fullscreenRTShader->setFloat("u_sampleCount", static_cast<float>(m_gpuRaytracer->getSampleCount()));
    m_fullscreenRTShader->setFloat("u_exposure", m_gpuExposure);
    m_fullscreenRTShader->setFloat("u_gamma", m_gpuGamma);
    m_fullscreenRTShader->setBool("u_enableACES", m_gpuEnableACES);
    m_fullscreenRTShader->setBool("u_flipV", true);   // GPU raytracer accum texture: pixels stored top-to-bottom
    m_fullscreenRTShader->setBool("u_enableOutline", m_outlineActive);

    m_fullscreenQuad->draw();
    m_fullscreenRTShader->unbind();

    glEnable(GL_DEPTH_TEST);

    m_framebuffer->unbind();
    m_drawCalls = 1;
}
#endif

std::pair<int,int> SceneRenderer::pick(Scene& scene, int pixelX, int pixelY)
{
#ifdef VEX_BACKEND_VULKAN
    // CPU ray-triangle intersection (Möller–Trumbore) — no GPU readback needed.
    // pixelX/Y come from ImGui in top-left origin space.
    const auto& spec = m_framebuffer->getSpec();
    float aspect = static_cast<float>(spec.width) / static_cast<float>(spec.height);

    glm::mat4 view  = scene.camera.getViewMatrix();
    glm::mat4 proj  = scene.camera.getProjectionMatrix(aspect);
    glm::mat4 invVP = glm::inverse(proj * view);

    // Convert pixel to NDC (y flipped: ImGui y=0 is top, NDC y=+1 is top).
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

    for (int gi = 0; gi < static_cast<int>(scene.meshGroups.size()); ++gi)
    {
        for (int si = 0; si < static_cast<int>(scene.meshGroups[gi].submeshes.size()); ++si)
        {
            const auto& md      = scene.meshGroups[gi].submeshes[si].meshData;
            const auto& verts   = md.vertices;
            const auto& indices = md.indices;

            for (size_t i = 0; i + 2 < indices.size(); i += 3)
            {
                glm::vec3 v0 = verts[indices[i + 0]].position;
                glm::vec3 v1 = verts[indices[i + 1]].position;
                glm::vec3 v2 = verts[indices[i + 2]].position;

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
                    bestGroup   = gi;
                    bestSubmesh = si;
                }
            }
        }
    }

    return { bestGroup, bestSubmesh };
#endif

#ifdef VEX_BACKEND_OPENGL
    if (!m_pickShader || !m_pickFB)
        return {-1, -1};

    const auto& mainSpec = m_framebuffer->getSpec();
    const auto& pickSpec = m_pickFB->getSpec();

    // Ensure pick FB matches main viewport size
    if (pickSpec.width != mainSpec.width || pickSpec.height != mainSpec.height)
        m_pickFB->resize(mainSpec.width, mainSpec.height);

    m_pickFB->bind();
    m_pickFB->clear(0.0f, 0.0f, 0.0f, 1.0f);

    float aspect = static_cast<float>(mainSpec.width)
                 / static_cast<float>(mainSpec.height);

    glm::mat4 view = scene.camera.getViewMatrix();
    glm::mat4 proj = scene.camera.getProjectionMatrix(aspect);

    m_pickShader->bind();
    m_pickShader->setMat4("u_view", view);
    m_pickShader->setMat4("u_projection", proj);

    // Build flat-draw-index -> {groupIdx, submeshIdx} mapping
    std::vector<std::pair<int,int>> drawToMesh;
    for (int gi = 0; gi < static_cast<int>(scene.meshGroups.size()); ++gi)
    {
        m_pickShader->setMat4("u_model", scene.meshGroups[gi].modelMatrix);
        for (int si = 0; si < static_cast<int>(scene.meshGroups[gi].submeshes.size()); ++si)
        {
            auto& sm = scene.meshGroups[gi].submeshes[si];
            int drawIdx = static_cast<int>(drawToMesh.size());
            m_pickShader->setInt("u_objectID", drawIdx);
            vex::Texture2D* tex = sm.diffuseTexture
                ? sm.diffuseTexture.get()
                : m_whiteTexture.get();
            m_pickShader->setTexture(0, tex);
            m_pickShader->setBool("u_alphaClip", sm.meshData.alphaClip);
            sm.mesh->draw();
            drawToMesh.push_back({gi, si});
        }
    }
    m_pickShader->unbind();

    // Read back the pixel under the cursor
    int objectID = m_pickFB->readPixel(pixelX, pixelY) - 1;

    m_pickFB->unbind();

    if (objectID >= 0 && objectID < static_cast<int>(drawToMesh.size()))
        return drawToMesh[objectID];

    return {-1, -1};
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Vulkan hardware ray tracing render path
// ─────────────────────────────────────────────────────────────────────────────

#ifdef VEX_BACKEND_VULKAN
void SceneRenderer::renderVKRaytrace(Scene& scene)
{
    if (!m_vkRaytracer)
        return;

    const auto& spec = m_framebuffer->getSpec();
    uint32_t w = spec.width;
    uint32_t h = spec.height;

    // ── Environment map change detection ──────────────────────────────────────
    bool envChanged     = false; // any env change → accumulation reset
    bool envDataChanged = false; // env SSBO data changed → full scene data re-upload

    bool customPathChanged = (scene.customEnvmapPath != m_prevCustomEnvmapPath);
    if (customPathChanged)
        m_prevCustomEnvmapPath = scene.customEnvmapPath;

    if (scene.currentEnvmap != m_prevEnvmapIndex || customPathChanged)
    {
        m_prevEnvmapIndex = scene.currentEnvmap;
        envChanged     = true;
        envDataChanged = true; // HDR pixel data (or its absence) changed in the SSBOs

        if (scene.currentEnvmap > Scene::SolidColor)
        {
            std::string envPath = (scene.currentEnvmap == Scene::CustomHDR)
                ? scene.customEnvmapPath
                : std::string(Scene::envmapPaths[scene.currentEnvmap]);

            int ew, eh, ech;
            stbi_set_flip_vertically_on_load(false);
            float* envData = stbi_loadf(envPath.c_str(), &ew, &eh, &ech, 3);
            if (envData)
            {
                m_vkEnvMapW = ew;
                m_vkEnvMapH = eh;
                m_vkEnvMapData.assign(envData, envData + ew * eh * 3);
                stbi_image_free(envData);

                // Build importance-sampling CDF for the environment map.
                // Layout: [marginalCDF: H floats][condCDF: W*H floats][totalIntegral: 1 float]
                int W = ew, H = eh;
                std::vector<float> luminance(W * H);
                for (int idx = 0; idx < W * H; ++idx)
                {
                    float r = m_vkEnvMapData[idx * 3 + 0];
                    float g = m_vkEnvMapData[idx * 3 + 1];
                    float b = m_vkEnvMapData[idx * 3 + 2];
                    luminance[idx] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
                }

                // Row integrals weighted by sin(theta) for solid-angle distribution
                std::vector<float> rowIntegral(H, 0.0f);
                for (int row = 0; row < H; ++row)
                {
                    float sinTheta = std::sin((float(row) + 0.5f) / float(H) * 3.14159265f);
                    for (int col = 0; col < W; ++col)
                        rowIntegral[row] += luminance[row * W + col] * sinTheta;
                }

                m_vkEnvCdfData.assign(H + W * H + 1, 0.0f);

                // Marginal CDF (H entries)
                float rowSum = 0.0f;
                for (int row = 0; row < H; ++row)
                {
                    rowSum += rowIntegral[row];
                    m_vkEnvCdfData[row] = rowSum;
                }
                float totalIntegral = rowSum;
                if (totalIntegral > 0.0f)
                    for (int row = 0; row < H; ++row) m_vkEnvCdfData[row] /= totalIntegral;

                // Conditional CDF per row (W entries each, stored starting at offset H)
                for (int row = 0; row < H; ++row)
                {
                    float sinTheta = std::sin((float(row) + 0.5f) / float(H) * 3.14159265f);
                    float colSum = 0.0f;
                    for (int col = 0; col < W; ++col)
                    {
                        colSum += luminance[row * W + col] * sinTheta;
                        m_vkEnvCdfData[H + row * W + col] = colSum;
                    }
                    if (colSum > 0.0f)
                        for (int col = 0; col < W; ++col)
                            m_vkEnvCdfData[H + row * W + col] /= colSum;
                }

                // Total integral stored at end (used in PDF formula in GLSL)
                m_vkEnvCdfData[H + W * H] = totalIntegral;
            }
            else
            {
                m_vkEnvMapData.clear();
                m_vkEnvCdfData.clear();
                m_vkEnvMapW = 0; m_vkEnvMapH = 0;
            }
        }
        else
        {
            m_vkEnvMapData.clear();
            m_vkEnvCdfData.clear();
            m_vkEnvMapW = 0; m_vkEnvMapH = 0;
        }
    }

    if (scene.currentEnvmap == Scene::SolidColor && scene.skyboxColor != m_prevSkyboxColor)
    {
        m_prevSkyboxColor = scene.skyboxColor;
        envChanged = true;
    }

    // ── Upload scene data if needed ───────────────────────────────────────────
    bool firstRender = (m_vkRTTexW == 0 && m_vkRTTexH == 0);
    bool needUpload  = m_vkGeomDirty || envDataChanged || firstRender;
    bool needImage   = needUpload || (w != m_vkRTTexW || h != m_vkRTTexH);

    if (needUpload)
    {
        m_vkRaytracer->uploadSceneData(
            m_vkTriShading, m_vkLights, m_vkTexData,
            m_vkEnvMapData, m_vkEnvCdfData, m_vkInstanceOffsets);
        m_vkGeomDirty = false;
    }

    if (needImage)
    {
        m_vkRaytracer->createOutputImage(w, h);
        m_vkRTTexW = w;
        m_vkRTTexH = h;
        m_vkRaytracer->reset();
        m_vkSampleCount = 0;
        // The RT output image view handle may be recycled by the driver after
        // destroy+create, so the cached descriptor set would silently point to
        // the old (freed) GPU allocation. Force re-creation of the descriptor.
        if (m_vkFullscreenRTShader)
            static_cast<vex::VKShader*>(m_vkFullscreenRTShader.get())->clearExternalTextureCache();
    }
    else if (envChanged)
    {
        m_vkRaytracer->reset();
        m_vkSampleCount = 0;
    }

    // ── Point light change detection ──────────────────────────────────────────
    bool lightChanged = (scene.showLight       != m_prevShowLight
                      || scene.lightPos        != m_prevLightPos
                      || scene.lightColor      != m_prevLightColor
                      || scene.lightIntensity  != m_prevLightIntensity);
    if (lightChanged)
    {
        m_vkRaytracer->reset();
        m_vkSampleCount = 0;
        m_prevShowLight      = scene.showLight;
        m_prevLightPos       = scene.lightPos;
        m_prevLightColor     = scene.lightColor;
        m_prevLightIntensity = scene.lightIntensity;
    }

    // ── Sun light change detection ────────────────────────────────────────────
    glm::vec3 sunDir = scene.getSunDirection();
    bool sunChanged = (scene.showSun          != m_prevShowSun
                    || sunDir                  != m_prevSunDirection
                    || scene.sunColor          != m_prevSunColor
                    || scene.sunIntensity      != m_prevSunIntensity
                    || scene.sunAngularRadius  != m_prevSunAngularRadius);
    if (sunChanged)
    {
        m_vkRaytracer->reset();
        m_vkSampleCount = 0;
        m_prevShowSun          = scene.showSun;
        m_prevSunDirection     = sunDir;
        m_prevSunColor         = scene.sunColor;
        m_prevSunIntensity     = scene.sunIntensity;
        m_prevSunAngularRadius = scene.sunAngularRadius;
    }

    // ── Camera change detection ───────────────────────────────────────────────
    float aspect = static_cast<float>(w) / static_cast<float>(h);
    glm::mat4 view = scene.camera.getViewMatrix();
    glm::mat4 proj = scene.camera.getProjectionMatrix(aspect);
    glm::vec3 camPos = scene.camera.getPosition();

    if (camPos != m_prevCameraPos || view != m_prevViewMatrix)
    {
        m_vkRaytracer->reset();
        m_vkSampleCount = 0;
        m_prevCameraPos  = camPos;
        m_prevViewMatrix = view;
    }

    if (scene.camera.aperture != m_prevAperture || scene.camera.focusDistance != m_prevFocusDistance)
    {
        m_vkRaytracer->reset();
        m_vkSampleCount = 0;
        m_prevAperture      = scene.camera.aperture;
        m_prevFocusDistance = scene.camera.focusDistance;
    }

    // ── Build RTUniforms ──────────────────────────────────────────────────────
    glm::vec3 right = glm::vec3(view[0][0], view[1][0], view[2][0]);
    glm::vec3 up    = glm::vec3(view[0][1], view[1][1], view[2][1]);
    glm::mat4 vp    = proj * view;

    vex::RTUniforms u{};
    vex::rtUniformsSetMat4(u.inverseVP,      glm::inverse(vp));
    vex::rtUniformsSetVec3(u.cameraOrigin,   camPos);
    u.aperture      = scene.camera.aperture;
    vex::rtUniformsSetVec3(u.cameraRight,    right);
    u.focusDistance = scene.camera.focusDistance;
    vex::rtUniformsSetVec3(u.cameraUp,       up);
    u.sampleCount   = m_vkSampleCount;
    u.width         = w;
    u.height        = h;
    u.maxDepth      = m_vkMaxDepth;
    u.rayEps        = m_vkRayEps;
    u.enableNEE             = m_vkEnableNEE             ? 1u : 0u;
    u.enableAA              = m_vkEnableAA              ? 1u : 0u;
    u.enableFireflyClamping = m_vkEnableFireflyClamping ? 1u : 0u;
    u.enableEnvLighting     = m_vkEnableEnvLighting     ? 1u : 0u;
    u.envLightMultiplier    = m_vkEnvLightMultiplier;
    u.flatShading           = m_vkFlatShading           ? 1u : 0u;
    u.enableNormalMapping   = m_vkEnableNormalMapping   ? 1u : 0u;
    u.enableEmissive        = m_vkEnableEmissive        ? 1u : 0u;
    u.bilinearFiltering     = m_vkBilinearFiltering     ? 1u : 0u;
    u.enableRR              = m_vkEnableRR              ? 1u : 0u;

    // Point light
    vex::rtUniformsSetVec3(u.pointLightPos,   scene.lightPos);
    vex::rtUniformsSetVec3(u.pointLightColor, scene.lightColor * scene.lightIntensity);
    u.pointLightEnabled = scene.showLight ? 1u : 0u;

    // Sun / directional light
    vex::rtUniformsSetVec3(u.sunDir,   sunDir);
    vex::rtUniformsSetVec3(u.sunColor, scene.sunColor * scene.sunIntensity);
    u.sunAngularRadius = scene.sunAngularRadius;
    u.sunEnabled       = scene.showSun ? 1u : 0u;

    // Environment
    vex::rtUniformsSetVec3(u.envColor, scene.skyboxColor);
    u.hasEnvMap    = (m_vkEnvMapW > 0) ? 1u : 0u;
    u.envMapWidth  = m_vkEnvMapW;
    u.envMapHeight = m_vkEnvMapH;
    u.hasEnvCDF    = m_vkEnvCdfData.empty() ? 0u : 1u;

    // Light count and total area (stored in m_vkLights header)
    u.lightCount     = 0;
    u.totalLightArea = 0.0f;
    if (m_vkLights.size() >= 2)
    {
        u.lightCount = m_vkLights[0];
        std::memcpy(&u.totalLightArea, &m_vkLights[1], sizeof(float));
    }

    m_vkRaytracer->setUniforms(u);

    // ── Trace one sample into the accumulation image (outside render pass) ────
    // Skip if there is no TLAS (scene is empty after all objects were deleted).
    // Tracing without a valid TLAS writes to an uninitialized descriptor and causes
    // a GPU fault / TDR reset.
    const bool hasTlas = (m_vkRaytracer->getTlas().handle != VK_NULL_HANDLE);

    auto cmd = vex::VKContext::get().getCurrentCommandBuffer();
    if (hasTlas && (m_gpuMaxSamples == 0 || m_vkSampleCount < m_gpuMaxSamples))
    {
        m_vkRaytracer->trace(cmd);
        ++m_vkSampleCount;
        m_vkRaytracer->postTraceBarrier(cmd);
    }

    // ── GPU-side display: sample accumulation image directly ──────────────────
    m_framebuffer->bind();
    m_framebuffer->clear(0.0f, 0.0f, 0.0f, 1.0f);

    if (m_vkFullscreenRTShader && hasTlas)
    {
        auto* rtShaderVK = static_cast<vex::VKShader*>(m_vkFullscreenRTShader.get());

        // Update push-constant floats before bind() pushes them
        m_vkFullscreenRTShader->setFloat("u_sampleCount", static_cast<float>(m_vkSampleCount));
        m_vkFullscreenRTShader->setFloat("u_exposure",    m_vkExposure);
        m_vkFullscreenRTShader->setFloat("u_gamma",       m_vkGamma);
        m_vkFullscreenRTShader->bind();
        // setBool pushes after bind, ensuring the final push contains all values
        m_vkFullscreenRTShader->setBool("u_enableACES",    m_vkEnableACES);
        m_vkFullscreenRTShader->setBool("u_flipV",         true);
        m_vkFullscreenRTShader->setBool("u_enableOutline", m_outlineActive);

        rtShaderVK->setExternalTextureVK(0,
            m_vkRaytracer->getOutputImageView(),
            m_vkRaytracer->getDisplaySampler(),
            VK_IMAGE_LAYOUT_GENERAL);

        auto* vkMaskFB = static_cast<vex::VKFramebuffer*>(m_outlineMaskFB.get());
        rtShaderVK->setExternalTextureVK(1,
            vkMaskFB->getColorImageView(),
            vkMaskFB->getColorSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        m_fullscreenQuad->draw();
        m_vkFullscreenRTShader->unbind();
    }

    m_framebuffer->unbind();
    m_drawCalls = 1;
}
#endif // VEX_BACKEND_VULKAN

// ---------------------------------------------------------------------------
// Shadow map debug display
// ---------------------------------------------------------------------------

uintptr_t SceneRenderer::getShadowMapDisplayHandle()
{
    if (!m_shadowFB || !m_shadowMapEverRendered)
        return 0;

#ifdef VEX_BACKEND_OPENGL
    auto* fb = static_cast<vex::GLFramebuffer*>(m_shadowFB.get());
    fb->prepareDepthForDisplay();
    return static_cast<uintptr_t>(fb->getDepthAttachment());
#else
    auto* fb = static_cast<vex::VKFramebuffer*>(m_shadowFB.get());
    return fb->getDepthImGuiHandle();
#endif
}

bool SceneRenderer::shadowMapFlipsUV() const
{
#ifdef VEX_BACKEND_VULKAN
    // Shadow map uses standard (non-Y-flipped) viewport → row 0 = NDC y=-1 = scene bottom.
    // ImGui's UV y=0 maps to image top, so the display appears inverted — flip to correct.
    return m_shadowFB != nullptr;
#else
    return m_shadowFB ? m_shadowFB->flipsUV() : false;
#endif
}
