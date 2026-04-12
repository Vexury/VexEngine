#include "render_mode_gpu_raytrace.h"
#include "scene.h"

#include <vex/graphics/shader.h>
#include <vex/graphics/framebuffer.h>
#include <vex/graphics/mesh.h>
#include <vex/graphics/texture.h>
#include <vex/core/log.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <cstring>

// =============================================================================
// OpenGL GPU Raytrace Mode
// =============================================================================
#ifdef VEX_BACKEND_OPENGL
#include <glad/glad.h>
#include <vex/opengl/gl_gpu_raytracer.h>

bool GPURaytraceMode::init(const RenderModeInitData& init)
{
    m_fullscreenQuad       = init.fullscreenQuad;
    m_whiteTexture         = init.whiteTexture;
    m_fullscreenRTShader   = init.fullscreenRTShader;
    m_bloomFB[0]           = init.bloomFB[0];
    m_bloomFB[1]           = init.bloomFB[1];
    m_bloomThresholdShader = init.bloomThresholdShader;
    m_bloomBlurShader      = init.bloomBlurShader;
    m_geomCache            = init.geomCache;

    m_raytracer = std::make_unique<vex::GLGPURaytracer>();
    if (!m_raytracer->init())
    {
        vex::Log::error("Failed to initialize GPU raytracer");
        m_raytracer.reset();
    }
    return true;
}

void GPURaytraceMode::shutdown()
{
    if (m_raytracer)
    {
        m_raytracer->shutdown();
        m_raytracer.reset();
    }
}

void GPURaytraceMode::deactivate() {}

void GPURaytraceMode::activate()
{
    m_samplesPerSec = 0.0f;
    if (m_raytracer)
    {
        m_raytracer->reset();
        m_geomDirty = true;
    }
}

void GPURaytraceMode::onGeometryRebuilt() { activate(); }

uint32_t GPURaytraceMode::getSampleCount() const
{
    return m_raytracer ? m_raytracer->getSampleCount() : 0;
}

bool GPURaytraceMode::reloadShader()
{
    return m_raytracer ? m_raytracer->reloadShader() : false;
}

void GPURaytraceMode::render(Scene& scene, const SharedRenderData& shared, const FrameChanges& changes)
{
    if (!m_raytracer || !m_fullscreenRTShader)
        return;

    const auto& spec = shared.outputFB->getSpec();
    uint32_t w = spec.width;
    uint32_t h = spec.height;

    m_raytracer->resize(w, h);

    // Upload geometry if dirty
    if (m_geomDirty && m_geomCache)
    {
        m_raytracer->uploadGeometry(m_geomCache->triangles(), m_geomCache->bvh(),
                                    m_geomCache->lightIndices(), m_geomCache->lightCDF(),
                                    m_geomCache->totalLightArea(), m_geomCache->textures());
        m_geomDirty = false;
    }

    // Environment
    if (changes.envChanged)
    {
        if (changes.glEnvMapData)
            m_raytracer->setEnvironmentMap(changes.glEnvMapData, changes.glEnvMapW, changes.glEnvMapH);
        else
            m_raytracer->clearEnvironmentMap();
        m_raytracer->reset();
    }

    if (changes.skyboxColorChanged)
    {
        m_raytracer->setEnvironmentColor(scene.skyboxColor);
        m_raytracer->reset();
    }

    if (changes.lightChanged)
    {
        m_raytracer->setPointLight(scene.lightPos, scene.lightColor * scene.lightIntensity, scene.showLight);
        m_raytracer->reset();
    }

    if (changes.sunChanged)
    {
        m_raytracer->setDirectionalLight(
            changes.sunDir,
            scene.sunColor * scene.sunIntensity,
            scene.sunAngularRadius,
            scene.showSun);
        m_raytracer->reset();
    }

    if (changes.cameraChanged)
    {
        m_raytracer->reset();
    }

    float aspect = static_cast<float>(w) / static_cast<float>(h);
    glm::mat4 view = changes.viewMatrix;
    glm::mat4 proj = scene.camera.getProjectionMatrix(aspect);

    glm::mat4 vp = proj * view;
    m_raytracer->setCamera(changes.camPos, glm::inverse(vp));

    {
        glm::vec3 right = glm::vec3(view[0][0], view[1][0], view[2][0]);
        glm::vec3 up    = glm::vec3(view[0][1], view[1][1], view[2][1]);
        if (changes.dofChanged)
            m_raytracer->reset();
        m_raytracer->setDoF(scene.camera.aperture, scene.camera.focusDistance, right, up);
    }

    if (shared.showDenoisedResult && *shared.showDenoisedResult) {
        // Denoised display handled in SceneRenderer::triggerDenoise
        // Nothing to trace
    } else {
        // Skip if sample limit reached
        if (shared.maxSamples == 0 || m_raytracer->getSampleCount() < shared.maxSamples)
        {
            auto now = std::chrono::steady_clock::now();
            if (m_raytracer->getSampleCount() == 0) {
                m_samplesPerSec = 0.0f;
            } else {
                float dt = std::chrono::duration<float>(now - m_lastSampleTime).count();
                if (dt > 1e-6f) {
                    float instant = 1.0f / dt;
                    m_samplesPerSec = m_samplesPerSec < 1e-6f
                        ? instant : m_samplesPerSec * 0.9f + instant * 0.1f;
                }
            }
            m_lastSampleTime = now;
            m_raytracer->traceSample();
        }
    }

    // Bloom pass
    uint32_t bloomTex = 0;
    if (shared.bloomEnabled && m_bloomThresholdShader && m_bloomBlurShader
        && m_bloomFB[0] && m_bloomFB[1])
    {
        const auto& outSpec = shared.outputFB->getSpec();
        uint32_t bw = std::max(1u, outSpec.width / 2);
        uint32_t bh = std::max(1u, outSpec.height / 2);
        if (m_bloomFB[0]->getSpec().width != bw || m_bloomFB[0]->getSpec().height != bh)
        {
            m_bloomFB[0]->resize(bw, bh);
            m_bloomFB[1]->resize(bw, bh);
        }

        glDisable(GL_DEPTH_TEST);

        m_bloomFB[0]->bind();
        m_bloomThresholdShader->bind();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_raytracer->getAccumTexture());
        m_bloomThresholdShader->setInt("u_hdrMap", 0);
        m_bloomThresholdShader->setFloat("u_threshold", shared.bloomThreshold);
        m_bloomThresholdShader->setFloat("u_sampleCount", static_cast<float>(m_raytracer->getSampleCount()));
        m_fullscreenQuad->draw();
        m_bloomThresholdShader->unbind();
        m_bloomFB[0]->unbind();

        bool horizontal = true;
        for (int i = 0; i < shared.bloomBlurPasses * 2; ++i)
        {
            int src = horizontal ? 0 : 1;
            int dst = horizontal ? 1 : 0;
            m_bloomFB[dst]->bind();
            m_bloomBlurShader->bind();
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(m_bloomFB[src]->getColorAttachmentHandle()));
            m_bloomBlurShader->setInt("u_image", 0);
            m_bloomBlurShader->setBool("u_horizontal", horizontal);
            m_fullscreenQuad->draw();
            m_bloomBlurShader->unbind();
            m_bloomFB[dst]->unbind();
            horizontal = !horizontal;
        }
        bloomTex = static_cast<GLuint>(m_bloomFB[0]->getColorAttachmentHandle());
    }

    // Display
    shared.outputFB->bind();
    shared.outputFB->clear(0.0f, 0.0f, 0.0f, 1.0f);

    glDisable(GL_DEPTH_TEST);

    vex::Shader* rtShader = m_fullscreenRTShader;
    rtShader->bind();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_raytracer->getAccumTexture());
    rtShader->setInt("u_accumMap", 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(shared.outlineMaskFB->getColorAttachmentHandle()));
    rtShader->setInt("u_outlineMask", 1);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, bloomTex != 0
        ? bloomTex
        : static_cast<GLuint>(m_whiteTexture->getNativeHandle()));
    rtShader->setInt("u_bloomMap", 2);

    rtShader->setFloat("u_sampleCount", static_cast<float>(m_raytracer->getSampleCount()));
    rtShader->setFloat("u_exposure",    m_settings.exposure);
    rtShader->setFloat("u_gamma",       m_settings.gamma);
    rtShader->setBool("u_enableACES",   m_settings.enableACES);
    rtShader->setBool("u_flipV", true);
    rtShader->setBool("u_enableOutline", shared.outlineActive);
    rtShader->setBool("u_enableBloom", bloomTex != 0);
    rtShader->setFloat("u_bloomIntensity", shared.bloomIntensity);

    m_fullscreenQuad->draw();
    rtShader->unbind();

    glEnable(GL_DEPTH_TEST);

    shared.outputFB->unbind();
    if (shared.drawCalls) *shared.drawCalls = 1;
}

#endif // VEX_BACKEND_OPENGL

// =============================================================================
// Vulkan HW RT Mode
// =============================================================================
#ifdef VEX_BACKEND_VULKAN
#include <vex/vulkan/vk_context.h>
#include <vex/vulkan/vk_shader.h>
#include <vex/vulkan/vk_framebuffer.h>
#include <vex/vulkan/vk_texture.h>
#include <vex/vulkan/vk_gpu_raytracer.h>

bool GPURaytraceMode::init(const RenderModeInitData& init)
{
    m_fullscreenQuad       = init.fullscreenQuad;
    m_whiteTexture         = init.whiteTexture;
    m_fullscreenRTShader   = init.fullscreenRTShader;
    m_bloomFB[0]           = init.bloomFB[0];
    m_bloomFB[1]           = init.bloomFB[1];
    m_bloomThresholdShader = init.bloomThresholdShader;
    m_bloomBlurShader      = init.bloomBlurShader;
    m_geomCache            = init.geomCache;

    m_raytracer = std::make_unique<vex::VKGpuRaytracer>();
    if (!m_raytracer->init())
    {
        vex::Log::error("Failed to initialize Vulkan RT raytracer");
        m_raytracer.reset();
    }
    return true;
}

void GPURaytraceMode::shutdown()
{
    if (m_raytracer)
    {
        m_raytracer->shutdown();
        m_raytracer.reset();
    }
}

void GPURaytraceMode::deactivate()
{
    if (m_raytracer)
        m_raytracer->freeSceneData();
    m_rtTexW = 0;
    m_rtTexH = 0;
}

void GPURaytraceMode::activate()
{
    m_samplesPerSec = 0.0f;
    if (m_raytracer)
    {
        m_raytracer->reset();
        m_sampleCount = 0;
    }
}

void GPURaytraceMode::onGeometryRebuilt()
{
    activate();
    m_geomDirty = true;
}

uint32_t GPURaytraceMode::getSampleCount() const { return m_sampleCount; }
bool     GPURaytraceMode::reloadShader()         { return false; }

void GPURaytraceMode::render(Scene& scene, const SharedRenderData& shared, const FrameChanges& changes)
{
    if (!m_raytracer)
        return;

    // Reset accumulator when settings change (user-facing knobs that affect the render)
    if (m_settings != m_prevSettings)
    {
        activate();
        m_prevSettings = m_settings;
    }

    const auto& spec = shared.outputFB->getSpec();
    uint32_t w = spec.width;
    uint32_t h = spec.height;

    bool firstRender = (m_rtTexW == 0 && m_rtTexH == 0);

    // Handle volume changes (passed via changes.volumesChanged)
    // Volume data is in shared.vkVolumesData

    // Upload scene data if dirty
    bool needUpload = m_geomDirty || changes.envDataChanged || firstRender;
    bool needImage  = needUpload || (w != m_rtTexW || h != m_rtTexH);

    if (needUpload)
    {
        if (m_geomDirty)
            vex::Log::info("  VK SSBO: uploading triangles to GPU (GPURaytraceMode)");

        m_raytracer->uploadSceneData(
            m_geomCache->vkTriShading(), m_geomCache->vkLights(),
            m_geomCache->vkTexData(),
            changes.vkEnvMapData ? *changes.vkEnvMapData : std::vector<float>{},
            changes.vkEnvCdfData ? *changes.vkEnvCdfData : std::vector<float>{},
            m_geomCache->vkInstanceOffsets(),
            *shared.vkVolumesData);
        m_geomDirty = false;
    }

    if (needImage)
    {
        m_raytracer->createOutputImage(w, h);
        m_rtTexW = w;
        m_rtTexH = h;
        m_raytracer->reset();
        m_sampleCount = 0;
        if (shared.showDenoisedResult) *shared.showDenoisedResult = false;
        if (m_fullscreenRTShader)
            static_cast<vex::VKShader*>(m_fullscreenRTShader)->clearExternalTextureCache();
    }
    else if (changes.envChanged)
    {
        m_raytracer->reset();
        m_sampleCount = 0;
        if (shared.showDenoisedResult) *shared.showDenoisedResult = false;
    }

    if (changes.lightChanged)
    {
        m_raytracer->reset();
        m_sampleCount = 0;
        if (shared.showDenoisedResult) *shared.showDenoisedResult = false;
    }

    if (changes.sunChanged)
    {
        m_raytracer->reset();
        m_sampleCount = 0;
        if (shared.showDenoisedResult) *shared.showDenoisedResult = false;
    }

    if (changes.cameraChanged || changes.dofChanged)
    {
        m_raytracer->reset();
        m_sampleCount = 0;
        if (shared.showDenoisedResult) *shared.showDenoisedResult = false;
    }

    if (changes.volumesChanged && !m_geomDirty)
    {
        m_raytracer->reset();
        m_sampleCount = 0;
        if (shared.showDenoisedResult) *shared.showDenoisedResult = false;
        m_raytracer->uploadVolumes(*shared.vkVolumesData);
    }

    // Build RTUniforms
    float aspect = static_cast<float>(w) / static_cast<float>(h);
    glm::mat4 view = changes.viewMatrix;
    glm::mat4 proj = scene.camera.getProjectionMatrix(aspect);
    glm::vec3 camPos = changes.camPos;
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
    u.sampleCount   = m_sampleCount;
    u.width         = w;
    u.height        = h;
    u.maxDepth      = m_settings.maxDepth;
    u.rayEps        = m_settings.rayEps;
    u.enableNEE             = m_settings.enableNEE             ? 1u : 0u;
    u.enableAA              = m_settings.enableAA              ? 1u : 0u;
    u.enableFireflyClamping  = m_settings.enableFireflyClamping ? 1u : 0u;
    u.fireflyClampThreshold  = m_settings.fireflyClampThreshold;
    u.enableEnvLighting     = m_settings.enableEnvLighting     ? 1u : 0u;
    u.envLightMultiplier    = m_settings.envLightMultiplier;
    u.flatShading           = m_settings.flatShading           ? 1u : 0u;
    u.enableNormalMapping   = m_settings.enableNormalMapping   ? 1u : 0u;
    u.enableEmissive        = m_settings.enableEmissive        ? 1u : 0u;
    u.bilinearFiltering     = m_settings.bilinearFiltering     ? 1u : 0u;
    u.samplerType           = static_cast<uint32_t>(m_settings.samplerType);
    u.enableRR              = m_settings.enableRR              ? 1u : 0u;
    u.useLuminanceCDF       = (m_geomCache && m_geomCache->useLuminanceCDF()) ? 1u : 0u;

    vex::rtUniformsSetVec3(u.pointLightPos,   scene.lightPos);
    vex::rtUniformsSetVec3(u.pointLightColor, scene.lightColor * scene.lightIntensity);
    u.pointLightEnabled = scene.showLight ? 1u : 0u;

    vex::rtUniformsSetVec3(u.sunDir,   changes.sunDir);
    vex::rtUniformsSetVec3(u.sunColor, scene.sunColor * scene.sunIntensity);
    u.sunAngularRadius = scene.sunAngularRadius;
    u.sunEnabled       = scene.showSun ? 1u : 0u;

    vex::rtUniformsSetVec3(u.envColor, scene.skyboxColor);
    u.envRotation  = scene.envRotation;
    u.hasEnvMap    = (changes.vkEnvMapW > 0) ? 1u : 0u;
    u.envMapWidth  = changes.vkEnvMapW;
    u.envMapHeight = changes.vkEnvMapH;
    u.hasEnvCDF    = (changes.vkEnvCdfData && !changes.vkEnvCdfData->empty()) ? 1u : 0u;

    u.lightCount     = 0;
    u.totalLightArea = 0.0f;
    if (m_geomCache && m_geomCache->vkLights().size() >= 2)
    {
        u.lightCount = m_geomCache->vkLights()[0];
        std::memcpy(&u.totalLightArea, &m_geomCache->vkLights()[1], sizeof(float));
    }

    m_raytracer->setUniforms(u);

    const bool hasTlas = (m_raytracer->getTlas().handle != VK_NULL_HANDLE);
    auto cmd = vex::VKContext::get().getCurrentCommandBuffer();

    bool showDenoised = shared.showDenoisedResult && *shared.showDenoisedResult;
    if (!showDenoised && hasTlas && (shared.maxSamples == 0 || m_sampleCount < shared.maxSamples))
    {
        m_raytracer->trace(cmd);

        auto now = std::chrono::steady_clock::now();
        if (m_sampleCount == 0) {
            m_samplesPerSec = 0.0f;
        } else {
            float dt = std::chrono::duration<float>(now - m_lastSampleTime).count();
            if (dt > 1e-6f) {
                float instant = 1.0f / dt;
                m_samplesPerSec = m_samplesPerSec < 1e-6f
                    ? instant : m_samplesPerSec * 0.9f + instant * 0.1f;
            }
        }
        m_lastSampleTime = now;
        ++m_sampleCount;
        m_raytracer->postTraceBarrier(cmd);
    }

    // Bloom pass
    VkImageView vkRTBloomView    = VK_NULL_HANDLE;
    VkSampler   vkRTBloomSampler = VK_NULL_HANDLE;
    vex::Texture2D* denoisedTex = shared.cpuAccumTex;
    bool vkRTBloomActive = shared.bloomEnabled && hasTlas
                           && m_bloomThresholdShader && m_bloomBlurShader
                           && m_bloomFB[0] && m_bloomFB[1];
    if (vkRTBloomActive)
    {
        const auto& outSpec = shared.outputFB->getSpec();
        uint32_t bw = std::max(1u, outSpec.width / 2);
        uint32_t bh = std::max(1u, outSpec.height / 2);
        bool needResize = (m_bloomFB[0]->getSpec().width != bw
                        || m_bloomFB[0]->getSpec().height != bh);
        if (needResize)
        {
            m_bloomFB[0]->resize(bw, bh);
            m_bloomFB[1]->resize(bw, bh);
            m_bloomThresholdShader->preparePipeline(*m_bloomFB[0]);
            m_bloomBlurShader->preparePipeline(*m_bloomFB[0]);
            static_cast<vex::VKShader*>(m_bloomThresholdShader)->clearExternalTextureCache();
            static_cast<vex::VKShader*>(m_bloomBlurShader)->clearExternalTextureCache();
        }

        // When denoised, compute bloom from the denoised HDR texture (pre-normalized)
        VkImageView   bloomSrcView;
        VkSampler     bloomSrcSampler;
        VkImageLayout bloomSrcLayout;
        float         bloomSampleCount;
        if (showDenoised && denoisedTex)
        {
            auto* vkTex = static_cast<vex::VKTexture2D*>(denoisedTex);
            bloomSrcView    = vkTex->getImageView();
            bloomSrcSampler = vkTex->getSampler();
            bloomSrcLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            bloomSampleCount = 1.0f;
        }
        else
        {
            bloomSrcView    = m_raytracer->getOutputImageView();
            bloomSrcSampler = m_raytracer->getDisplaySampler();
            bloomSrcLayout  = VK_IMAGE_LAYOUT_GENERAL;
            bloomSampleCount = static_cast<float>(m_sampleCount);
        }

        auto* vkThreshVK = static_cast<vex::VKShader*>(m_bloomThresholdShader);
        m_bloomFB[0]->bind();
        m_bloomFB[0]->clear(0.0f, 0.0f, 0.0f, 1.0f);
        m_bloomThresholdShader->setFloat("u_threshold",   shared.bloomThreshold);
        m_bloomThresholdShader->setFloat("u_sampleCount", bloomSampleCount);
        m_bloomThresholdShader->bind();
        vkThreshVK->setExternalTextureVK(0, bloomSrcView, bloomSrcSampler, bloomSrcLayout);
        m_fullscreenQuad->draw();
        m_bloomThresholdShader->unbind();
        m_bloomFB[0]->unbind();

        auto* vkBlurVK = static_cast<vex::VKShader*>(m_bloomBlurShader);
        bool horizontal = true;
        for (int i = 0; i < shared.bloomBlurPasses * 2; ++i)
        {
            int src = horizontal ? 0 : 1;
            int dst = horizontal ? 1 : 0;
            auto* srcFBVK = static_cast<vex::VKFramebuffer*>(m_bloomFB[src]);
            m_bloomFB[dst]->bind();
            m_bloomFB[dst]->clear(0.0f, 0.0f, 0.0f, 1.0f);
            m_bloomBlurShader->bind();
            vkBlurVK->setExternalTextureVK(0,
                srcFBVK->getColorImageView(),
                srcFBVK->getColorSampler(),
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            m_bloomBlurShader->setBool("u_horizontal", horizontal);
            m_fullscreenQuad->draw();
            m_bloomBlurShader->unbind();
            m_bloomFB[dst]->unbind();
            horizontal = !horizontal;
        }
        auto* vkBloom0 = static_cast<vex::VKFramebuffer*>(m_bloomFB[0]);
        vkRTBloomView    = vkBloom0->getColorImageView();
        vkRTBloomSampler = vkBloom0->getColorSampler();
    }

    // Display
    shared.outputFB->bind();
    shared.outputFB->clear(0.0f, 0.0f, 0.0f, 1.0f);

    if (showDenoised && m_fullscreenRTShader && denoisedTex)
    {
        auto* rtShaderVK = static_cast<vex::VKShader*>(m_fullscreenRTShader);
        auto* vkTex      = static_cast<vex::VKTexture2D*>(denoisedTex);
        auto* vkMaskFB   = static_cast<vex::VKFramebuffer*>(shared.outlineMaskFB);

        m_fullscreenRTShader->setFloat("u_sampleCount",    1.0f);
        m_fullscreenRTShader->setFloat("u_exposure",       m_settings.exposure);
        m_fullscreenRTShader->setFloat("u_gamma",          m_settings.gamma);
        m_fullscreenRTShader->setFloat("u_bloomIntensity", shared.bloomIntensity);
        m_fullscreenRTShader->bind();
        m_fullscreenRTShader->setBool("u_enableACES",    m_settings.enableACES);
        m_fullscreenRTShader->setBool("u_flipV",         true);
        m_fullscreenRTShader->setBool("u_enableOutline", shared.outlineActive);
        m_fullscreenRTShader->setBool("u_enableBloom",   vkRTBloomActive);

        rtShaderVK->setExternalTextureVK(0,
            vkTex->getImageView(), vkTex->getSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        rtShaderVK->setExternalTextureVK(1,
            vkMaskFB->getColorImageView(), vkMaskFB->getColorSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        rtShaderVK->setExternalTextureVK(2,
            vkRTBloomActive ? vkRTBloomView    : vkMaskFB->getColorImageView(),
            vkRTBloomActive ? vkRTBloomSampler : vkMaskFB->getColorSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        m_fullscreenQuad->draw();
        m_fullscreenRTShader->unbind();
    }
    else if (m_fullscreenRTShader && hasTlas)
    {
        auto* rtShaderVK = static_cast<vex::VKShader*>(m_fullscreenRTShader);

        m_fullscreenRTShader->setFloat("u_sampleCount",    static_cast<float>(m_sampleCount));
        m_fullscreenRTShader->setFloat("u_exposure",       m_settings.exposure);
        m_fullscreenRTShader->setFloat("u_gamma",          m_settings.gamma);
        m_fullscreenRTShader->setFloat("u_bloomIntensity", shared.bloomIntensity);
        m_fullscreenRTShader->bind();
        m_fullscreenRTShader->setBool("u_enableACES",    m_settings.enableACES);
        m_fullscreenRTShader->setBool("u_flipV",         true);
        m_fullscreenRTShader->setBool("u_enableOutline", shared.outlineActive);
        m_fullscreenRTShader->setBool("u_enableBloom",   vkRTBloomActive);

        rtShaderVK->setExternalTextureVK(0,
            m_raytracer->getOutputImageView(),
            m_raytracer->getDisplaySampler(),
            VK_IMAGE_LAYOUT_GENERAL);

        auto* vkMaskFB = static_cast<vex::VKFramebuffer*>(shared.outlineMaskFB);
        rtShaderVK->setExternalTextureVK(1,
            vkMaskFB->getColorImageView(), vkMaskFB->getColorSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        rtShaderVK->setExternalTextureVK(2,
            vkRTBloomActive ? vkRTBloomView    : vkMaskFB->getColorImageView(),
            vkRTBloomActive ? vkRTBloomSampler : vkMaskFB->getColorSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        m_fullscreenQuad->draw();
        m_fullscreenRTShader->unbind();
    }

    shared.outputFB->unbind();
    if (shared.drawCalls) *shared.drawCalls = 1;
}

#endif // VEX_BACKEND_VULKAN
