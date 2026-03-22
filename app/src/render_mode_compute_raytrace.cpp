#ifdef VEX_BACKEND_VULKAN

#include "render_mode_compute_raytrace.h"
#include "scene.h"

#include <vex/graphics/shader.h>
#include <vex/graphics/framebuffer.h>
#include <vex/graphics/mesh.h>
#include <vex/graphics/texture.h>
#include <vex/core/log.h>

#include <vex/vulkan/vk_context.h>
#include <vex/vulkan/vk_shader.h>
#include <vex/vulkan/vk_framebuffer.h>
#include <vex/vulkan/vk_texture.h>
#include <vex/vulkan/vk_compute_raytracer.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <cstring>

bool VKComputeRaytraceMode::init(const RenderModeInitData& init)
{
    m_fullscreenQuad       = init.fullscreenQuad;
    m_fullscreenRTShader   = init.fullscreenRTShader;
    m_bloomFB[0]           = init.bloomFB[0];
    m_bloomFB[1]           = init.bloomFB[1];
    m_bloomThresholdShader = init.bloomThresholdShader;
    m_bloomBlurShader      = init.bloomBlurShader;
    m_geomCache            = init.geomCache;
    m_vkRTSettings         = init.vkRTSettings;

    m_vkComputeRaytracer = std::make_unique<vex::VKComputeRaytracer>();
    if (!m_vkComputeRaytracer->init())
    {
        vex::Log::error("Failed to initialize Vulkan compute path tracer");
        m_vkComputeRaytracer.reset();
    }
    return true;
}

void VKComputeRaytraceMode::shutdown()
{
    if (m_vkComputeRaytracer)
    {
        m_vkComputeRaytracer->shutdown();
        m_vkComputeRaytracer.reset();
    }
}

void VKComputeRaytraceMode::deactivate()
{
    if (m_vkComputeRaytracer)
        m_vkComputeRaytracer->freeSceneData();
    m_vkComputeRTTexW = 0;
    m_vkComputeRTTexH = 0;
}

void VKComputeRaytraceMode::activate()
{
    m_vkComputeSamplesPerSec = 0.0f;
    if (m_vkComputeRaytracer)
    {
        m_vkComputeRaytracer->reset();
        m_vkComputeSampleCount = 0;
    }
}

void VKComputeRaytraceMode::onGeometryRebuilt()
{
    activate();
    m_vkComputeGeomDirty = true;
}

void VKComputeRaytraceMode::render(Scene& scene, const SharedRenderData& shared, const FrameChanges& changes)
{
    if (!m_vkComputeRaytracer)
        return;

    const auto& spec = shared.outputFB->getSpec();
    uint32_t w = spec.width;
    uint32_t h = spec.height;

    bool firstRender = (m_vkComputeRTTexW == 0 && m_vkComputeRTTexH == 0);

    // VK RT settings come from m_vkRTSettings (points into GPURaytraceMode)
    const VKRTSettings* s = m_vkRTSettings;
    // Fall back to default settings if pointer is null (shouldn't happen in practice)
    VKRTSettings defaultSettings;
    if (!s) s = &defaultSettings;

    // Environment data changes
    if (changes.envDataChanged && !firstRender)
    {
        if (changes.vkEnvMapData && !changes.vkEnvMapData->empty())
            m_vkComputeRaytracer->uploadEnvironmentMap(
                *changes.vkEnvMapData, changes.vkEnvMapW, changes.vkEnvMapH,
                *changes.vkEnvCdfData);
        else
            m_vkComputeRaytracer->clearEnvironmentMap();
    }

    if (m_vkComputeGeomDirty || firstRender)
    {
        m_vkComputeRaytracer->uploadGeometry(
            m_geomCache->triangles(), m_geomCache->bvh(),
            m_geomCache->lightIndices(), m_geomCache->lightCDF(),
            m_geomCache->totalLightArea(), m_geomCache->textures());
        m_vkComputeGeomDirty = false;

        if (changes.vkEnvMapData && !changes.vkEnvMapData->empty())
            m_vkComputeRaytracer->uploadEnvironmentMap(
                *changes.vkEnvMapData, changes.vkEnvMapW, changes.vkEnvMapH,
                *changes.vkEnvCdfData);
        else
            m_vkComputeRaytracer->clearEnvironmentMap();
    }

    // Resize output image if needed
    if (w != m_vkComputeRTTexW || h != m_vkComputeRTTexH)
    {
        m_vkComputeRaytracer->createOutputImage(w, h);
        m_vkComputeRTTexW = w;
        m_vkComputeRTTexH = h;
        m_vkComputeRaytracer->reset();
        m_vkComputeSampleCount = 0;
        if (shared.showDenoisedResult) *shared.showDenoisedResult = false;
        if (m_fullscreenRTShader)
            static_cast<vex::VKShader*>(m_fullscreenRTShader)->clearExternalTextureCache();
    }
    else if (changes.envChanged)
    {
        m_vkComputeRaytracer->reset();
        m_vkComputeSampleCount = 0;
        if (shared.showDenoisedResult) *shared.showDenoisedResult = false;
    }

    if (changes.lightChanged)
    {
        m_vkComputeRaytracer->reset();
        m_vkComputeSampleCount = 0;
        if (shared.showDenoisedResult) *shared.showDenoisedResult = false;
    }

    if (changes.sunChanged)
    {
        m_vkComputeRaytracer->reset();
        m_vkComputeSampleCount = 0;
        if (shared.showDenoisedResult) *shared.showDenoisedResult = false;
    }

    if (changes.cameraChanged || changes.dofChanged)
    {
        m_vkComputeRaytracer->reset();
        m_vkComputeSampleCount = 0;
        if (shared.showDenoisedResult) *shared.showDenoisedResult = false;
    }

    // Build VKComputeUniforms
    float aspect = static_cast<float>(w) / static_cast<float>(h);
    glm::mat4 view = changes.viewMatrix;
    glm::mat4 proj = scene.camera.getProjectionMatrix(aspect);
    glm::vec3 camPos = changes.camPos;
    glm::vec3 right = glm::vec3(view[0][0], view[1][0], view[2][0]);
    glm::vec3 up    = glm::vec3(view[0][1], view[1][1], view[2][1]);
    glm::mat4 vp    = proj * view;

    vex::VKComputeUniforms u{};
    vex::vkComputeUniformsSetMat4(u.inverseVP,      glm::inverse(vp));
    vex::vkComputeUniformsSetVec3(u.cameraOrigin,   camPos);
    u.aperture      = scene.camera.aperture;
    vex::vkComputeUniformsSetVec3(u.cameraRight,    right);
    u.focusDistance = scene.camera.focusDistance;
    vex::vkComputeUniformsSetVec3(u.cameraUp,       up);
    u.sampleCount   = m_vkComputeSampleCount;
    u.width         = w;
    u.height        = h;
    u.maxDepth      = s->maxDepth;
    u.rayEps        = s->rayEps;
    u.enableNEE             = s->enableNEE             ? 1u : 0u;
    u.enableAA              = s->enableAA              ? 1u : 0u;
    u.enableFireflyClamping  = s->enableFireflyClamping ? 1u : 0u;
    u.fireflyClampThreshold  = s->fireflyClampThreshold;
    u.enableEnvLighting     = s->enableEnvLighting     ? 1u : 0u;
    u.envLightMultiplier    = s->envLightMultiplier;
    u.flatShading           = s->flatShading           ? 1u : 0u;
    u.enableNormalMapping   = s->enableNormalMapping   ? 1u : 0u;
    u.enableEmissive        = s->enableEmissive        ? 1u : 0u;
    u.bilinearFiltering     = s->bilinearFiltering     ? 1u : 0u;
    u.samplerType           = static_cast<uint32_t>(s->samplerType);
    u.enableRR              = s->enableRR              ? 1u : 0u;
    u.useLuminanceCDF       = (m_geomCache && m_geomCache->useLuminanceCDF()) ? 1u : 0u;

    vex::vkComputeUniformsSetVec3(u.pointLightPos,   scene.lightPos);
    vex::vkComputeUniformsSetVec3(u.pointLightColor, scene.lightColor * scene.lightIntensity);
    u.pointLightEnabled = scene.showLight ? 1u : 0u;

    vex::vkComputeUniformsSetVec3(u.sunDir,   changes.sunDir);
    vex::vkComputeUniformsSetVec3(u.sunColor, scene.sunColor * scene.sunIntensity);
    u.sunAngularRadius = scene.sunAngularRadius;
    u.sunEnabled       = scene.showSun ? 1u : 0u;

    vex::vkComputeUniformsSetVec3(u.envColor, scene.skyboxColor);
    u.hasEnvMap    = (changes.vkEnvMapW > 0) ? 1u : 0u;
    u.envMapWidth  = changes.vkEnvMapW;
    u.envMapHeight = changes.vkEnvMapH;
    u.hasEnvCDF    = (changes.vkEnvCdfData && !changes.vkEnvCdfData->empty()) ? 1u : 0u;

    u.lightCount     = m_geomCache ? static_cast<uint32_t>(m_geomCache->lightIndices().size()) : 0u;
    u.totalLightArea = m_geomCache ? m_geomCache->totalLightArea() : 0.0f;

    u.triangleCount = m_vkComputeRaytracer->getTriangleCount();
    u.bvhNodeCount  = m_vkComputeRaytracer->getBvhNodeCount();

    m_vkComputeRaytracer->setUniforms(u);

    // Trace one sample
    auto cmd = vex::VKContext::get().getCurrentCommandBuffer();
    bool showDenoised = shared.showDenoisedResult && *shared.showDenoisedResult;
    if (!showDenoised && (shared.maxSamples == 0 || m_vkComputeSampleCount < shared.maxSamples))
    {
        m_vkComputeRaytracer->traceSample(cmd);

        auto now = std::chrono::steady_clock::now();
        if (m_vkComputeSampleCount == 0) {
            m_vkComputeSamplesPerSec = 0.0f;
        } else {
            float dt = std::chrono::duration<float>(now - m_vkComputeLastSampleTime).count();
            if (dt > 1e-6f) {
                float instant = 1.0f / dt;
                m_vkComputeSamplesPerSec = m_vkComputeSamplesPerSec < 1e-6f
                    ? instant : m_vkComputeSamplesPerSec * 0.9f + instant * 0.1f;
            }
        }
        m_vkComputeLastSampleTime = now;
        ++m_vkComputeSampleCount;
        m_vkComputeRaytracer->postTraceBarrier(cmd);
    }

    // Bloom pass
    VkImageView vkRTBloomView    = VK_NULL_HANDLE;
    VkSampler   vkRTBloomSampler = VK_NULL_HANDLE;
    vex::Texture2D* denoisedTex = shared.cpuAccumTex;
    bool vkRTBloomActive = shared.bloomEnabled
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
            bloomSrcView    = m_vkComputeRaytracer->getOutputImageView();
            bloomSrcSampler = m_vkComputeRaytracer->getDisplaySampler();
            bloomSrcLayout  = VK_IMAGE_LAYOUT_GENERAL;
            bloomSampleCount = static_cast<float>(m_vkComputeSampleCount);
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
        m_fullscreenRTShader->setFloat("u_exposure",       s->exposure);
        m_fullscreenRTShader->setFloat("u_gamma",          s->gamma);
        m_fullscreenRTShader->setFloat("u_bloomIntensity", shared.bloomIntensity);
        m_fullscreenRTShader->bind();
        m_fullscreenRTShader->setBool("u_enableACES",    s->enableACES);
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
    else if (m_fullscreenRTShader && m_vkComputeRaytracer->getOutputImageView())
    {
        auto* rtShaderVK = static_cast<vex::VKShader*>(m_fullscreenRTShader);

        m_fullscreenRTShader->setFloat("u_sampleCount",    static_cast<float>(m_vkComputeSampleCount));
        m_fullscreenRTShader->setFloat("u_exposure",       s->exposure);
        m_fullscreenRTShader->setFloat("u_gamma",          s->gamma);
        m_fullscreenRTShader->setFloat("u_bloomIntensity", shared.bloomIntensity);
        m_fullscreenRTShader->bind();
        m_fullscreenRTShader->setBool("u_enableACES",    s->enableACES);
        m_fullscreenRTShader->setBool("u_flipV",         true);
        m_fullscreenRTShader->setBool("u_enableOutline", shared.outlineActive);
        m_fullscreenRTShader->setBool("u_enableBloom",   vkRTBloomActive);

        rtShaderVK->setExternalTextureVK(0,
            m_vkComputeRaytracer->getOutputImageView(),
            m_vkComputeRaytracer->getDisplaySampler(),
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
