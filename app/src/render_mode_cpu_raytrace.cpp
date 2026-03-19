#include "render_mode_cpu_raytrace.h"
#include "scene.h"

#include <vex/graphics/shader.h>
#include <vex/graphics/framebuffer.h>
#include <vex/graphics/mesh.h>
#include <vex/graphics/texture.h>
#include <vex/raytracing/cpu_raytracer.h>
#include <vex/core/log.h>

#ifdef VEX_BACKEND_OPENGL
#include <glad/glad.h>
#endif

#ifdef VEX_BACKEND_VULKAN
#include <vex/vulkan/vk_shader.h>
#include <vex/vulkan/vk_framebuffer.h>
#include <vex/vulkan/vk_texture.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>

bool CPURaytraceMode::init(const RenderModeInitData& init)
{
    m_fullscreenQuad       = init.fullscreenQuad;
    m_whiteTexture         = init.whiteTexture;
    m_fullscreenRTShader   = init.fullscreenRTShader;
    m_bloomFB[0]           = init.bloomFB[0];
    m_bloomFB[1]           = init.bloomFB[1];
    m_bloomThresholdShader = init.bloomThresholdShader;
    m_bloomBlurShader      = init.bloomBlurShader;
    m_resizeCPUAccumTex    = init.resizeCPUAccumTex;
    m_cpuRaytracer         = init.cpuRaytracer;
    return true;
}

void CPURaytraceMode::activate()
{
    m_samplesPerSec = 0.0f;
    if (m_cpuRaytracer)
        m_cpuRaytracer->reset();
}

void CPURaytraceMode::render(Scene& scene, const SharedRenderData& shared, const FrameChanges& changes)
{
    if (!m_cpuRaytracer)
        return;

    const auto& spec = shared.outputFB->getSpec();
    uint32_t w = spec.width;
    uint32_t h = spec.height;

    // Resize raytracer if viewport changed
    m_cpuRaytracer->resize(w, h);

    // Recreate/resize the CPU accumulation texture if needed
    vex::Texture2D* raytraceTex = m_resizeCPUAccumTex ? m_resizeCPUAccumTex(w, h) : shared.cpuAccumTex;

    // React to environment changes (pre-computed in FrameChanges)
    if (changes.envChanged)
    {
        m_cpuRaytracer->reset();
        if (shared.showDenoisedResult) *shared.showDenoisedResult = false;
    }

    // React to light changes
    if (changes.lightChanged)
    {
        m_cpuRaytracer->setPointLight(scene.lightPos,
                                      scene.lightColor * scene.lightIntensity,
                                      scene.showLight);
        m_cpuRaytracer->reset();
        if (shared.showDenoisedResult) *shared.showDenoisedResult = false;
    }

    // React to sun changes
    if (changes.sunChanged)
    {
        m_cpuRaytracer->setDirectionalLight(
            changes.sunDir,
            scene.sunColor * scene.sunIntensity,
            scene.sunAngularRadius,
            scene.showSun);
        m_cpuRaytracer->reset();
        if (shared.showDenoisedResult) *shared.showDenoisedResult = false;
    }

    // React to camera changes
    if (changes.cameraChanged)
    {
        m_cpuRaytracer->reset();
        if (shared.showDenoisedResult) *shared.showDenoisedResult = false;
    }

    float aspect = static_cast<float>(w) / static_cast<float>(h);
    glm::mat4 view = changes.viewMatrix;
    glm::mat4 proj = scene.camera.getProjectionMatrix(aspect);
    glm::vec3 camPos = changes.camPos;

    glm::mat4 vp = proj * view;
    m_cpuRaytracer->setCamera(camPos, glm::inverse(vp));

    // DoF
    {
        glm::vec3 right = glm::vec3(view[0][0], view[1][0], view[2][0]);
        glm::vec3 up    = glm::vec3(view[0][1], view[1][1], view[2][1]);
        m_cpuRaytracer->setDoF(scene.camera.aperture, scene.camera.focusDistance, right, up);
    }

    bool showDenoised = shared.showDenoisedResult && *shared.showDenoisedResult;
    if (!showDenoised &&
        (shared.maxSamples == 0 || m_cpuRaytracer->getSampleCount() < shared.maxSamples))
    {
        auto now = std::chrono::steady_clock::now();
        if (m_cpuRaytracer->getSampleCount() == 0) {
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
        m_cpuRaytracer->traceSample();

        // Upload result to texture
        if (raytraceTex)
        {
            const auto& pixels = m_cpuRaytracer->getPixelBuffer();
            raytraceTex->setData(pixels.data(), w, h, 4);
        }
    }

#ifdef VEX_BACKEND_OPENGL
    // Bloom pass
    uint32_t bloomTex = 0;
    if (raytraceTex && shared.bloomEnabled && m_bloomThresholdShader && m_bloomBlurShader
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
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(raytraceTex->getNativeHandle()));
        m_bloomThresholdShader->setInt("u_hdrMap", 0);
        m_bloomThresholdShader->setFloat("u_threshold",   shared.bloomThreshold);
        m_bloomThresholdShader->setFloat("u_sampleCount", 1.0f);
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
    if (rtShader && raytraceTex)
    {
        rtShader->bind();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(raytraceTex->getNativeHandle()));
        rtShader->setInt("u_accumMap", 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(shared.outlineMaskFB->getColorAttachmentHandle()));
        rtShader->setInt("u_outlineMask", 1);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, bloomTex != 0
            ? bloomTex
            : static_cast<GLuint>(m_whiteTexture->getNativeHandle()));
        rtShader->setInt("u_bloomMap", 2);
        rtShader->setFloat("u_sampleCount",    1.0f);
        rtShader->setFloat("u_exposure",       0.0f);
        rtShader->setFloat("u_gamma",          1.0f);
        rtShader->setBool("u_enableACES",      false);
        rtShader->setBool("u_flipV",           true);
        rtShader->setBool("u_enableOutline",   shared.outlineActive);
        rtShader->setBool("u_enableBloom",     bloomTex != 0);
        rtShader->setFloat("u_bloomIntensity", shared.bloomIntensity);
        m_fullscreenQuad->draw();
        rtShader->unbind();
    }

    glEnable(GL_DEPTH_TEST);
    shared.outputFB->unbind();
#endif

#ifdef VEX_BACKEND_VULKAN
    if (m_fullscreenRTShader && raytraceTex)
    {
        auto* rtShaderVK = static_cast<vex::VKShader*>(m_fullscreenRTShader);
        auto* vkTex      = static_cast<vex::VKTexture2D*>(raytraceTex);
        auto* vkMaskFB   = static_cast<vex::VKFramebuffer*>(shared.outlineMaskFB);

        // Bloom pass
        VkImageView bloomView    = VK_NULL_HANDLE;
        VkSampler   bloomSampler = VK_NULL_HANDLE;
        bool bloomActive = shared.bloomEnabled
                           && m_bloomThresholdShader && m_bloomBlurShader
                           && m_bloomFB[0] && m_bloomFB[1];
        if (bloomActive)
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

            auto* vkThreshVK = static_cast<vex::VKShader*>(m_bloomThresholdShader);
            m_bloomFB[0]->bind();
            m_bloomFB[0]->clear(0.0f, 0.0f, 0.0f, 1.0f);
            m_bloomThresholdShader->setFloat("u_threshold",   shared.bloomThreshold);
            m_bloomThresholdShader->setFloat("u_sampleCount", 1.0f);
            m_bloomThresholdShader->bind();
            vkThreshVK->setExternalTextureVK(0,
                vkTex->getImageView(), vkTex->getSampler(),
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
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
            bloomView    = vkBloom0->getColorImageView();
            bloomSampler = vkBloom0->getColorSampler();
        }

        // Display
        shared.outputFB->bind();
        shared.outputFB->clear(0.0f, 0.0f, 0.0f, 1.0f);

        m_fullscreenRTShader->setFloat("u_sampleCount",    1.0f);
        m_fullscreenRTShader->setFloat("u_exposure",       0.0f);
        m_fullscreenRTShader->setFloat("u_gamma",          1.0f);
        m_fullscreenRTShader->setFloat("u_bloomIntensity", shared.bloomIntensity);
        m_fullscreenRTShader->bind();
        m_fullscreenRTShader->setBool("u_enableACES",    false);
        m_fullscreenRTShader->setBool("u_flipV",         true);
        m_fullscreenRTShader->setBool("u_enableOutline", shared.outlineActive);
        m_fullscreenRTShader->setBool("u_enableBloom",   bloomActive);

        rtShaderVK->setExternalTextureVK(0,
            vkTex->getImageView(), vkTex->getSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        rtShaderVK->setExternalTextureVK(1,
            vkMaskFB->getColorImageView(), vkMaskFB->getColorSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        rtShaderVK->setExternalTextureVK(2,
            bloomActive ? bloomView    : vkMaskFB->getColorImageView(),
            bloomActive ? bloomSampler : vkMaskFB->getColorSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        m_fullscreenQuad->draw();
        m_fullscreenRTShader->unbind();
        shared.outputFB->unbind();
    }
#endif

    if (shared.drawCalls) *shared.drawCalls = 1;
}
