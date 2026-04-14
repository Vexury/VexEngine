#include "render_mode_rasterize.h"
#include "scene.h"

#include <vex/graphics/shader.h>
#include <vex/graphics/framebuffer.h>
#include <vex/graphics/mesh.h>
#include <vex/graphics/texture.h>
#include <vex/raytracing/bvh.h>
#include <vex/core/log.h>

#ifdef VEX_BACKEND_OPENGL
#include <glad/glad.h>
#include <vex/opengl/gl_framebuffer.h>
#endif

#ifdef VEX_BACKEND_VULKAN
#include <vex/vulkan/vk_context.h>
#include <vex/vulkan/vk_shader.h>
#include <vex/vulkan/vk_framebuffer.h>
#include <vex/vulkan/vk_texture.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <stb_image_write.h>

#ifdef VEX_BACKEND_OPENGL
#include <glad/glad.h>
#endif

#include <algorithm>
#include <limits>

// ---------------------------------------------------------------------------
// init / shutdown
// ---------------------------------------------------------------------------

bool RasterizeMode::init(const RenderModeInitData& init)
{
    m_fullscreenQuad       = init.fullscreenQuad;
    m_whiteTexture         = init.whiteTexture;
    m_flatNormalTexture    = init.flatNormalTexture;
    m_meshShader           = init.meshShader;
    m_fullscreenRTShader   = init.fullscreenRTShader;
    m_bloomFB[0]           = init.bloomFB[0];
    m_bloomFB[1]           = init.bloomFB[1];
    m_bloomThresholdShader = init.bloomThresholdShader;
    m_bloomBlurShader      = init.bloomBlurShader;
    m_geomCache            = init.geomCache;

    std::string dir = vex::Shader::shaderDir();
    std::string ext = vex::Shader::shaderExt();

#ifdef VEX_BACKEND_OPENGL
    m_rasterHDRFB = vex::Framebuffer::create({ .width = 1280, .height = 720, .hasDepth = true });

    m_pickShader = vex::Shader::create();
    if (!m_pickShader->loadFromFiles(dir + "pick.vert" + ext, dir + "pick.frag" + ext))
        return false;
    m_pickFB = vex::Framebuffer::create({ .width = 1280, .height = 720, .hasDepth = true });
#endif

#ifdef VEX_BACKEND_VULKAN
    m_rasterHDRFB = vex::Framebuffer::create({ .width = 1280, .height = 720, .hasDepth = true, .hdrColor = true });
#endif

    return true;
}

void RasterizeMode::lateInitVK([[maybe_unused]] const RenderModeInitData& init)
{
    // Called after the main framebuffer and mesh shader are ready (VK only).
    // The mesh shader pipeline for the HDR FB must be prepared here because
    // SceneRenderer does it on the HDR FB before calling our init().
    // Nothing needed here because SceneRenderer::init() calls
    //   m_meshShader->preparePipeline(*m_rasterHDRFB);
    // using SceneRenderer's own m_rasterHDRFB.
    // After the move, SceneRenderer no longer owns m_rasterHDRFB, so we need
    // to re-prepare using our own m_rasterHDRFB.
    //
    // However the wiring is: SceneRenderer passes m_rasterHDRFB to us during init.
    // We keep our own m_rasterHDRFB so we do the pipeline prepare here.
#ifdef VEX_BACKEND_VULKAN
    if (m_meshShader && m_rasterHDRFB)
        m_meshShader->preparePipeline(*m_rasterHDRFB);
#endif
}

void RasterizeMode::shutdown()
{
#ifdef VEX_BACKEND_OPENGL
    if (m_rasterEnvMapTex) { glDeleteTextures(1, &m_rasterEnvMapTex); m_rasterEnvMapTex = 0; }
    m_pickShader.reset();
    m_pickFB.reset();
#endif
#ifdef VEX_BACKEND_VULKAN
    m_vkRasterEnvTex.reset();
#endif
    m_rasterHDRFB.reset();
}

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------

void RasterizeMode::render(Scene& scene,
                           const SharedRenderData& shared,
                           const FrameChanges& /*changes*/)
{
    renderWithSelection(scene, shared, shared.selectedNodeIdx, shared.selectedSubmesh);
}

void RasterizeMode::renderWithSelection(Scene& scene, const SharedRenderData& shared,
                                        [[maybe_unused]] int selectedNodeIdx,
                                        [[maybe_unused]] int selectedSubmesh)
{
    // Keep the intermediate HDR framebuffer in sync with the output framebuffer size
    bool hdrFBResized = false;
    {
        const auto& outSpec = shared.outputFB->getSpec();
        const auto& hdrSpec = m_rasterHDRFB->getSpec();
        if (hdrSpec.width != outSpec.width || hdrSpec.height != outSpec.height)
        {
            m_rasterHDRFB->resize(outSpec.width, outSpec.height);
            hdrFBResized = true;
#ifdef VEX_BACKEND_VULKAN
            // Wait for all in-flight frames to complete before recreating pipelines:
            // beginFrame() only waits for frame N%2, so frame N-1 may still be executing.
            // vkDestroyPipeline on an in-flight pipeline triggers VUID-vkDestroyPipeline-00765.
            vkDeviceWaitIdle(vex::VKContext::get().getDevice());
            if (m_meshShader)
                m_meshShader->preparePipeline(*m_rasterHDRFB);
            if (scene.skybox)
                scene.skybox->preparePipeline(*m_rasterHDRFB);
#endif
        }
    }
    vex::Framebuffer* renderFB = m_rasterHDRFB.get();
    const bool isDebugView = (shared.debugMode != 0); // DebugMode::None == 0

    // Shadow map rendered by SceneRenderer's shared pre-pass; just read its results.
    const glm::mat4 lightVP         = shared.shadowLightVP;
    const float     shadowNormalBias = shared.shadowNormalBias;

    renderFB->bind();

#ifdef VEX_BACKEND_VULKAN
    if (hdrFBResized && m_fullscreenRTShader)
        static_cast<vex::VKShader*>(m_fullscreenRTShader)->clearExternalTextureCache();
#endif

    bool useSolidColor = (scene.currentEnvmap == Scene::SolidColor);

    if (useSolidColor)
        renderFB->clear(scene.skyboxColor.r, scene.skyboxColor.g, scene.skyboxColor.b, 1.0f);
    else
        renderFB->clear(0.0f, 0.0f, 0.0f, 1.0f);

    float aspect = static_cast<float>(shared.outputFB->getSpec().width)
                 / static_cast<float>(shared.outputFB->getSpec().height);

    glm::mat4 view = scene.camera.getViewMatrix();
    glm::mat4 proj = scene.camera.getProjectionMatrix(aspect);

    if (shared.drawCalls) *shared.drawCalls = 0;

    if (scene.showSkybox && scene.skybox && !useSolidColor)
    {
        scene.skybox->setEnvRotation(scene.envRotation);
        scene.skybox->draw(glm::inverse(proj * view));
        if (shared.drawCalls) ++(*shared.drawCalls);
    }

    [[maybe_unused]] bool hasSelection = (selectedNodeIdx >= 0
                                          && selectedNodeIdx < static_cast<int>(scene.nodes.size()));

    // --- Main mesh pass ---
    vex::Shader* meshShader = m_meshShader;
    meshShader->bind();
    meshShader->setMat4("u_view",       view);
    meshShader->setMat4("u_projection", proj);
    meshShader->setVec3("u_cameraPos", scene.camera.getPosition());
    meshShader->setVec3("u_lightPos", scene.lightPos);
    meshShader->setVec3("u_lightColor", scene.showLight ? scene.lightColor * scene.lightIntensity : glm::vec3(0.0f));
    meshShader->setVec3("u_sunDirection", scene.getSunDirection());
    meshShader->setVec3("u_sunColor", scene.showSun ? scene.sunColor * scene.sunIntensity : glm::vec3(0.0f));

    meshShader->setInt("u_debugMode", shared.debugMode);
    meshShader->setFloat("u_nearPlane", scene.camera.nearPlane);
    meshShader->setFloat("u_farPlane", scene.camera.farPlane);

    if (shared.debugMode == 1) // DebugMode::Wireframe
        meshShader->setWireframe(true);

#ifdef VEX_BACKEND_OPENGL
    {
        bool hasEnvMap = (shared.rasterEnvMapTex != 0);
        glActiveTexture(GL_TEXTURE5);
        if (hasEnvMap)
            glBindTexture(GL_TEXTURE_2D, shared.rasterEnvMapTex);
        else
            glBindTexture(GL_TEXTURE_2D, 0);
        meshShader->setInt("u_envMap", 5);
        meshShader->setBool("u_hasEnvMap", hasEnvMap);
        meshShader->setBool("u_enableEnvLighting", m_rasterEnableEnvLighting);
        glm::vec3 envCol = useSolidColor ? scene.skyboxColor : shared.rasterEnvColor;
        meshShader->setVec3("u_envColor", envCol);
        meshShader->setFloat("u_envLightMultiplier", m_rasterEnvLightMultiplier);
        meshShader->setFloat("u_envRotation", scene.envRotation);

        if (shared.shadowFB && shared.shadowEverRendered)
        {
            auto* glShadowFB = static_cast<vex::GLFramebuffer*>(shared.shadowFB);
            glActiveTexture(GL_TEXTURE6);
            glBindTexture(GL_TEXTURE_2D, glShadowFB->getDepthAttachment());
            meshShader->setInt("u_shadowMap", 6);
            meshShader->setMat4("u_shadowViewProj", lightVP);
            meshShader->setBool("u_enableShadows", scene.showSun && m_rasterEnableShadows);
            meshShader->setFloat("u_shadowNormalBias", shadowNormalBias);
            meshShader->setFloat("u_shadowStrength", m_shadowStrength);
            meshShader->setVec3("u_shadowColor", m_shadowColor);
        }
    }
#endif

#ifdef VEX_BACKEND_VULKAN
    {
        bool hasEnvMap = (shared.vkRasterEnvTex != nullptr);
        glm::vec3 envCol = useSolidColor ? scene.skyboxColor : shared.rasterEnvColor;
        meshShader->setVec3("u_envColor", envCol);
        meshShader->setFloat("u_envLightMultiplier", m_rasterEnvLightMultiplier);
        meshShader->setBool("u_enableEnvLighting", m_rasterEnableEnvLighting);
        meshShader->setBool("u_hasEnvMap", hasEnvMap && m_rasterEnableEnvLighting);
        meshShader->setFloat("u_envRotation", scene.envRotation);
        meshShader->setTexture(5, hasEnvMap ? shared.vkRasterEnvTex : m_whiteTexture);

        meshShader->setMat4("u_shadowViewProj", lightVP);
        meshShader->setBool("u_enableShadows", scene.showSun && m_rasterEnableShadows);
        meshShader->setFloat("u_shadowNormalBias", shadowNormalBias);
        meshShader->setFloat("u_shadowStrength", m_shadowStrength);
        meshShader->setVec3("u_shadowColor", m_shadowColor);
        if (shared.shadowFB && shared.shadowEverRendered)
        {
            auto* vkShadowFB   = static_cast<vex::VKFramebuffer*>(shared.shadowFB);
            auto* vkMeshShader = static_cast<vex::VKShader*>(meshShader);
            vkMeshShader->setExternalTextureVK(6,
                vkShadowFB->getDepthImageView(),
                vkShadowFB->getDepthCompSampler(),
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        }
    }
#endif

    for (int ni = 0; ni < static_cast<int>(scene.nodes.size()); ++ni)
    {
        const glm::mat4 nodeWorld = scene.getWorldMatrix(ni);
        for (int si = 0; si < static_cast<int>(scene.nodes[ni].submeshes.size()); ++si)
        {
            auto& sm = scene.nodes[ni].submeshes[si];
            meshShader->setMat4("u_model", nodeWorld * sm.modelMatrix);

#ifdef VEX_BACKEND_OPENGL
            bool isSelectedNode = hasSelection && ni == selectedNodeIdx;
            bool writeStencil = isSelectedNode && (selectedSubmesh < 0 || si == selectedSubmesh);
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
                : m_whiteTexture;
            meshShader->setTexture(0, tex);

            bool hasNorm = shared.enableNormalMapping && sm.normalTexture != nullptr;
            vex::Texture2D* normTex = hasNorm
                ? sm.normalTexture.get()
                : m_flatNormalTexture;
            meshShader->setTexture(1, normTex);
            meshShader->setBool("u_hasNormalMap", hasNorm);

            bool hasRoughMap = sm.roughnessTexture != nullptr;
            meshShader->setTexture(2, hasRoughMap ? sm.roughnessTexture.get() : m_whiteTexture);
            meshShader->setBool("u_hasRoughnessMap", hasRoughMap);

            bool hasMetalMap = sm.metallicTexture != nullptr;
            meshShader->setTexture(3, hasMetalMap ? sm.metallicTexture.get() : m_whiteTexture);
            meshShader->setBool("u_hasMetallicMap", hasMetalMap);

            bool hasEmissive = sm.emissiveTexture != nullptr;
            meshShader->setTexture(4, hasEmissive ? sm.emissiveTexture.get() : m_whiteTexture);
            meshShader->setBool("u_hasEmissiveMap", hasEmissive);

            bool hasAO = sm.aoTexture != nullptr;
            meshShader->setTexture(7, hasAO ? sm.aoTexture.get() : m_whiteTexture);
            meshShader->setBool("u_hasAOMap", hasAO);

            bool hasAlpha = sm.alphaTexture != nullptr;
            meshShader->setTexture(8, hasAlpha ? sm.alphaTexture.get() : m_whiteTexture);
            meshShader->setBool("u_hasAlphaMap", hasAlpha);

            meshShader->setVec3("u_baseColor", sm.meshData.baseColor);
            meshShader->setVec3("u_emissiveColor", sm.meshData.emissiveColor);
            meshShader->setFloat("u_emissiveStrength", sm.meshData.emissiveStrength);
            meshShader->setInt("u_materialType", sm.meshData.materialType);
            meshShader->setFloat("u_roughness", sm.meshData.roughness);
            meshShader->setFloat("u_metallic", sm.meshData.metallic);
            meshShader->setBool("u_alphaClip", sm.meshData.alphaClip);
            sm.mesh->draw();
            if (shared.drawCalls) ++(*shared.drawCalls);

#ifdef VEX_BACKEND_OPENGL
            if (writeStencil)
            {
                glStencilMask(0x00);
                glDisable(GL_STENCIL_TEST);
            }
#endif
        }
    }

    if (shared.debugMode == 1) // DebugMode::Wireframe
        meshShader->setWireframe(false);

    meshShader->unbind();
    renderFB->unbind();

#ifdef VEX_BACKEND_OPENGL
    uint32_t bloomTex = 0;
    if (!isDebugView && shared.bloomEnabled && m_bloomThresholdShader && m_bloomBlurShader
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
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(m_rasterHDRFB->getColorAttachmentHandle()));
        m_bloomThresholdShader->setInt("u_hdrMap", 0);
        m_bloomThresholdShader->setFloat("u_threshold", shared.bloomThreshold);
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
            glBindTexture(GL_TEXTURE_2D,
                static_cast<GLuint>(m_bloomFB[src]->getColorAttachmentHandle()));
            m_bloomBlurShader->setInt("u_image", 0);
            m_bloomBlurShader->setBool("u_horizontal", horizontal);
            m_fullscreenQuad->draw();
            m_bloomBlurShader->unbind();
            m_bloomFB[dst]->unbind();
            horizontal = !horizontal;
        }
        bloomTex = static_cast<GLuint>(m_bloomFB[0]->getColorAttachmentHandle());
    }

    // Tone-map blit: HDR intermediate buffer -> output framebuffer
    vex::Shader* rtShader = m_fullscreenRTShader;
    if (rtShader)
    {
        shared.outputFB->bind();
        shared.outputFB->clear(0.0f, 0.0f, 0.0f, 1.0f);

        glDisable(GL_DEPTH_TEST);

        rtShader->bind();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(m_rasterHDRFB->getColorAttachmentHandle()));
        rtShader->setInt("u_accumMap", 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(shared.outlineMaskFB->getColorAttachmentHandle()));
        rtShader->setInt("u_outlineMask", 1);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, bloomTex != 0
            ? bloomTex
            : static_cast<GLuint>(m_whiteTexture->getNativeHandle()));
        rtShader->setInt("u_bloomMap", 2);
        rtShader->setFloat("u_sampleCount", 1.0f);
        rtShader->setFloat("u_exposure",    isDebugView ? 0.0f : m_rasterExposure);
        rtShader->setFloat("u_gamma",       isDebugView ? 1.0f : m_rasterGamma);
        rtShader->setBool("u_enableACES",   !isDebugView && m_rasterEnableACES);
        rtShader->setBool("u_flipV", false);
        rtShader->setBool("u_enableOutline", shared.outlineActive);
        rtShader->setBool("u_enableBloom",  !isDebugView && bloomTex != 0);
        rtShader->setFloat("u_bloomIntensity", shared.bloomIntensity);
        m_fullscreenQuad->draw();
        rtShader->unbind();

        glEnable(GL_DEPTH_TEST);
        shared.outputFB->unbind();
    }
#endif

#ifdef VEX_BACKEND_VULKAN
    VkImageView vkBloomView = VK_NULL_HANDLE;
    VkSampler   vkBloomSampler = VK_NULL_HANDLE;
    bool vkBloomActive = !isDebugView && shared.bloomEnabled
                         && m_bloomThresholdShader && m_bloomBlurShader
                         && m_bloomFB[0] && m_bloomFB[1];
    if (vkBloomActive)
    {
        const auto& outSpec = shared.outputFB->getSpec();
        uint32_t bw = std::max(1u, outSpec.width / 2);
        uint32_t bh = std::max(1u, outSpec.height / 2);
        bool needResize = (m_bloomFB[0]->getSpec().width != bw
                        || m_bloomFB[0]->getSpec().height != bh);
        if (needResize)
        {
            vkDeviceWaitIdle(vex::VKContext::get().getDevice());
            m_bloomFB[0]->resize(bw, bh);
            m_bloomFB[1]->resize(bw, bh);
            m_bloomThresholdShader->preparePipeline(*m_bloomFB[0]);
            m_bloomBlurShader->preparePipeline(*m_bloomFB[0]);
            static_cast<vex::VKShader*>(m_bloomThresholdShader)->clearExternalTextureCache();
            static_cast<vex::VKShader*>(m_bloomBlurShader)->clearExternalTextureCache();
        }

        auto* vkHDRFB2   = static_cast<vex::VKFramebuffer*>(m_rasterHDRFB.get());
        auto* vkThreshVK = static_cast<vex::VKShader*>(m_bloomThresholdShader);

        m_bloomFB[0]->bind();
        m_bloomFB[0]->clear(0.0f, 0.0f, 0.0f, 1.0f);
        m_bloomThresholdShader->setFloat("u_threshold", shared.bloomThreshold);
        m_bloomThresholdShader->setFloat("u_sampleCount", 1.0f);
        m_bloomThresholdShader->bind();
        vkThreshVK->setExternalTextureVK(0,
            vkHDRFB2->getColorImageView(),
            vkHDRFB2->getColorSampler(),
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
        vkBloomView   = vkBloom0->getColorImageView();
        vkBloomSampler = vkBloom0->getColorSampler();
    }

    if (m_fullscreenRTShader)
    {
        auto* vkHDRFB    = static_cast<vex::VKFramebuffer*>(m_rasterHDRFB.get());
        auto* rtShaderVK = static_cast<vex::VKShader*>(m_fullscreenRTShader);

        shared.outputFB->bind();
        shared.outputFB->clear(0.0f, 0.0f, 0.0f, 1.0f);

        m_fullscreenRTShader->setFloat("u_sampleCount",    1.0f);
        m_fullscreenRTShader->setFloat("u_exposure",       isDebugView ? 0.0f : m_rasterExposure);
        m_fullscreenRTShader->setFloat("u_gamma",          isDebugView ? 1.0f : m_rasterGamma);
        m_fullscreenRTShader->setFloat("u_bloomIntensity", shared.bloomIntensity);
        m_fullscreenRTShader->bind();
        m_fullscreenRTShader->setBool("u_enableACES",    !isDebugView && m_rasterEnableACES);
        m_fullscreenRTShader->setBool("u_flipV",         true);

        rtShaderVK->setExternalTextureVK(0,
            vkHDRFB->getColorImageView(),
            vkHDRFB->getColorSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        auto* vkMaskFB = static_cast<vex::VKFramebuffer*>(shared.outlineMaskFB);
        rtShaderVK->setExternalTextureVK(1,
            vkMaskFB->getColorImageView(),
            vkMaskFB->getColorSampler(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        if (vkBloomActive)
            rtShaderVK->setExternalTextureVK(2, vkBloomView, vkBloomSampler,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        else
            rtShaderVK->setExternalTextureVK(2, vkMaskFB->getColorImageView(),
                vkMaskFB->getColorSampler(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        m_fullscreenRTShader->setBool("u_enableOutline", shared.outlineActive);
        m_fullscreenRTShader->setBool("u_enableBloom",   vkBloomActive);

        m_fullscreenQuad->draw();
        m_fullscreenRTShader->unbind();

        shared.outputFB->unbind();
    }
#endif
}

// ---------------------------------------------------------------------------
// Picking (GL only)
// ---------------------------------------------------------------------------

std::pair<int,int> RasterizeMode::pick(Scene& scene, const SharedRenderData& shared,
                                        int pixelX, int pixelY)
{
#ifdef VEX_BACKEND_OPENGL
    if (!m_pickShader || !m_pickFB)
        return {-1, -1};

    const auto& mainSpec = shared.outputFB->getSpec();
    const auto& pickSpec = m_pickFB->getSpec();
    if (pickSpec.width != mainSpec.width || pickSpec.height != mainSpec.height)
        m_pickFB->resize(mainSpec.width, mainSpec.height);

    m_pickFB->bind();
    m_pickFB->clear(0.0f, 0.0f, 0.0f, 1.0f);

    float aspect = static_cast<float>(mainSpec.width) / static_cast<float>(mainSpec.height);
    glm::mat4 view = scene.camera.getViewMatrix();
    glm::mat4 proj = scene.camera.getProjectionMatrix(aspect);

    m_pickShader->bind();
    m_pickShader->setMat4("u_view", view);
    m_pickShader->setMat4("u_projection", proj);

    std::vector<std::pair<int,int>> drawToMesh;
    for (int ni = 0; ni < static_cast<int>(scene.nodes.size()); ++ni)
    {
        const glm::mat4 nodeWorld = scene.getWorldMatrix(ni);
        for (int si = 0; si < static_cast<int>(scene.nodes[ni].submeshes.size()); ++si)
        {
            auto& sm = scene.nodes[ni].submeshes[si];
            m_pickShader->setMat4("u_model", nodeWorld * sm.modelMatrix);
            int drawIdx = static_cast<int>(drawToMesh.size());
            m_pickShader->setInt("u_objectID", drawIdx);
            vex::Texture2D* tex = sm.diffuseTexture
                ? sm.diffuseTexture.get()
                : m_whiteTexture;
            m_pickShader->setTexture(0, tex);
            m_pickShader->setBool("u_alphaClip", sm.meshData.alphaClip);
            sm.mesh->draw();
            drawToMesh.push_back({ni, si});
        }
    }
    m_pickShader->unbind();

    int objectID = m_pickFB->readPixel(pixelX, pixelY) - 1;
    m_pickFB->unbind();

    if (objectID >= 0 && objectID < static_cast<int>(drawToMesh.size()))
        return drawToMesh[objectID];
    return {-1, -1};
#else
    (void)scene; (void)shared; (void)pixelX; (void)pixelY;
    return {-1, -1};
#endif
}

