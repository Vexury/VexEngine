#include "scene_renderer.h"
#include "scene.h"

#include <vex/graphics/mesh.h>
#include <vex/graphics/skybox.h>
#include <vex/scene/mesh_data.h>
#include <vex/core/log.h>

#include <stb_image.h>
#include <stb_image_write.h>

#ifdef VEX_BACKEND_OPENGL
#include <glad/glad.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdio>
#include <limits>
#include <unordered_map>
#include <vector>

static constexpr float OUTLINE_WIDTH = 0.012f;
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

    m_outlineShader = vex::Shader::create();
    if (!m_outlineShader->loadFromFiles(dir + "outline.vert" + ext, dir + "outline.frag" + ext))
        return false;
#endif

    // Fullscreen shader and quad for raytracing display
    m_fullscreenShader = vex::Shader::create();
    if (!m_fullscreenShader->loadFromFiles(dir + "fullscreen.vert" + ext, dir + "fullscreen.frag" + ext))
        return false;

    m_fullscreenQuad = vex::Mesh::create();
    m_fullscreenQuad->upload(buildFullscreenQuadData());

    m_framebuffer = vex::Framebuffer::create({ .width = 1280, .height = 720, .hasDepth = true });
#ifdef VEX_BACKEND_OPENGL
    m_pickFB = vex::Framebuffer::create({ .width = 1280, .height = 720, .hasDepth = true });
#endif

    // Create backend-specific pipelines for the offscreen framebuffer
    m_meshShader->preparePipeline(*m_framebuffer);
    m_fullscreenShader->preparePipeline(*m_framebuffer);
    if (scene.skybox)
        scene.skybox->preparePipeline(*m_framebuffer);

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

    return true;
}

void SceneRenderer::shutdown()
{
#ifdef VEX_BACKEND_OPENGL
    if (m_rasterEnvMapTex) { glDeleteTextures(1, &m_rasterEnvMapTex); m_rasterEnvMapTex = 0; }
    m_rasterHDRFB.reset();
    if (m_gpuRaytracer)
        m_gpuRaytracer->shutdown();
    m_gpuRaytracer.reset();
    m_fullscreenRTShader.reset();
#endif
    m_cpuRaytracer.reset();
    m_raytraceTexture.reset();
    m_fullscreenQuad.reset();
    m_fullscreenShader.reset();
    m_meshShader.reset();
    m_whiteTexture.reset();
    m_flatNormalTexture.reset();
    m_outlineShader.reset();
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

    if (mode == RenderMode::CPURaytrace && m_cpuRaytracer)
        m_cpuRaytracer->reset();

#ifdef VEX_BACKEND_OPENGL
    if (mode == RenderMode::GPURaytrace && m_gpuRaytracer)
    {
        m_gpuRaytracer->reset();
        m_gpuGeometryDirty = true; // force re-upload
    }
#endif
}

uint32_t SceneRenderer::getRaytraceSampleCount() const
{
#ifdef VEX_BACKEND_OPENGL
    if (m_renderMode == RenderMode::GPURaytrace && m_gpuRaytracer)
        return m_gpuRaytracer->getSampleCount();
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
}

int SceneRenderer::getGPUMaxDepth() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getMaxDepth() : 5;
#else
    return 5;
#endif
}

void SceneRenderer::setGPUEnableNEE([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setEnableNEE(v);
#endif
}

bool SceneRenderer::getGPUEnableNEE() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getEnableNEE() : true;
#else
    return true;
#endif
}

void SceneRenderer::setGPUEnableAA([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setEnableAA(v);
#endif
}

bool SceneRenderer::getGPUEnableAA() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getEnableAA() : true;
#else
    return true;
#endif
}

void SceneRenderer::setGPUEnableFireflyClamping([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setEnableFireflyClamping(v);
#endif
}

bool SceneRenderer::getGPUEnableFireflyClamping() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getEnableFireflyClamping() : true;
#else
    return true;
#endif
}

void SceneRenderer::setGPUEnableEnvironment([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setEnableEnvironment(v);
#endif
}

bool SceneRenderer::getGPUEnableEnvironment() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getEnableEnvironment() : false;
#else
    return false;
#endif
}

void SceneRenderer::setGPUEnvLightMultiplier([[maybe_unused]] float v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setEnvLightMultiplier(v);
#endif
}

float SceneRenderer::getGPUEnvLightMultiplier() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getEnvLightMultiplier() : 1.0f;
#else
    return 1.0f;
#endif
}

void  SceneRenderer::setRasterEnableEnvLighting(bool v)   { m_rasterEnableEnvLighting = v; }
bool  SceneRenderer::getRasterEnableEnvLighting() const    { return m_rasterEnableEnvLighting; }
void  SceneRenderer::setRasterEnvLightMultiplier(float v)  { m_rasterEnvLightMultiplier = v; }
float SceneRenderer::getRasterEnvLightMultiplier() const   { return m_rasterEnvLightMultiplier; }

void  SceneRenderer::setRasterExposure([[maybe_unused]] float v)
{
#ifdef VEX_BACKEND_OPENGL
    m_rasterExposure = v;
#endif
}
float SceneRenderer::getRasterExposure() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_rasterExposure;
#else
    return 0.0f;
#endif
}

void  SceneRenderer::setRasterGamma([[maybe_unused]] float v)
{
#ifdef VEX_BACKEND_OPENGL
    m_rasterGamma = v;
#endif
}
float SceneRenderer::getRasterGamma() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_rasterGamma;
#else
    return 2.2f;
#endif
}

void  SceneRenderer::setRasterEnableACES([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    m_rasterEnableACES = v;
#endif
}
bool  SceneRenderer::getRasterEnableACES() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_rasterEnableACES;
#else
    return true;
#endif
}

void SceneRenderer::setGPUFlatShading([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setFlatShading(v);
#endif
}

bool SceneRenderer::getGPUFlatShading() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getFlatShading() : false;
#else
    return false;
#endif
}

void SceneRenderer::setGPUEnableNormalMapping([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setEnableNormalMapping(v);
#endif
}

bool SceneRenderer::getGPUEnableNormalMapping() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getEnableNormalMapping() : true;
#else
    return true;
#endif
}

void SceneRenderer::setGPUEnableEmissive([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setEnableEmissive(v);
#endif
}

bool SceneRenderer::getGPUEnableEmissive() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getEnableEmissive() : true;
#else
    return true;
#endif
}

void SceneRenderer::setGPUExposure([[maybe_unused]] float v)
{
#ifdef VEX_BACKEND_OPENGL
    m_gpuExposure = v;
#endif
}

float SceneRenderer::getGPUExposure() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuExposure;
#else
    return 0.0f;
#endif
}

void SceneRenderer::setGPUGamma([[maybe_unused]] float v)
{
#ifdef VEX_BACKEND_OPENGL
    m_gpuGamma = v;
#endif
}

float SceneRenderer::getGPUGamma() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuGamma;
#else
    return 2.2f;
#endif
}

void SceneRenderer::setGPUEnableACES([[maybe_unused]] bool v)
{
#ifdef VEX_BACKEND_OPENGL
    m_gpuEnableACES = v;
#endif
}

bool SceneRenderer::getGPUEnableACES() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuEnableACES;
#else
    return true;
#endif
}

void SceneRenderer::setGPURayEps([[maybe_unused]] float v)
{
#ifdef VEX_BACKEND_OPENGL
    if (m_gpuRaytracer) m_gpuRaytracer->setRayEps(v);
#endif
}

float SceneRenderer::getGPURayEps() const
{
#ifdef VEX_BACKEND_OPENGL
    return m_gpuRaytracer ? m_gpuRaytracer->getRayEps() : 1e-4f;
#else
    return 1e-4f;
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

void SceneRenderer::rebuildRaytraceGeometry(Scene& scene)
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
        int tw, th, tch;
        stbi_set_flip_vertically_on_load(false);
        unsigned char* texData = stbi_load(path.c_str(), &tw, &th, &tch, 4);
        int idx = -1;
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

    for (const auto& group : scene.meshGroups)
    {
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

                glm::vec3 edge1 = v1.position - v0.position;
                glm::vec3 edge2 = v2.position - v0.position;
                glm::vec3 cross = glm::cross(edge1, edge2);
                float len = glm::length(cross);

                vex::CPURaytracer::Triangle tri;
                tri.v0 = v0.position;
                tri.v1 = v1.position;
                tri.v2 = v2.position;
                tri.n0 = v0.normal;
                tri.n1 = v1.normal;
                tri.n2 = v2.normal;
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

    // Store copies for GPU upload before moving to CPU raytracer
    m_rtTextures = textures; // copy
    m_cpuRaytracer->setGeometry(std::move(triangles), std::move(textures));

    // Build BVH and reorder triangles for GPU (mirrors CPU raytracer internal logic)
    {
        // Re-extract triangles from scene (CPU raytracer consumed them)
        // Actually re-use the CPU raytracer approach: build our own BVH
        std::vector<vex::CPURaytracer::Triangle> gpuTriangles;
        std::vector<std::pair<int,int>> gpuTriangleSrc; // {gi, si} per triangle
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

                    gpuTriangles.push_back(tri);
                    gpuTriangleSrc.push_back({gi, si});
                }
            }
        }

        // Build BVH
        uint32_t count = static_cast<uint32_t>(gpuTriangles.size());
        std::vector<vex::AABB> triBounds(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            triBounds[i].grow(gpuTriangles[i].v0);
            triBounds[i].grow(gpuTriangles[i].v1);
            triBounds[i].grow(gpuTriangles[i].v2);
        }
        m_rtBVH.build(triBounds);

        // Reorder triangles (and source tracking) to match BVH index order
        const auto& bvhIndices = m_rtBVH.indices();
        std::vector<vex::CPURaytracer::Triangle> reordered(count);
        std::vector<std::pair<int,int>> reorderedSrc(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            reordered[i]    = gpuTriangles[bvhIndices[i]];
            reorderedSrc[i] = gpuTriangleSrc[bvhIndices[i]];
        }
        m_rtTriangles           = std::move(reordered);
        m_rtTriangleSrcSubmesh  = std::move(reorderedSrc);

        // Build light data
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
        {
            for (float& c : m_rtLightCDF)
                c /= m_rtTotalLightArea;
        }
    }

    char sahBuf[32];
    std::snprintf(sahBuf, sizeof(sahBuf), "%.1f", m_cpuRaytracer->getBVHSAHCost());
    vex::Log::info("  BVH built: " + std::to_string(m_cpuRaytracer->getBVHNodeCount()) + " nodes, "
                  + std::to_string(m_rtTriangles.size()) + " triangles, SAH cost " + sahBuf);
    if (!m_rtLightIndices.empty())
        vex::Log::info("  " + std::to_string(m_rtLightIndices.size()) + " emissive triangle(s)");

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
}

void SceneRenderer::renderScene(Scene& scene, int selectedGroup, int selectedSubmesh)
{
    // Full geometry rebuild (new mesh loaded, etc.)
    if (scene.geometryDirty)
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

    switch (m_renderMode)
    {
        case RenderMode::CPURaytrace:
            renderCPURaytrace(scene);
            break;
        case RenderMode::GPURaytrace:
#ifdef VEX_BACKEND_OPENGL
            renderGPURaytrace(scene);
            break;
#endif
            // Fall through to rasterize if not OpenGL
        case RenderMode::Rasterize:
            renderRasterize(scene, selectedGroup, selectedSubmesh);
            break;
    }
}

void SceneRenderer::renderRasterize(Scene& scene, int selectedGroup, int selectedSubmesh)
{
#ifdef VEX_BACKEND_OPENGL
    // Keep the intermediate HDR framebuffer in sync with the output framebuffer size
    {
        const auto& outSpec = m_framebuffer->getSpec();
        const auto& hdrSpec = m_rasterHDRFB->getSpec();
        if (hdrSpec.width != outSpec.width || hdrSpec.height != outSpec.height)
            m_rasterHDRFB->resize(outSpec.width, outSpec.height);
    }
    vex::Framebuffer* renderFB = m_rasterHDRFB.get();
#else
    vex::Framebuffer* renderFB = m_framebuffer.get();
#endif

    renderFB->bind();

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
    m_meshShader->setMat4("u_view", view);
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

    }
#endif

    for (int gi = 0; gi < static_cast<int>(scene.meshGroups.size()); ++gi)
    {
        bool isSelectedGroup = hasSelection && gi == selectedGroup;

        for (int si = 0; si < static_cast<int>(scene.meshGroups[gi].submeshes.size()); ++si)
        {
            auto& sm = scene.meshGroups[gi].submeshes[si];

#ifdef VEX_BACKEND_OPENGL
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

#ifdef VEX_BACKEND_OPENGL
    // --- Outline pass for selected group/submesh (rendered into same HDR buffer) ---
    if (hasSelection)
    {
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
        glStencilMask(0x00);
        glDepthMask(GL_FALSE);

        m_outlineShader->bind();
        m_outlineShader->setMat4("u_view", view);
        m_outlineShader->setMat4("u_projection", proj);
        m_outlineShader->setVec3("u_outlineColor", glm::vec3(1.0f, 0.5f, 0.0f));

        auto& outlineSubmeshes = scene.meshGroups[selectedGroup].submeshes;
        if (selectedSubmesh >= 0 && selectedSubmesh < static_cast<int>(outlineSubmeshes.size()))
        {
            // Draw outline for single submesh only
            auto& sm = outlineSubmeshes[selectedSubmesh];
            m_outlineShader->setFloat("u_outlineWidth", OUTLINE_WIDTH);
            sm.mesh->draw();
            m_outlineShader->setFloat("u_outlineWidth", -OUTLINE_WIDTH);
            sm.mesh->draw();
        }
        else
        {
            // Draw outline for all submeshes in the group
            for (auto& sm : outlineSubmeshes)
            {
                m_outlineShader->setFloat("u_outlineWidth", OUTLINE_WIDTH);
                sm.mesh->draw();
                m_outlineShader->setFloat("u_outlineWidth", -OUTLINE_WIDTH);
                sm.mesh->draw();
            }
        }

        m_outlineShader->unbind();

        glDepthMask(GL_TRUE);
        glStencilMask(0xFF);
        glDisable(GL_STENCIL_TEST);
    }
#endif

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
        m_fullscreenRTShader->setFloat("u_sampleCount", 1.0f);
        m_fullscreenRTShader->setFloat("u_exposure", m_rasterExposure);
        m_fullscreenRTShader->setFloat("u_gamma", m_rasterGamma);
        m_fullscreenRTShader->setBool("u_enableACES", m_rasterEnableACES);
        m_fullscreenRTShader->setBool("u_flipV", false); // GL framebuffer: natural bottom-left origin, no flip needed
        m_fullscreenQuad->draw();
        m_fullscreenRTShader->unbind();

        glEnable(GL_DEPTH_TEST);

        m_framebuffer->unbind();
    }
#endif
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

        if (scene.currentEnvmap >= Scene::Sky)
        {
            std::string envPath;

            // Use custom path if set and this is the custom entry
            if (scene.currentEnvmap == Scene::CustomHDR)
            {
                envPath = scene.customEnvmapPath;
            }
            else
            {
                // Try .hdr first, fall back to .jpg
                std::string base = "assets/textures/envmaps/"
                                 + std::string(scene.envmapNames[scene.currentEnvmap]) + "/"
                                 + std::string(scene.envmapNames[scene.currentEnvmap]);
                envPath = base + ".hdr";

                // Test if .hdr exists, fall back to .jpg
                int testW, testH, testCh;
                if (!stbi_info(envPath.c_str(), &testW, &testH, &testCh))
                    envPath = base + ".jpg";
            }

            int ew, eh, ech;
            stbi_set_flip_vertically_on_load(false);
            float* envData = stbi_loadf(envPath.c_str(), &ew, &eh, &ech, 3);
            if (envData)
            {
                m_cpuRaytracer->setEnvironmentMap(envData, ew, eh);

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
                // Compute average env color for ambient diffuse
                float rSum = 0, gSum = 0, bSum = 0;
                int n = ew * eh;
                for (int i = 0; i < n; ++i) { rSum += envData[3*i]; gSum += envData[3*i+1]; bSum += envData[3*i+2]; }
                m_rasterEnvColor = glm::clamp(glm::vec3(rSum, gSum, bSum) / float(n), 0.0f, 1.0f);
#endif

                stbi_image_free(envData);
            }
        }
        else
        {
            m_cpuRaytracer->clearEnvironmentMap();
#ifdef VEX_BACKEND_OPENGL
            if (m_rasterEnvMapTex) { glDeleteTextures(1, &m_rasterEnvMapTex); m_rasterEnvMapTex = 0; }
            m_rasterEnvColor = scene.skyboxColor;
#endif
        }
    }

    if (scene.currentEnvmap == Scene::SolidColor && scene.skyboxColor != m_prevSkyboxColor)
    {
        m_prevSkyboxColor = scene.skyboxColor;
        m_cpuRaytracer->setEnvironmentColor(scene.skyboxColor);
#ifdef VEX_BACKEND_OPENGL
        m_rasterEnvColor = scene.skyboxColor;
#endif
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

    // Trace one sample
    m_cpuRaytracer->traceSample();

    // Upload result to texture
    const auto& pixels = m_cpuRaytracer->getPixelBuffer();
    m_raytraceTexture->setData(pixels.data(), w, h, 4);

    // Render fullscreen quad to framebuffer
    m_framebuffer->bind();
    m_framebuffer->clear(0.0f, 0.0f, 0.0f, 1.0f);

#ifdef VEX_BACKEND_OPENGL
    glDisable(GL_DEPTH_TEST);
#endif

    m_fullscreenShader->bind();
    m_fullscreenShader->setTexture(0, m_raytraceTexture.get());
    m_fullscreenShader->setBool("u_flipV", true);  // CPU raytracer pixels are top-to-bottom
    m_fullscreenQuad->draw();
    m_fullscreenShader->unbind();

#ifdef VEX_BACKEND_OPENGL
    glEnable(GL_DEPTH_TEST);
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

        if (scene.currentEnvmap >= Scene::Sky)
        {
            std::string envPath;
            if (scene.currentEnvmap == Scene::CustomHDR)
            {
                envPath = scene.customEnvmapPath;
            }
            else
            {
                std::string base = "assets/textures/envmaps/"
                                 + std::string(scene.envmapNames[scene.currentEnvmap]) + "/"
                                 + std::string(scene.envmapNames[scene.currentEnvmap]);
                envPath = base + ".hdr";
                int testW, testH, testCh;
                if (!stbi_info(envPath.c_str(), &testW, &testH, &testCh))
                    envPath = base + ".jpg";
            }

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

    // Dispatch compute shader
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

    m_fullscreenRTShader->setFloat("u_sampleCount", static_cast<float>(m_gpuRaytracer->getSampleCount()));
    m_fullscreenRTShader->setFloat("u_exposure", m_gpuExposure);
    m_fullscreenRTShader->setFloat("u_gamma", m_gpuGamma);
    m_fullscreenRTShader->setBool("u_enableACES", m_gpuEnableACES);
    m_fullscreenRTShader->setBool("u_flipV", true);   // GPU raytracer accum texture: pixels stored top-to-bottom

    m_fullscreenQuad->draw();
    m_fullscreenRTShader->unbind();

    glEnable(GL_DEPTH_TEST);

    m_framebuffer->unbind();
    m_drawCalls = 1;
}
#endif

std::pair<int,int> SceneRenderer::pick(Scene& scene, int pixelX, int pixelY)
{
    if (!m_pickShader || !m_pickFB)
        return {-1, -1};

    const auto& mainSpec = m_framebuffer->getSpec();
    const auto& spec = m_pickFB->getSpec();

    // Ensure pick FB matches main viewport size
    if (spec.width != mainSpec.width || spec.height != mainSpec.height)
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
}
