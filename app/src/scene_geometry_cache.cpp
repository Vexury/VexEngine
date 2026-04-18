#include "scene_geometry_cache.h"
#include "scene.h"
#include "scene_importer.h"

#include <vex/graphics/mesh.h>
#include <vex/scene/mesh_data.h>
#include <vex/core/log.h>

#include <stb_image.h>
#include <tinyexr.h>

#ifdef VEX_BACKEND_VULKAN
#include <vex/vulkan/vk_mesh.h>
#include <vex/vulkan/vk_gpu_raytracer.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <vector>

static constexpr float GEOMETRY_EPSILON = 1e-8f;
static constexpr int   RT_TEX_MAX       = 1024;

static std::vector<uint8_t> downsampleNearest(
    const uint8_t* src, int sw, int sh, int dw, int dh)
{
    std::vector<uint8_t> out(static_cast<size_t>(dw) * dh * 4);
    float invX = static_cast<float>(sw) / dw;
    float invY = static_cast<float>(sh) / dh;
    for (int y = 0; y < dh; ++y)
    {
        int sy = std::min(static_cast<int>(y * invY), sh - 1);
        for (int x = 0; x < dw; ++x)
        {
            int sx = std::min(static_cast<int>(x * invX), sw - 1);
            const uint8_t* s = src  + (sy * sw + sx) * 4;
            uint8_t*       d = out.data() + (y  * dw + x)  * 4;
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// SceneGeometryCache::rebuild
// ---------------------------------------------------------------------------

void SceneGeometryCache::rebuild(const Scene& scene, vex::CPURaytracer& cpuRT,
                                  bool luminanceCDF, ProgressFn progress)
{
    m_luminanceCDF = luminanceCDF;
    m_blasTlasReady = false;

    auto t_total = std::chrono::steady_clock::now();

    std::vector<vex::CPURaytracer::TextureData> textures;

    std::unordered_map<std::string, int> textureMap;
    int texFromCache = 0;
    int texFromDisk  = 0;

    auto addTextureData = [&](const uint8_t* pixels, int tw, int th) -> int
    {
        int dw = tw, dh = th;
        if (tw > RT_TEX_MAX || th > RT_TEX_MAX)
        {
            float scale = std::min(static_cast<float>(RT_TEX_MAX) / tw,
                                   static_cast<float>(RT_TEX_MAX) / th);
            dw = std::max(1, static_cast<int>(tw * scale));
            dh = std::max(1, static_cast<int>(th * scale));
        }
        int idx = static_cast<int>(textures.size());
        vex::CPURaytracer::TextureData td;
        td.width  = dw;
        td.height = dh;
        if (dw == tw && dh == th)
            td.pixels.assign(pixels, pixels + static_cast<size_t>(tw) * th * 4);
        else
            td.pixels = downsampleNearest(pixels, tw, th, dw, dh);
        textures.push_back(std::move(td));
        return idx;
    };

    auto resolveTexture = [&](const std::string& path) -> int
    {
        if (path.empty()) return -1;
        auto it = textureMap.find(path);
        if (it != textureMap.end()) return it->second;
        int idx = -1;

        auto cacheIt = scene.importedTexPixels.find(path);
        if (cacheIt != scene.importedTexPixels.end())
        {
            const auto& cached = cacheIt->second;
            idx = addTextureData(cached.pixels.data(), cached.width, cached.height);
            ++texFromCache;
        }
        else if (path.size() >= 4 &&
            (path.compare(path.size() - 4, 4, ".exr") == 0 ||
             path.compare(path.size() - 4, 4, ".EXR") == 0))
        {
            float* exrRGBA = nullptr;
            int tw = 0, th = 0;
            const char* err = nullptr;
            if (LoadEXR(&exrRGBA, &tw, &th, path.c_str(), &err) == TINYEXR_SUCCESS)
            {
                std::vector<uint8_t> px(static_cast<size_t>(tw) * th * 4);
                for (size_t i = 0; i < px.size(); ++i)
                    px[i] = static_cast<unsigned char>(
                        std::clamp(exrRGBA[i], 0.0f, 1.0f) * 255.0f + 0.5f);
                free(exrRGBA);
                idx = addTextureData(px.data(), tw, th);
                ++texFromDisk;
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
                idx = addTextureData(texData, tw, th);
                stbi_image_free(texData);
                ++texFromDisk;
            }
        }

        textureMap[path] = idx;
        return idx;
    };

    // Rebuild per-node local-space AABBs
    m_nodeLocalAABBs.clear();
    m_nodeLocalAABBs.resize(scene.nodes.size());
    for (size_t ni = 0; ni < scene.nodes.size(); ++ni)
        for (const auto& sm : scene.nodes[ni].submeshes)
            for (const auto& v : sm.meshData.vertices)
                m_nodeLocalAABBs[ni].grow(v.position);

    if (scene.importedTexPixels.empty())
        SceneImporter::prefetchTextures(const_cast<Scene&>(scene));

    // -----------------------------------------------------------------------
    // Texture resolution pre-pass (sequential) + task list building
    // -----------------------------------------------------------------------
    if (progress) progress("Resolving textures...", 0.35f);
    auto t_flatten = std::chrono::steady_clock::now();

    // Each SubmeshTask captures all inputs a parallel worker needs: pre-resolved
    // texture indices, pre-computed matrices, and the output slice offset.
    struct SubmeshTask {
        int nodeIdx, smIdx;
        int triOffset, triCount;
        glm::mat4 worldMat;
        glm::mat3 normalMat;
        int texIdx, emissiveTexIdx, normalTexIdx, roughnessTexIdx, metallicTexIdx, alphaTexIdx;
    };

    std::vector<SubmeshTask> tasks;
    int globalTriOffset = 0;

#ifdef VEX_BACKEND_VULKAN
    auto iBF = [](int   v) -> float    { float    f; std::memcpy(&f, &v, sizeof(f)); return f; };
    auto fBU = [](float v) -> uint32_t { uint32_t u; std::memcpy(&u, &v, sizeof(u)); return u; };
    m_vkInstanceOffsets.clear();
#endif

    for (int ni = 0; ni < (int)scene.nodes.size(); ++ni)
    {
        const glm::mat4 nodeWorld = scene.getWorldMatrix(ni);
        for (int si = 0; si < (int)scene.nodes[ni].submeshes.size(); ++si)
        {
            const auto& sm = scene.nodes[ni].submeshes[si];
            const glm::mat4 combined = nodeWorld * sm.modelMatrix;
            const glm::mat3 normalM  = glm::mat3(glm::transpose(glm::inverse(combined)));
            int tc = (int)(sm.meshData.indices.size() / 3);

            SubmeshTask task;
            task.nodeIdx      = ni;
            task.smIdx        = si;
            task.triOffset    = globalTriOffset;
            task.triCount     = tc;
            task.worldMat     = combined;
            task.normalMat    = normalM;
            task.texIdx          = resolveTexture(sm.meshData.diffuseTexturePath);
            task.emissiveTexIdx  = resolveTexture(sm.meshData.emissiveTexturePath);
            task.normalTexIdx    = resolveTexture(sm.meshData.normalTexturePath);
            task.roughnessTexIdx = resolveTexture(sm.meshData.roughnessTexturePath);
            task.metallicTexIdx  = resolveTexture(sm.meshData.metallicTexturePath);
            task.alphaTexIdx     = resolveTexture(sm.meshData.alphaTexturePath);
            tasks.push_back(task);

#ifdef VEX_BACKEND_VULKAN
            m_vkInstanceOffsets.push_back(static_cast<uint32_t>(globalTriOffset));
#endif
            globalTriOffset += tc;

        }  // end for(si)
    }  // end for(ni)

    // -----------------------------------------------------------------------
    // Parallel triangle flatten (Improvements 1 + 2)
    // -----------------------------------------------------------------------

    // Pre-allocate output arrays; workers write to exclusive slices.
    std::vector<vex::CPURaytracer::Triangle> flatTris(static_cast<size_t>(globalTriOffset));
    std::vector<std::pair<int,int>>          flatSrc(static_cast<size_t>(globalTriOffset));
    std::vector<int>                         flatSrcIdx(static_cast<size_t>(globalTriOffset));

#ifdef VEX_BACKEND_VULKAN
    static constexpr size_t FLOATS_PER_TRI = 52;
    std::vector<float> flatShading(static_cast<size_t>(globalTriOffset) * FLOATS_PER_TRI, 0.0f);

    struct LightEntry { uint32_t globalIdx; float weight; };
    std::vector<std::vector<LightEntry>> taskLights(tasks.size());
#endif

    // Workers atomically claim tasks by index. Each taskIdx maps to an exclusive
    // slice of the output arrays (flatTris[task.triOffset .. +task.triCount]),
    // so no locks are needed. taskLights is also indexed by taskIdx — NOT by
    // thread id — so the post-join merge preserves submesh order.
    {
        std::atomic<int> nextTask{0};
        const int numThreads = std::max(1, (int)std::thread::hardware_concurrency());
        std::vector<std::thread> workers;
        workers.reserve(numThreads);

        for (int t = 0; t < numThreads; ++t)
        {
            workers.emplace_back([&]()
            {
                for (;;)
                {
                    int taskIdx = nextTask.fetch_add(1, std::memory_order_relaxed);
                    if (taskIdx >= (int)tasks.size()) break;

                    const SubmeshTask& task = tasks[taskIdx];
                    const auto& sm      = scene.nodes[task.nodeIdx].submeshes[task.smIdx];
                    const auto& verts   = sm.meshData.vertices;
                    const auto& indices = sm.meshData.indices;
                    const auto& md      = sm.meshData;

#ifdef VEX_BACKEND_VULKAN
                    const bool smEmissive = glm::length(md.emissiveColor) > 0.001f;
#endif

                    for (size_t j = 0; j + 2 < indices.size(); j += 3)
                    {
                        const int localTri = static_cast<int>(j / 3);
                        const int outIdx   = task.triOffset + localTri;

                        const auto& v0 = verts[indices[j + 0]];
                        const auto& v1 = verts[indices[j + 1]];
                        const auto& v2 = verts[indices[j + 2]];

                        glm::vec3 p0 = glm::vec3(task.worldMat * glm::vec4(v0.position, 1.0f));
                        glm::vec3 p1 = glm::vec3(task.worldMat * glm::vec4(v1.position, 1.0f));
                        glm::vec3 p2 = glm::vec3(task.worldMat * glm::vec4(v2.position, 1.0f));

                        glm::vec3 edge1 = p1 - p0;
                        glm::vec3 edge2 = p2 - p0;
                        glm::vec3 cr    = glm::cross(edge1, edge2);
                        float len       = glm::length(cr);
                        glm::vec3 geoN  = (len > GEOMETRY_EPSILON) ? (cr / len) : glm::vec3(0, 1, 0);
                        float area      = len * 0.5f;

                        glm::vec2 dUV1 = v1.uv - v0.uv;
                        glm::vec2 dUV2 = v2.uv - v0.uv;
                        float det = dUV1.x * dUV2.y - dUV2.x * dUV1.y;
                        glm::vec3 tangent(1, 0, 0);
                        float bitangentSign = 1.0f;
                        if (std::abs(det) > GEOMETRY_EPSILON)
                        {
                            float f = 1.0f / det;
                            tangent = glm::normalize(f * (dUV2.y * edge1 - dUV1.y * edge2));
                            glm::vec3 B = f * (-dUV2.x * edge1 + dUV1.x * edge2);
                            bitangentSign = (glm::dot(glm::cross(geoN, tangent), B) < 0.0f) ? -1.0f : 1.0f;
                        }

                        vex::CPURaytracer::Triangle tri;
                        tri.v0 = p0; tri.v1 = p1; tri.v2 = p2;
                        tri.n0 = glm::normalize(task.normalMat * v0.normal);
                        tri.n1 = glm::normalize(task.normalMat * v1.normal);
                        tri.n2 = glm::normalize(task.normalMat * v2.normal);
                        tri.uv0 = v0.uv; tri.uv1 = v1.uv; tri.uv2 = v2.uv;
                        tri.color            = v0.color * md.baseColor;
                        tri.emissive         = md.emissiveColor * md.emissiveStrength;
                        tri.emissiveStrength = md.emissiveStrength;
                        tri.geometricNormal  = geoN;
                        tri.area             = area;
                        tri.textureIndex          = task.texIdx;
                        tri.emissiveTextureIndex  = task.emissiveTexIdx;
                        tri.normalMapTextureIndex = task.normalTexIdx;
                        tri.roughnessTextureIndex = task.roughnessTexIdx;
                        tri.metallicTextureIndex  = task.metallicTexIdx;
                        tri.alphaTextureIndex     = task.alphaTexIdx;
                        tri.alphaClip    = md.alphaClip;
                        tri.materialType = md.materialType;
                        tri.ior          = md.ior;
                        tri.roughness    = md.roughness;
                        tri.metallic     = md.metallic;
                        tri.tangent      = tangent;
                        tri.bitangentSign = bitangentSign;

                        flatTris[outIdx]   = tri;
                        flatSrc[outIdx]    = {task.nodeIdx, task.smIdx};
                        flatSrcIdx[outIdx] = localTri;

#ifdef VEX_BACKEND_VULKAN
                        float* sh = &flatShading[static_cast<size_t>(outIdx) * FLOATS_PER_TRI];
                        // [0] n0.xyz + roughnessTexIdx
                        sh[ 0]=tri.n0.x; sh[ 1]=tri.n0.y; sh[ 2]=tri.n0.z; sh[ 3]=iBF(task.roughnessTexIdx);
                        // [1] n1.xyz + metallicTexIdx
                        sh[ 4]=tri.n1.x; sh[ 5]=tri.n1.y; sh[ 6]=tri.n1.z; sh[ 7]=iBF(task.metallicTexIdx);
                        // [2] n2.xyz + emissiveStrength
                        sh[ 8]=tri.n2.x; sh[ 9]=tri.n2.y; sh[10]=tri.n2.z; sh[11]=md.emissiveStrength;
                        // [3] uv0.xy + uv1.xy
                        sh[12]=v0.uv.x; sh[13]=v0.uv.y; sh[14]=v1.uv.x; sh[15]=v1.uv.y;
                        // [4] uv2.xy + roughness + metallic
                        sh[16]=v2.uv.x; sh[17]=v2.uv.y; sh[18]=md.roughness; sh[19]=md.metallic;
                        // [5] color.xyz (tinted) + texIdx
                        sh[20]=v0.color.x*md.baseColor.x; sh[21]=v0.color.y*md.baseColor.y;
                        sh[22]=v0.color.z*md.baseColor.z; sh[23]=iBF(task.texIdx);
                        // [6] emissive.xyz (scaled) + area
                        sh[24]=md.emissiveColor.x*md.emissiveStrength;
                        sh[25]=md.emissiveColor.y*md.emissiveStrength;
                        sh[26]=md.emissiveColor.z*md.emissiveStrength;
                        sh[27]=area;
                        // [7] geoNormal.xyz + normalMapTexIdx
                        sh[28]=geoN.x; sh[29]=geoN.y; sh[30]=geoN.z; sh[31]=iBF(task.normalTexIdx);
                        // [8] alphaEnc + materialType + ior + emissiveTexIdx
                        // alphaEnc encodes alpha clip: -1=no clip, -2=use diffuse.a, >=0=alpha tex idx
                        { int alphaEnc = md.alphaClip
                              ? (task.alphaTexIdx >= 0 ? task.alphaTexIdx : -2)
                              : -1;
                          sh[32]=iBF(alphaEnc); }
                        sh[33]=static_cast<float>(md.materialType);
                        sh[34]=md.ior; sh[35]=iBF(task.emissiveTexIdx);
                        // [9] tangent.xyz + bitangentSign
                        sh[36]=tangent.x; sh[37]=tangent.y; sh[38]=tangent.z; sh[39]=bitangentSign;
                        // [10] v0.xyz + pad
                        sh[40]=p0.x; sh[41]=p0.y; sh[42]=p0.z; sh[43]=0.0f;
                        // [11] v1.xyz + pad
                        sh[44]=p1.x; sh[45]=p1.y; sh[46]=p1.z; sh[47]=0.0f;
                        // [12] v2.xyz + pad
                        sh[48]=p2.x; sh[49]=p2.y; sh[50]=p2.z; sh[51]=0.0f;

                        if (smEmissive)
                        {
                            const auto& ec = md.emissiveColor;
                            float vkW = m_luminanceCDF
                                ? (0.2126f*ec.r + 0.7152f*ec.g + 0.0722f*ec.b) * md.emissiveStrength * area
                                : area;
                            // taskIdx (not thread id) keeps lights in submesh order after the join.
                            taskLights[taskIdx].push_back({static_cast<uint32_t>(outIdx), vkW});
                        }
#endif
                    }
                }
            });
        }
        for (auto& w : workers) w.join();
    }

    {
        float t_flatten_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t_flatten).count();
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "  CPU triangle flatten + texture resolve: %.0f ms  (%d tris, %d cached / %d disk)",
            t_flatten_ms, globalTriOffset, texFromCache, texFromDisk);
        vex::Log::info(buf);
    }

    // -----------------------------------------------------------------------
    // BVH build + post-join: getReorderedTriangles + src-mapping reorder
    // (Improvement 1: eliminates the old Stage 3 second flatten pass)
    // -----------------------------------------------------------------------
    if (progress) progress("Building BVH...", 0.45f);
    m_rtTextures = textures;  // copy before move

    {
        auto t_cpu_bvh = std::chrono::steady_clock::now();
        cpuRT.setGeometry(std::move(flatTris), std::move(textures));
        {
            float ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - t_cpu_bvh).count();
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "  CPURaytracer::setGeometry (BVH build + reorder): %.0f ms", ms);
            vex::Log::info(buf);
        }

        auto t_rt_bvh = std::chrono::steady_clock::now();
        m_rtBVH = cpuRT.getBVH();

        // Get BVH-ordered triangles directly from CPURaytracer (no second flatten pass).
        cpuRT.getReorderedTriangles(m_rtTriangles);

        // Reorder src-mapping arrays using the same BVH permutation.
        const auto& bvhIndices = m_rtBVH.indices();
        const size_t triCount  = bvhIndices.size();
        m_rtTriangleSrcSubmesh.resize(triCount);
        m_rtTriangleSrcTriIdx.resize(triCount);
        for (size_t i = 0; i < triCount; ++i)
        {
            m_rtTriangleSrcSubmesh[i] = flatSrc[bvhIndices[i]];
            m_rtTriangleSrcTriIdx[i]  = flatSrcIdx[bvhIndices[i]];
        }

        // Build CPU RT light CDF from BVH-ordered m_rtTriangles.
        m_rtLightIndices.clear();
        m_rtLightCDF.clear();
        m_rtTotalLightArea = 0.0f;
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_rtTriangles.size()); ++i)
        {
            if (glm::length(m_rtTriangles[i].emissive) > 0.001f)
            {
                m_rtLightIndices.push_back(i);
                const auto& em = m_rtTriangles[i].emissive;
                float w = m_luminanceCDF
                    ? (0.2126f * em.r + 0.7152f * em.g + 0.0722f * em.b) * m_rtTriangles[i].area
                    : m_rtTriangles[i].area;
                m_rtTotalLightArea += w;
                m_rtLightCDF.push_back(m_rtTotalLightArea);
            }
        }
        if (m_rtTotalLightArea > 0.0f)
            for (float& c : m_rtLightCDF) c /= m_rtTotalLightArea;

        {
            float t_rt_bvh_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - t_rt_bvh).count();
            char rtbuf[128];
            std::snprintf(rtbuf, sizeof(rtbuf),
                "  GPU-tri reorder (shared CPU BVH): %.0f ms  (%zu tris)", t_rt_bvh_ms, triCount);
            vex::Log::info(rtbuf);
        }

        char sahBuf[32];
        std::snprintf(sahBuf, sizeof(sahBuf), "%.1f", cpuRT.getBVHSAHCost());
        std::string emissiveStr = m_rtLightIndices.empty() ? ""
            : ", " + std::to_string(m_rtLightIndices.size()) + " emissive";
        vex::Log::info("  CPU BVH: " + std::to_string(cpuRT.getBVHNodeCount()) + " nodes, "
                      + std::to_string(m_rtTriangles.size()) + " triangles, SAH " + sahBuf + emissiveStr);
    }

#ifdef VEX_BACKEND_VULKAN
    {
        if (progress) progress("Packing shading data...", 0.58f);

        // Shading data was packed in parallel; just move it into the member vector.
        m_vkTriShading = std::move(flatShading);

        // Merge per-task light entries in task order, which equals submesh order.
        // The VK shading SSBO is indexed by instanceOffsets[BLAS] + gl_PrimitiveID,
        // so light indices must be in that same submesh-contiguous order.
        std::vector<uint32_t> vkLightIndices;
        std::vector<float>    vkLightCDF;
        float vkTotalLightArea = 0.0f;
        for (const auto& lights : taskLights)
        {
            for (const auto& le : lights)
            {
                vkLightIndices.push_back(le.globalIdx);
                vkTotalLightArea += le.weight;
                vkLightCDF.push_back(vkTotalLightArea);
            }
        }
        if (vkTotalLightArea > 0.0f)
            for (float& c : vkLightCDF) c /= vkTotalLightArea;

        uint32_t lightCount = static_cast<uint32_t>(vkLightIndices.size());
        {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "  VK triShading SSBO pack: parallel  (%zu tris, %u emissive)",
                m_vkTriShading.size() / FLOATS_PER_TRI, lightCount);
            vex::Log::info(buf);
        }

        m_vkLights.clear();
        m_vkLights.push_back(lightCount);
        m_vkLights.push_back(fBU(vkTotalLightArea));
        m_vkLights.push_back(0); m_vkLights.push_back(0);
        for (uint32_t idx : vkLightIndices) m_vkLights.push_back(idx);
        for (float c : vkLightCDF) m_vkLights.push_back(fBU(c));

        // Note: textures are no longer packed into a CPU SSBO here.
        // They are uploaded as individual VkImages by VKGpuRaytracer::uploadSceneData()
        // using m_rtTextures directly via the textures() accessor.

        vex::Log::info("  VK CPU pack done: " + std::to_string(m_vkInstanceOffsets.size()) + " submeshes, "
                      + std::to_string(m_vkTriShading.size() / FLOATS_PER_TRI) + " tris, "
                      + std::to_string(lightCount) + " emissive - call buildAccelerationStructures() to commit GPU");
    }
#endif // VEX_BACKEND_VULKAN

    const_cast<Scene&>(scene).importedTexPixels.clear();

    {
        float t_total_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t_total).count();
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "  rebuildRaytraceGeometry total: %.0f ms  (%.1f s)", t_total_ms, t_total_ms / 1000.0f);
        vex::Log::info(buf);
    }

    m_texturePathToIndex = std::move(textureMap);
    m_ready = true;
}

// ---------------------------------------------------------------------------
// SceneGeometryCache::buildAccelerationStructures  (VK only)
// ---------------------------------------------------------------------------

#ifdef VEX_BACKEND_VULKAN
void SceneGeometryCache::buildAccelerationStructures(const Scene& scene,
                                                      vex::VKGpuRaytracer* vkRaytracer,
                                                      ProgressFn progress)
{
    if (!vkRaytracer) return;

    vkRaytracer->clearAccelerationStructures();

    std::vector<glm::mat4> blasTransforms;
    blasTransforms.reserve(m_vkInstanceOffsets.size());

    for (int ni = 0; ni < (int)scene.nodes.size(); ++ni)
    {
        const glm::mat4 nodeWorld = scene.getWorldMatrix(ni);
        for (const auto& sm : scene.nodes[ni].submeshes)
        {
            const glm::mat4 combinedMat = nodeWorld * sm.modelMatrix;
            auto* vkMesh = static_cast<vex::VKMesh*>(sm.mesh.get());
            vkRaytracer->addBlas(
                vkMesh->getVertexBuffer(), vkMesh->getVertexCount(), sizeof(vex::Vertex),
                vkMesh->getIndexBuffer(),  vkMesh->getIndexCount());
            blasTransforms.push_back(combinedMat);
        }
    }

    if (progress) progress("Building BLASes...", 0.7f);
    {
        auto t_blas = std::chrono::steady_clock::now();
        vkRaytracer->commitBlasBuild();
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t_blas).count();
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "  VK BLAS build (GPU): %.0f ms  (%zu BLASes)",
            ms, blasTransforms.size());
        vex::Log::info(buf);
    }

    if (progress) progress("Building TLAS...", 0.9f);
    {
        auto t_tlas = std::chrono::steady_clock::now();
        vkRaytracer->buildTlas(blasTransforms);
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t_tlas).count();
        char buf[64];
        std::snprintf(buf, sizeof(buf), "  VK TLAS build (GPU): %.0f ms", ms);
        vex::Log::info(buf);
    }

    vex::Log::info("  VK GPU: " + std::to_string(blasTransforms.size()) + " BLASes + TLAS built");
    m_blasTlasReady = true;
}
#endif // VEX_BACKEND_VULKAN

// ---------------------------------------------------------------------------
// SceneGeometryCache::rebuildMaterials
// ---------------------------------------------------------------------------

void SceneGeometryCache::rebuildMaterials(const Scene& scene, vex::CPURaytracer* cpuRT,
                                           bool luminanceCDF)
{
    m_luminanceCDF = luminanceCDF;

    for (size_t i = 0; i < m_rtTriangles.size(); ++i)
    {
        auto [gi, si] = m_rtTriangleSrcSubmesh[i];
        int triIdx    = m_rtTriangleSrcTriIdx[i];
        const auto& sm = scene.nodes[gi].submeshes[si];
        const auto& md = sm.meshData;
        const auto& v0 = md.vertices[md.indices[triIdx * 3]];
        m_rtTriangles[i].color            = v0.color   * md.baseColor;
        m_rtTriangles[i].emissive         = md.emissiveColor * md.emissiveStrength;
        m_rtTriangles[i].emissiveStrength = md.emissiveStrength;
        m_rtTriangles[i].materialType     = md.materialType;
        m_rtTriangles[i].ior              = md.ior;
        m_rtTriangles[i].roughness        = md.roughness;
        m_rtTriangles[i].metallic         = md.metallic;
        m_rtTriangles[i].alphaClip        = md.alphaClip;
    }

    m_rtLightIndices.clear();
    m_rtLightCDF.clear();
    m_rtTotalLightArea = 0.0f;
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_rtTriangles.size()); ++i)
    {
        if (glm::length(m_rtTriangles[i].emissive) > 0.001f)
        {
            m_rtLightIndices.push_back(i);
            const auto& em = m_rtTriangles[i].emissive;
            float w = m_luminanceCDF
                ? (0.2126f * em.r + 0.7152f * em.g + 0.0722f * em.b) * m_rtTriangles[i].area
                : m_rtTriangles[i].area;
            m_rtTotalLightArea += w;
            m_rtLightCDF.push_back(m_rtTotalLightArea);
        }
    }
    if (m_rtTotalLightArea > 0.0f)
        for (float& c : m_rtLightCDF) c /= m_rtTotalLightArea;

    if (cpuRT)
        cpuRT->updateMaterials(m_rtTriangles);

#ifdef VEX_BACKEND_VULKAN
    if (!m_vkTriShading.empty())
    {
        static constexpr size_t FLOATS_PER_TRI = 52;
        auto iBF = [](int   v) -> float    { float    f; std::memcpy(&f, &v, sizeof(f)); return f; };
        auto fBU = [](float v) -> uint32_t { uint32_t u; std::memcpy(&u, &v, sizeof(u)); return u; };

        size_t blasIdx = 0;
        for (const auto& node : scene.nodes)
        {
            for (const auto& sm : node.submeshes)
            {
                uint32_t triStart = m_vkInstanceOffsets[blasIdx];
                size_t   triCount = sm.meshData.indices.size() / 3;
                for (size_t t = 0; t < triCount; ++t)
                {
                    size_t base = (triStart + t) * FLOATS_PER_TRI;
                    // [2].w = emissiveStrength
                    m_vkTriShading[base + 11] = sm.meshData.emissiveStrength;
                    // [4].z/.w = roughness/metallic
                    m_vkTriShading[base + 18] = sm.meshData.roughness;
                    m_vkTriShading[base + 19] = sm.meshData.metallic;
                    // [5].xyz = color tinted by baseColor
                    const auto& v0 = sm.meshData.vertices[sm.meshData.indices[t * 3]];
                    m_vkTriShading[base + 20] = v0.color.x * sm.meshData.baseColor.x;
                    m_vkTriShading[base + 21] = v0.color.y * sm.meshData.baseColor.y;
                    m_vkTriShading[base + 22] = v0.color.z * sm.meshData.baseColor.z;
                    // [6].xyz = emissive scaled by emissiveStrength
                    m_vkTriShading[base + 24] = sm.meshData.emissiveColor.x * sm.meshData.emissiveStrength;
                    m_vkTriShading[base + 25] = sm.meshData.emissiveColor.y * sm.meshData.emissiveStrength;
                    m_vkTriShading[base + 26] = sm.meshData.emissiveColor.z * sm.meshData.emissiveStrength;
                    // [8].x encodes alpha clip: -1=no clip, -2=use diffuse.a, >=0=alpha tex idx.
                    { auto it = m_texturePathToIndex.find(sm.meshData.alphaTexturePath);
                      int alphaTexIdx = (it != m_texturePathToIndex.end()) ? it->second : -1;
                      int alphaEnc = sm.meshData.alphaClip
                          ? (alphaTexIdx >= 0 ? alphaTexIdx : -2)
                          : -1;
                      m_vkTriShading[base + 32] = iBF(alphaEnc); }
                    m_vkTriShading[base + 33] = static_cast<float>(sm.meshData.materialType);
                    m_vkTriShading[base + 34] = sm.meshData.ior;
                }
                ++blasIdx;
            }
        }

        // Rebuild m_vkLights SSBO from updated emissive data
        std::vector<uint32_t> newLightIdx;
        std::vector<float>    newLightCDF;
        float newTotal = 0.0f;
        size_t bIdx = 0;
        for (const auto& nd : scene.nodes)
        {
            for (const auto& sm2 : nd.submeshes)
            {
                if (glm::length(sm2.meshData.emissiveColor) > 0.001f)
                {
                    uint32_t triStart2 = m_vkInstanceOffsets[bIdx];
                    uint32_t triCount2 = static_cast<uint32_t>(sm2.meshData.indices.size() / 3);
                    for (uint32_t t = 0; t < triCount2; ++t)
                    {
                        uint32_t globalTri = triStart2 + t;
                        float area = m_vkTriShading[globalTri * FLOATS_PER_TRI + 27];
                        const auto& ec = sm2.meshData.emissiveColor;
                        float w = m_luminanceCDF
                            ? (0.2126f * ec.r + 0.7152f * ec.g + 0.0722f * ec.b)
                              * sm2.meshData.emissiveStrength * area
                            : area;
                        newLightIdx.push_back(globalTri);
                        newTotal += w;
                        newLightCDF.push_back(newTotal);
                    }
                }
                ++bIdx;
            }
        }
        if (newTotal > 0.0f)
            for (float& c : newLightCDF) c /= newTotal;
        m_vkLights.clear();
        m_vkLights.push_back(static_cast<uint32_t>(newLightIdx.size()));
        m_vkLights.push_back(fBU(newTotal));
        m_vkLights.push_back(0u); m_vkLights.push_back(0u);
        for (uint32_t idx : newLightIdx) m_vkLights.push_back(idx);
        for (float c : newLightCDF) m_vkLights.push_back(fBU(c));
    }
#endif
}

// ---------------------------------------------------------------------------
// SceneGeometryCache::rebuildLightCDF
// ---------------------------------------------------------------------------

void SceneGeometryCache::rebuildLightCDF(bool luminanceCDF)
{
    m_luminanceCDF = luminanceCDF;

    // Rebuild CPU/compute light CDF from existing triangle data
    m_rtLightIndices.clear();
    m_rtLightCDF.clear();
    m_rtTotalLightArea = 0.0f;
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_rtTriangles.size()); ++i)
    {
        if (glm::length(m_rtTriangles[i].emissive) > 0.001f)
        {
            m_rtLightIndices.push_back(i);
            const auto& em = m_rtTriangles[i].emissive;
            float w = m_luminanceCDF
                ? (0.2126f * em.r + 0.7152f * em.g + 0.0722f * em.b) * m_rtTriangles[i].area
                : m_rtTriangles[i].area;
            m_rtTotalLightArea += w;
            m_rtLightCDF.push_back(m_rtTotalLightArea);
        }
    }
    if (m_rtTotalLightArea > 0.0f)
        for (float& c : m_rtLightCDF) c /= m_rtTotalLightArea;

#ifdef VEX_BACKEND_VULKAN
    // Rebuild VK HW RT light SSBO from m_vkTriShading (emissive at [6].xyz, area at [6].w)
    if (!m_vkTriShading.empty())
    {
        auto fBU = [](float f) -> uint32_t { uint32_t u; std::memcpy(&u, &f, sizeof(u)); return u; };
        constexpr size_t kFloatsPerTri = 52;
        std::vector<uint32_t> vkIdx;
        std::vector<float>    vkCDF;
        float vkTotal = 0.0f;
        uint32_t triCount = static_cast<uint32_t>(m_vkTriShading.size() / kFloatsPerTri);
        for (uint32_t i = 0; i < triCount; ++i)
        {
            const float* p = &m_vkTriShading[i * kFloatsPerTri];
            glm::vec3 em(p[24], p[25], p[26]);
            float area = p[27];
            if (glm::length(em) > 0.001f)
            {
                vkIdx.push_back(i);
                float w = m_luminanceCDF
                    ? (0.2126f * em.r + 0.7152f * em.g + 0.0722f * em.b) * area
                    : area;
                vkTotal += w;
                vkCDF.push_back(vkTotal);
            }
        }
        if (vkTotal > 0.0f)
            for (float& c : vkCDF) c /= vkTotal;
        m_vkLights.clear();
        m_vkLights.push_back(static_cast<uint32_t>(vkIdx.size()));
        m_vkLights.push_back(fBU(vkTotal));
        m_vkLights.push_back(0); m_vkLights.push_back(0);
        for (uint32_t idx : vkIdx)  m_vkLights.push_back(idx);
        for (float  c   : vkCDF)    m_vkLights.push_back(fBU(c));
    }
#endif
}
