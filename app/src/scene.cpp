#include "scene.h"

#include <vex/scene/mesh_data.h>
#include <vex/core/log.h>

#include <cfloat>

bool Scene::importOBJ(const std::string& path, const std::string& name)
{
    auto submeshes = vex::MeshData::loadOBJ(path);
    if (submeshes.empty())
        return false;

    vex::Log::info("Uploading " + std::to_string(submeshes.size()) + " submesh(es) to GPU...");

    MeshGroup group;
    group.name = name;

    int texCount = 0;
    glm::vec3 bboxMin(FLT_MAX), bboxMax(-FLT_MAX);
    for (size_t i = 0; i < submeshes.size(); ++i)
    {
        for (const auto& v : submeshes[i].vertices)
        {
            bboxMin = glm::min(bboxMin, v.position);
            bboxMax = glm::max(bboxMax, v.position);
        }

        auto mesh = vex::Mesh::create();
        mesh->upload(submeshes[i]);

        std::unique_ptr<vex::Texture2D> tex;
        if (!submeshes[i].diffuseTexturePath.empty())
        {
            tex = vex::Texture2D::createFromFile(submeshes[i].diffuseTexturePath);
            if (tex) ++texCount;
        }

        std::unique_ptr<vex::Texture2D> normTex;
        if (!submeshes[i].normalTexturePath.empty())
        {
            normTex = vex::Texture2D::createFromFile(submeshes[i].normalTexturePath);
            if (normTex) ++texCount;
        }

        std::unique_ptr<vex::Texture2D> roughTex;
        if (!submeshes[i].roughnessTexturePath.empty())
        {
            roughTex = vex::Texture2D::createFromFile(submeshes[i].roughnessTexturePath);
            if (roughTex) ++texCount;
        }

        std::unique_ptr<vex::Texture2D> metalTex;
        if (!submeshes[i].metallicTexturePath.empty())
        {
            metalTex = vex::Texture2D::createFromFile(submeshes[i].metallicTexturePath);
            if (metalTex) ++texCount;
        }

        std::unique_ptr<vex::Texture2D> emissiveTex;
        if (!submeshes[i].emissiveTexturePath.empty())
        {
            emissiveTex = vex::Texture2D::createFromFile(submeshes[i].emissiveTexturePath);
            if (emissiveTex) ++texCount;
        }

        uint32_t vc = static_cast<uint32_t>(submeshes[i].vertices.size());
        uint32_t ic = static_cast<uint32_t>(submeshes[i].indices.size());

        SceneMesh sm;
        sm.name = submeshes[i].name.empty()
            ? "Submesh " + std::to_string(i)
            : submeshes[i].name;
        sm.mesh = std::move(mesh);
        sm.diffuseTexture = std::move(tex);
        sm.normalTexture = std::move(normTex);
        sm.roughnessTexture = std::move(roughTex);
        sm.metallicTexture = std::move(metalTex);
        sm.emissiveTexture = std::move(emissiveTex);
        sm.meshData = submeshes[i]; // retain CPU-side data for raytracing
        sm.vertexCount = vc;
        sm.indexCount = ic;
        group.submeshes.push_back(std::move(sm));
    }
    group.center = (bboxMin + bboxMax) * 0.5f;
    group.radius = glm::length(bboxMax - bboxMin) * 0.5f;
    meshGroups.push_back(std::move(group));
    geometryDirty = true;

    if (texCount > 0)
        vex::Log::info("  Loaded " + std::to_string(texCount) + " texture(s)");

    return true;
}
