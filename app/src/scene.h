#pragma once

#include <vex/core/camera.h>
#include <vex/graphics/mesh.h>
#include <vex/graphics/skybox.h>
#include <vex/graphics/texture.h>
#include <vex/scene/mesh_data.h>

#include "mesh_group_save.h"

#include <glm/glm.hpp>

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// CPU-side pixel data for a texture, populated during importOBJ and consumed
// (then cleared) by rebuildRaytraceGeometry to avoid a second stbi_load pass.
struct TexPixels
{
    std::vector<uint8_t> pixels; // unflipped RGBA8
    int width  = 0;
    int height = 0;
};

struct SceneMesh
{
    std::string name;
    std::unique_ptr<vex::Mesh> mesh;
    std::shared_ptr<vex::Texture2D> diffuseTexture;
    std::shared_ptr<vex::Texture2D> normalTexture;
    std::shared_ptr<vex::Texture2D> roughnessTexture;
    std::shared_ptr<vex::Texture2D> metallicTexture;
    std::shared_ptr<vex::Texture2D> emissiveTexture;
    std::shared_ptr<vex::Texture2D> aoTexture;
    vex::MeshData meshData;
    glm::mat4 modelMatrix = glm::mat4(1.0f);  // local transform relative to node
    uint32_t vertexCount = 0;
    uint32_t indexCount  = 0;
};

struct SceneNode
{
    std::string name;
    std::vector<SceneMesh> submeshes;
    glm::vec3 center { 0.0f };
    float     radius = 1.0f;
    glm::mat4 localMatrix = glm::mat4(1.0f);  // local transform (relative to parent)
    int       parentIndex   = -1;              // index into scene.nodes; -1 = root
    std::vector<int> childIndices;             // ordered children
};

struct SceneVolume
{
    std::string name     = "Volume";
    glm::vec3   center   = { 0.0f, 0.0f, 0.0f };
    glm::vec3   halfSize = { 1.0f, 1.0f, 1.0f };
    float       density  = 0.5f;               // sigma_t (extinction coefficient)
    glm::vec3   albedo   = {0.8f, 0.8f, 0.8f}; // per-channel scattering albedo (sigma_s = albedo * density)
    float       aniso    = 0.0f;               // Henyey-Greenstein g [-1=back, 0=iso, 1=forward]
    bool        infinite = false;  // true = global fog (no AABB clip)
    bool        enabled  = true;
};

struct Scene
{
    vex::Camera camera;
    std::vector<SceneNode> nodes;
    std::vector<SceneVolume> volumes;
    std::unique_ptr<vex::Skybox> skybox;

    glm::vec3 lightPos       { 0.0f, 1.95f, 0.0f };
    glm::vec3 lightColor     { 1.0f, 0.9f, 0.7f };
    float     lightIntensity = 1.0f;

    // Directional (sun) light
    float     sunAzimuth       = 40.0f;  // degrees, compass direction
    float     sunElevation     = 45.0f;  // degrees above horizon
    glm::vec3 sunColor         { 1.0f, 0.95f, 0.85f };
    float     sunIntensity     = 1.0f;
    float     sunAngularRadius = 0.00873f; // ~0.5 degrees in radians
    bool      showSun          = true;

    glm::vec3 getSunDirection() const
    {
        float azRad  = glm::radians(sunAzimuth);
        float elRad  = glm::radians(sunElevation);
        float cosEl  = std::cos(elRad);
        // Direction light travels (from sun toward ground)
        return glm::vec3(-cosEl * std::cos(azRad),
                         -std::sin(elRad),
                         -cosEl * std::sin(azRad));
    }

    enum Envmap { SolidColor = 0, HDRI0 = 1, HDRI1 = 2, HDRI2 = 3, HDRI3 = 4, CustomHDR = 5, EnvmapCount = 6 };
    static constexpr const char* envmapNames[] = {
        "Solid Color",
        "Brown Photo Studio 02",
        "Kloofendal Puresky",
        "Lilienstein",
        "Moonless Golf"
    };
    static constexpr const char* envmapPaths[] = {
        "",
        "VexAssetsCC0/HDRIs/2K/brown_photostudio_02_2k.hdr",
        "VexAssetsCC0/HDRIs/2K/kloofendal_48d_partly_cloudy_puresky_2k.hdr",
        "VexAssetsCC0/HDRIs/2K/lilienstein_2k.hdr",
        "VexAssetsCC0/HDRIs/2K/moonless_golf_2k.hdr"
    };

    int currentEnvmap = SolidColor;
    glm::vec3 skyboxColor { 0.9f, 0.78f, 0.65f };
    std::string customEnvmapPath;

    bool showSkybox = true;
    bool showLight  = false;

    bool geometryDirty = false;
    bool materialDirty = false;

    // Pixel cache populated during importOBJ / addNodeFromSave.
    // Used by SceneRenderer::rebuildRaytraceGeometry to skip the second stbi_load
    // per texture. Cleared by rebuildRaytraceGeometry after consumption.
    std::unordered_map<std::string, TexPixels> importedTexPixels;

    // Returns the accumulated world-space matrix of a node (product of all ancestor localMatrices).
    glm::mat4 getWorldMatrix(int nodeIdx) const;

    using ProgressFn = std::function<void(const std::string& stage, float progress)>;
    bool importOBJ(const std::string& path, const std::string& name,
                   ProgressFn onProgress = nullptr);
    bool importGLTF(const std::string& path, const std::string& name,
                    ProgressFn onProgress = nullptr);

    // Recreate GPU resources from a CPU save and add the node to the scene.
    // insertAt = -1 → append; otherwise inserts at that index.
    // When insertAt >= 0, calls fixRefsAfterInsert first so existing refs stay valid.
    void addNodeFromSave(const NodeSave& save, int insertAt = -1);

    // Fill importedTexPixels via parallel stbi_load over all unique texture paths in
    // the current scene nodes. Called by rebuildRaytraceGeometry when the cache is
    // empty (e.g. after a render-mode switch discarded it).
    void prefetchTextures();
};

// ── Index fixup helpers ───────────────────────────────────────────────────────
// Keep parentIndex/childIndices coherent after flat-vector insert or erase.

// After erasing node at removedIdx: decrement all refs > removedIdx.
void fixRefsAfterRemove(Scene& scene, int removedIdx);

// Before inserting at insertedIdx: increment all refs >= insertedIdx.
void fixRefsAfterInsert(Scene& scene, int insertedIdx);

// Collect subtree rooted at nodeIdx; returned in DESCENDING index order (safe erase order).
std::vector<int> collectSubtree(const Scene& scene, int nodeIdx);
