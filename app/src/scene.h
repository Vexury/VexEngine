#pragma once

#include <vex/core/camera.h>
#include <vex/graphics/mesh.h>
#include <vex/graphics/skybox.h>
#include <vex/graphics/texture.h>
#include <vex/scene/mesh_data.h>

#include <glm/glm.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

struct SceneMesh
{
    std::string name;
    std::unique_ptr<vex::Mesh> mesh;
    std::unique_ptr<vex::Texture2D> diffuseTexture;
    std::unique_ptr<vex::Texture2D> normalTexture;
    std::unique_ptr<vex::Texture2D> roughnessTexture;
    std::unique_ptr<vex::Texture2D> metallicTexture;
    std::unique_ptr<vex::Texture2D> emissiveTexture;
    vex::MeshData meshData;
    uint32_t vertexCount = 0;
    uint32_t indexCount  = 0;
};

struct MeshGroup
{
    std::string name;
    std::vector<SceneMesh> submeshes;
    glm::vec3 center { 0.0f };
    float     radius = 1.0f;
    glm::mat4 modelMatrix = glm::mat4(1.0f);
};

struct Scene
{
    vex::Camera camera;
    std::vector<MeshGroup> meshGroups;
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

    bool importOBJ(const std::string& path, const std::string& name);
};
