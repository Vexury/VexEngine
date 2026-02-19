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

    enum Envmap { SolidColor = 0, Sky = 1, Warehouse = 2, CustomHDR = 3, EnvmapCount };
    static constexpr const char* envmapNames[] = { "Solid Color", "sky", "warehouse" };

    int currentEnvmap = SolidColor;
    glm::vec3 skyboxColor { 0.86f, 0.64f, 0.40f };
    std::string customEnvmapPath;

    bool showSkybox = true;
    bool showLight  = false;

    bool geometryDirty = false;
    bool materialDirty = false;

    bool importOBJ(const std::string& path, const std::string& name);
};
