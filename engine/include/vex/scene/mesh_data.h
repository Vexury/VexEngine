#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace vex
{

struct GLTFNodeInfo
{
    std::string  nodeName;
    glm::mat4    localTransform = glm::mat4(1.0f); // from TRS or matrix
    int          parentIndex    = -1;              // into the returned outNodes vector, -1=root
    std::vector<int> meshDataIndices;              // which flat MeshData entries belong here
};

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
    glm::vec3 emissive;
    glm::vec2 uv{0.0f, 0.0f};
    glm::vec4 tangent{0.0f};  // xyz = tangent, w = bitangent sign (+1 or -1)
};

struct MeshData
{
    std::string name;       // material group name (from MTL usemtl)
    std::string objectName; // parent object/shape name (from OBJ o/g tag)
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::string diffuseTexturePath;
    std::string emissiveTexturePath;
    std::string normalTexturePath;
    std::string roughnessTexturePath;
    std::string metallicTexturePath;
    std::string aoTexturePath;
    glm::vec3 baseColor = {1.0f, 1.0f, 1.0f}; // albedo tint multiplied on top of vertex/texture color
    float emissiveStrength = 1.0f;             // multiplier on emissive color and emissive texture
    bool alphaClip = false;
    int materialType = 0;   // 0=Microfacet (GGX), 1=Mirror, 2=Dielectric
    float ior = 1.5f;       // index of refraction (dielectric only)
    float roughness = 0.5f;
    float metallic = 0.0f;

    static std::vector<MeshData> loadOBJ(const std::string& path);
    static std::vector<MeshData> loadGLTF(const std::string& path,
                                          std::vector<GLTFNodeInfo>& outNodes);
};

} // namespace vex
