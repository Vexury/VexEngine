#pragma once

#include <cfloat>
#include <cstdint>
#include <glm/glm.hpp>

namespace vex
{

struct HitRecord
{
    float t = FLT_MAX;
    glm::vec3 position;
    glm::vec3 normal;           // interpolated shading normal
    glm::vec3 geometricNormal;  // face normal (for light pdf)
    glm::vec3 color;
    glm::vec3 emissive;
    glm::vec2 uv;
    int textureIndex = -1;
    int emissiveTextureIndex = -1;
    int normalMapTextureIndex = -1;
    int roughnessTextureIndex = -1;
    int metallicTextureIndex = -1;
    uint32_t triangleIndex = UINT32_MAX;
    int materialType = 0;
    float ior = 1.5f;
    float roughness = 0.5f;
    float metallic = 0.0f;
    glm::vec3 tangent{1, 0, 0};
    float bitangentSign = 1.0f;
    bool hit = false;
};

} // namespace vex
