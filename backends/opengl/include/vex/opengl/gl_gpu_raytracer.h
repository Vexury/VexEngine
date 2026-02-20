#pragma once

#include <vex/raytracing/cpu_raytracer.h>
#include <vex/raytracing/bvh.h>

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace vex
{

class GLGPURaytracer
{
public:
    bool init();
    void shutdown();

    // Geometry upload (called when scene changes)
    void uploadGeometry(const std::vector<CPURaytracer::Triangle>& triangles,
                        const BVH& bvh,
                        const std::vector<uint32_t>& lightIndices,
                        const std::vector<float>& lightCDF,
                        float totalLightArea,
                        const std::vector<CPURaytracer::TextureData>& textures);

    // Environment
    void setEnvironmentMap(const float* data, int w, int h);
    void clearEnvironmentMap();
    void setEnvironmentColor(const glm::vec3& color);

    // Per-frame state
    void setCamera(const glm::vec3& origin, const glm::mat4& inverseVP);
    void setPointLight(const glm::vec3& pos, const glm::vec3& color, bool enabled);
    void setDirectionalLight(const glm::vec3& dir, const glm::vec3& color,
                             float angularRadius, bool enabled);

    // Settings
    void setMaxDepth(int d);
    void setEnableNEE(bool v);
    void setEnableAA(bool v);
    void setEnableFireflyClamping(bool v);
    void setEnableEnvironment(bool v);
    void setEnvLightMultiplier(float v);
    void setFlatShading(bool v);
    void setEnableNormalMapping(bool v);
    void setEnableEmissive(bool v);

    void setRayEps(float v);
    float getRayEps() const { return m_rayEps; }

    void setEnableRR(bool v);
    bool getEnableRR() const { return m_enableRR; }

    // Depth of field (aperture=0 → pinhole; reset must be called externally when changed)
    void setDoF(float aperture, float focusDistance, glm::vec3 right, glm::vec3 up);

    int  getMaxDepth() const { return m_maxDepth; }
    bool getEnableNEE() const { return m_enableNEE; }
    bool getEnableAA() const { return m_enableAA; }
    bool getEnableFireflyClamping() const { return m_enableFireflyClamping; }
    bool  getEnableEnvironment() const { return m_enableEnvironment; }
    float getEnvLightMultiplier() const { return m_envLightMultiplier; }
    bool  getFlatShading() const { return m_flatShading; }
    bool getEnableNormalMapping() const { return m_enableNormalMapping; }
    bool getEnableEmissive() const { return m_enableEmissive; }

    void resize(uint32_t w, uint32_t h);
    void reset();
    void traceSample();

    // Reload the compute shader from disk (keeps old shader on failure)
    bool reloadShader();

    uint32_t getAccumTexture() const { return m_accumTexture; }
    uint32_t getSampleCount() const { return m_sampleCount; }
    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }

private:
    bool compileComputeShader(const std::string& path);
    void createAccumTexture();
    void cacheUniformLocations();

    uint32_t m_computeProgram = 0;
    uint32_t m_accumTexture = 0;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_sampleCount = 0;

    // SSBOs
    uint32_t m_bvhSSBO = 0;
    uint32_t m_triVertsSSBO = 0;    // hot: v0,v1,v2 (3 vec4s per tri)
    uint32_t m_triShadingSSBO = 0;  // cold: normals, UVs, material (9 vec4s per tri)
    uint32_t m_lightSSBO = 0;
    uint32_t m_texDataSSBO = 0;
    uint32_t m_envMapSSBO = 0;
    uint32_t m_envCdfSSBO = 0;

    // Cached uniform locations
    int32_t m_locCameraOrigin = -1;
    int32_t m_locInverseVP = -1;
    int32_t m_locSampleCount = -1;
    int32_t m_locWidth = -1;
    int32_t m_locHeight = -1;
    int32_t m_locMaxDepth = -1;
    int32_t m_locEnableNEE = -1;
    int32_t m_locEnableAA = -1;
    int32_t m_locEnableFireflyClamping = -1;
    int32_t m_locEnableEnvLighting = -1;
    int32_t m_locEnvLightMultiplier = -1;
    int32_t m_locPointLightPos = -1;
    int32_t m_locPointLightColor = -1;
    int32_t m_locPointLightEnabled = -1;
    int32_t m_locSunDir = -1;
    int32_t m_locSunColor = -1;
    int32_t m_locSunAngularRadius = -1;
    int32_t m_locSunEnabled = -1;
    int32_t m_locEnvColor = -1;
    int32_t m_locEnvMapWidth = -1;
    int32_t m_locEnvMapHeight = -1;
    int32_t m_locHasEnvMap = -1;
    int32_t m_locHasEnvCDF = -1;
    int32_t m_locFlatShading = -1;
    int32_t m_locEnableNormalMapping = -1;
    int32_t m_locEnableEmissive = -1;
    int32_t m_locTriangleCount = -1;
    int32_t m_locBvhNodeCount = -1;
    int32_t m_locRayEps = -1;
    int32_t m_locEnableRR = -1;
    int32_t m_locAperture = -1;
    int32_t m_locFocusDistance = -1;
    int32_t m_locCameraRight = -1;
    int32_t m_locCameraUp = -1;
    // Settings
    int  m_maxDepth = 5;
    bool m_enableNEE = true;
    bool m_enableAA = true;
    bool m_enableFireflyClamping = true;
    bool  m_enableEnvironment = false;
    float m_envLightMultiplier = 1.0f;
    bool  m_flatShading = false;
    bool m_enableNormalMapping = true;
    bool m_enableEmissive = true;

    // Camera
    glm::vec3 m_cameraOrigin{0.0f};
    glm::mat4 m_inverseVP{1.0f};

    // Lights
    glm::vec3 m_pointLightPos{0.0f};
    glm::vec3 m_pointLightColor{1.0f};
    bool m_pointLightEnabled = false;

    glm::vec3 m_sunDir{0.0f, -1.0f, 0.0f};
    glm::vec3 m_sunColor{1.0f};
    float m_sunAngularRadius = 0.00873f;
    bool m_sunEnabled = false;

    // Environment
    glm::vec3 m_envColor{0.0f};
    int m_envMapWidth = 0;
    int m_envMapHeight = 0;
    bool m_hasEnvMap = false;
    bool m_hasEnvCDF = false;

    float m_rayEps = 1e-4f;
    bool  m_enableRR = true;

    // Depth of field
    float     m_aperture      = 0.0f;
    float     m_focusDistance = 10.0f;
    glm::vec3 m_cameraRight   = {1.0f, 0.0f, 0.0f};
    glm::vec3 m_cameraUp      = {0.0f, 1.0f, 0.0f};

    // Geometry counts for dispatch
    uint32_t m_triangleCount = 0;
    uint32_t m_bvhNodeCount = 0;
};

} // namespace vex
