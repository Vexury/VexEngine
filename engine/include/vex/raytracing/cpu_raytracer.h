#pragma once

#include <vex/raytracing/ray.h>
#include <vex/raytracing/hit.h>
#include <vex/raytracing/bvh.h>

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace vex
{

class CPURaytracer
{
public:
    struct Triangle
    {
        glm::vec3 v0, v1, v2;
        glm::vec3 n0, n1, n2;
        glm::vec2 uv0, uv1, uv2;
        glm::vec3 color;
        glm::vec3 emissive;
        glm::vec3 geometricNormal;
        float area;
        int textureIndex = -1;         // index into texture array, -1 = none
        int emissiveTextureIndex = -1; // index into texture array, -1 = none
        int normalMapTextureIndex = -1;
        int roughnessTextureIndex = -1;
        int metallicTextureIndex = -1;
        bool alphaClip = false;
        int materialType = 0;   // 0=Diffuse, 1=Mirror, 2=Dielectric
        float ior = 1.5f;
        float roughness = 0.5f;
        float metallic = 0.0f;
        glm::vec3 tangent{1, 0, 0};
        float bitangentSign = 1.0f;
    };

    struct TextureData
    {
        std::vector<uint8_t> pixels; // RGBA, 4 bytes per pixel
        int width = 0;
        int height = 0;
    };

    void setGeometry(std::vector<Triangle> triangles, std::vector<TextureData> textures = {});
    void updateMaterials(const std::vector<Triangle>& triangles);
    void setCamera(const glm::vec3& origin, const glm::mat4& inverseVP);

    void resize(uint32_t width, uint32_t height);
    void reset();
    void traceSample();

    const std::vector<uint8_t>& getPixelBuffer() const { return m_pixelBuffer; }

    uint32_t getSampleCount() const { return m_sampleCount; }
    uint32_t getWidth()  const { return m_width; }
    uint32_t getHeight() const { return m_height; }

    // Settings (each resets accumulation when changed)
    void setMaxDepth(int depth);
    int  getMaxDepth() const { return m_maxDepth; }

    void setEnableNEE(bool v);
    bool getEnableNEE() const { return m_enableNEE; }

    void setEnableFireflyClamping(bool v);
    bool getEnableFireflyClamping() const { return m_enableFireflyClamping; }

    void setEnableAA(bool v);
    bool getEnableAA() const { return m_enableAA; }

    void setEnableEnvironment(bool v);
    bool getEnableEnvironment() const { return m_enableEnvironment; }

    void  setEnvLightMultiplier(float v);
    float getEnvLightMultiplier() const { return m_envLightMultiplier; }

    void setFlatShading(bool v);
    bool getFlatShading() const { return m_flatShading; }

    void setEnableNormalMapping(bool v);
    bool getEnableNormalMapping() const { return m_enableNormalMapping; }

    void setEnableEmissive(bool v);
    bool getEnableEmissive() const { return m_enableEmissive; }

    void setExposure(float v);
    float getExposure() const { return m_exposure; }

    void setGamma(float v);
    float getGamma() const { return m_gamma; }

    void setEnableACES(bool v);
    bool getEnableACES() const { return m_enableACES; }

    void setRayEps(float v);
    float getRayEps() const { return m_rayEps; }

    // Depth of field (resets accumulation when changed; aperture=0 → pinhole)
    void setDoF(float aperture, float focusDistance, glm::vec3 right, glm::vec3 up);

    // BVH stats
    uint32_t getBVHNodeCount() const { return m_bvh.nodeCount(); }
    size_t   getBVHMemoryBytes() const { return m_bvh.memoryBytes(); }
    AABB     getBVHRootAABB() const { return m_bvh.rootAABB(); }
    float    getBVHSAHCost() const { return m_bvh.sahCost(); }

    // Point light (caller is responsible for calling reset() after changes)
    void setPointLight(const glm::vec3& pos, const glm::vec3& color, bool enabled);

    // Directional (sun) light (caller is responsible for calling reset() after changes)
    void setDirectionalLight(const glm::vec3& direction, const glm::vec3& color,
                             float angularRadius, bool enabled);

    // Environment data (caller is responsible for calling reset() after changes)
    void setEnvironmentColor(const glm::vec3& color);
    void setEnvironmentMap(const float* data, int width, int height);
    void clearEnvironmentMap();

private:
    struct RNG
    {
        uint32_t state;
        explicit RNG(uint32_t seed) : state(seed) {}

        float next()
        {
            state = state * 747796405u + 2891336453u;
            uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
            word = (word >> 22u) ^ word;
            return static_cast<float>(word) / 4294967296.0f;
        }
    };

    static uint32_t hash(uint32_t x);

    // Hot intersection data — compact for cache-efficient BVH traversal (36 bytes)
    struct TriVerts
    {
        glm::vec3 v0, v1, v2;
    };

    // Cold shading data — only accessed on confirmed hits
    struct TriData
    {
        glm::vec3 n0, n1, n2;
        glm::vec2 uv0, uv1, uv2;
        glm::vec3 color;
        glm::vec3 emissive;
        glm::vec3 geometricNormal;
        float area;
        int textureIndex = -1;
        int emissiveTextureIndex = -1;
        int normalMapTextureIndex = -1;
        int roughnessTextureIndex = -1;
        int metallicTextureIndex = -1;
        bool alphaClip = false;
        int materialType = 0;
        float ior = 1.5f;
        float roughness = 0.5f;
        float metallic = 0.0f;
        glm::vec3 tangent{1, 0, 0};
        float bitangentSign = 1.0f;
    };

    bool intersectTriangle(const Ray& ray, const TriVerts& verts,
                           float& t, float& u, float& v) const;

    HitRecord traceRay(const Ray& ray) const;
    bool traceShadowRay(const Ray& ray, float maxDist) const;
    Ray generateRay(int x, int y, float jitterX, float jitterY, RNG& rng) const;
    glm::vec3 pathTrace(const Ray& ray, RNG& rng) const;
    glm::vec3 sampleEnvironment(const glm::vec3& direction) const;
    glm::vec4 sampleTexture(int textureIndex, const glm::vec2& uv) const;

    // Environment map importance sampling
    void buildEnvMapCDF();
    glm::vec3 sampleEnvMap(RNG& rng, glm::vec3& outDir, float& outPdf) const;
    float envMapPdf(const glm::vec3& dir) const;

    // Acceleration structure
    void buildBVH();

    // Light sampling
    void buildLightData();
    glm::vec3 sampleLightPoint(RNG& rng, uint32_t& outTriIndex) const;

    BVH m_bvh;
    std::vector<TriVerts> m_triVerts;   // hot: intersection only
    std::vector<TriData>  m_triData;    // cold: shading only
    std::vector<TextureData> m_textures;
    uint32_t m_width = 0, m_height = 0;

    std::vector<glm::vec3> m_accumBuffer;
    std::vector<uint8_t> m_pixelBuffer;
    uint32_t m_sampleCount = 0;

    glm::vec3 m_cameraOrigin{0.0f};
    glm::mat4 m_inverseVP{1.0f};

    // Settings
    int  m_maxDepth = 5;
    bool m_enableNEE = true;
    bool m_enableFireflyClamping = true;
    bool m_enableAA = true;
    bool  m_enableEnvironment = false;
    float m_envLightMultiplier = 1.0f;
    bool  m_flatShading = false;
    bool m_enableNormalMapping = true;
    bool m_enableEmissive = true;
    float m_exposure = 0.0f;
    float m_gamma = 2.2f;
    bool m_enableACES = true;
    float m_rayEps = 1e-4f;

    // Depth of field
    float      m_aperture      = 0.0f;
    float      m_focusDistance = 10.0f;
    glm::vec3  m_cameraRight   = {1.0f, 0.0f, 0.0f};
    glm::vec3  m_cameraUp      = {0.0f, 1.0f, 0.0f};

    // Point light
    glm::vec3 m_pointLightPos{0.0f};
    glm::vec3 m_pointLightColor{1.0f};
    bool m_pointLightEnabled = false;

    // Directional (sun) light
    glm::vec3 m_sunDir{0.0f, -1.0f, 0.0f};
    glm::vec3 m_sunColor{1.0f};
    float m_sunAngularRadius = 0.00873f; // ~0.5 degrees
    float m_sunCosAngle = 0.99996f;      // cos(0.00873)
    bool m_sunEnabled = false;

    // Environment
    glm::vec3 m_envColor{0.0f};
    std::vector<float> m_envMapPixels;
    int m_envMapWidth = 0;
    int m_envMapHeight = 0;
    bool m_hasEnvMap = false;

    // Environment map CDF for importance sampling
    std::vector<float> m_envCondCDF;    // per-row conditional CDF [width * height]
    std::vector<float> m_envMarginalCDF; // row marginal CDF [height]
    float m_envTotalIntegral = 0.0f;

    // Light data (emissive triangles)
    std::vector<uint32_t> m_lightIndices;
    std::vector<float> m_lightCDF;
    float m_totalLightArea = 0.0f;
};

} // namespace vex
