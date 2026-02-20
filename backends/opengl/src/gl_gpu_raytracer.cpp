#include <vex/opengl/gl_gpu_raytracer.h>
#include <vex/core/log.h>

#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring>

namespace vex
{

bool GLGPURaytracer::compileComputeShader(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        Log::error("Failed to open compute shader: " + path);
        return false;
    }

    std::stringstream ss;
    ss << file.rdbuf();
    std::string source = ss.str();

    uint32_t shader = glCreateShader(GL_COMPUTE_SHADER);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint result;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE)
    {
        GLint length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string infoLog(static_cast<size_t>(length), '\0');
        glGetShaderInfoLog(shader, length, &length, infoLog.data());
        Log::error("Compute shader compile error: " + infoLog);
        glDeleteShader(shader);
        return false;
    }

    m_computeProgram = glCreateProgram();
    glAttachShader(m_computeProgram, shader);
    glLinkProgram(m_computeProgram);

    glGetProgramiv(m_computeProgram, GL_LINK_STATUS, &result);
    if (result == GL_FALSE)
    {
        GLint length;
        glGetProgramiv(m_computeProgram, GL_INFO_LOG_LENGTH, &length);
        std::string infoLog(static_cast<size_t>(length), '\0');
        glGetProgramInfoLog(m_computeProgram, length, &length, infoLog.data());
        Log::error("Compute shader link error: " + infoLog);
        glDeleteProgram(m_computeProgram);
        m_computeProgram = 0;
        glDeleteShader(shader);
        return false;
    }

    glDeleteShader(shader);
    return true;
}

bool GLGPURaytracer::reloadShader()
{
    uint32_t oldProgram = m_computeProgram;
    m_computeProgram = 0; // prevent compileComputeShader from leaking the old handle

    if (!compileComputeShader("shaders/opengl/pathtracer.comp"))
    {
        // Restore old program so rendering continues unchanged
        m_computeProgram = oldProgram;
        Log::error("Shader reload failed, keeping old shader.");
        return false;
    }

    if (oldProgram)
        glDeleteProgram(oldProgram);

    cacheUniformLocations();
    reset();
    Log::info("GPU shader reloaded successfully.");
    return true;
}

void GLGPURaytracer::createAccumTexture()
{
    if (m_accumTexture)
        glDeleteTextures(1, &m_accumTexture);

    glGenTextures(1, &m_accumTexture);
    glBindTexture(GL_TEXTURE_2D, m_accumTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, static_cast<GLsizei>(m_width),
                 static_cast<GLsizei>(m_height), 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void GLGPURaytracer::cacheUniformLocations()
{
    auto loc = [this](const char* name) { return glGetUniformLocation(m_computeProgram, name); };

    m_locCameraOrigin        = loc("u_cameraOrigin");
    m_locInverseVP           = loc("u_inverseVP");
    m_locSampleCount         = loc("u_sampleCount");
    m_locWidth               = loc("u_width");
    m_locHeight              = loc("u_height");
    m_locMaxDepth            = loc("u_maxDepth");
    m_locEnableNEE           = loc("u_enableNEE");
    m_locEnableAA            = loc("u_enableAA");
    m_locEnableFireflyClamping = loc("u_enableFireflyClamping");
    m_locEnableEnvLighting   = loc("u_enableEnvLighting");
    m_locEnvLightMultiplier  = loc("u_envLightMultiplier");
    m_locPointLightPos       = loc("u_pointLightPos");
    m_locPointLightColor     = loc("u_pointLightColor");
    m_locPointLightEnabled   = loc("u_pointLightEnabled");
    m_locSunDir              = loc("u_sunDir");
    m_locSunColor            = loc("u_sunColor");
    m_locSunAngularRadius    = loc("u_sunAngularRadius");
    m_locSunEnabled          = loc("u_sunEnabled");
    m_locEnvColor            = loc("u_envColor");
    m_locEnvMapWidth         = loc("u_envMapWidth");
    m_locEnvMapHeight        = loc("u_envMapHeight");
    m_locHasEnvMap           = loc("u_hasEnvMap");
    m_locHasEnvCDF           = loc("u_hasEnvCDF");
    m_locFlatShading         = loc("u_flatShading");
    m_locEnableNormalMapping = loc("u_enableNormalMapping");
    m_locEnableEmissive      = loc("u_enableEmissive");
    m_locTriangleCount       = loc("u_triangleCount");
    m_locBvhNodeCount        = loc("u_bvhNodeCount");
    m_locRayEps              = loc("u_rayEps");
    m_locEnableRR            = loc("u_enableRR");
    m_locAperture            = loc("u_aperture");
    m_locFocusDistance       = loc("u_focusDistance");
    m_locCameraRight         = loc("u_cameraRight");
    m_locCameraUp            = loc("u_cameraUp");
}

bool GLGPURaytracer::init()
{
    if (!compileComputeShader("shaders/opengl/pathtracer.comp"))
        return false;

    cacheUniformLocations();

    // Create SSBOs
    glGenBuffers(1, &m_bvhSSBO);
    glGenBuffers(1, &m_triVertsSSBO);
    glGenBuffers(1, &m_triShadingSSBO);
    glGenBuffers(1, &m_lightSSBO);
    glGenBuffers(1, &m_texDataSSBO);
    glGenBuffers(1, &m_envMapSSBO);
    glGenBuffers(1, &m_envCdfSSBO);

    // Initialize light SSBO with empty data
    struct { uint32_t count; float area; uint32_t pad0; uint32_t pad1; } emptyLight = {0, 0.0f, 0, 0};
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_lightSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(emptyLight), &emptyLight, GL_DYNAMIC_DRAW);

    // Initialize texture SSBO with zero textures
    uint32_t zeroTexCount = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_texDataSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(uint32_t), &zeroTexCount, GL_DYNAMIC_DRAW);

    // Initialize env map SSBO with minimal data
    float emptyEnv = 0.0f;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_envMapSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float), &emptyEnv, GL_DYNAMIC_DRAW);

    // Initialize env CDF SSBO with minimal data
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_envCdfSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float), &emptyEnv, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    Log::info("GPU Raytracer initialized");
    return true;
}

void GLGPURaytracer::shutdown()
{
    if (m_accumTexture)    { glDeleteTextures(1, &m_accumTexture);  m_accumTexture = 0; }
    if (m_bvhSSBO)        { glDeleteBuffers(1, &m_bvhSSBO);        m_bvhSSBO = 0; }
    if (m_triVertsSSBO)   { glDeleteBuffers(1, &m_triVertsSSBO);   m_triVertsSSBO = 0; }
    if (m_triShadingSSBO) { glDeleteBuffers(1, &m_triShadingSSBO); m_triShadingSSBO = 0; }
    if (m_lightSSBO)      { glDeleteBuffers(1, &m_lightSSBO);      m_lightSSBO = 0; }
    if (m_texDataSSBO)    { glDeleteBuffers(1, &m_texDataSSBO);    m_texDataSSBO = 0; }
    if (m_envMapSSBO)     { glDeleteBuffers(1, &m_envMapSSBO);     m_envMapSSBO = 0; }
    if (m_envCdfSSBO)     { glDeleteBuffers(1, &m_envCdfSSBO);     m_envCdfSSBO = 0; }
    if (m_computeProgram) { glDeleteProgram(m_computeProgram);     m_computeProgram = 0; }
}

void GLGPURaytracer::uploadGeometry(
    const std::vector<CPURaytracer::Triangle>& triangles,
    const BVH& bvh,
    const std::vector<uint32_t>& lightIndices,
    const std::vector<float>& lightCDF,
    float totalLightArea,
    const std::vector<CPURaytracer::TextureData>& textures)
{
    m_triangleCount = static_cast<uint32_t>(triangles.size());
    m_bvhNodeCount  = bvh.nodeCount();

    // ── Upload BVH nodes ───────────────────────────────────────────
    // GPU layout: vec3 boundsMin, uint leftFirst, vec3 boundsMax, uint triCount (32 bytes)
    struct GPUBVHNode {
        float minX, minY, minZ;
        uint32_t leftFirst;
        float maxX, maxY, maxZ;
        uint32_t triCount;
    };

    const auto& nodes = bvh.nodes();
    std::vector<GPUBVHNode> gpuNodes(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        gpuNodes[i].minX = nodes[i].bounds.min.x;
        gpuNodes[i].minY = nodes[i].bounds.min.y;
        gpuNodes[i].minZ = nodes[i].bounds.min.z;
        gpuNodes[i].leftFirst = nodes[i].leftFirst;
        gpuNodes[i].maxX = nodes[i].bounds.max.x;
        gpuNodes[i].maxY = nodes[i].bounds.max.y;
        gpuNodes[i].maxZ = nodes[i].bounds.max.z;
        gpuNodes[i].triCount = nodes[i].triCount;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_bvhSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 static_cast<GLsizeiptr>(gpuNodes.size() * sizeof(GPUBVHNode)),
                 gpuNodes.data(), GL_STATIC_DRAW);

    // ── Upload triangle vertices (hot — intersection only) ──────────
    // 3 vec4s per triangle (48 bytes): v0+pad, v1+pad, v2+pad
    std::vector<float> vertsBuffer(triangles.size() * 12); // 3 vec4s * 4 floats
    for (size_t i = 0; i < triangles.size(); ++i)
    {
        const auto& tri = triangles[i];
        float* p = &vertsBuffer[i * 12];
        p[0]  = tri.v0.x; p[1]  = tri.v0.y; p[2]  = tri.v0.z; p[3]  = 0.0f;
        p[4]  = tri.v1.x; p[5]  = tri.v1.y; p[6]  = tri.v1.z; p[7]  = 0.0f;
        p[8]  = tri.v2.x; p[9]  = tri.v2.y; p[10] = tri.v2.z; p[11] = 0.0f;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_triVertsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 static_cast<GLsizeiptr>(vertsBuffer.size() * sizeof(float)),
                 vertsBuffer.data(), GL_STATIC_DRAW);

    // ── Upload triangle shading data (cold — only on confirmed hits) ─
    // 10 vec4s per triangle (160 bytes)
    std::vector<float> shadingBuffer(triangles.size() * 40); // 10 vec4s * 4 floats
    for (size_t i = 0; i < triangles.size(); ++i)
    {
        const auto& tri = triangles[i];
        float* p = &shadingBuffer[i * 40];
        // vec4 0: n0 + roughnessTextureIndex (int bits)
        p[0]  = tri.n0.x; p[1]  = tri.n0.y; p[2]  = tri.n0.z;
        { uint32_t bits; std::memcpy(&bits, &tri.roughnessTextureIndex, sizeof(int)); std::memcpy(&p[3], &bits, sizeof(float)); }
        // vec4 1: n1 + metallicTextureIndex (int bits)
        p[4]  = tri.n1.x; p[5]  = tri.n1.y; p[6]  = tri.n1.z;
        { uint32_t bits; std::memcpy(&bits, &tri.metallicTextureIndex, sizeof(int)); std::memcpy(&p[7], &bits, sizeof(float)); }
        // vec4 2: n2 + pad
        p[8]  = tri.n2.x; p[9]  = tri.n2.y; p[10] = tri.n2.z; p[11] = 0.0f;
        // vec4 3: uv0.xy, uv1.xy
        p[12] = tri.uv0.x; p[13] = tri.uv0.y; p[14] = tri.uv1.x; p[15] = tri.uv1.y;
        // vec4 4: uv2.xy, roughness, metallic
        p[16] = tri.uv2.x; p[17] = tri.uv2.y; p[18] = tri.roughness; p[19] = tri.metallic;
        // vec4 5: color + texIndex (as int bits)
        p[20] = tri.color.x; p[21] = tri.color.y; p[22] = tri.color.z;
        uint32_t texBits;
        std::memcpy(&texBits, &tri.textureIndex, sizeof(int));
        std::memcpy(&p[23], &texBits, sizeof(float));
        // vec4 6: emissive + area
        p[24] = tri.emissive.x; p[25] = tri.emissive.y; p[26] = tri.emissive.z; p[27] = tri.area;
        // vec4 7: geometricNormal + normalMapTexIndex (as int bits)
        p[28] = tri.geometricNormal.x; p[29] = tri.geometricNormal.y; p[30] = tri.geometricNormal.z;
        uint32_t normalTexBits;
        std::memcpy(&normalTexBits, &tri.normalMapTextureIndex, sizeof(int));
        std::memcpy(&p[31], &normalTexBits, sizeof(float));
        // vec4 8: alphaClip, materialType (as float), ior, emissiveTexIndex (as int bits)
        p[32] = tri.alphaClip ? 1.0f : 0.0f;
        p[33] = static_cast<float>(tri.materialType);
        p[34] = tri.ior;
        uint32_t emissiveTexBits;
        std::memcpy(&emissiveTexBits, &tri.emissiveTextureIndex, sizeof(int));
        std::memcpy(&p[35], &emissiveTexBits, sizeof(float));
        // vec4 9: tangent.xyz + bitangentSign
        p[36] = tri.tangent.x; p[37] = tri.tangent.y; p[38] = tri.tangent.z; p[39] = tri.bitangentSign;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_triShadingSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 static_cast<GLsizeiptr>(shadingBuffer.size() * sizeof(float)),
                 shadingBuffer.data(), GL_STATIC_DRAW);

    // ── Upload light data ──────────────────────────────────────────
    // Header: lightCount (uint), totalLightArea (float), pad, pad
    // Then: lightIndices[lightCount], lightCDF[lightCount] (as uint bits)
    uint32_t lc = static_cast<uint32_t>(lightIndices.size());
    size_t lightBufSize = 4 + lc * 2; // header (4 uints) + indices + CDF
    std::vector<uint32_t> lightBuf(lightBufSize);
    lightBuf[0] = lc;
    float totalArea = totalLightArea;
    std::memcpy(&lightBuf[1], &totalArea, sizeof(float));
    lightBuf[2] = 0;
    lightBuf[3] = 0;

    for (uint32_t i = 0; i < lc; ++i)
        lightBuf[4 + i] = lightIndices[i];

    for (uint32_t i = 0; i < lc; ++i)
    {
        float cdf = lightCDF[i];
        std::memcpy(&lightBuf[4 + lc + i], &cdf, sizeof(float));
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_lightSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 static_cast<GLsizeiptr>(lightBuf.size() * sizeof(uint32_t)),
                 lightBuf.data(), GL_STATIC_DRAW);

    // ── Upload texture data ────────────────────────────────────────
    // Header: [0] = texCount
    //         [1+i*4+0] = pixelOffset (in uint index into this same buffer)
    //         [1+i*4+1] = width
    //         [1+i*4+2] = height
    //         [1+i*4+3] = pad
    // Then RGBA pixel data: 4 bytes per pixel = 1 uint per pixel
    uint32_t texCount = static_cast<uint32_t>(textures.size());
    uint32_t headerSize = 1 + texCount * 4;

    // Calculate total pixel data size in uints (1 uint per pixel for RGBA)
    uint32_t totalPixelUints = 0;
    for (const auto& tex : textures)
        totalPixelUints += static_cast<uint32_t>(tex.width) * static_cast<uint32_t>(tex.height);

    std::vector<uint32_t> texBuf(headerSize + totalPixelUints, 0);
    texBuf[0] = texCount;

    uint32_t currentPixelOffset = headerSize;

    for (uint32_t i = 0; i < texCount; ++i)
    {
        const auto& tex = textures[i];
        uint32_t pixelCount = static_cast<uint32_t>(tex.width) * static_cast<uint32_t>(tex.height);
        texBuf[1 + i * 4 + 0] = currentPixelOffset;
        texBuf[1 + i * 4 + 1] = static_cast<uint32_t>(tex.width);
        texBuf[1 + i * 4 + 2] = static_cast<uint32_t>(tex.height);
        texBuf[1 + i * 4 + 3] = 0;

        // Copy RGBA pixels — 4 bytes per pixel maps directly to 1 uint per pixel
        std::memcpy(&texBuf[currentPixelOffset], tex.pixels.data(), pixelCount * 4);

        currentPixelOffset += pixelCount;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_texDataSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 static_cast<GLsizeiptr>(texBuf.size() * sizeof(uint32_t)),
                 texBuf.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    reset();
}

void GLGPURaytracer::setEnvironmentMap(const float* data, int w, int h)
{
    m_envMapWidth  = w;
    m_envMapHeight = h;
    m_hasEnvMap    = true;

    size_t size = static_cast<size_t>(w) * h * 3;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_envMapSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(size * sizeof(float)),
                 data, GL_STATIC_DRAW);

    // Build CDF on CPU
    const float PI_VAL = 3.14159265358979323846f;
    std::vector<float> condCDF(static_cast<size_t>(w) * h);
    std::vector<float> marginalCDF(h);
    float totalIntegral = 0.0f;

    for (int y = 0; y < h; ++y)
    {
        float sinTheta = std::sin(PI_VAL * (static_cast<float>(y) + 0.5f) / static_cast<float>(h));
        float rowSum = 0.0f;

        for (int x = 0; x < w; ++x)
        {
            int idx = (y * w + x) * 3;
            float r = data[idx], g = data[idx + 1], b = data[idx + 2];
            float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            rowSum += lum * sinTheta;
            condCDF[y * w + x] = rowSum;
        }

        if (rowSum > 0.0f)
        {
            for (int x = 0; x < w; ++x)
                condCDF[y * w + x] /= rowSum;
        }
        else
        {
            for (int x = 0; x < w; ++x)
                condCDF[y * w + x] = static_cast<float>(x + 1) / static_cast<float>(w);
        }

        totalIntegral += rowSum;
        marginalCDF[y] = totalIntegral;
    }

    if (totalIntegral > 0.0f)
    {
        for (int y = 0; y < h; ++y)
            marginalCDF[y] /= totalIntegral;
    }

    // Upload CDF to SSBO: [marginalCDF: H floats][condCDF: W*H floats][totalIntegral: 1 float]
    size_t cdfSize = static_cast<size_t>(h) + static_cast<size_t>(w) * h + 1;
    std::vector<float> cdfBuf(cdfSize);
    std::memcpy(&cdfBuf[0], marginalCDF.data(), h * sizeof(float));
    std::memcpy(&cdfBuf[h], condCDF.data(), static_cast<size_t>(w) * h * sizeof(float));
    cdfBuf[h + static_cast<size_t>(w) * h] = totalIntegral;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_envCdfSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(cdfSize * sizeof(float)),
                 cdfBuf.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    m_hasEnvCDF = (totalIntegral > 0.0f);
}

void GLGPURaytracer::clearEnvironmentMap()
{
    m_envMapWidth = 0;
    m_envMapHeight = 0;
    m_hasEnvMap = false;
    m_hasEnvCDF = false;

    float emptyEnv = 0.0f;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_envMapSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float), &emptyEnv, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_envCdfSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float), &emptyEnv, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void GLGPURaytracer::setEnvironmentColor(const glm::vec3& color)
{
    m_envColor = color;
}

void GLGPURaytracer::setCamera(const glm::vec3& origin, const glm::mat4& inverseVP)
{
    m_cameraOrigin = origin;
    m_inverseVP = inverseVP;
}

void GLGPURaytracer::setPointLight(const glm::vec3& pos, const glm::vec3& color, bool enabled)
{
    m_pointLightPos = pos;
    m_pointLightColor = color;
    m_pointLightEnabled = enabled;
}

void GLGPURaytracer::setDirectionalLight(const glm::vec3& dir, const glm::vec3& color,
                                          float angularRadius, bool enabled)
{
    m_sunDir = glm::normalize(dir);
    m_sunColor = color;
    m_sunAngularRadius = angularRadius;
    m_sunEnabled = enabled;
}

void GLGPURaytracer::setRayEps(float v)
{
    if (m_rayEps == v) return;
    m_rayEps = v;
    reset();
}

void GLGPURaytracer::setEnableRR(bool v)
{
    if (m_enableRR == v) return;
    m_enableRR = v;
    reset();
}

void GLGPURaytracer::setDoF(float aperture, float focusDistance, glm::vec3 right, glm::vec3 up)
{
    m_aperture      = aperture;
    m_focusDistance = focusDistance;
    m_cameraRight   = right;
    m_cameraUp      = up;
}

void GLGPURaytracer::setMaxDepth(int d)
{
    if (m_maxDepth == d) return;
    m_maxDepth = d;
    reset();
}

void GLGPURaytracer::setEnableNEE(bool v)
{
    if (m_enableNEE == v) return;
    m_enableNEE = v;
    reset();
}

void GLGPURaytracer::setEnableAA(bool v)
{
    if (m_enableAA == v) return;
    m_enableAA = v;
    reset();
}

void GLGPURaytracer::setEnableFireflyClamping(bool v)
{
    if (m_enableFireflyClamping == v) return;
    m_enableFireflyClamping = v;
    reset();
}

void GLGPURaytracer::setEnableEnvironment(bool v)
{
    if (m_enableEnvironment == v) return;
    m_enableEnvironment = v;
    reset();
}

void GLGPURaytracer::setEnvLightMultiplier(float v)
{
    if (m_envLightMultiplier == v) return;
    m_envLightMultiplier = v;
    reset();
}

void GLGPURaytracer::setFlatShading(bool v)
{
    if (m_flatShading == v) return;
    m_flatShading = v;
    reset();
}

void GLGPURaytracer::setEnableNormalMapping(bool v)
{
    if (m_enableNormalMapping == v) return;
    m_enableNormalMapping = v;
    reset();
}

void GLGPURaytracer::setEnableEmissive(bool v)
{
    if (m_enableEmissive == v) return;
    m_enableEmissive = v;
    reset();
}

void GLGPURaytracer::resize(uint32_t w, uint32_t h)
{
    if (m_width == w && m_height == h)
        return;

    m_width = w;
    m_height = h;
    createAccumTexture();
    reset();
}

void GLGPURaytracer::reset()
{
    m_sampleCount = 0;

    if (m_accumTexture && m_width > 0 && m_height > 0)
    {
        // Clear accumulation texture to zero using glClearTexImage (GL 4.4+)
        float zeros[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        glClearTexImage(m_accumTexture, 0, GL_RGBA, GL_FLOAT, zeros);
    }
}

void GLGPURaytracer::traceSample()
{
    if (m_width == 0 || m_height == 0 || m_computeProgram == 0)
        return;

    glUseProgram(m_computeProgram);

    // Bind accumulation image
    glBindImageTexture(0, m_accumTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

    // Bind SSBOs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_bvhSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_triVertsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_lightSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_texDataSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_envMapSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_triShadingSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_envCdfSSBO);

    // Set uniforms (cached locations)
    glUniform3fv(m_locCameraOrigin, 1, glm::value_ptr(m_cameraOrigin));
    glUniformMatrix4fv(m_locInverseVP, 1, GL_FALSE, glm::value_ptr(m_inverseVP));
    glUniform1ui(m_locSampleCount, m_sampleCount);
    glUniform1ui(m_locWidth, m_width);
    glUniform1ui(m_locHeight, m_height);
    glUniform1i(m_locMaxDepth, m_maxDepth);
    glUniform1i(m_locEnableNEE, m_enableNEE ? 1 : 0);
    glUniform1i(m_locEnableAA, m_enableAA ? 1 : 0);
    glUniform1i(m_locEnableFireflyClamping, m_enableFireflyClamping ? 1 : 0);
    glUniform1i(m_locEnableEnvLighting, m_enableEnvironment ? 1 : 0);
    glUniform1f(m_locEnvLightMultiplier, m_envLightMultiplier);
    glUniform1i(m_locFlatShading, m_flatShading ? 1 : 0);
    glUniform1i(m_locEnableNormalMapping, m_enableNormalMapping ? 1 : 0);
    glUniform1i(m_locEnableEmissive, m_enableEmissive ? 1 : 0);

    glUniform3fv(m_locPointLightPos, 1, glm::value_ptr(m_pointLightPos));
    glUniform3fv(m_locPointLightColor, 1, glm::value_ptr(m_pointLightColor));
    glUniform1i(m_locPointLightEnabled, m_pointLightEnabled ? 1 : 0);

    glUniform3fv(m_locSunDir, 1, glm::value_ptr(m_sunDir));
    glUniform3fv(m_locSunColor, 1, glm::value_ptr(m_sunColor));
    glUniform1f(m_locSunAngularRadius, m_sunAngularRadius);
    glUniform1i(m_locSunEnabled, m_sunEnabled ? 1 : 0);

    glUniform3fv(m_locEnvColor, 1, glm::value_ptr(m_envColor));
    glUniform1i(m_locEnvMapWidth, m_envMapWidth);
    glUniform1i(m_locEnvMapHeight, m_envMapHeight);
    glUniform1i(m_locHasEnvMap, m_hasEnvMap ? 1 : 0);
    glUniform1i(m_locHasEnvCDF, m_hasEnvCDF ? 1 : 0);

    glUniform1ui(m_locTriangleCount, m_triangleCount);
    glUniform1ui(m_locBvhNodeCount, m_bvhNodeCount);
    glUniform1f(m_locRayEps, m_rayEps);
    glUniform1i(m_locEnableRR, m_enableRR ? 1 : 0);
    glUniform1f(m_locAperture, m_aperture);
    glUniform1f(m_locFocusDistance, m_focusDistance);
    glUniform3fv(m_locCameraRight, 1, glm::value_ptr(m_cameraRight));
    glUniform3fv(m_locCameraUp, 1, glm::value_ptr(m_cameraUp));
    // Dispatch compute shader
    uint32_t groupsX = (m_width  + 7) / 8;
    uint32_t groupsY = (m_height + 7) / 8;
    glDispatchCompute(groupsX, groupsY, 1);

    // Memory barrier for image access
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    glUseProgram(0);

    ++m_sampleCount;
}

} // namespace vex
