#include <vex/raytracing/cpu_raytracer.h>
#include <vex/raytracing/bsdf.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

namespace vex
{

// --- Utility ---

uint32_t CPURaytracer::hash(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    return x;
}

// --- Setup ---

void CPURaytracer::setGeometry(std::vector<Triangle> triangles, std::vector<TextureData> textures)
{
    // Split into hot (intersection) and cold (shading) arrays
    size_t count = triangles.size();
    m_triVerts.resize(count);
    m_triData.resize(count);
    for (size_t i = 0; i < count; ++i)
    {
        const auto& tri = triangles[i];
        m_triVerts[i] = { tri.v0, tri.v1, tri.v2 };
        m_triData[i]  = { tri.n0, tri.n1, tri.n2,
                          tri.uv0, tri.uv1, tri.uv2,
                          tri.color, tri.emissive, tri.geometricNormal,
                          tri.area, tri.textureIndex, tri.emissiveTextureIndex,
                          tri.normalMapTextureIndex,
                          tri.roughnessTextureIndex, tri.metallicTextureIndex,
                          tri.alphaClip, tri.materialType, tri.ior,
                          tri.roughness, tri.metallic,
                          tri.tangent, tri.bitangentSign };
    }
    m_textures = std::move(textures);
    buildBVH();
    buildLightData();
    reset();
}

void CPURaytracer::updateMaterials(const std::vector<Triangle>& triangles)
{
    for (size_t i = 0; i < triangles.size() && i < m_triData.size(); ++i)
    {
        m_triData[i].materialType = triangles[i].materialType;
        m_triData[i].ior          = triangles[i].ior;
        m_triData[i].roughness    = triangles[i].roughness;
        m_triData[i].metallic     = triangles[i].metallic;
    }
    reset();
}

void CPURaytracer::buildBVH()
{
    uint32_t count = static_cast<uint32_t>(m_triVerts.size());

    // Compute per-triangle AABBs from the compact vertex array
    std::vector<AABB> triBounds(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        triBounds[i].grow(m_triVerts[i].v0);
        triBounds[i].grow(m_triVerts[i].v1);
        triBounds[i].grow(m_triVerts[i].v2);
    }

    m_bvh.build(triBounds);

    // Reorder both arrays to match BVH spatial ordering so leaf nodes
    // can reference contiguous ranges directly (better cache coherency).
    const auto& indices = m_bvh.indices();
    std::vector<TriVerts> reorderedVerts(count);
    std::vector<TriData>  reorderedData(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        reorderedVerts[i] = m_triVerts[indices[i]];
        reorderedData[i]  = m_triData[indices[i]];
    }
    m_triVerts = std::move(reorderedVerts);
    m_triData  = std::move(reorderedData);
}

void CPURaytracer::setCamera(const glm::vec3& origin, const glm::mat4& inverseVP)
{
    m_cameraOrigin = origin;
    m_inverseVP = inverseVP;
}

void CPURaytracer::resize(uint32_t width, uint32_t height)
{
    if (m_width == width && m_height == height)
        return;

    m_width = width;
    m_height = height;
    m_accumBuffer.assign(width * height, glm::vec3(0.0f));
    m_pixelBuffer.assign(width * height * 4, 0);
    m_sampleCount = 0;
}

void CPURaytracer::reset()
{
    std::fill(m_accumBuffer.begin(), m_accumBuffer.end(), glm::vec3(0.0f));
    std::fill(m_pixelBuffer.begin(), m_pixelBuffer.end(), uint8_t(0));
    m_sampleCount = 0;
}

// --- Settings (auto-reset on change) ---

void CPURaytracer::setMaxDepth(int depth)
{
    if (m_maxDepth == depth) return;
    m_maxDepth = depth;
    reset();
}

void CPURaytracer::setEnableNEE(bool v)
{
    if (m_enableNEE == v) return;
    m_enableNEE = v;
    reset();
}

void CPURaytracer::setEnableFireflyClamping(bool v)
{
    if (m_enableFireflyClamping == v) return;
    m_enableFireflyClamping = v;
    reset();
}

void CPURaytracer::setEnableAA(bool v)
{
    if (m_enableAA == v) return;
    m_enableAA = v;
    reset();
}

void CPURaytracer::setEnableEnvironment(bool v)
{
    if (m_enableEnvironment == v) return;
    m_enableEnvironment = v;
    reset();
}

void CPURaytracer::setEnvLightMultiplier(float v)
{
    if (m_envLightMultiplier == v) return;
    m_envLightMultiplier = v;
    reset();
}

void CPURaytracer::setFlatShading(bool v)
{
    if (m_flatShading == v) return;
    m_flatShading = v;
    reset();
}

void CPURaytracer::setEnableNormalMapping(bool v)
{
    if (m_enableNormalMapping == v) return;
    m_enableNormalMapping = v;
    reset();
}

void CPURaytracer::setEnableEmissive(bool v)
{
    if (m_enableEmissive == v) return;
    m_enableEmissive = v;
    reset();
}

void CPURaytracer::setExposure(float v)  { m_exposure = v; }
void CPURaytracer::setGamma(float v)     { m_gamma = v; }
void CPURaytracer::setEnableACES(bool v) { m_enableACES = v; }

void CPURaytracer::setRayEps(float v)
{
    if (m_rayEps == v) return;
    m_rayEps = v;
    reset();
}

void CPURaytracer::setEnableRR(bool v)
{
    if (m_enableRR == v) return;
    m_enableRR = v;
    reset();
}

void CPURaytracer::setDoF(float aperture, float focusDistance, glm::vec3 right, glm::vec3 up)
{
    if (m_aperture == aperture && m_focusDistance == focusDistance &&
        m_cameraRight == right && m_cameraUp == up)
        return;
    m_aperture      = aperture;
    m_focusDistance = focusDistance;
    m_cameraRight   = right;
    m_cameraUp      = up;
    reset();
}

// --- Point light (caller resets) ---

void CPURaytracer::setPointLight(const glm::vec3& pos, const glm::vec3& color, bool enabled)
{
    m_pointLightPos = pos;
    m_pointLightColor = color;
    m_pointLightEnabled = enabled;
}

// --- Directional light (caller resets) ---

void CPURaytracer::setDirectionalLight(const glm::vec3& direction, const glm::vec3& color,
                                       float angularRadius, bool enabled)
{
    m_sunDir = glm::normalize(direction);
    m_sunColor = color;
    m_sunAngularRadius = angularRadius;
    m_sunCosAngle = std::cos(angularRadius);
    m_sunEnabled = enabled;
}

// --- Environment (caller resets) ---

void CPURaytracer::setEnvironmentColor(const glm::vec3& color)
{
    m_envColor = color;
}

void CPURaytracer::setEnvironmentMap(const float* data, int width, int height)
{
    m_envMapWidth = width;
    m_envMapHeight = height;
    size_t size = static_cast<size_t>(width) * height * 3;
    m_envMapPixels.assign(data, data + size);
    m_hasEnvMap = true;
    buildEnvMapCDF();
}

void CPURaytracer::clearEnvironmentMap()
{
    m_envMapPixels.clear();
    m_envMapWidth = 0;
    m_envMapHeight = 0;
    m_hasEnvMap = false;
    m_envCondCDF.clear();
    m_envMarginalCDF.clear();
    m_envTotalIntegral = 0.0f;
}

void CPURaytracer::buildEnvMapCDF()
{
    int W = m_envMapWidth;
    int H = m_envMapHeight;
    m_envCondCDF.resize(static_cast<size_t>(W) * H);
    m_envMarginalCDF.resize(H);
    m_envTotalIntegral = 0.0f;

    for (int y = 0; y < H; ++y)
    {
        float sinTheta = std::sin(PI * (static_cast<float>(y) + 0.5f) / static_cast<float>(H));
        float rowSum = 0.0f;

        for (int x = 0; x < W; ++x)
        {
            int idx = (y * W + x) * 3;
            float r = m_envMapPixels[idx];
            float g = m_envMapPixels[idx + 1];
            float b = m_envMapPixels[idx + 2];
            float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            float weight = lum * sinTheta;
            rowSum += weight;
            m_envCondCDF[y * W + x] = rowSum;
        }

        // Normalize conditional CDF for this row
        if (rowSum > 0.0f)
        {
            for (int x = 0; x < W; ++x)
                m_envCondCDF[y * W + x] /= rowSum;
        }
        else
        {
            // Uniform distribution for zero-energy rows
            for (int x = 0; x < W; ++x)
                m_envCondCDF[y * W + x] = static_cast<float>(x + 1) / static_cast<float>(W);
        }

        m_envTotalIntegral += rowSum;
        m_envMarginalCDF[y] = m_envTotalIntegral;
    }

    // Normalize marginal CDF
    if (m_envTotalIntegral > 0.0f)
    {
        for (int y = 0; y < H; ++y)
            m_envMarginalCDF[y] /= m_envTotalIntegral;
    }
}

glm::vec3 CPURaytracer::sampleEnvMap(RNG& rng, glm::vec3& outDir, float& outPdf) const
{
    int W = m_envMapWidth;
    int H = m_envMapHeight;

    // Sample row via marginal CDF
    float u1 = rng.next();
    auto rowIt = std::lower_bound(m_envMarginalCDF.begin(), m_envMarginalCDF.end(), u1);
    int row = static_cast<int>(std::distance(m_envMarginalCDF.begin(), rowIt));
    if (row >= H) row = H - 1;

    // Sample column via conditional CDF for this row
    float u2 = rng.next();
    auto colBegin = m_envCondCDF.begin() + row * W;
    auto colIt = std::lower_bound(colBegin, colBegin + W, u2);
    int col = static_cast<int>(std::distance(colBegin, colIt));
    if (col >= W) col = W - 1;

    // Convert pixel to texture coordinates (center of pixel)
    float texU = (static_cast<float>(col) + 0.5f) / static_cast<float>(W);
    float texV = (static_cast<float>(row) + 0.5f) / static_cast<float>(H);

    // Convert to direction (consistent with sampleEnvironment mapping)
    float phi = (texU - 0.5f) * 2.0f * PI;
    float theta = texV * PI;
    float sinTheta = std::sin(theta);
    float cosTheta = std::cos(theta);

    outDir = glm::vec3(sinTheta * std::cos(phi), cosTheta, sinTheta * std::sin(phi));

    // Look up radiance at this pixel
    int idx = (row * W + col) * 3;
    glm::vec3 radiance(m_envMapPixels[idx], m_envMapPixels[idx + 1], m_envMapPixels[idx + 2]);

    // Compute PDF
    float lum = 0.2126f * radiance.r + 0.7152f * radiance.g + 0.0722f * radiance.b;
    if (sinTheta < 1e-8f || m_envTotalIntegral < 1e-8f || lum < 1e-8f)
    {
        outPdf = 0.0f;
        return radiance;
    }

    outPdf = (lum * static_cast<float>(W) * static_cast<float>(H))
           / (2.0f * PI * PI * sinTheta * m_envTotalIntegral);

    return radiance;
}

float CPURaytracer::envMapPdf(const glm::vec3& dir) const
{
    int W = m_envMapWidth;
    int H = m_envMapHeight;

    // Direction to UV (same as sampleEnvironment)
    float u = 0.5f + std::atan2(dir.z, dir.x) / (2.0f * PI);
    float v = 0.5f - std::asin(glm::clamp(dir.y, -1.0f, 1.0f)) / PI;

    int px = std::clamp(static_cast<int>(u * W), 0, W - 1);
    int py = std::clamp(static_cast<int>(v * H), 0, H - 1);
    int idx = (py * W + px) * 3;

    float r = m_envMapPixels[idx];
    float g = m_envMapPixels[idx + 1];
    float b = m_envMapPixels[idx + 2];
    float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;

    float sinTheta = std::sin(PI * (static_cast<float>(py) + 0.5f) / static_cast<float>(H));
    if (sinTheta < 1e-8f || m_envTotalIntegral < 1e-8f)
        return 0.0f;

    return (lum * static_cast<float>(W) * static_cast<float>(H))
         / (2.0f * PI * PI * sinTheta * m_envTotalIntegral);
}

glm::vec3 CPURaytracer::sampleEnvironment(const glm::vec3& direction) const
{
    if (m_hasEnvMap && !m_envMapPixels.empty())
    {
        // Equirectangular mapping
        float u = 0.5f + std::atan2(direction.z, direction.x) / (2.0f * PI);
        float v = 0.5f - std::asin(glm::clamp(direction.y, -1.0f, 1.0f)) / PI;

        int px = std::clamp(static_cast<int>(u * m_envMapWidth),  0, m_envMapWidth - 1);
        int py = std::clamp(static_cast<int>(v * m_envMapHeight), 0, m_envMapHeight - 1);
        int idx = (py * m_envMapWidth + px) * 3;

        return glm::vec3(m_envMapPixels[idx],
                         m_envMapPixels[idx + 1],
                         m_envMapPixels[idx + 2]);
    }

    return m_envColor;
}

glm::vec4 CPURaytracer::sampleTexture(int textureIndex, const glm::vec2& uv) const
{
    const auto& tex = m_textures[textureIndex];

    // Wrap UVs to [0,1)
    float u = uv.x - std::floor(uv.x);
    float v = 1.0f - (uv.y - std::floor(uv.y)); // flip V: OBJ V=0 is bottom, texture row 0 is top

    int px = std::clamp(static_cast<int>(u * tex.width),  0, tex.width - 1);
    int py = std::clamp(static_cast<int>(v * tex.height), 0, tex.height - 1);
    int idx = (py * tex.width + px) * 4;

    return glm::vec4(tex.pixels[idx]     / 255.0f,
                     tex.pixels[idx + 1] / 255.0f,
                     tex.pixels[idx + 2] / 255.0f,
                     tex.pixels[idx + 3] / 255.0f);
}

// --- Light data ---

void CPURaytracer::buildLightData()
{
    m_lightIndices.clear();
    m_lightCDF.clear();
    m_totalLightArea = 0.0f;

    for (uint32_t i = 0; i < static_cast<uint32_t>(m_triData.size()); ++i)
    {
        const auto& data = m_triData[i];
        if (glm::length(data.emissive) > 0.001f)
        {
            m_lightIndices.push_back(i);
            m_totalLightArea += data.area;
            m_lightCDF.push_back(m_totalLightArea);
        }
    }

    if (m_totalLightArea > 0.0f)
    {
        for (float& c : m_lightCDF)
            c /= m_totalLightArea;
    }
}

glm::vec3 CPURaytracer::sampleLightPoint(RNG& rng, uint32_t& outTriIndex) const
{
    float u = rng.next();
    auto it = std::lower_bound(m_lightCDF.begin(), m_lightCDF.end(), u);
    uint32_t lightIdx = static_cast<uint32_t>(std::distance(m_lightCDF.begin(), it));
    if (lightIdx >= m_lightIndices.size())
        lightIdx = static_cast<uint32_t>(m_lightIndices.size()) - 1;

    outTriIndex = m_lightIndices[lightIdx];
    const auto& verts = m_triVerts[outTriIndex];

    float u1 = rng.next();
    float u2 = rng.next();
    float su0 = std::sqrt(u1);

    return verts.v0 * (1.0f - su0) + verts.v1 * (su0 * (1.0f - u2)) + verts.v2 * (su0 * u2);
}

// --- Ray generation and intersection ---

static glm::vec2 sampleConcentricDisk(float u1, float u2)
{
    float a = 2.0f * u1 - 1.0f;
    float b = 2.0f * u2 - 1.0f;
    if (a == 0.0f && b == 0.0f)
        return glm::vec2(0.0f);
    float r, phi;
    if (std::abs(a) > std::abs(b))
    {
        r   = a;
        phi = (PI / 4.0f) * (b / a);
    }
    else
    {
        r   = b;
        phi = (PI / 2.0f) - (PI / 4.0f) * (a / b);
    }
    return glm::vec2(r * std::cos(phi), r * std::sin(phi));
}

Ray CPURaytracer::generateRay(int x, int y, float jitterX, float jitterY, RNG& rng) const
{
    float ndcX = (2.0f * (static_cast<float>(x) + jitterX) / static_cast<float>(m_width)) - 1.0f;
    float ndcY = 1.0f - (2.0f * (static_cast<float>(y) + jitterY) / static_cast<float>(m_height));

    glm::vec4 nearClip = m_inverseVP * glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
    glm::vec4 farClip  = m_inverseVP * glm::vec4(ndcX, ndcY,  1.0f, 1.0f);

    glm::vec3 nearWorld = glm::vec3(nearClip) / nearClip.w;
    glm::vec3 farWorld  = glm::vec3(farClip) / farClip.w;

    Ray ray;
    ray.origin    = m_cameraOrigin;
    ray.direction = glm::normalize(farWorld - nearWorld);

    if (m_aperture > 0.0f)
    {
        glm::vec3 focalPoint = ray.origin + ray.direction * m_focusDistance;
        glm::vec2 disk = sampleConcentricDisk(rng.next(), rng.next()) * m_aperture;
        ray.origin    += disk.x * m_cameraRight + disk.y * m_cameraUp;
        ray.direction  = glm::normalize(focalPoint - ray.origin);
    }

    return ray;
}

bool CPURaytracer::intersectTriangle(const Ray& ray, const TriVerts& verts,
                                     float& t, float& u, float& v) const
{
    constexpr float EPSILON = 1e-7f;

    glm::vec3 edge1 = verts.v1 - verts.v0;
    glm::vec3 edge2 = verts.v2 - verts.v0;
    glm::vec3 h = glm::cross(ray.direction, edge2);
    float a = glm::dot(edge1, h);

    if (a > -EPSILON && a < EPSILON)
        return false;

    float f = 1.0f / a;
    glm::vec3 s = ray.origin - verts.v0;
    u = f * glm::dot(s, h);

    if (u < 0.0f || u > 1.0f)
        return false;

    glm::vec3 q = glm::cross(s, edge1);
    v = f * glm::dot(ray.direction, q);

    if (v < 0.0f || u + v > 1.0f)
        return false;

    t = f * glm::dot(edge2, q);
    return t > EPSILON;
}

HitRecord CPURaytracer::traceRay(const Ray& ray) const
{
    HitRecord closest;

    if (m_bvh.empty())
        return closest;

    const auto& nodes = m_bvh.nodes();
    glm::vec3 invDir = 1.0f / ray.direction;

    uint32_t stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    while (stackPtr > 0)
    {
        const auto& node = nodes[stack[--stackPtr]];

        if (!intersectAABB(node.bounds, ray.origin, invDir, closest.t))
            continue;

        if (node.isLeaf())
        {
            for (uint32_t i = node.leftFirst; i < node.leftFirst + node.triCount; ++i)
            {
                float t, u, v;
                if (intersectTriangle(ray, m_triVerts[i], t, u, v) && t < closest.t)
                {
                    const auto& data = m_triData[i];
                    float w = 1.0f - u - v;

                    // Alpha clip: skip transparent intersections
                    if (data.alphaClip && data.textureIndex >= 0)
                    {
                        glm::vec2 hitUV = w * data.uv0 + u * data.uv1 + v * data.uv2;
                        if (sampleTexture(data.textureIndex, hitUV).a < 0.5f)
                            continue;
                    }

                    closest.t = t;
                    closest.hit = true;
                    closest.position = ray.at(t);
                    closest.normal = m_flatShading
                        ? data.geometricNormal
                        : glm::normalize(w * data.n0 + u * data.n1 + v * data.n2);
                    closest.geometricNormal = data.geometricNormal;
                    closest.color = data.color;
                    closest.emissive = data.emissive;
                    closest.uv = w * data.uv0 + u * data.uv1 + v * data.uv2;
                    closest.textureIndex = data.textureIndex;
                    closest.emissiveTextureIndex = data.emissiveTextureIndex;
                    closest.normalMapTextureIndex = data.normalMapTextureIndex;
                    closest.roughnessTextureIndex = data.roughnessTextureIndex;
                    closest.metallicTextureIndex = data.metallicTextureIndex;
                    closest.triangleIndex = i;
                    closest.materialType = data.materialType;
                    closest.ior = data.ior;
                    closest.roughness = data.roughness;
                    closest.metallic = data.metallic;
                    closest.tangent = data.tangent;
                    closest.bitangentSign = data.bitangentSign;
                }
            }
        }
        else
        {
            stack[stackPtr++] = node.leftFirst;
            stack[stackPtr++] = node.leftFirst + 1;
        }
    }

    return closest;
}

bool CPURaytracer::traceShadowRay(const Ray& ray, float maxDist) const
{
    if (m_bvh.empty())
        return false;

    const auto& nodes = m_bvh.nodes();
    glm::vec3 invDir = 1.0f / ray.direction;

    uint32_t stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    while (stackPtr > 0)
    {
        const auto& node = nodes[stack[--stackPtr]];

        if (!intersectAABB(node.bounds, ray.origin, invDir, maxDist))
            continue;

        if (node.isLeaf())
        {
            for (uint32_t i = node.leftFirst; i < node.leftFirst + node.triCount; ++i)
            {
                float t, u, v;
                if (intersectTriangle(ray, m_triVerts[i], t, u, v) && t < maxDist)
                {
                    // Alpha clip: transparent surfaces don't occlude
                    const auto& data = m_triData[i];
                    if (data.alphaClip && data.textureIndex >= 0)
                    {
                        float w = 1.0f - u - v;
                        glm::vec2 hitUV = w * data.uv0 + u * data.uv1 + v * data.uv2;
                        if (sampleTexture(data.textureIndex, hitUV).a < 0.5f)
                            continue;
                    }
                    return true; // occluded
                }
            }
        }
        else
        {
            stack[stackPtr++] = node.leftFirst;
            stack[stackPtr++] = node.leftFirst + 1;
        }
    }

    return false;
}

// --- Path tracing ---

glm::vec3 CPURaytracer::pathTrace(const Ray& initialRay, RNG& rng) const
{
    glm::vec3 radiance(0.0f);
    glm::vec3 throughput(1.0f);
    Ray ray = initialRay;
    float prevBsdfPdf = 0.0f;
    bool prevWasDelta = false;
    bool hasLights = !m_lightIndices.empty();

    for (int depth = 0; depth < m_maxDepth; ++depth)
    {
        // Russian Roulette — terminate low-throughput paths after the first 2 bounces
        if (m_enableRR && depth >= 2)
        {
            float p = std::min(0.2126f * throughput.r + 0.7152f * throughput.g + 0.0722f * throughput.b, 0.95f);
            if (rng.next() > p)
                break;
            throughput /= p;
        }

        HitRecord hit = traceRay(ray);

        if (!hit.hit)
        {
            // Sun contribution when ray misses geometry
            // m_sunColor stores irradiance; radiance of the disk = irradiance / solidAngle
            if (m_sunEnabled && glm::dot(ray.direction, -m_sunDir) > m_sunCosAngle)
            {
                float sunSolidAngle = 2.0f * PI * (1.0f - m_sunCosAngle);
                float sunRadiance   = 1.0f / sunSolidAngle;
                float lightPdf      = 1.0f / sunSolidAngle;

                if (depth == 0 || !m_enableNEE || prevWasDelta)
                {
                    radiance += throughput * m_sunColor * sunRadiance;
                }
                else
                {
                    // MIS: BSDF hit the sun disk
                    float weight = prevBsdfPdf / (prevBsdfPdf + lightPdf);
                    radiance += throughput * m_sunColor * sunRadiance * weight;
                }
            }

            {
                glm::vec3 envContrib = sampleEnvironment(ray.direction);
                if (depth == 0)
                {
                    // Background always visible regardless of enableEnvironment toggle
                    radiance += throughput * envContrib;
                }
                else if (m_enableEnvironment)
                {
                    glm::vec3 scaledEnv = envContrib * m_envLightMultiplier;
                    bool hasEnvCDF = m_hasEnvMap && m_envTotalIntegral > 0.0f;
                    if (m_enableNEE && !prevWasDelta && hasEnvCDF)
                    {
                        float ePdf = envMapPdf(ray.direction);
                        if (ePdf > 1e-8f)
                            scaledEnv *= prevBsdfPdf / (prevBsdfPdf + ePdf);
                    }
                    radiance += throughput * scaledEnv;
                }
            }
            break;
        }

        // Determine front/back face
        bool frontFace = glm::dot(hit.geometricNormal, -ray.direction) > 0.0f;

        // Opaque back-face hit — mesh has inverted normals (e.g. CAD export with outward-facing
        // inner surfaces). Bouncing from here with offsetNormal = -Ng sends the ray toward the
        // arch interior where it oscillates forever. Pass through instead: advance the origin
        // past the surface and keep the same direction. Dielectrics are exempt because they
        // legitimately need back-face handling for refraction.
        if (!frontFace && hit.materialType != 2) // 2 = Dielectric
        {
            ray.origin = hit.position + ray.direction * m_rayEps;
            continue;
        }

        // Ray origin offsets must follow the geometric normal, not the shading normal.
        // After normal mapping the shading normal can be nearly tangent to the surface,
        // so hit.normal * eps barely moves the origin away from the actual surface —
        // no amount of EPS increase helps. The geometric normal always points cleanly
        // away from the real surface, so it is the correct offset direction.
        const glm::vec3 offsetNormal = frontFace ? hit.geometricNormal : -hit.geometricNormal;

        // Ensure the shading normal is on the same side as the geometric normal.
        // Covers front-face hits where interpolated vertex normals cross the geometric
        // boundary, and dielectric back-face hits (opaque back-face hits are handled above).
        // The NdotL guard in sample() and evaluate() requires N to agree with Ng.
        if (glm::dot(hit.normal, offsetNormal) < 0.0f)
            hit.normal = -hit.normal;

        // --- Hit emissive surface ---
        glm::vec3 emission(0.0f);
        if (m_enableEmissive)
        {
            emission = hit.emissive;
            if (hit.emissiveTextureIndex >= 0)
                emission = glm::vec3(sampleTexture(hit.emissiveTextureIndex, hit.uv));
        }

        if (glm::length(emission) > 0.001f)
        {
            float cosLight = glm::dot(hit.geometricNormal, -ray.direction);
            bool isTexturedEmitter = (hit.emissiveTextureIndex >= 0);

            if (depth == 0 || prevWasDelta || isTexturedEmitter)
            {
                // Direct view, delta bounce, or textured emitter (not in light CDF) — full contribution
                if (cosLight > 0.0f)
                    radiance += throughput * emission;
            }
            else if (m_enableNEE && hasLights && cosLight > 0.0f)
            {
                // MIS weight for BSDF path hitting a light
                float pdfLight = (hit.t * hit.t) / (cosLight * m_totalLightArea);
                float weight = prevBsdfPdf / (prevBsdfPdf + pdfLight);
                radiance += throughput * emission * weight;
            }
            else if (!m_enableNEE)
            {
                // No NEE — BSDF is the only strategy, weight = 1
                if (cosLight > 0.0f)
                    radiance += throughput * emission;
            }

            // Textured emitters continue to scatter via their base material;
            // solid emitters (actual light sources in the CDF) terminate.
            if (!isTexturedEmitter)
                break;
        }

        glm::vec3 albedo = hit.color;
        if (hit.textureIndex >= 0)
            albedo *= glm::vec3(sampleTexture(hit.textureIndex, hit.uv));

        // Normal map perturbation
        if (m_enableNormalMapping && hit.normalMapTextureIndex >= 0)
        {
            glm::vec3 N = hit.normal;
            glm::vec4 mapSample = sampleTexture(hit.normalMapTextureIndex, hit.uv);
            glm::vec3 mapN(mapSample.x * 2.0f - 1.0f,
                           mapSample.y * 2.0f - 1.0f,
                           mapSample.z * 2.0f - 1.0f);
            mapN = glm::normalize(mapN);

            glm::vec3 T = hit.tangent;
            T = glm::normalize(T - glm::dot(T, N) * N);  // re-orthogonalize
            glm::vec3 B = glm::cross(N, T) * hit.bitangentSign;

            hit.normal = glm::normalize(T * mapN.x + B * mapN.y + N * mapN.z);

            // Re-apply alignment after normal map perturbation.
            if (glm::dot(hit.normal, offsetNormal) < 0.0f)
                hit.normal = -hit.normal;
        }

        // Sample roughness/metallic textures
        float roughness = hit.roughness;
        if (hit.roughnessTextureIndex >= 0)
            roughness = sampleTexture(hit.roughnessTextureIndex, hit.uv).x;

        float metallic = hit.metallic;
        if (hit.metallicTextureIndex >= 0)
            metallic = sampleTexture(hit.metallicTextureIndex, hit.uv).x;

        // --- Material dispatch ---
        if (hit.materialType == 2)
        {
            // Dielectric: Fresnel reflect/refract
            DielectricBSDF glassBsdf{ albedo, hit.ior };
            glm::vec3 wo = -ray.direction;
            BSDFSample sample = glassBsdf.sample(hit.normal, wo, frontFace, rng.next());

            throughput *= sample.throughput;
            prevBsdfPdf = sample.pdf;
            prevWasDelta = true;

            // Offset origin: same side for reflection, opposite for refraction
            if (glm::dot(sample.direction, offsetNormal) > 0.0f)
                ray.origin = hit.position + offsetNormal * m_rayEps;
            else
                ray.origin = hit.position - offsetNormal * m_rayEps;
            ray.direction = sample.direction;
        }
        else if (hit.materialType == 1 || (metallic > 0.99f && roughness < 0.01f))
        {
            // Mirror: explicit mirror material, or perfect-metallic PBR params (delta BRDF, no NEE)
            MirrorBSDF mirrorBsdf{ albedo };
            glm::vec3 wo = -ray.direction;
            BSDFSample sample = mirrorBsdf.sample(hit.normal, wo);

            throughput *= sample.throughput;
            prevBsdfPdf = sample.pdf;
            prevWasDelta = true;

            ray.origin    = hit.position + offsetNormal * m_rayEps;
            ray.direction = sample.direction;
        }
        else
        {
            // Cook-Torrance GGX (handles both diffuse and metallic materials)
            glm::vec3 wo = -ray.direction;
            CookTorranceBSDF bsdf{ albedo, roughness, metallic, hit.ior };

            // --- NEE: emissive triangle sampling ---
            if (m_enableNEE && m_enableEmissive && hasLights)
            {
                uint32_t lightTriIdx;
                glm::vec3 lightPos = sampleLightPoint(rng, lightTriIdx);
                const auto& lightData = m_triData[lightTriIdx];

                glm::vec3 toLight = lightPos - hit.position;
                float dist = glm::length(toLight);
                glm::vec3 lightDir = toLight / dist;

                float cosSurface = glm::dot(hit.normal, lightDir);
                float cosLight   = glm::dot(lightData.geometricNormal, -lightDir);

                if (cosSurface > 0.0f && cosLight > 0.0f && glm::dot(offsetNormal, lightDir) > 0.0f)
                {
                    Ray shadowRay;
                    shadowRay.origin    = hit.position + offsetNormal * m_rayEps;
                    shadowRay.direction = lightDir;

                    if (!traceShadowRay(shadowRay, dist - 2.0f * m_rayEps))
                    {
                        float pdfLight = (dist * dist) / (cosLight * m_totalLightArea);
                        float pdfBsdf  = bsdf.pdf(hit.normal, wo, lightDir);
                        float misWeight = pdfLight / (pdfLight + pdfBsdf);

                        glm::vec3 brdf = bsdf.evaluate(hit.normal, wo, lightDir);
                        radiance += throughput * brdf * lightData.emissive * cosSurface / pdfLight * misWeight;
                    }
                }
            }

            // --- NEE: point light sampling ---
            if (m_enableNEE && m_pointLightEnabled)
            {
                glm::vec3 toLight = m_pointLightPos - hit.position;
                float dist = glm::length(toLight);
                glm::vec3 lightDir = toLight / dist;

                float cosSurface = glm::dot(hit.normal, lightDir);

                if (cosSurface > 0.0f && glm::dot(offsetNormal, lightDir) > 0.0f)
                {
                    Ray shadowRay;
                    shadowRay.origin    = hit.position + offsetNormal * m_rayEps;
                    shadowRay.direction = lightDir;

                    if (!traceShadowRay(shadowRay, dist - 2.0f * m_rayEps))
                    {
                        glm::vec3 brdf = bsdf.evaluate(hit.normal, wo, lightDir);
                        // Point light: no MIS (delta distribution, BSDF can never hit it)
                        // Inverse-square attenuation
                        radiance += throughput * brdf * m_pointLightColor * cosSurface / (dist * dist);
                    }
                }
            }

            // --- NEE: directional (sun) light sampling ---
            if (m_enableNEE && m_sunEnabled)
            {
                // Sample direction uniformly within sun cone
                float sunSolidAngle = 2.0f * PI * (1.0f - m_sunCosAngle);
                float u1 = rng.next();
                float u2 = rng.next();

                float cosTheta = 1.0f - u1 * (1.0f - m_sunCosAngle);
                float sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);
                float phi = 2.0f * PI * u2;

                // Build ONB around -sunDir (direction toward sun)
                glm::vec3 toSun = -m_sunDir;
                glm::vec3 t, b;
                buildONB(toSun, t, b);

                glm::vec3 lightDir = t * (std::cos(phi) * sinTheta)
                                   + b * (std::sin(phi) * sinTheta)
                                   + toSun * cosTheta;
                lightDir = glm::normalize(lightDir);

                float cosSurface = glm::dot(hit.normal, lightDir);

                if (cosSurface > 0.0f && glm::dot(offsetNormal, lightDir) > 0.0f)
                {
                    Ray shadowRay;
                    shadowRay.origin    = hit.position + offsetNormal * m_rayEps;
                    shadowRay.direction = lightDir;

                    if (!traceShadowRay(shadowRay, std::numeric_limits<float>::max()))
                    {
                        float lightPdf  = 1.0f / sunSolidAngle;
                        float bsdfPdf   = bsdf.pdf(hit.normal, wo, lightDir);
                        float misWeight = lightPdf / (lightPdf + bsdfPdf);

                        glm::vec3 brdf = bsdf.evaluate(hit.normal, wo, lightDir);
                        radiance += throughput * brdf * m_sunColor * cosSurface * misWeight;
                    }
                }
            }

            // --- NEE: environment map importance sampling ---
            if (m_enableNEE && m_enableEnvironment && m_hasEnvMap && m_envTotalIntegral > 0.0f)
            {
                glm::vec3 envDir;
                float envPdf;
                glm::vec3 envRad = sampleEnvMap(rng, envDir, envPdf);

                float cosSurface = glm::dot(hit.normal, envDir);

                if (cosSurface > 0.0f && envPdf > 1e-8f && glm::dot(offsetNormal, envDir) > 0.0f)
                {
                    Ray shadowRay;
                    shadowRay.origin    = hit.position + offsetNormal * m_rayEps;
                    shadowRay.direction = envDir;

                    if (!traceShadowRay(shadowRay, std::numeric_limits<float>::max()))
                    {
                        float bsdfPdf   = bsdf.pdf(hit.normal, wo, envDir);
                        float misWeight = envPdf / (envPdf + bsdfPdf);

                        glm::vec3 brdf = bsdf.evaluate(hit.normal, wo, envDir);
                        radiance += throughput * brdf * envRad * m_envLightMultiplier * cosSurface / envPdf * misWeight;
                    }
                }
            }

            // --- BSDF sampling for next bounce ---
            BSDFSample sample = bsdf.sample(hit.normal, offsetNormal, wo, rng.next(), rng.next(), rng.next());

            if (sample.pdf < 1e-8f)
                break;

            // Don't bounce below the actual geometric surface — prevents self-intersection
            if (glm::dot(sample.direction, offsetNormal) < 0.0f)
                break;

            throughput *= sample.throughput;
            prevBsdfPdf = sample.pdf;
            prevWasDelta = false;

            ray.origin    = hit.position + offsetNormal * m_rayEps;
            ray.direction = sample.direction;
        }
    }

    return radiance;
}

// --- Sample dispatch ---

void CPURaytracer::traceSample()
{
    if (m_width == 0 || m_height == 0)
        return;

    uint32_t threadCount = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> threads(threadCount);

    auto traceRows = [this](uint32_t startRow, uint32_t endRow)
    {
        for (uint32_t y = startRow; y < endRow; ++y)
        {
            for (uint32_t x = 0; x < m_width; ++x)
            {
                uint32_t seed = hash(x + y * m_width) ^ hash(m_sampleCount);
                RNG rng(seed);

                float jx = m_enableAA ? rng.next() : 0.5f;
                float jy = m_enableAA ? rng.next() : 0.5f;
                Ray ray = generateRay(static_cast<int>(x), static_cast<int>(y), jx, jy, rng);

                glm::vec3 color = pathTrace(ray, rng);

                // NaN/Inf guard — protect accumulation buffer
                if (std::isnan(color.r) || std::isnan(color.g) || std::isnan(color.b) ||
                    std::isinf(color.r) || std::isinf(color.g) || std::isinf(color.b))
                    color = glm::vec3(0.0f);

                if (m_enableFireflyClamping)
                {
                    float lum = 0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b;
                    if (lum > 10.0f)
                        color *= 10.0f / lum;
                }

                m_accumBuffer[y * m_width + x] += color;
            }
        }
    };

    uint32_t rowsPerThread = m_height / threadCount;
    uint32_t remainder = m_height % threadCount;

    uint32_t startRow = 0;
    for (uint32_t i = 0; i < threadCount; ++i)
    {
        uint32_t endRow = startRow + rowsPerThread + (i < remainder ? 1 : 0);
        threads[i] = std::thread(traceRows, startRow, endRow);
        startRow = endRow;
    }

    for (auto& t : threads)
        t.join();

    ++m_sampleCount;

    float invSamples = 1.0f / static_cast<float>(m_sampleCount);
    float exposureMul = std::pow(2.0f, m_exposure);
    float invGamma = 1.0f / m_gamma;
    for (uint32_t i = 0; i < m_width * m_height; ++i)
    {
        glm::vec3 c = m_accumBuffer[i] * invSamples * exposureMul;

        if (m_enableACES)
        {
            // ACES filmic tone mapping (Narkowicz fit)
            const float a = 2.51f, b = 0.03f, cc = 2.43f, d = 0.59f, e = 0.14f;
            c = glm::clamp((c * (a * c + b)) / (c * (cc * c + d) + e), 0.0f, 1.0f);
        }
        else
        {
            c = glm::clamp(c, 0.0f, 1.0f);
        }

        c = glm::pow(c, glm::vec3(invGamma));

        m_pixelBuffer[i * 4 + 0] = static_cast<uint8_t>(c.r * 255.0f);
        m_pixelBuffer[i * 4 + 1] = static_cast<uint8_t>(c.g * 255.0f);
        m_pixelBuffer[i * 4 + 2] = static_cast<uint8_t>(c.b * 255.0f);
        m_pixelBuffer[i * 4 + 3] = 255;
    }
}

} // namespace vex
