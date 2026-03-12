#include <vex/scene/primitives.h>

#include <cmath>
#include <cstdint>

namespace vex {

static constexpr float PI = 3.14159265358979323846f;

static Vertex makeVtx(glm::vec3 pos, glm::vec3 norm, glm::vec2 uv, glm::vec4 tan)
{
    Vertex v;
    v.position = pos;
    v.normal   = norm;
    v.uv       = uv;
    v.tangent  = tan;
    v.color    = {1.0f, 1.0f, 1.0f};
    v.emissive = {0.0f, 0.0f, 0.0f};
    return v;
}

static MeshData baseData(const char* name)
{
    MeshData md;
    md.name      = name;
    md.roughness = 0.5f;
    md.metallic  = 0.0f;
    return md;
}

// ── Plane ────────────────────────────────────────────────────────────────────
// Face-up (+Y), centred at origin in XZ plane. UVs 0 to 1.
// Winding: CCW from above (normal = +Y).
MeshData Primitives::makePlane(float w, float h)
{
    MeshData md = baseData("Plane");
    float hw = w * 0.5f, hh = h * 0.5f;
    // Normal = +Y, Tangent = +X, Bitangent = N×T = (0,1,0)×(1,0,0) = (0,0,-1)
    // V goes from -Z to +Z, U goes from -X to +X
    const glm::vec3 n = {0, 1, 0};
    const glm::vec4 t = {1, 0, 0, 1};
    md.vertices = {
        makeVtx({-hw, 0.f, -hh}, n, {0.f, 0.f}, t),
        makeVtx({ hw, 0.f, -hh}, n, {1.f, 0.f}, t),
        makeVtx({ hw, 0.f,  hh}, n, {1.f, 1.f}, t),
        makeVtx({-hw, 0.f,  hh}, n, {0.f, 1.f}, t),
    };
    // {0,2,1} and {0,3,2}: CCW from above (+Y) verified by cross products
    md.indices = {0, 2, 1,  0, 3, 2};
    return md;
}

// ── Cube ─────────────────────────────────────────────────────────────────────
// 6 faces × 4 vertices, per-face normals/UVs/tangents.
// Winding: CCW from outside (verified per-face below).
MeshData Primitives::makeCube(float size)
{
    MeshData md = baseData("Cube");
    float s = size * 0.5f;

    struct FaceDesc {
        glm::vec3 n;       // face normal
        glm::vec3 t;       // tangent (along U = v1-v0 direction)
        glm::vec3 v[4];    // positions, CCW from outside; UVs (0,0)(1,0)(1,1)(0,1)
    };

    // All tangents verified: t = normalize(v[1]-v[0])
    // All bitangents verified: B = N×T = normalize(v[3]-v[0])
    const FaceDesc faces[] = {
        // +Z  N×T = (0,0,1)×(1,0,0) = (0,1,0) = v[3]-v[0] ✓
        { {0,0,1}, {1,0,0}, { {-s,-s,s},{s,-s,s},{s,s,s},{-s,s,s} } },
        // -Z  N×T = (0,0,-1)×(-1,0,0) = (0,1,0) ✓
        { {0,0,-1}, {-1,0,0}, { {s,-s,-s},{-s,-s,-s},{-s,s,-s},{s,s,-s} } },
        // +X  N×T = (1,0,0)×(0,0,-1) = (0,1,0) ✓
        { {1,0,0}, {0,0,-1}, { {s,-s,s},{s,-s,-s},{s,s,-s},{s,s,s} } },
        // -X  N×T = (-1,0,0)×(0,0,1) = (0,1,0) ✓
        { {-1,0,0}, {0,0,1}, { {-s,-s,-s},{-s,-s,s},{-s,s,s},{-s,s,-s} } },
        // +Y  N×T = (0,1,0)×(1,0,0) = (0,0,-1) = v[3]-v[0] ✓
        { {0,1,0}, {1,0,0}, { {-s,s,s},{s,s,s},{s,s,-s},{-s,s,-s} } },
        // -Y  N×T = (0,-1,0)×(1,0,0) = (0,0,1) = v[3]-v[0] ✓
        { {0,-1,0}, {1,0,0}, { {-s,-s,-s},{s,-s,-s},{s,-s,s},{-s,-s,s} } },
    };

    const glm::vec2 uvs[4] = { {0,0},{1,0},{1,1},{0,1} };

    for (const auto& f : faces)
    {
        auto base = static_cast<uint32_t>(md.vertices.size());
        glm::vec4 tan4 = {f.t.x, f.t.y, f.t.z, 1.0f};
        for (int k = 0; k < 4; ++k)
            md.vertices.push_back(makeVtx(f.v[k], f.n, uvs[k], tan4));
        md.indices.insert(md.indices.end(), {
            base+0, base+1, base+2,
            base+0, base+2, base+3
        });
    }
    return md;
}

// ── UV Sphere ────────────────────────────────────────────────────────────────
// Standard lat/lon sphere. Winding CCW from outside (verified analytically).
MeshData Primitives::makeUVSphere(float r, int stacks, int slices)
{
    MeshData md = baseData("Sphere");

    // (stacks+1) rings × (slices+1) columns (first and last column share position)
    for (int i = 0; i <= stacks; ++i)
    {
        float theta = static_cast<float>(i) / static_cast<float>(stacks) * PI;
        float sinT  = std::sin(theta);
        float cosT  = std::cos(theta);
        float v     = static_cast<float>(i) / static_cast<float>(stacks);

        for (int j = 0; j <= slices; ++j)
        {
            float phi  = static_cast<float>(j) / static_cast<float>(slices) * 2.f * PI;
            float sinP = std::sin(phi);
            float cosP = std::cos(phi);
            float u    = static_cast<float>(j) / static_cast<float>(slices);

            glm::vec3 pos  = { r * sinT * cosP, r * cosT, r * sinT * sinP };
            glm::vec3 norm = { sinT * cosP, cosT, sinT * sinP };
            // Tangent = d(pos)/d(phi) normalised = (-sinP, 0, cosP)
            glm::vec4 tan4 = { -sinP, 0.f, cosP, 1.f };

            md.vertices.push_back(makeVtx(pos, norm, {u, v}, tan4));
        }
    }

    // Indices: quads {a,d,c} {a,c,b} give outward CCW normals (verified)
    for (int i = 0; i < stacks; ++i)
    {
        for (int j = 0; j < slices; ++j)
        {
            uint32_t a = static_cast<uint32_t>(i     * (slices + 1) + j);
            uint32_t b = static_cast<uint32_t>((i+1) * (slices + 1) + j);
            uint32_t c = static_cast<uint32_t>((i+1) * (slices + 1) + j + 1);
            uint32_t d = static_cast<uint32_t>(i     * (slices + 1) + j + 1);

            // Skip degenerate triangles at poles
            if (i != 0)
                md.indices.insert(md.indices.end(), {a, d, c});
            if (i != stacks - 1)
                md.indices.insert(md.indices.end(), {a, c, b});
        }
    }
    return md;
}

// ── Cylinder ─────────────────────────────────────────────────────────────────
// Body (smooth normals) + flat top/bottom caps. Winding CCW from outside.
MeshData Primitives::makeCylinder(float r, float h, int slices)
{
    MeshData md = baseData("Cylinder");

    float hh = h * 0.5f;

    // ── Body ─────────────────────────────────────────────────────────────────
    // 2*(slices+1) vertices: bottom then top per column
    for (int j = 0; j <= slices; ++j)
    {
        float phi  = static_cast<float>(j) / static_cast<float>(slices) * 2.f * PI;
        float cosP = std::cos(phi);
        float sinP = std::sin(phi);
        float u    = static_cast<float>(j) / static_cast<float>(slices);

        glm::vec3 norm = { cosP, 0.f, sinP };
        // Tangent = increasing phi direction = (-sinP, 0, cosP), w=-1 → B=(0,1,0)
        glm::vec4 tan4 = { -sinP, 0.f, cosP, -1.f };

        md.vertices.push_back(makeVtx({r*cosP, -hh, r*sinP}, norm, {u, 0.f}, tan4));
        md.vertices.push_back(makeVtx({r*cosP,  hh, r*sinP}, norm, {u, 1.f}, tan4));
    }

    // Body quads: {bl,tl,tr} {bl,tr,br} — verified CCW from outside
    for (int j = 0; j < slices; ++j)
    {
        uint32_t bl = static_cast<uint32_t>(2 * j);
        uint32_t tl = static_cast<uint32_t>(2 * j + 1);
        uint32_t br = static_cast<uint32_t>(2 * (j + 1));
        uint32_t tr = static_cast<uint32_t>(2 * (j + 1) + 1);

        md.indices.insert(md.indices.end(), {bl, tl, tr, bl, tr, br});
    }

    // ── Caps ─────────────────────────────────────────────────────────────────
    uint32_t topCenterIdx = static_cast<uint32_t>(md.vertices.size());
    md.vertices.push_back(makeVtx({0.f, hh, 0.f}, {0,1,0}, {0.5f,0.5f}, {1,0,0,1}));

    uint32_t topRimBase = static_cast<uint32_t>(md.vertices.size());
    for (int j = 0; j < slices; ++j)
    {
        float phi  = static_cast<float>(j) / static_cast<float>(slices) * 2.f * PI;
        float cosP = std::cos(phi);
        float sinP = std::sin(phi);
        glm::vec2 uv = {0.5f + 0.5f * cosP, 0.5f - 0.5f * sinP};
        md.vertices.push_back(makeVtx({r*cosP, hh, r*sinP}, {0,1,0}, uv, {1,0,0,1}));
    }
    // Top triangles: {center, rim[j+1], rim[j]} — CCW from above (+Y) verified
    for (int j = 0; j < slices; ++j)
    {
        uint32_t a = topCenterIdx;
        uint32_t b = topRimBase + static_cast<uint32_t>(j);
        uint32_t c = topRimBase + static_cast<uint32_t>((j + 1) % slices);
        md.indices.insert(md.indices.end(), {a, c, b});
    }

    uint32_t botCenterIdx = static_cast<uint32_t>(md.vertices.size());
    md.vertices.push_back(makeVtx({0.f, -hh, 0.f}, {0,-1,0}, {0.5f,0.5f}, {1,0,0,1}));

    uint32_t botRimBase = static_cast<uint32_t>(md.vertices.size());
    for (int j = 0; j < slices; ++j)
    {
        float phi  = static_cast<float>(j) / static_cast<float>(slices) * 2.f * PI;
        float cosP = std::cos(phi);
        float sinP = std::sin(phi);
        glm::vec2 uv = {0.5f + 0.5f * cosP, 0.5f + 0.5f * sinP};
        md.vertices.push_back(makeVtx({r*cosP, -hh, r*sinP}, {0,-1,0}, uv, {1,0,0,1}));
    }
    // Bottom triangles: {center, rim[j], rim[j+1]} — CCW from below (-Y) verified
    for (int j = 0; j < slices; ++j)
    {
        uint32_t a = botCenterIdx;
        uint32_t b = botRimBase + static_cast<uint32_t>(j);
        uint32_t c = botRimBase + static_cast<uint32_t>((j + 1) % slices);
        md.indices.insert(md.indices.end(), {a, b, c});
    }

    return md;
}

} // namespace vex
