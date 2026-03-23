#include <doctest/doctest.h>
#include <vex/scene/primitives.h>

#include <cmath>

using namespace vex;

static bool normalsNormalized(const MeshData& m, float eps = 1e-4f)
{
    for (const auto& v : m.vertices)
    {
        float len = glm::length(v.normal);
        if (std::abs(len - 1.0f) > eps)
            return false;
    }
    return true;
}

static bool uvsInRange(const MeshData& m, float eps = 0.001f)
{
    for (const auto& v : m.vertices)
    {
        if (v.uv.x < -eps || v.uv.x > 1.0f + eps) return false;
        if (v.uv.y < -eps || v.uv.y > 1.0f + eps) return false;
    }
    return true;
}

static bool indicesInBounds(const MeshData& m)
{
    for (uint32_t idx : m.indices)
        if (idx >= m.vertices.size()) return false;
    return true;
}

// ── makePlane ────────────────────────────────────────────────────────────────

TEST_SUITE("makePlane")
{

TEST_CASE("vertex and index counts")
{
    auto m = Primitives::makePlane(2.0f, 2.0f);
    // 4 vertices, 2 triangles = 6 indices
    CHECK(m.vertices.size() == 4);
    CHECK(m.indices.size() == 6);
}

TEST_CASE("index count is divisible by 3")
{
    CHECK(Primitives::makePlane().indices.size() % 3 == 0);
}

TEST_CASE("all normals point straight up")
{
    auto m = Primitives::makePlane(2.0f, 4.0f);
    for (const auto& v : m.vertices)
    {
        CHECK(v.normal.x == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(v.normal.y == doctest::Approx(1.0f).epsilon(1e-5f));
        CHECK(v.normal.z == doctest::Approx(0.0f).epsilon(1e-5f));
    }
}

TEST_CASE("UVs are in [0, 1]")
{
    CHECK(uvsInRange(Primitives::makePlane()));
}

TEST_CASE("all indices are in bounds")
{
    CHECK(indicesInBounds(Primitives::makePlane()));
}

TEST_CASE("vertices are centred at origin in XZ")
{
    auto m = Primitives::makePlane(4.0f, 6.0f);
    for (const auto& v : m.vertices)
        CHECK(v.position.y == doctest::Approx(0.0f).epsilon(1e-5f));
    float minX = FLT_MAX, maxX = -FLT_MAX;
    for (const auto& v : m.vertices) { minX = std::min(minX, v.position.x); maxX = std::max(maxX, v.position.x); }
    CHECK(minX == doctest::Approx(-2.0f).epsilon(1e-5f));
    CHECK(maxX == doctest::Approx( 2.0f).epsilon(1e-5f));
}

} // TEST_SUITE("makePlane")

// ── makeCube ─────────────────────────────────────────────────────────────────

TEST_SUITE("makeCube")
{

TEST_CASE("vertex and index counts")
{
    auto m = Primitives::makeCube(1.0f);
    // 6 faces × 4 verts = 24; 6 faces × 6 indices = 36
    CHECK(m.vertices.size() == 24);
    CHECK(m.indices.size() == 36);
}

TEST_CASE("index count is divisible by 3")
{
    CHECK(Primitives::makeCube().indices.size() % 3 == 0);
}

TEST_CASE("all indices are in bounds")
{
    CHECK(indicesInBounds(Primitives::makeCube()));
}

TEST_CASE("normals are unit length")
{
    CHECK(normalsNormalized(Primitives::makeCube()));
}

TEST_CASE("UVs are in [0, 1]")
{
    CHECK(uvsInRange(Primitives::makeCube()));
}

TEST_CASE("all vertices within half-size of origin")
{
    float sz = 2.0f;
    auto m = Primitives::makeCube(sz);
    float half = sz * 0.5f;
    for (const auto& v : m.vertices)
    {
        CHECK(std::abs(v.position.x) <= half + 1e-5f);
        CHECK(std::abs(v.position.y) <= half + 1e-5f);
        CHECK(std::abs(v.position.z) <= half + 1e-5f);
    }
}

} // TEST_SUITE("makeCube")

// ── makeUVSphere ─────────────────────────────────────────────────────────────

TEST_SUITE("makeUVSphere")
{

TEST_CASE("index count is divisible by 3")
{
    CHECK(Primitives::makeUVSphere(1.0f, 8, 16).indices.size() % 3 == 0);
}

TEST_CASE("all indices are in bounds")
{
    CHECK(indicesInBounds(Primitives::makeUVSphere(1.0f, 8, 16)));
}

TEST_CASE("all vertices lie on the sphere surface")
{
    float r = 1.5f;
    auto m = Primitives::makeUVSphere(r, 8, 16);
    for (const auto& v : m.vertices)
    {
        float len = glm::length(v.position);
        CHECK(len == doctest::Approx(r).epsilon(1e-4f));
    }
}

TEST_CASE("normals are unit length")
{
    CHECK(normalsNormalized(Primitives::makeUVSphere(1.0f, 8, 16)));
}

TEST_CASE("UVs are in [0, 1]")
{
    CHECK(uvsInRange(Primitives::makeUVSphere(1.0f, 8, 16)));
}

} // TEST_SUITE("makeUVSphere")

// ── makeCylinder ─────────────────────────────────────────────────────────────

TEST_SUITE("makeCylinder")
{

TEST_CASE("index count is divisible by 3")
{
    CHECK(Primitives::makeCylinder(0.5f, 2.0f, 16).indices.size() % 3 == 0);
}

TEST_CASE("all indices are in bounds")
{
    CHECK(indicesInBounds(Primitives::makeCylinder(0.5f, 2.0f, 16)));
}

TEST_CASE("normals are unit length")
{
    CHECK(normalsNormalized(Primitives::makeCylinder(0.5f, 2.0f, 16)));
}

TEST_CASE("UVs are in [0, 1]")
{
    CHECK(uvsInRange(Primitives::makeCylinder(0.5f, 2.0f, 16)));
}

TEST_CASE("no vertex exceeds the bounding box")
{
    float r = 0.5f, h = 2.0f;
    auto m = Primitives::makeCylinder(r, h, 16);
    float hh = h * 0.5f;
    for (const auto& v : m.vertices)
    {
        CHECK(v.position.y >= -hh - 1e-5f);
        CHECK(v.position.y <=  hh + 1e-5f);
        float radial = std::sqrt(v.position.x * v.position.x + v.position.z * v.position.z);
        CHECK(radial <= r + 1e-5f);
    }
}

} // TEST_SUITE("makeCylinder")
