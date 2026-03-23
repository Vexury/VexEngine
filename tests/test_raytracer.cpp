#include <doctest/doctest.h>
#include <vex/raytracing/cpu_raytracer.h>
#include <vex/raytracing/ray.h>
#include <vex/raytracing/hit.h>

#include <glm/glm.hpp>
#include <algorithm>
#include <cmath>

using namespace vex;

// Build a minimal Triangle with computed geometric normal and area.
static CPURaytracer::Triangle makeTri(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2,
                                      glm::vec3 color = {1, 1, 1})
{
    CPURaytracer::Triangle t;
    t.v0 = v0; t.v1 = v1; t.v2 = v2;
    t.n0 = t.n1 = t.n2 = t.color = color;
    glm::vec3 e1 = v1 - v0, e2 = v2 - v0;
    glm::vec3 crossed = glm::cross(e1, e2);
    t.area = 0.5f * glm::length(crossed);
    t.geometricNormal = glm::normalize(crossed);
    t.n0 = t.n1 = t.n2 = t.geometricNormal;
    return t;
}

TEST_SUITE("CPURaytracer")
{

TEST_CASE("setGeometry builds a non-empty BVH")
{
    CPURaytracer rt;
    rt.setGeometry({
        makeTri({-1, 0, -1}, {1, 0, -1}, {0, 0,  1}),
        makeTri({ 0, 1, -1}, {2, 1, -1}, {1, 1,  1}),
        makeTri({-2, 2, -1}, {0, 2, -1}, {-1, 2, 1}),
    });
    CHECK(rt.getBVHNodeCount() > 0);
}

TEST_CASE("BVH root AABB encloses all input vertices")
{
    // Triangles at known extremes
    std::vector<CPURaytracer::Triangle> tris = {
        makeTri({-5,  0,  0}, {-4,  0,  0}, {-4.5f, 1, 0}),
        makeTri({ 5,  0,  0}, { 6,  0,  0}, { 5.5f, 1, 0}),
        makeTri({ 0, -3,  0}, { 1, -3,  0}, { 0.5f, -2, 0}),
        makeTri({ 0,  4,  0}, { 1,  4,  0}, { 0.5f,  5, 0}),
    };

    CPURaytracer rt;
    rt.setGeometry(tris);

    AABB root = rt.getBVHRootAABB();
    CHECK(root.min.x <= -5.0f + 1e-3f);
    CHECK(root.max.x >=  6.0f - 1e-3f);
    CHECK(root.min.y <= -3.0f + 1e-3f);
    CHECK(root.max.y >=  5.0f - 1e-3f);
}

TEST_CASE("getReorderedTriangles preserves the total triangle count")
{
    const int N = 10;
    std::vector<CPURaytracer::Triangle> tris;
    for (int i = 0; i < N; ++i)
    {
        float x = float(i) * 3.0f;
        tris.push_back(makeTri({x, 0, 0}, {x+1, 0, 0}, {x, 1, 0}));
    }

    CPURaytracer rt;
    rt.setGeometry(tris);

    std::vector<CPURaytracer::Triangle> reordered;
    rt.getReorderedTriangles(reordered);

    CHECK(reordered.size() == static_cast<size_t>(N));
}

TEST_CASE("getReorderedTriangles is a permutation of the original triangles")
{
    // Give each triangle a unique x-position so we can identify them.
    const int N = 8;
    std::vector<CPURaytracer::Triangle> tris;
    for (int i = 0; i < N; ++i)
    {
        float x = float(i) * 4.0f;
        tris.push_back(makeTri({x, 0, 0}, {x+1, 0, 0}, {x, 1, 0}));
    }

    CPURaytracer rt;
    rt.setGeometry(tris);

    std::vector<CPURaytracer::Triangle> reordered;
    rt.getReorderedTriangles(reordered);

    REQUIRE(reordered.size() == static_cast<size_t>(N));

    // Every original v0.x must appear exactly once in the reordered list.
    for (const auto& orig : tris)
    {
        int count = 0;
        for (const auto& r : reordered)
            if (std::abs(r.v0.x - orig.v0.x) < 1e-4f)
                ++count;
        CHECK(count == 1);
    }
}

TEST_CASE("setGeometry with zero triangles produces empty BVH")
{
    CPURaytracer rt;
    rt.setGeometry({});
    CHECK(rt.getBVHNodeCount() == 0);
}

TEST_CASE("SAH cost is finite and positive after a valid geometry upload")
{
    CPURaytracer rt;
    rt.setGeometry({
        makeTri({ 0, 0, 0}, {1, 0, 0}, {0, 1, 0}),
        makeTri({10, 0, 0}, {11, 0, 0}, {10, 1, 0}),
        makeTri({ 5, 5, 0}, {6,  5, 0}, {5,  6, 0}),
    });
    float cost = rt.getBVHSAHCost();
    CHECK(std::isfinite(cost));
    CHECK(cost > 0.0f);
}

} // TEST_SUITE("CPURaytracer")

// ── intersectTriangle (via traceRay) ─────────────────────────────────────────
//
// Triangle at z=5, vertices (0,0,5) (0,1,5) (1,0,5).
// Winding gives geometric normal = (0,0,-1), facing toward rays from z < 5.
// Interior in XY: the set { x>=0, y>=0, x+y<=1 }.

TEST_SUITE("intersectTriangle")
{

static CPURaytracer::Triangle frontTri()
{
    // makeTri winding: cross((0,1,0),(1,0,0)) = (0,0,-1) → normal faces -Z
    return makeTri({0,0,5}, {0,1,5}, {1,0,5});
}

TEST_CASE("direct hit returns correct t")
{
    CPURaytracer rt;
    rt.setGeometry({frontTri()});
    vex::HitRecord h = rt.traceRay({{0.25f, 0.25f, 0.0f}, {0,0,1}});
    REQUIRE(h.hit);
    CHECK(h.t == doctest::Approx(5.0f).epsilon(1e-4f));
}

TEST_CASE("hit position lies on the ray")
{
    CPURaytracer rt;
    rt.setGeometry({frontTri()});
    vex::HitRecord h = rt.traceRay({{0.3f, 0.2f, 0.0f}, {0,0,1}});
    REQUIRE(h.hit);
    CHECK(h.position.x == doctest::Approx(0.3f).epsilon(1e-4f));
    CHECK(h.position.y == doctest::Approx(0.2f).epsilon(1e-4f));
    CHECK(h.position.z == doctest::Approx(5.0f).epsilon(1e-4f));
}

TEST_CASE("miss: ray displaced outside triangle")
{
    CPURaytracer rt;
    rt.setGeometry({frontTri()});
    // (0.6, 0.6) is outside — x+y = 1.2 > 1
    vex::HitRecord h = rt.traceRay({{0.6f, 0.6f, 0.0f}, {0,0,1}});
    CHECK_FALSE(h.hit);
}

TEST_CASE("miss: ray pointing away from triangle")
{
    CPURaytracer rt;
    rt.setGeometry({frontTri()});
    vex::HitRecord h = rt.traceRay({{0.25f, 0.25f, 0.0f}, {0,0,-1}});
    CHECK_FALSE(h.hit);
}

TEST_CASE("miss: ray parallel to triangle plane")
{
    CPURaytracer rt;
    rt.setGeometry({frontTri()});
    // Direction along X at z=5 — determinant ≈ 0 → parallel reject
    vex::HitRecord h = rt.traceRay({{-1.0f, 0.25f, 5.0f}, {1,0,0}});
    CHECK_FALSE(h.hit);
}

TEST_CASE("miss: origin on triangle plane (t below self-hit threshold)")
{
    CPURaytracer rt;
    rt.setGeometry({frontTri()});
    // Ray starts exactly on the triangle surface — t = 0, rejected by t > 1e-7
    vex::HitRecord h = rt.traceRay({{0.25f, 0.25f, 5.0f}, {0,0,1}});
    CHECK_FALSE(h.hit);
}

TEST_CASE("back-face culled for opaque material")
{
    CPURaytracer rt;
    rt.setGeometry({frontTri()}); // normal = (0,0,-1)
    // Ray from z=10 pointing -Z hits the back face
    vex::HitRecord h = rt.traceRay({{0.25f, 0.25f, 10.0f}, {0,0,-1}});
    CHECK_FALSE(h.hit);
}

TEST_CASE("back-face NOT culled for dielectric")
{
    auto tri = frontTri();
    tri.materialType = 2; // Dielectric — back hits needed for refraction
    CPURaytracer rt;
    rt.setGeometry({tri});
    vex::HitRecord h = rt.traceRay({{0.25f, 0.25f, 10.0f}, {0,0,-1}});
    CHECK(h.hit);
}

TEST_CASE("two overlapping triangles: nearest hit returned")
{
    auto near = frontTri();                               // at z=5
    auto far  = makeTri({0,0,10}, {0,1,10}, {1,0,10});   // at z=10
    CPURaytracer rt;
    rt.setGeometry({near, far});
    vex::HitRecord h = rt.traceRay({{0.25f, 0.25f, 0.0f}, {0,0,1}});
    REQUIRE(h.hit);
    CHECK(h.t == doctest::Approx(5.0f).epsilon(1e-4f));
}

} // TEST_SUITE("intersectTriangle")
