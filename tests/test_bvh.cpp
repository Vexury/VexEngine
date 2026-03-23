#include <doctest/doctest.h>
#include <vex/raytracing/bvh.h>

#include <cfloat>
#include <vector>

using namespace vex;

static AABB makeBox(glm::vec3 mn, glm::vec3 mx)
{
    AABB b;
    b.grow(mn);
    b.grow(mx);
    return b;
}

// ── AABB ─────────────────────────────────────────────────────────────────────

TEST_SUITE("AABB")
{

TEST_CASE("default state is invalid (min > max)")
{
    AABB a;
    CHECK(a.min.x == doctest::Approx(FLT_MAX));
    CHECK(a.max.x == doctest::Approx(-FLT_MAX));
}

TEST_CASE("grow with points")
{
    AABB a;
    a.grow(glm::vec3(1.0f, 2.0f, 3.0f));
    a.grow(glm::vec3(-1.0f, 4.0f, 0.0f));

    CHECK(a.min.x == doctest::Approx(-1.0f));
    CHECK(a.min.y == doctest::Approx(2.0f));
    CHECK(a.min.z == doctest::Approx(0.0f));
    CHECK(a.max.x == doctest::Approx(1.0f));
    CHECK(a.max.y == doctest::Approx(4.0f));
    CHECK(a.max.z == doctest::Approx(3.0f));
}

TEST_CASE("grow with another AABB")
{
    AABB a, b;
    a.grow(glm::vec3(0.0f, 0.0f, 0.0f));
    a.grow(glm::vec3(1.0f, 1.0f, 1.0f));
    b.grow(glm::vec3(-2.0f, -2.0f, -2.0f));
    b.grow(glm::vec3(0.5f, 0.5f, 0.5f));
    a.grow(b);

    CHECK(a.min.x == doctest::Approx(-2.0f));
    CHECK(a.max.x == doctest::Approx(1.0f));
}

TEST_CASE("surface area of a unit cube is 6")
{
    AABB a;
    a.grow(glm::vec3(0.0f, 0.0f, 0.0f));
    a.grow(glm::vec3(1.0f, 1.0f, 1.0f));
    CHECK(a.surfaceArea() == doctest::Approx(6.0f));
}

TEST_CASE("surface area scales with size")
{
    AABB a;
    a.grow(glm::vec3(-2.0f, -2.0f, -2.0f));
    a.grow(glm::vec3(2.0f, 2.0f, 2.0f));
    // Side length = 4, SA = 6 * 4^2 = 96
    CHECK(a.surfaceArea() == doctest::Approx(96.0f));
}

TEST_CASE("centroid is the midpoint")
{
    AABB a;
    a.grow(glm::vec3(-1.0f, -1.0f, -1.0f));
    a.grow(glm::vec3(1.0f, 1.0f, 1.0f));
    glm::vec3 c = a.centroid();
    CHECK(c.x == doctest::Approx(0.0f));
    CHECK(c.y == doctest::Approx(0.0f));
    CHECK(c.z == doctest::Approx(0.0f));
}

} // TEST_SUITE("AABB")

// ── intersectAABB ─────────────────────────────────────────────────────────────

TEST_SUITE("intersectAABB")
{

TEST_CASE("ray along Z hits unit box")
{
    AABB box = makeBox({-1, -1, -1}, {1, 1, 1});
    glm::vec3 origin(0, 0, -5), dir(0, 0, 1);
    glm::vec3 invDir(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    CHECK(intersectAABB(box, origin, invDir, 100.0f));
}

TEST_CASE("ray misses box to the side")
{
    AABB box = makeBox({-1, -1, -1}, {1, 1, 1});
    glm::vec3 origin(5, 5, -5), dir(0, 0, 1);
    glm::vec3 invDir(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    CHECK_FALSE(intersectAABB(box, origin, invDir, 100.0f));
}

TEST_CASE("ray pointing away from box")
{
    AABB box = makeBox({-1, -1, -1}, {1, 1, 1});
    glm::vec3 origin(0, 0, 5), dir(0, 0, 1);
    glm::vec3 invDir(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    CHECK_FALSE(intersectAABB(box, origin, invDir, 100.0f));
}

TEST_CASE("tMax clips before box")
{
    // Box at z=-1..1, origin at z=-5 → box entry at t=4. tMax=3 → should miss.
    AABB box = makeBox({-1, -1, -1}, {1, 1, 1});
    glm::vec3 origin(0, 0, -5), dir(0, 0, 1);
    glm::vec3 invDir(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    CHECK_FALSE(intersectAABB(box, origin, invDir, 3.0f));
}

TEST_CASE("ray from inside box always hits")
{
    AABB box = makeBox({-1, -1, -1}, {1, 1, 1});
    glm::vec3 origin(0, 0, 0), dir(0, 0, 1);
    glm::vec3 invDir(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    CHECK(intersectAABB(box, origin, invDir, 100.0f));
}

} // TEST_SUITE("intersectAABB")

// ── BVH ──────────────────────────────────────────────────────────────────────

TEST_SUITE("BVH")
{

TEST_CASE("build from empty list produces empty BVH")
{
    BVH bvh;
    bvh.build({});
    CHECK(bvh.empty());
    CHECK(bvh.nodeCount() == 0);
    CHECK(bvh.indices().empty());
}

TEST_CASE("single triangle")
{
    AABB tri;
    tri.grow(glm::vec3(0, 0, 0));
    tri.grow(glm::vec3(1, 0, 0));
    tri.grow(glm::vec3(0, 1, 0));

    BVH bvh;
    bvh.build({tri});

    CHECK_FALSE(bvh.empty());
    CHECK(bvh.nodeCount() >= 1);
    REQUIRE(bvh.indices().size() == 1);
    CHECK(bvh.indices()[0] == 0);
}

TEST_CASE("indices are a permutation of 0..N-1")
{
    const int N = 20;
    std::vector<AABB> bounds(N);
    for (int i = 0; i < N; ++i)
    {
        bounds[i].grow(glm::vec3(float(i * 2), 0, 0));
        bounds[i].grow(glm::vec3(float(i * 2 + 1), 1, 1));
    }

    BVH bvh;
    bvh.build(bounds);

    REQUIRE(bvh.indices().size() == N);

    std::vector<bool> seen(N, false);
    for (uint32_t idx : bvh.indices())
    {
        REQUIRE(idx < static_cast<uint32_t>(N));
        seen[idx] = true;
    }
    for (int i = 0; i < N; ++i)
        CHECK(seen[i]);
}

TEST_CASE("root AABB encloses all input AABBs")
{
    std::vector<AABB> bounds(3);
    bounds[0].grow({0, 0, 0});  bounds[0].grow({1, 1, 1});
    bounds[1].grow({5, 5, 5});  bounds[1].grow({6, 6, 6});
    bounds[2].grow({-3, 0, 0}); bounds[2].grow({-2, 1, 1});

    BVH bvh;
    bvh.build(bounds);

    AABB root = bvh.rootAABB();
    CHECK(root.min.x <= -3.0f + 1e-4f);
    CHECK(root.min.y <= 0.0f  + 1e-4f);
    CHECK(root.max.x >= 6.0f  - 1e-4f);
    CHECK(root.max.y >= 6.0f  - 1e-4f);
}

TEST_CASE("all leaf tri counts sum to total triangle count")
{
    const int N = 10;
    std::vector<AABB> bounds(N);
    for (int i = 0; i < N; ++i)
    {
        bounds[i].grow(glm::vec3(float(i * 2), 0, 0));
        bounds[i].grow(glm::vec3(float(i * 2 + 1), 1, 1));
    }

    BVH bvh;
    bvh.build(bounds);

    uint32_t leafTriSum = 0;
    for (const auto& node : bvh.nodes())
    {
        if (node.isLeaf())
            leafTriSum += node.triCount;
    }
    CHECK(leafTriSum == static_cast<uint32_t>(N));
}

TEST_CASE("leaf node indices are in bounds")
{
    const int N = 15;
    std::vector<AABB> bounds(N);
    for (int i = 0; i < N; ++i)
    {
        bounds[i].grow(glm::vec3(float(i), float(i), 0));
        bounds[i].grow(glm::vec3(float(i) + 0.5f, float(i) + 0.5f, 0.5f));
    }

    BVH bvh;
    bvh.build(bounds);

    for (const auto& node : bvh.nodes())
    {
        if (!node.isLeaf()) continue;
        for (uint32_t j = 0; j < node.triCount; ++j)
            CHECK(node.leftFirst + j < bvh.indices().size());
    }
}

TEST_CASE("SAH cost is positive for non-empty build")
{
    std::vector<AABB> bounds(5);
    for (int i = 0; i < 5; ++i)
    {
        bounds[i].grow(glm::vec3(float(i), 0, 0));
        bounds[i].grow(glm::vec3(float(i) + 1, 1, 1));
    }
    BVH bvh;
    bvh.build(bounds);
    CHECK(bvh.sahCost() > 0.0f);
}

TEST_CASE("all internal nodes have triCount=0 and all leaves have triCount>0")
{
    const int N = 12;
    std::vector<AABB> bounds(N);
    for (int i = 0; i < N; ++i)
    {
        bounds[i].grow(glm::vec3(float(i * 3), 0, 0));
        bounds[i].grow(glm::vec3(float(i * 3 + 2), 1, 1));
    }
    BVH bvh;
    bvh.build(bounds);

    for (const auto& node : bvh.nodes())
    {
        if (node.isLeaf())
            CHECK(node.triCount > 0);
        else
            CHECK(node.triCount == 0);
    }
}

TEST_CASE("node count never exceeds 2*N-1")
{
    for (int N : {1, 2, 3, 4, 8, 16, 64})
    {
        std::vector<AABB> bounds(N);
        for (int i = 0; i < N; ++i)
        {
            bounds[i].grow(glm::vec3(float(i * 2), 0, 0));
            bounds[i].grow(glm::vec3(float(i * 2 + 1), 1, 1));
        }
        BVH bvh;
        bvh.build(bounds);
        CHECK(bvh.nodeCount() <= static_cast<uint32_t>(2 * N - 1));
    }
}

TEST_CASE("degenerate: all centroids identical collapses to a single leaf")
{
    // All triangles overlap at the same centroid — no axis has boundsMin != boundsMax.
    // The BVH should keep them as a single leaf rather than splitting.
    const int N = 8;
    std::vector<AABB> bounds(N);
    for (int i = 0; i < N; ++i)
    {
        bounds[i].grow(glm::vec3(-0.5f, -0.5f, -0.5f));
        bounds[i].grow(glm::vec3( 0.5f,  0.5f,  0.5f));
    }
    BVH bvh;
    bvh.build(bounds);

    // Root must be a leaf (triCount == N, no children created)
    CHECK(bvh.nodeCount() == 1);
    CHECK(bvh.nodes()[0].isLeaf());
    CHECK(bvh.nodes()[0].triCount == static_cast<uint32_t>(N));
}

TEST_CASE("node bounds enclose all triangles assigned to that node")
{
    const int N = 16;
    std::vector<AABB> triBounds(N);
    for (int i = 0; i < N; ++i)
    {
        float x = float(i);
        triBounds[i].grow(glm::vec3(x, float(i % 3), float(i % 5)));
        triBounds[i].grow(glm::vec3(x + 1, float(i % 3) + 1, float(i % 5) + 1));
    }

    BVH bvh;
    bvh.build(triBounds);

    const float eps = 1e-4f;
    for (const auto& node : bvh.nodes())
    {
        if (!node.isLeaf()) continue;
        for (uint32_t j = 0; j < node.triCount; ++j)
        {
            uint32_t idx = bvh.indices()[node.leftFirst + j];
            const AABB& tb = triBounds[idx];
            CHECK(node.bounds.min.x <= tb.min.x + eps);
            CHECK(node.bounds.min.y <= tb.min.y + eps);
            CHECK(node.bounds.min.z <= tb.min.z + eps);
            CHECK(node.bounds.max.x >= tb.max.x - eps);
            CHECK(node.bounds.max.y >= tb.max.y - eps);
            CHECK(node.bounds.max.z >= tb.max.z - eps);
        }
    }
}

} // TEST_SUITE("BVH")
