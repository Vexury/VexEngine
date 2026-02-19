#pragma once

#include <glm/glm.hpp>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <vector>

namespace vex
{

struct AABB
{
    glm::vec3 min{FLT_MAX, FLT_MAX, FLT_MAX};
    glm::vec3 max{-FLT_MAX, -FLT_MAX, -FLT_MAX};

    void grow(const glm::vec3& p)
    {
        min = glm::min(min, p);
        max = glm::max(max, p);
    }

    void grow(const AABB& other)
    {
        min = glm::min(min, other.min);
        max = glm::max(max, other.max);
    }

    float surfaceArea() const
    {
        glm::vec3 d = max - min;
        return 2.0f * (d.x * d.y + d.y * d.z + d.x * d.z);
    }

    glm::vec3 centroid() const
    {
        return (min + max) * 0.5f;
    }
};

// Ray-AABB intersection using the slab method.
// Expects precomputed invDir = 1/ray.direction.
inline bool intersectAABB(const AABB& box, const glm::vec3& origin,
                          const glm::vec3& invDir, float tMax)
{
    float t1 = (box.min.x - origin.x) * invDir.x;
    float t2 = (box.max.x - origin.x) * invDir.x;
    float tmin = std::min(t1, t2);
    float tmax = std::max(t1, t2);

    t1 = (box.min.y - origin.y) * invDir.y;
    t2 = (box.max.y - origin.y) * invDir.y;
    tmin = std::max(tmin, std::min(t1, t2));
    tmax = std::min(tmax, std::max(t1, t2));

    t1 = (box.min.z - origin.z) * invDir.z;
    t2 = (box.max.z - origin.z) * invDir.z;
    tmin = std::max(tmin, std::min(t1, t2));
    tmax = std::min(tmax, std::max(t1, t2));

    return tmax >= std::max(tmin, 0.0f) && tmin < tMax;
}

class BVH
{
public:
    struct Node
    {
        AABB bounds;
        uint32_t leftFirst; // internal: left child index; leaf: first tri index
        uint32_t triCount;  // 0 = internal node, >0 = leaf

        bool isLeaf() const { return triCount > 0; }
    };

    // Build from per-triangle AABBs. After build, use indices() to
    // reorder your triangle array for direct leaf-node access.
    void build(const std::vector<AABB>& triBounds);

    const std::vector<Node>& nodes() const { return m_nodes; }
    const std::vector<uint32_t>& indices() const { return m_indices; }
    bool empty() const { return m_nodes.empty(); }

    uint32_t nodeCount() const { return static_cast<uint32_t>(m_nodes.size()); }

    size_t memoryBytes() const
    {
        return m_nodes.capacity() * sizeof(Node) + m_indices.capacity() * sizeof(uint32_t);
    }

    AABB rootAABB() const { return m_nodes.empty() ? AABB{} : m_nodes[0].bounds; }

    float sahCost() const
    {
        if (m_nodes.empty()) return 0.0f;
        float rootArea = m_nodes[0].bounds.surfaceArea();
        if (rootArea <= 0.0f) return 0.0f;
        float cost = 0.0f;
        for (const auto& n : m_nodes)
        {
            if (n.isLeaf())
                cost += n.bounds.surfaceArea() * static_cast<float>(n.triCount) * INTERSECT_COST;
            else
                cost += n.bounds.surfaceArea() * TRAVERSAL_COST;
        }
        return cost / rootArea;
    }

private:
    static constexpr uint32_t SAH_BINS = 12;
    static constexpr float TRAVERSAL_COST = 1.0f;
    static constexpr float INTERSECT_COST = 1.0f;

    void updateNodeBounds(uint32_t nodeIdx);
    void subdivide(uint32_t nodeIdx);

    std::vector<Node> m_nodes;
    std::vector<uint32_t> m_indices;
    uint32_t m_nodesUsed = 0;

    // Temporary build data (cleared after build)
    std::vector<AABB> m_triBounds;
    std::vector<glm::vec3> m_centroids;
};

} // namespace vex
