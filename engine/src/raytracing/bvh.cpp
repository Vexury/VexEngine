#include <vex/raytracing/bvh.h>

#include <algorithm>
#include <numeric>

namespace vex
{

void BVH::build(const std::vector<AABB>& triBounds)
{
    uint32_t triCount = static_cast<uint32_t>(triBounds.size());
    if (triCount == 0)
    {
        m_nodes.clear();
        m_nodes.shrink_to_fit();
        m_indices.clear();
        m_indices.shrink_to_fit();
        return;
    }

    // Store build data
    m_triBounds = triBounds;
    m_centroids.resize(triCount);
    for (uint32_t i = 0; i < triCount; ++i)
        m_centroids[i] = triBounds[i].centroid();

    // Initialize index array [0, 1, 2, ..., N-1]
    m_indices.resize(triCount);
    std::iota(m_indices.begin(), m_indices.end(), 0u);

    // Allocate worst-case node count (2N - 1)
    m_nodes.resize(2 * triCount);
    m_nodesUsed = 1;

    // Create root
    Node& root = m_nodes[0];
    root.leftFirst = 0;
    root.triCount = triCount;
    updateNodeBounds(0);

    subdivide(0);

    // Trim to actual size
    m_nodes.resize(m_nodesUsed);

    // Free temporary build data
    m_triBounds.clear();
    m_triBounds.shrink_to_fit();
    m_centroids.clear();
    m_centroids.shrink_to_fit();
}

void BVH::updateNodeBounds(uint32_t nodeIdx)
{
    Node& node = m_nodes[nodeIdx];
    node.bounds = AABB{};
    for (uint32_t i = 0; i < node.triCount; ++i)
        node.bounds.grow(m_triBounds[m_indices[node.leftFirst + i]]);
}

void BVH::subdivide(uint32_t nodeIdx)
{
    Node& node = m_nodes[nodeIdx];

    if (node.triCount <= 2)
        return;

    // Centroid bounds for the triangles in this node
    AABB centroidBounds;
    for (uint32_t i = 0; i < node.triCount; ++i)
        centroidBounds.grow(m_centroids[m_indices[node.leftFirst + i]]);

    float parentArea = node.bounds.surfaceArea();
    float bestCost = FLT_MAX;
    int bestAxis = -1;
    float bestSplitPos = 0.0f;

    // Evaluate SAH for each axis using binning
    for (int axis = 0; axis < 3; ++axis)
    {
        float boundsMin = centroidBounds.min[axis];
        float boundsMax = centroidBounds.max[axis];
        if (boundsMin == boundsMax)
            continue;

        // Bin triangles by centroid position
        struct Bin
        {
            AABB bounds;
            uint32_t count = 0;
        };
        Bin bins[SAH_BINS] = {};
        float scale = static_cast<float>(SAH_BINS) / (boundsMax - boundsMin);

        for (uint32_t i = 0; i < node.triCount; ++i)
        {
            uint32_t triIdx = m_indices[node.leftFirst + i];
            uint32_t binIdx = std::min(
                SAH_BINS - 1,
                static_cast<uint32_t>((m_centroids[triIdx][axis] - boundsMin) * scale));
            bins[binIdx].count++;
            bins[binIdx].bounds.grow(m_triBounds[triIdx]);
        }

        // Sweep left-to-right and right-to-left to build prefix sums
        float leftArea[SAH_BINS - 1], rightArea[SAH_BINS - 1];
        uint32_t leftCount[SAH_BINS - 1], rightCount[SAH_BINS - 1];

        AABB leftBounds, rightBounds;
        uint32_t leftSum = 0, rightSum = 0;

        for (uint32_t i = 0; i < SAH_BINS - 1; ++i)
        {
            leftSum += bins[i].count;
            leftBounds.grow(bins[i].bounds);
            leftCount[i] = leftSum;
            leftArea[i] = leftBounds.surfaceArea();

            uint32_t ri = SAH_BINS - 1 - i;
            rightSum += bins[ri].count;
            rightBounds.grow(bins[ri].bounds);
            rightCount[ri - 1] = rightSum;
            rightArea[ri - 1] = rightBounds.surfaceArea();
        }

        // Evaluate each split position
        for (uint32_t i = 0; i < SAH_BINS - 1; ++i)
        {
            float cost = TRAVERSAL_COST +
                INTERSECT_COST * (leftCount[i] * leftArea[i] +
                                  rightCount[i] * rightArea[i]) / parentArea;
            if (cost < bestCost)
            {
                bestCost = cost;
                bestAxis = axis;
                bestSplitPos = boundsMin + static_cast<float>(i + 1) / scale;
            }
        }
    }

    // If no split improves over leaf cost, keep as leaf
    float leafCost = static_cast<float>(node.triCount) * INTERSECT_COST;
    if (bestAxis == -1 || bestCost >= leafCost)
        return;

    // Partition triangle indices around the split position
    int left = static_cast<int>(node.leftFirst);
    int right = left + static_cast<int>(node.triCount) - 1;
    while (left <= right)
    {
        if (m_centroids[m_indices[left]][bestAxis] < bestSplitPos)
            left++;
        else
            std::swap(m_indices[left], m_indices[right--]);
    }

    uint32_t leftTriCount = static_cast<uint32_t>(left) - node.leftFirst;
    if (leftTriCount == 0 || leftTriCount == node.triCount)
        return; // degenerate split — keep as leaf

    // Allocate child nodes (consecutive pair)
    uint32_t leftIdx = m_nodesUsed++;
    uint32_t rightIdx = m_nodesUsed++;

    m_nodes[leftIdx].leftFirst = node.leftFirst;
    m_nodes[leftIdx].triCount = leftTriCount;

    m_nodes[rightIdx].leftFirst = static_cast<uint32_t>(left);
    m_nodes[rightIdx].triCount = node.triCount - leftTriCount;

    // Convert current node to internal
    node.leftFirst = leftIdx;
    node.triCount = 0;

    updateNodeBounds(leftIdx);
    updateNodeBounds(rightIdx);

    subdivide(leftIdx);
    subdivide(rightIdx);
}

} // namespace vex
