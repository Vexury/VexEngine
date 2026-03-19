#include "scene.h"

#include <algorithm>

// ── getWorldMatrix ────────────────────────────────────────────────────────────

glm::mat4 Scene::getWorldMatrix(int nodeIdx) const
{
    if (nodeIdx < 0 || nodeIdx >= (int)nodes.size()) return glm::mat4(1.0f);
    const SceneNode& node = nodes[nodeIdx];
    if (node.parentIndex < 0) return node.localMatrix;
    return getWorldMatrix(node.parentIndex) * node.localMatrix;
}

// ── Index fixup helpers ───────────────────────────────────────────────────────

void fixRefsAfterRemove(Scene& scene, int removedIdx)
{
    for (auto& n : scene.nodes)
    {
        if (n.parentIndex > removedIdx) --n.parentIndex;
        for (auto& c : n.childIndices)
            if (c > removedIdx) --c;
    }
}

void fixRefsAfterInsert(Scene& scene, int insertedIdx)
{
    for (auto& n : scene.nodes)
    {
        if (n.parentIndex >= insertedIdx) ++n.parentIndex;
        for (auto& c : n.childIndices)
            if (c >= insertedIdx) ++c;
    }
}

std::vector<int> collectSubtree(const Scene& scene, int nodeIdx)
{
    std::vector<int> result;
    std::vector<int> stack = { nodeIdx };
    while (!stack.empty())
    {
        int cur = stack.back();
        stack.pop_back();
        result.push_back(cur);
        for (int child : scene.nodes[cur].childIndices)
            stack.push_back(child);
    }
    std::sort(result.rbegin(), result.rend()); // descending — safe erase order
    return result;
}
