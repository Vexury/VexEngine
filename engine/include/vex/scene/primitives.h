#pragma once

#include <vex/scene/mesh_data.h>

namespace vex {

struct Primitives
{
    static MeshData makePlane(float w = 2.0f, float h = 2.0f);
    static MeshData makeCube(float size = 1.0f);
    static MeshData makeUVSphere(float r = 1.0f, int stacks = 16, int slices = 32);
    static MeshData makeCylinder(float r = 0.5f, float h = 2.0f, int slices = 32);
};

} // namespace vex
