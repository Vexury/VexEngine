#pragma once

#include <memory>

namespace vex
{

struct MeshData;

class Mesh
{
public:
    virtual ~Mesh() = default;

    virtual void upload(const MeshData& data) = 0;
    virtual void draw() const = 0;

    static std::unique_ptr<Mesh> create();
};

} // namespace vex
