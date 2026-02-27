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

    // Batch all mesh uploads into a single GPU transfer submit.
    // No-op on OpenGL. Call beginBatchUpload() before a bulk upload loop
    // and endBatchUpload() after — reduces N fence waits to 1.
    static void beginBatchUpload();
    static void endBatchUpload();
};

} // namespace vex
