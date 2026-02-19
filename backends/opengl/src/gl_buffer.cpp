#include <vex/opengl/gl_buffer.h>
#include <glad/glad.h>

namespace vex
{

// Factory methods
std::unique_ptr<VertexBuffer> VertexBuffer::create(const void* data, size_t size, BufferUsage usage)
{
    return std::make_unique<GLVertexBuffer>(data, size, usage);
}

std::unique_ptr<IndexBuffer> IndexBuffer::create(const uint32_t* indices, uint32_t count)
{
    return std::make_unique<GLIndexBuffer>(indices, count);
}

static GLenum toGLUsage(BufferUsage usage)
{
    switch (usage)
    {
        case BufferUsage::Static:  return GL_STATIC_DRAW;
        case BufferUsage::Dynamic: return GL_DYNAMIC_DRAW;
        case BufferUsage::Stream:  return GL_STREAM_DRAW;
    }
    return GL_STATIC_DRAW;
}

// VertexBuffer

GLVertexBuffer::GLVertexBuffer(const void* data, size_t size, BufferUsage usage)
{
    glGenBuffers(1, &m_id);
    glBindBuffer(GL_ARRAY_BUFFER, m_id);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(size), data, toGLUsage(usage));
}

GLVertexBuffer::~GLVertexBuffer()
{
    if (m_id) glDeleteBuffers(1, &m_id);
}

void GLVertexBuffer::bind()   { glBindBuffer(GL_ARRAY_BUFFER, m_id); }
void GLVertexBuffer::unbind() { glBindBuffer(GL_ARRAY_BUFFER, 0); }

void GLVertexBuffer::setData(const void* data, size_t size)
{
    glBindBuffer(GL_ARRAY_BUFFER, m_id);
    glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(size), data);
}

// IndexBuffer

GLIndexBuffer::GLIndexBuffer(const uint32_t* indices, uint32_t count)
    : m_count(count)
{
    glGenBuffers(1, &m_id);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_id);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(count * sizeof(uint32_t)),
                 indices, GL_STATIC_DRAW);
}

GLIndexBuffer::~GLIndexBuffer()
{
    if (m_id) glDeleteBuffers(1, &m_id);
}

void GLIndexBuffer::bind()   { glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_id); }
void GLIndexBuffer::unbind() { glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); }

} // namespace vex
