#include <vex/core/profiler.h>
#include <vex/core/log.h>

#include <glad/glad.h>

#include <vector>

namespace vex
{
namespace
{

// GL_TIME_ELAPSED queries cannot nest, so this uses glQueryCounter with
// GL_TIMESTAMP instead. That maps 1:1 onto vkCmdWriteTimestamp and supports
// arbitrary nesting.
class GLProfilerBackend final : public IProfilerBackend
{
public:
    bool init() override
    {
        GLint bits = 0;
        glGetQueryiv(GL_TIMESTAMP, GL_QUERY_COUNTER_BITS, &bits);
        m_supported = bits > 0;
        if (!m_supported)
        {
            Log::warn("Profiler: GL_TIMESTAMP counter has zero bits");
            return true;
        }

        m_queries.resize(k_queriesPerSlot * Profiler::k_ringSlots);
        glGenQueries(static_cast<GLsizei>(m_queries.size()), m_queries.data());

        m_labels = glPushDebugGroup != nullptr && glPopDebugGroup != nullptr;
        return true;
    }

    void destroy() override
    {
        if (!m_queries.empty())
        {
            glDeleteQueries(static_cast<GLsizei>(m_queries.size()), m_queries.data());
            m_queries.clear();
        }
    }

    bool  supportsTimestamps() const override { return m_supported; }
    float ticksToMs() const override          { return 1e-6f; } // GL reports ns

    // OpenGL query objects are reused directly, so there is nothing to reset.
    void resetQueries(uint32_t, uint32_t) override {}

    void writeTimestamp(uint32_t slot, uint32_t query) override
    {
        if (!m_supported) return;
        glQueryCounter(m_queries[slot * k_queriesPerSlot + query], GL_TIMESTAMP);
    }

    bool resolve(uint32_t slot, uint32_t queryCount,
                 std::vector<uint64_t>& out) override
    {
        if (!m_supported || queryCount == 0) return false;

        const uint32_t base = slot * k_queriesPerSlot;

        // Query objects complete in submission order, so the last one being
        // available implies every earlier one in this slot is too.
        GLint available = 0;
        glGetQueryObjectiv(m_queries[base + queryCount - 1],
                           GL_QUERY_RESULT_AVAILABLE, &available);
        if (!available) return false;

        out.resize(queryCount);
        for (uint32_t i = 0; i < queryCount; ++i)
        {
            GLuint64 value = 0;
            glGetQueryObjectui64v(m_queries[base + i], GL_QUERY_RESULT, &value);
            out[i] = static_cast<uint64_t>(value);
        }
        return true;
    }

    void pushLabel(const char* name) override
    {
        if (!m_labels) return;
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, name);
    }

    void popLabel() override
    {
        if (!m_labels) return;
        glPopDebugGroup();
    }

private:
    static constexpr uint32_t k_queriesPerSlot = Profiler::k_maxZones * 2;

    std::vector<GLuint> m_queries;
    bool m_supported = false;
    bool m_labels    = false;
};

} // namespace

std::unique_ptr<IProfilerBackend> IProfilerBackend::create()
{
    return std::make_unique<GLProfilerBackend>();
}

} // namespace vex
