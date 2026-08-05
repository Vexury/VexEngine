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

        // GL_QUERY_RESULT blocks until the timestamp has landed, which is
        // deliberate. OpenGL has no frames-in-flight fence, so nothing bounds
        // how far the CPU may run ahead of the GPU. The profiler resolves the
        // slot from k_ringSlots frames back, and on a GPU-bound frame the
        // driver keeps roughly that many frames queued, so that slot is
        // typically the one still executing. Polling GL_QUERY_RESULT_AVAILABLE
        // and giving up therefore never succeeds in a GPU-bound mode: the whole
        // result set freezes at the last frame that happened to resolve.
        // Waiting caps the lag at the ring depth instead, which is exactly the
        // invariant the ring already assumes. The wait runs inside beginFrame,
        // outside every zone, so it is never attributed to a measured time, and
        // it costs nothing when the CPU is the bottleneck because the results
        // are already there.
        //
        // Every query is read rather than probing one and inferring the rest.
        // Zones nest, so the queries do not complete in index order: the
        // outermost zone's end timestamp is submitted last but lives at index 1.
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
