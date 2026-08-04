#include <doctest/doctest.h>
#include <vex/core/profiler.h>

#include <memory>
#include <string>
#include <vector>

namespace
{

// Returns scripted tick values so the core can be exercised without a GPU.
// Each call to resolve() hands back m_ticks, or reports "not ready" when
// m_ready is false.
class MockProfilerBackend final : public vex::IProfilerBackend
{
public:
    bool  init() override                     { return true; }
    void  destroy() override                  {}
    bool  supportsTimestamps() const override { return m_supported; }
    float ticksToMs() const override          { return 1e-6f; } // ticks are ns

    void resetQueries(uint32_t slot, uint32_t count) override
    {
        resetCalls.push_back({slot, count});
    }

    void writeTimestamp(uint32_t slot, uint32_t query) override
    {
        writes.push_back({slot, query});
    }

    bool resolve(uint32_t slot, uint32_t queryCount,
                 std::vector<uint64_t>& out) override
    {
        if (!m_ready) return false;
        out.assign(queryCount, 0u);
        for (uint32_t i = 0; i < queryCount; ++i)
            out[i] = tickFor(slot, i);
        return true;
    }

    void pushLabel(const char* name) override { labels.push_back(name); }
    void popLabel() override                  { ++popCount; }

    // Test knobs
    bool m_supported = true;
    bool m_ready     = true;
    // Zone i in slot s spans [i*2000000, i*2000000 + 1000000] ns, so every
    // zone measures exactly 1.0 ms regardless of slot.
    static uint64_t tickFor(uint32_t, uint32_t query)
    {
        const uint32_t zone = query / 2;
        const bool     isEnd = (query & 1u) != 0u;
        return static_cast<uint64_t>(zone) * 2000000ull
             + (isEnd ? 1000000ull : 0ull);
    }

    struct Call { uint32_t slot; uint32_t count; };
    std::vector<Call>        resetCalls;
    std::vector<Call>        writes;
    std::vector<std::string> labels;
    int                      popCount = 0;
};

// The Profiler is a singleton; each test starts from a clean instance.
MockProfilerBackend* installMock()
{
    auto  mock = std::make_unique<MockProfilerBackend>();
    auto* raw  = mock.get();
    vex::Profiler::get().shutdown();
    vex::Profiler::get().init(std::move(mock));
    return raw;
}

} // namespace

TEST_CASE("profiler records nested zones with correct depth and order")
{
    installMock();
    auto& p = vex::Profiler::get();

    // Frame 0 records the zones; results only surface k_ringSlots frames later.
    for (int i = 0; i < static_cast<int>(vex::Profiler::k_ringSlots) + 1; ++i)
    {
        p.beginFrame();
        p.beginZone("Frame");
          p.beginZone("Shadow");
          p.endZone();
          p.beginZone("Raster");
            p.beginZone("Meshes");
            p.endZone();
          p.endZone();
        p.endZone();
        p.endFrame();
    }

    const auto& r = p.results();
    REQUIRE(r.size() == 4);
    CHECK(std::string(r[0].name) == "Frame");   CHECK(r[0].depth == 0);
    CHECK(std::string(r[1].name) == "Shadow");  CHECK(r[1].depth == 1);
    CHECK(std::string(r[2].name) == "Raster");  CHECK(r[2].depth == 1);
    CHECK(std::string(r[3].name) == "Meshes");  CHECK(r[3].depth == 2);
    CHECK(r[0].gpuMs == doctest::Approx(1.0f));
}

TEST_CASE("profiler results lag by exactly k_ringSlots frames")
{
    installMock();
    auto& p = vex::Profiler::get();

    p.beginFrame();
    p.beginZone("Only");
    p.endZone();
    p.endFrame();

    // Frames 1 and 2 land on the other two ring slots, which hold no data.
    for (uint32_t i = 1; i < vex::Profiler::k_ringSlots; ++i)
    {
        p.beginFrame();
        p.endFrame();
        CHECK(p.results().empty());
    }

    // Frame k_ringSlots returns to slot 0 and resolves frame 0's zones.
    p.beginFrame();
    REQUIRE(p.results().size() == 1);
    CHECK(std::string(p.results()[0].name) == "Only");
    p.endFrame();
}

TEST_CASE("unresolved query results leave the previous values untouched")
{
    auto* mock = installMock();
    auto& p    = vex::Profiler::get();

    for (uint32_t i = 0; i < vex::Profiler::k_ringSlots + 1; ++i)
    {
        p.beginFrame();
        p.beginZone("Stable");
        p.endZone();
        p.endFrame();
    }
    REQUIRE(p.results().size() == 1);
    const float before = p.results()[0].gpuMs;

    mock->m_ready = false;
    for (uint32_t i = 0; i < vex::Profiler::k_ringSlots; ++i)
    {
        p.beginFrame();
        p.beginZone("Different");
        p.endZone();
        p.endFrame();
    }

    REQUIRE(p.results().size() == 1);
    CHECK(std::string(p.results()[0].name) == "Stable");
    CHECK(p.results()[0].gpuMs == doctest::Approx(before));
}

TEST_CASE("zone overflow past k_maxZones drops extras without corruption")
{
    installMock();
    auto& p = vex::Profiler::get();

    const uint32_t over = vex::Profiler::k_maxZones + 10;
    for (uint32_t f = 0; f < vex::Profiler::k_ringSlots + 1; ++f)
    {
        p.beginFrame();
        for (uint32_t i = 0; i < over; ++i)
        {
            p.beginZone("Zone");
            p.endZone();
        }
        p.endFrame();
    }

    CHECK(p.results().size() == vex::Profiler::k_maxZones);
}

TEST_CASE("unbalanced endZone cannot drive depth negative")
{
    installMock();
    auto& p = vex::Profiler::get();

    p.beginFrame();
    p.endZone();          // stray, must be ignored
    p.endZone();          // stray, must be ignored
    p.beginZone("After");
    p.endZone();
    p.endFrame();

    for (uint32_t i = 0; i < vex::Profiler::k_ringSlots; ++i)
    {
        p.beginFrame();
        p.endFrame();
    }

    REQUIRE(p.results().size() == 1);
    CHECK(std::string(p.results()[0].name) == "After");
    CHECK(p.results()[0].depth == 0);
}

TEST_CASE("cpu-only zones report cpu time and no gpu time")
{
    installMock();
    auto& p = vex::Profiler::get();

    for (uint32_t f = 0; f < vex::Profiler::k_ringSlots + 1; ++f)
    {
        p.beginFrame();
        p.beginCPUZone("CpuOnly");
        p.endCPUZone();
        p.endFrame();
    }

    REQUIRE(p.results().size() == 1);
    CHECK(p.results()[0].gpuMs < 0.0f);
    CHECK(p.results()[0].cpuMs >= 0.0f);
}

TEST_CASE("gpu zones emit debug labels in matched pairs")
{
    auto* mock = installMock();
    auto& p    = vex::Profiler::get();

    p.beginFrame();
    p.beginZone("Outer");
      p.beginZone("Inner");
      p.endZone();
      p.beginCPUZone("NoLabel");
      p.endCPUZone();
    p.endZone();
    p.endFrame();

    REQUIRE(mock->labels.size() == 2);
    CHECK(mock->labels[0] == "Outer");
    CHECK(mock->labels[1] == "Inner");
    CHECK(mock->popCount == 2);
}

TEST_CASE("one-shot records accumulate independently of frames")
{
    installMock();
    auto& p = vex::Profiler::get();

    p.clearOneShots();
    p.recordOneShot("BVH build", 1620.4f);
    p.recordOneShot("TLAS build", 212.7f);

    REQUIRE(p.oneShots().size() == 2);
    CHECK(p.oneShots()[0].name == "BVH build");
    CHECK(p.oneShots()[0].ms == doctest::Approx(1620.4f));
    CHECK(p.oneShots()[1].name == "TLAS build");
}

TEST_CASE("gpu timing unavailable degrades to cpu-only without crashing")
{
    auto mock = std::make_unique<MockProfilerBackend>();
    mock->m_supported = false;
    vex::Profiler::get().shutdown();
    vex::Profiler::get().init(std::move(mock));

    auto& p = vex::Profiler::get();
    CHECK_FALSE(p.gpuTimingAvailable());

    for (uint32_t f = 0; f < vex::Profiler::k_ringSlots + 1; ++f)
    {
        p.beginFrame();
        p.beginZone("Frame");
        p.endZone();
        p.endFrame();
    }

    REQUIRE(p.results().size() == 1);
    CHECK(p.results()[0].gpuMs < 0.0f);
    CHECK(p.results()[0].cpuMs >= 0.0f);
}
