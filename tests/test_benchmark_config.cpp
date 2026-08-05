#include <doctest/doctest.h>
#include "benchmark_config.h"

TEST_CASE("parse minimal config applies defaults")
{
    std::string err;
    auto cfg = parseBenchmarkConfig(R"({"scene":"a.obj"})", err);
    REQUIRE(cfg.has_value());
    CHECK(cfg->scenePath == "a.obj");
    CHECK(cfg->mode == "rasterize");
    CHECK(cfg->width == 1920);
    CHECK(cfg->height == 1080);
    CHECK(cfg->warmupFrames == 30);
    CHECK(cfg->measureFrames == 300);
    CHECK(cfg->maxSamples == 0);
    CHECK(cfg->vsync == false);
    CHECK(cfg->orbitDegrees == doctest::Approx(0.0f));
    CHECK(cfg->camera.empty());
}

TEST_CASE("parse full config reads every field")
{
    std::string err;
    auto cfg = parseBenchmarkConfig(R"({
        "name": "run1", "scene": "b.gltf", "sceneFormat": "gltf",
        "mode": "gpu_raytrace", "width": 2560, "height": 1440,
        "warmupFrames": 5, "measureFrames": 50, "maxSamples": 64,
        "vsync": true, "orbitDegrees": 90.0,
        "camera": [
            {"target":[1,2,3],"distance":4,"yaw":0.5,"pitch":-0.25},
            {"target":[4,5,6],"distance":7,"yaw":1.5,"pitch":-0.75}
        ]
    })", err);
    REQUIRE(cfg.has_value());
    CHECK(cfg->name == "run1");
    CHECK(cfg->sceneFormat == "gltf");
    CHECK(cfg->mode == "gpu_raytrace");
    CHECK(cfg->width == 2560);
    CHECK(cfg->height == 1440);
    CHECK(cfg->warmupFrames == 5);
    CHECK(cfg->measureFrames == 50);
    CHECK(cfg->maxSamples == 64);
    CHECK(cfg->vsync == true);
    CHECK(cfg->orbitDegrees == doctest::Approx(90.0f));
    REQUIRE(cfg->camera.size() == 2);
    CHECK(cfg->camera[0].target.y == doctest::Approx(2.0f));
    CHECK(cfg->camera[1].distance == doctest::Approx(7.0f));
}

TEST_CASE("parse rejects missing scene and malformed json")
{
    std::string err;
    CHECK_FALSE(parseBenchmarkConfig(R"({"mode":"rasterize"})", err).has_value());
    CHECK_FALSE(err.empty());

    err.clear();
    CHECK_FALSE(parseBenchmarkConfig("{not json", err).has_value());
    CHECK_FALSE(err.empty());
}

TEST_CASE("parse rejects wrong-typed fields without throwing")
{
    std::string err;
    auto badScalar = parseBenchmarkConfig(R"({"scene":"a.obj","width":"wide"})", err);
    CHECK_FALSE(badScalar.has_value());
    CHECK_FALSE(err.empty());

    err.clear();
    auto badString = parseBenchmarkConfig(R"({"scene":"a.obj","mode":123})", err);
    CHECK_FALSE(badString.has_value());
    CHECK_FALSE(err.empty());

    err.clear();
    auto badCameraField = parseBenchmarkConfig(
        R"({"scene":"a.obj","camera":[{"distance":"far"}]})", err);
    CHECK_FALSE(badCameraField.has_value());
    CHECK_FALSE(err.empty());
}

TEST_CASE("camera interpolation handles 0, 1, 2 and 3 keyframes")
{
    CHECK(interpolateCamera({}, 0.5f).distance == doctest::Approx(10.0f));

    BenchCamKey only;
    only.distance = 42.0f;
    CHECK(interpolateCamera({only}, 0.0f).distance == doctest::Approx(42.0f));
    CHECK(interpolateCamera({only}, 1.0f).distance == doctest::Approx(42.0f));

    BenchCamKey a; a.distance = 0.0f; a.yaw = 0.0f;
    BenchCamKey b; b.distance = 10.0f; b.yaw = 2.0f;
    CHECK(interpolateCamera({a, b}, 0.0f).distance == doctest::Approx(0.0f));
    CHECK(interpolateCamera({a, b}, 1.0f).distance == doctest::Approx(10.0f));
    CHECK(interpolateCamera({a, b}, 0.5f).distance == doctest::Approx(5.0f));
    CHECK(interpolateCamera({a, b}, 0.25f).yaw == doctest::Approx(0.5f));
    CHECK(interpolateCamera({a, b}, -0.5f).distance == doctest::Approx(0.0f));
    CHECK(interpolateCamera({a, b}, 1.5f).distance == doctest::Approx(10.0f));

    BenchCamKey c; c.distance = 20.0f;
    CHECK(interpolateCamera({a, b, c}, 0.5f).distance == doctest::Approx(10.0f));
    CHECK(interpolateCamera({a, b, c}, 1.0f).distance == doctest::Approx(20.0f));
    CHECK(interpolateCamera({a, b, c}, 0.25f).distance == doctest::Approx(5.0f));
}

TEST_CASE("statistics handle empty, single and small sample sets")
{
    ProfileStats empty = computeStats({});
    CHECK(empty.mean == doctest::Approx(0.0f));
    CHECK(empty.p99 == doctest::Approx(0.0f));

    ProfileStats one = computeStats({5.0f});
    CHECK(one.mean == doctest::Approx(5.0f));
    CHECK(one.min == doctest::Approx(5.0f));
    CHECK(one.max == doctest::Approx(5.0f));
    CHECK(one.p50 == doctest::Approx(5.0f));
    CHECK(one.p99 == doctest::Approx(5.0f));
    CHECK(one.stddev == doctest::Approx(0.0f));

    ProfileStats two = computeStats({2.0f, 4.0f});
    CHECK(two.mean == doctest::Approx(3.0f));
    CHECK(two.p50 == doctest::Approx(2.0f));  // nearest-rank: ceil(0.5*2)-1 = 0
    CHECK(two.p95 == doctest::Approx(4.0f));
    CHECK(two.p99 == doctest::Approx(4.0f));
}

TEST_CASE("statistics compute percentiles by nearest rank")
{
    std::vector<float> v;
    for (int i = 1; i <= 100; ++i) v.push_back(static_cast<float>(i));

    ProfileStats s = computeStats(v);
    CHECK(s.mean == doctest::Approx(50.5f));
    CHECK(s.min == doctest::Approx(1.0f));
    CHECK(s.max == doctest::Approx(100.0f));
    CHECK(s.p50 == doctest::Approx(50.0f));
    CHECK(s.p95 == doctest::Approx(95.0f));
    CHECK(s.p99 == doctest::Approx(99.0f));
}

TEST_CASE("csv escaping quotes fields containing separators")
{
    CHECK(csvEscape("plain") == "plain");
    CHECK(csvEscape("has,comma") == "\"has,comma\"");
    CHECK(csvEscape("has\"quote") == "\"has\"\"quote\"");
    CHECK(csvEscape("has\nnewline") == "\"has\nnewline\"");
    CHECK(csvEscape("\r") == "\"\r\"");
}

// The staleness guard is the safety net against publishing statistics computed
// over repeated copies of one profiler sample, which has happened on this
// project. BenchmarkRunner itself depends on the app and engine layers and
// cannot live in this target, so the two decision points are pure functions
// here and are tested directly.

TEST_CASE("duplicate frame detection compares the whole recorded frame")
{
    const std::vector<float> a{1.0f, 2.0f, -1.0f};
    const std::vector<float> b{1.0f, 2.0f, -1.0f};
    const std::vector<float> c{1.0f, 2.5f, -1.0f};

    CHECK(benchFrameIsDuplicate(a, 0.5f, 3.0f, b, 0.5f, 3.0f));

    // Any single differing component makes it a fresh sample.
    CHECK_FALSE(benchFrameIsDuplicate(a, 0.5f, 3.0f, c, 0.5f, 3.0f));
    CHECK_FALSE(benchFrameIsDuplicate(a, 0.5f, 3.0f, b, 0.6f, 3.0f));
    CHECK_FALSE(benchFrameIsDuplicate(a, 0.5f, 3.0f, b, 0.5f, 3.1f));

    // A differing zone count is a different frame, not a repeat.
    CHECK_FALSE(benchFrameIsDuplicate(a, 0.5f, 3.0f, {1.0f, 2.0f}, 0.5f, 3.0f));

    // Unmeasured sentinels compare like any other value.
    CHECK(benchFrameIsDuplicate({-1.0f}, -1.0f, -1.0f, {-1.0f}, -1.0f, -1.0f));
}

TEST_CASE("a run with no duplicate frames is accepted")
{
    CHECK_FALSE(benchRunIsStale(300, 0));
    CHECK_FALSE(benchRunIsStale(2, 0));
}

TEST_CASE("a run whose frames all repeat is rejected")
{
    // The shipped failure: 300 recorded frames, 299 comparisons, all repeats.
    CHECK(benchRunIsStale(300, 299));
    CHECK(benchRunIsStale(100, 99));
}

TEST_CASE("rejection threshold is exact on either side")
{
    // 101 frames yield 100 comparisons, so the 25 percent bar is exactly 25.
    CHECK_FALSE(benchRunIsStale(101, 24));
    CHECK(benchRunIsStale(101, 25));
    CHECK(benchRunIsStale(101, 26));

    // 401 frames yield 400 comparisons, bar exactly 100.
    CHECK_FALSE(benchRunIsStale(401, 99));
    CHECK(benchRunIsStale(401, 100));

    // A fractional bar rounds in favour of accepting: 11 frames yield 10
    // comparisons and a bar of 2.5, so 2 passes and 3 fails.
    CHECK_FALSE(benchRunIsStale(11, 2));
    CHECK(benchRunIsStale(11, 3));
}

TEST_CASE("runs too short to judge are not rejected, but an empty run is")
{
    // One frame yields no comparisons, so there is nothing to detect.
    CHECK_FALSE(benchRunIsStale(1, 0));

    // No frames at all means nothing was measured, which is not a measurement
    // either: the statistics would print as a confident row of zeros.
    CHECK(benchRunIsStale(0, 0));
}
