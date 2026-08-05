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
}
