#include <doctest/doctest.h>
#include <vex/core/camera.h>

#include <glm/glm.hpp>
#include <cmath>

using namespace vex;

static constexpr float PI = 3.14159265358979323846f;

TEST_SUITE("Camera")
{

TEST_CASE("position is at the correct distance from target")
{
    Camera cam;
    for (float dist : {0.5f, 1.0f, 5.0f, 100.0f})
    {
        cam.setOrbit({0, 0, 0}, dist, 0.3f, 0.4f);
        float actual = glm::length(cam.getPosition() - cam.getTarget());
        CHECK(actual == doctest::Approx(dist).epsilon(1e-4f));
    }
}

TEST_CASE("position at yaw=0 pitch=0 is directly behind target on +Z")
{
    // Spherical: x=dist*cos(0)*sin(0)=0, y=dist*sin(0)=0, z=dist*cos(0)*cos(0)=dist
    Camera cam;
    cam.setOrbit({0, 0, 0}, 5.0f, 0.0f, 0.0f);
    glm::vec3 pos = cam.getPosition();
    CHECK(pos.x == doctest::Approx(0.0f).epsilon(1e-5f));
    CHECK(pos.y == doctest::Approx(0.0f).epsilon(1e-5f));
    CHECK(pos.z == doctest::Approx(5.0f).epsilon(1e-5f));
}

TEST_CASE("position at yaw=PI/2 pitch=0 is beside target on +X")
{
    // Spherical: x=dist*cos(0)*sin(PI/2)=dist, y=0, z=dist*cos(0)*cos(PI/2)=0
    Camera cam;
    cam.setOrbit({0, 0, 0}, 5.0f, PI / 2.0f, 0.0f);
    glm::vec3 pos = cam.getPosition();
    CHECK(pos.x == doctest::Approx(5.0f).epsilon(1e-4f));
    CHECK(pos.y == doctest::Approx(0.0f).epsilon(1e-4f));
    CHECK(pos.z == doctest::Approx(0.0f).epsilon(1e-4f));
}

TEST_CASE("position distance is preserved after full yaw rotation")
{
    Camera cam;
    cam.setOrbit({1, 2, 3}, 7.0f, 0.0f, 0.5f);
    cam.rotate(2.0f * PI, 0.0f); // full circle
    float dist = glm::length(cam.getPosition() - cam.getTarget());
    CHECK(dist == doctest::Approx(7.0f).epsilon(1e-3f));
}

TEST_CASE("pitch is clamped to [-1.5, 1.5]")
{
    Camera cam;
    cam.setOrbit({0, 0, 0}, 1.0f, 0.0f, 0.0f);
    cam.rotate(0.0f,  100.0f); // clamp up
    CHECK(cam.getPosition().y <= 1.0f + 1e-4f); // y = sin(pitch) → sin(1.5) < 1
    cam.rotate(0.0f, -200.0f); // clamp down
    CHECK(cam.getPosition().y >= -1.0f - 1e-4f);
}

TEST_CASE("zoom never drops distance below 0.01")
{
    Camera cam;
    cam.setOrbit({0, 0, 0}, 1.0f, 0.0f, 0.0f);
    for (int i = 0; i < 200; ++i)
        cam.zoom(10.0f); // aggressive zoom in
    CHECK(cam.getDistance() >= 0.01f - 1e-6f);
}

TEST_CASE("zoom out increases distance")
{
    Camera cam;
    cam.setOrbit({0, 0, 0}, 2.0f, 0.0f, 0.0f);
    float before = cam.getDistance();
    cam.zoom(-1.0f); // negative = zoom out
    CHECK(cam.getDistance() > before);
}

TEST_CASE("view matrix upper-3x3 is orthonormal")
{
    Camera cam;
    cam.setOrbit({1, 2, 3}, 5.0f, 0.7f, 0.3f);
    glm::mat3 R(cam.getViewMatrix());
    glm::mat3 RRt = R * glm::transpose(R);

    // Diagonal should be ~1, off-diagonal ~0
    CHECK(RRt[0][0] == doctest::Approx(1.0f).epsilon(1e-4f));
    CHECK(RRt[1][1] == doctest::Approx(1.0f).epsilon(1e-4f));
    CHECK(RRt[2][2] == doctest::Approx(1.0f).epsilon(1e-4f));
    CHECK(RRt[0][1] == doctest::Approx(0.0f).epsilon(1e-4f));
    CHECK(RRt[0][2] == doctest::Approx(0.0f).epsilon(1e-4f));
    CHECK(RRt[1][2] == doctest::Approx(0.0f).epsilon(1e-4f));
}

TEST_CASE("view matrix transforms camera position to the origin")
{
    // V * eye = (0,0,0) — eye is always at origin in view space
    Camera cam;
    cam.setOrbit({0, 1, 0}, 4.0f, 1.0f, 0.4f);
    glm::mat4 V   = cam.getViewMatrix();
    glm::vec4 pos = V * glm::vec4(cam.getPosition(), 1.0f);
    CHECK(pos.x == doctest::Approx(0.0f).epsilon(1e-4f));
    CHECK(pos.y == doctest::Approx(0.0f).epsilon(1e-4f));
    CHECK(pos.z == doctest::Approx(0.0f).epsilon(1e-4f));
}

TEST_CASE("target is in front of the camera (negative Z in view space)")
{
    // glm uses right-handed coords: forward = -Z in view space
    Camera cam;
    cam.setOrbit({0, 0, 0}, 5.0f, 0.5f, 0.2f);
    glm::mat4    V      = cam.getViewMatrix();
    glm::vec4    target = V * glm::vec4(cam.getTarget(), 1.0f);
    CHECK(target.z < 0.0f);
}

TEST_CASE("target is at the expected depth in view space")
{
    // V * target should be at (0, 0, -distance) since target is directly in front
    Camera cam;
    float dist = 6.0f;
    cam.setOrbit({0, 0, 0}, dist, 0.0f, 0.0f);
    glm::mat4 V      = cam.getViewMatrix();
    glm::vec4 target = V * glm::vec4(cam.getTarget(), 1.0f);
    CHECK(target.x == doctest::Approx(0.0f).epsilon(1e-4f));
    CHECK(target.y == doctest::Approx(0.0f).epsilon(1e-4f));
    CHECK(target.z == doctest::Approx(-dist).epsilon(1e-3f));
}

} // TEST_SUITE("Camera")
