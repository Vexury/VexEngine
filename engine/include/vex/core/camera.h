#pragma once

#include <glm/glm.hpp>

namespace vex
{

class Camera
{
public:
    void setOrbit(const glm::vec3& target, float distance, float yaw, float pitch);
    void rotate(float deltaYaw, float deltaPitch);
    void zoom(float delta);

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix(float aspectRatio) const;
    glm::vec3 getPosition() const;

    const glm::vec3& getTarget() const { return m_target; }
    glm::vec3& getTarget() { return m_target; }

    float getDistance() const { return m_distance; }
    float& getDistance() { return m_distance; }

    float fov = 45.0f;
    float nearPlane = 0.01f;
    float farPlane = 1000.0f;

private:
    glm::vec3 m_target = { 0.0f, 1.0f, 0.0f };
    float m_distance = 4.0f;
    float m_yaw = 0.0f;
    float m_pitch = 0.0f;
};

} // namespace vex
