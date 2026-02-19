#include <vex/core/camera.h>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>

namespace vex
{

void Camera::setOrbit(const glm::vec3& target, float distance, float yaw, float pitch)
{
    m_target = target;
    m_distance = distance;
    m_yaw = yaw;
    m_pitch = pitch;
}

void Camera::rotate(float deltaYaw, float deltaPitch)
{
    m_yaw += deltaYaw;
    m_pitch = std::clamp(m_pitch + deltaPitch, -1.5f, 1.5f);
}

void Camera::zoom(float delta)
{
    float factor = 1.0f - delta * 0.15f;
    m_distance = std::max(0.01f, m_distance * factor);
}

glm::vec3 Camera::getPosition() const
{
    float x = m_distance * std::cos(m_pitch) * std::sin(m_yaw);
    float y = m_distance * std::sin(m_pitch);
    float z = m_distance * std::cos(m_pitch) * std::cos(m_yaw);
    return m_target + glm::vec3(x, y, z);
}

glm::mat4 Camera::getViewMatrix() const
{
    return glm::lookAt(getPosition(), m_target, glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 Camera::getProjectionMatrix(float aspectRatio) const
{
    return glm::perspective(glm::radians(fov), aspectRatio, nearPlane, farPlane);
}

} // namespace vex
