#include <vex/core/input.h>
#include <GLFW/glfw3.h>

namespace vex
{

bool Input::isKeyPressed(GLFWwindow* window, int key)
{
    return glfwGetKey(window, key) == GLFW_PRESS;
}

bool Input::isMouseButtonPressed(GLFWwindow* window, MouseButton button)
{
    int glfwButton = GLFW_MOUSE_BUTTON_LEFT;
    switch (button)
    {
        case MouseButton::Left:   glfwButton = GLFW_MOUSE_BUTTON_LEFT;   break;
        case MouseButton::Right:  glfwButton = GLFW_MOUSE_BUTTON_RIGHT;  break;
        case MouseButton::Middle: glfwButton = GLFW_MOUSE_BUTTON_MIDDLE; break;
        default: return false;
    }
    return glfwGetMouseButton(window, glfwButton) == GLFW_PRESS;
}

void Input::getCursorPosition(GLFWwindow* window, double& x, double& y)
{
    glfwGetCursorPos(window, &x, &y);
}

} // namespace vex
