#pragma once

struct GLFWwindow;

namespace vex
{

enum class MouseButton { None, Left, Right, Middle };

class Input
{
public:
    static bool isKeyPressed(GLFWwindow* window, int key);
    static bool isMouseButtonPressed(GLFWwindow* window, MouseButton button);
    static void getCursorPosition(GLFWwindow* window, double& x, double& y);
};

} // namespace vex
