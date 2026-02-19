#version 430 core
layout(location = 0) in vec2 aPos;

out vec2 vScreenPos;

void main()
{
    vScreenPos = aPos;
    gl_Position = vec4(aPos, 1.0, 1.0); // z=1 (far plane after perspective divide)
}
