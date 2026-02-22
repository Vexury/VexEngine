#version 430 core
layout(location = 0) in vec3 aPos;

uniform mat4 u_lightViewProj;

void main()
{
    gl_Position = u_lightViewProj * vec4(aPos, 1.0);
}
