#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec3 aEmissive;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_outlineWidth;

void main()
{
    vec3 pos = aPos + normalize(aNormal) * u_outlineWidth;
    gl_Position = u_projection * u_view * vec4(pos, 1.0);
}
