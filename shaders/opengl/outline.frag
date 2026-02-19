#version 430 core
out vec4 FragColor;

uniform vec3 u_outlineColor;

void main()
{
    FragColor = vec4(u_outlineColor, 1.0);
}
