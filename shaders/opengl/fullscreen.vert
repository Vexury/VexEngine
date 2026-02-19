#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec3 aEmissive;
layout(location = 4) in vec2 aUV;

out vec2 TexCoords;

void main()
{
    TexCoords = aUV;
    gl_Position = vec4(aPos, 1.0);
}
