#version 450

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec3 aEmissive;
layout(location = 4) in vec2 aUV;

layout(set = 0, binding = 0) uniform UBO {
    mat4 view;
    mat4 projection;
    vec3 cameraPos;
    float _pad0;
    vec3 lightPos;
    float _pad1;
    vec3 lightColor;
    float _pad2;
};

layout(location = 0) out vec2 vUV;

void main()
{
    vUV = aUV;
    gl_Position = vec4(aPos, 1.0);
}
