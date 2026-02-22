#version 450

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec3 aEmissive;
layout(location = 4) in vec2 aUV;
layout(location = 5) in vec4 aTangent;

layout(set = 0, binding = 0) uniform UBO {
    mat4 view;
    mat4 projection;
    vec3 cameraPos;
    float _pad0;
    vec3 lightPos;
    float _pad1;
    vec3 lightColor;
    float _pad2;
    vec3 sunDirection;
    float _pad3;
    vec3 sunColor;
    float _pad4;
    vec3  envColor;
    float _pad5;
    uint  enableEnvLighting;
    float envLightMultiplier;
    uint  hasEnvMap;
    float _pad6;
    mat4  sunShadowVP;
    uint  enableShadows;
    float _pad7a;
    float _pad7b;
    float _pad7c;
};

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec3 vColor;
layout(location = 3) out vec3 vEmissive;
layout(location = 4) out vec2 vUV;
layout(location = 5) out vec4 vTangent;

void main()
{
    vWorldPos = aPos;
    vNormal = aNormal;
    vColor = aColor;
    vEmissive = aEmissive;
    vUV = aUV;
    vTangent = aTangent;
    gl_Position = projection * view * vec4(aPos, 1.0);
}
