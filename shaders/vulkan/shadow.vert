#version 450

layout(location = 0) in vec3 aPos;

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
    mat4  model;
};

void main()
{
    gl_Position = sunShadowVP * model * vec4(aPos, 1.0);
}
