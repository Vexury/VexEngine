#version 450

layout(location = 0) in vec2 vUV;

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

layout(set = 1, binding = 0) uniform sampler2D u_diffuseMap;

layout(push_constant) uniform PC {
    layout(offset = 40) uint flipV;
} pc;

layout(location = 0) out vec4 FragColor;

void main()
{
    vec2 uv = pc.flipV != 0u ? vec2(vUV.x, 1.0 - vUV.y) : vUV;
    FragColor = texture(u_diffuseMap, uv);
}
