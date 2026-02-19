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

layout(location = 0) out vec4 FragColor;

void main()
{
    FragColor = texture(u_diffuseMap, vUV);
}
