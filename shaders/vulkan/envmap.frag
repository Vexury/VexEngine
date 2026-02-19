#version 450

layout(location = 0) in vec2 vScreenPos;

layout(set = 0, binding = 0) uniform EnvUBO {
    mat4 inverseVP;
};

layout(set = 0, binding = 1) uniform sampler2D u_envmap;

layout(location = 0) out vec4 FragColor;

const float PI = 3.14159265359;

void main()
{
    // Reconstruct world-space ray direction from screen position
    vec4 worldPos = inverseVP * vec4(vScreenPos, 1.0, 1.0);
    vec3 dir = normalize(worldPos.xyz / worldPos.w);

    // Equirectangular mapping
    float u = atan(dir.z, dir.x) / (2.0 * PI) + 0.5;
    float v = -asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5;

    FragColor = texture(u_envmap, vec2(u, v));
}
