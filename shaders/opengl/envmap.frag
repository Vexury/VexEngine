#version 430 core
out vec4 FragColor;

in vec2 vScreenPos;

uniform mat4 u_inverseVP;
uniform sampler2D u_envmap;

const float PI = 3.14159265359;

void main()
{
    // Reconstruct world-space ray direction from screen position
    vec4 worldPos = u_inverseVP * vec4(vScreenPos, 1.0, 1.0);
    vec3 dir = normalize(worldPos.xyz / worldPos.w);

    // Equirectangular mapping
    float u = atan(dir.z, dir.x) / (2.0 * PI) + 0.5;
    float v = asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5;

    FragColor = texture(u_envmap, vec2(u, v));
}
