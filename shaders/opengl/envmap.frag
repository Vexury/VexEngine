#version 430 core
out vec4 FragColor;

in vec2 vScreenPos;

uniform mat4  u_inverseVP;
uniform float u_envRotation;
uniform sampler2D u_envmap;

const float PI = 3.14159265359;

void main()
{
    // Reconstruct world-space ray direction from screen position
    vec4 worldPos = u_inverseVP * vec4(vScreenPos, 1.0, 1.0);
    vec3 dir = normalize(worldPos.xyz / worldPos.w);

    // Equirectangular mapping (with Y-axis rotation)
    float u = fract(0.5 + (atan(dir.z, dir.x) + u_envRotation) / (2.0 * PI));
    float v = asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5;

    // textureLod bypasses derivative-based mip selection.
    // At the atan2 seam, dFdx(u) ≈ -1 which would select the highest mip
    // (a blurry band). Mip 0 is always the right choice for a skybox.
    FragColor = textureLod(u_envmap, vec2(u, v), 0.0);
}
