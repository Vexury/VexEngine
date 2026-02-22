#version 450

layout(location = 0) in vec2 vUV;

layout(set = 1, binding = 0) uniform sampler2D u_accumMap;
layout(set = 2, binding = 0) uniform sampler2D u_outlineMask;

// Push constant block — only the fields this shader uses.
// Must match MeshPushConstant offsets in vk_shader.h.
layout(push_constant) uniform PC {
    layout(offset = 40) uint  flipV;         // offset 40
    layout(offset = 44) float sampleCount;   // offset 44
    layout(offset = 48) float exposure;      // offset 48
    layout(offset = 52) float gamma;         // offset 52
    layout(offset = 56) uint  enableACES;    // offset 56
    layout(offset = 64) uint  enableOutline; // offset 64
} pc;

layout(location = 0) out vec4 FragColor;

void main()
{
    vec2 uv = pc.flipV != 0u ? vec2(vUV.x, 1.0 - vUV.y) : vUV;
    vec3 c = texture(u_accumMap, uv).rgb;

    // Divide by sample count
    float s = max(pc.sampleCount, 1.0);
    c /= s;

    // Exposure (same convention as OpenGL version: pow(2, exposure))
    c *= pow(2.0, pc.exposure);

    // Tone mapping
    if (pc.enableACES != 0u)
    {
        // ACES filmic (Narkowicz fit)
        const float a = 2.51, b = 0.03, cc = 2.43, d = 0.59, e = 0.14;
        c = clamp((c * (a * c + b)) / (c * (cc * c + d) + e), 0.0, 1.0);
    }
    else
    {
        c = clamp(c, 0.0, 1.0);
    }

    // Gamma correction
    c = pow(c, vec3(1.0 / pc.gamma));

    // Screen-space outline composite (display-space overlay)
    if (pc.enableOutline != 0u)
    {
        vec2  ts   = 1.0 / vec2(textureSize(u_outlineMask, 0));
        float orig = texture(u_outlineMask, uv).r;
        float dil  = 0.0;
        for (int dx = -2; dx <= 2; dx++)
            for (int dy = -2; dy <= 2; dy++)
                dil = max(dil, texture(u_outlineMask, uv + vec2(dx, dy) * ts).r);
        float ring = dil * (1.0 - orig);
        c = mix(c, vec3(1.0, 0.5, 0.0), ring);
    }

    FragColor = vec4(c, 1.0);
}
