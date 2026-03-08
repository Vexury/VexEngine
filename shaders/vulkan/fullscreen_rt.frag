#version 450

layout(location = 0) in vec2 vUV;

layout(set = 1, binding = 0) uniform sampler2D u_accumMap;
layout(set = 2, binding = 0) uniform sampler2D u_outlineMask;
layout(set = 3, binding = 0) uniform sampler2D u_bloomMap;

// Push constant block — fragment fields start at offset 64 (after 64-byte vertex model matrix).
// Must match MeshPushConstant offsets in vk_shader.h (+64).
layout(push_constant) uniform PC {
    layout(offset = 104) uint  flipV;          // MeshPC offset 40 + 64
    layout(offset = 108) float sampleCount;    // MeshPC offset 44 + 64
    layout(offset = 112) float exposure;       // MeshPC offset 48 + 64
    layout(offset = 116) float gamma;          // MeshPC offset 52 + 64
    layout(offset = 120) uint  enableACES;     // MeshPC offset 56 + 64
    layout(offset = 128) uint  enableOutline;  // MeshPC offset 64 + 64
    layout(offset = 132) uint  enableBloom;    // MeshPC offset 68 + 64
    layout(offset = 136) float bloomIntensity; // MeshPC offset 72 + 64
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

    // Bloom composite (HDR linear space, before tone mapping)
    // The bloom map is produced by sampling a source texture through the Y-flip viewport,
    // which inverts its Y relative to geometry-rendered textures (accumMap, outlineMask).
    // It must always be sampled with raw vUV — never with the flipV-corrected uv.
    if (pc.enableBloom != 0u)
        c += texture(u_bloomMap, vUV).rgb * pc.bloomIntensity;

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
    // The outline mask is rendered by geometry (same pipeline as the HDR framebuffer),
    // so row 0 = top of scene — standard Vulkan UV orientation, same as accumMap.
    // Must be sampled with the flipV-corrected uv, not raw vUV.
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
