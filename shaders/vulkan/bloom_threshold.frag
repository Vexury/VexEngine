#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 FragColor;

layout(set = 1, binding = 0) uniform sampler2D u_hdrMap;

// Uses bloomThreshold field from the shared MeshPushConstant.
// Must match offset in vk_shader.h.
// Fragment push constants start at offset 64 (after 64-byte vertex model matrix).
layout(push_constant) uniform PC {
    layout(offset = 108) float sampleCount; // MeshPC offset 44 + 64
    layout(offset = 140) float threshold;   // MeshPC offset 76 + 64
} pc;

void main()
{
    vec3  color      = texture(u_hdrMap, vUV).rgb / max(pc.sampleCount, 1.0);
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    // Soft threshold: smoothly ramp contribution above the threshold
    float factor     = max(0.0, brightness - pc.threshold) / max(brightness, 0.0001);
    FragColor = vec4(color * factor, 1.0);
}
