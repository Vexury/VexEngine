#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 FragColor;

layout(set = 1, binding = 0) uniform sampler2D u_image;

// Uses bloomHorizontal field from the shared MeshPushConstant.
// Must match offset in vk_shader.h.
layout(push_constant) uniform PC {
    layout(offset = 80) uint horizontal;
} pc;

// 9-tap separable Gaussian (σ≈1.5)
const float weight[5] = float[](0.2270270, 0.1945946, 0.1216216, 0.0540541, 0.0162162);

void main()
{
    vec2 texOffset = 1.0 / textureSize(u_image, 0);
    vec3 result = texture(u_image, vUV).rgb * weight[0];

    if (pc.horizontal != 0u)
    {
        for (int i = 1; i < 5; ++i)
        {
            result += texture(u_image, vUV + vec2(texOffset.x * i, 0.0)).rgb * weight[i];
            result += texture(u_image, vUV - vec2(texOffset.x * i, 0.0)).rgb * weight[i];
        }
    }
    else
    {
        for (int i = 1; i < 5; ++i)
        {
            result += texture(u_image, vUV + vec2(0.0, texOffset.y * i)).rgb * weight[i];
            result += texture(u_image, vUV - vec2(0.0, texOffset.y * i)).rgb * weight[i];
        }
    }

    FragColor = vec4(result, 1.0);
}
