#version 430 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D u_accumMap;
uniform float u_sampleCount;
uniform float u_exposure;
uniform float u_gamma;
uniform bool  u_enableACES;
uniform bool  u_flipV;   // true when sampling an OpenGL framebuffer texture (row 0 = bottom)
uniform sampler2D u_outlineMask;
uniform bool  u_enableOutline;

void main()
{
    vec2 uv = u_flipV ? vec2(TexCoords.x, 1.0 - TexCoords.y) : TexCoords;
    vec3 c = texture(u_accumMap, uv).rgb;

    // Divide by sample count
    if (u_sampleCount > 0.0)
        c /= u_sampleCount;

    // Exposure
    c *= pow(2.0, u_exposure);

    // Tone mapping
    if (u_enableACES)
    {
        // ACES filmic tone mapping (Narkowicz fit)
        const float a = 2.51, b = 0.03, cc = 2.43, d = 0.59, e = 0.14;
        c = clamp((c * (a * c + b)) / (c * (cc * c + d) + e), 0.0, 1.0);
    }
    else
    {
        c = clamp(c, 0.0, 1.0);
    }

    // Gamma correction
    float invGamma = 1.0 / u_gamma;
    c = pow(c, vec3(invGamma));

    // Screen-space outline composite (display-space overlay)
    if (u_enableOutline)
    {
        vec2 ts   = 1.0 / vec2(textureSize(u_outlineMask, 0));
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
