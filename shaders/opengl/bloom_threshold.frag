#version 430 core
out vec4 FragColor;
in  vec2 TexCoords;

uniform sampler2D u_hdrMap;
uniform float     u_threshold;
uniform float     u_sampleCount;

void main()
{
    vec3  color      = texture(u_hdrMap, TexCoords).rgb / max(u_sampleCount, 1.0);
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    // Soft threshold: smoothly ramp contribution above the threshold
    float factor     = max(0.0, brightness - u_threshold) / max(brightness, 0.0001);
    FragColor = vec4(color * factor, 1.0);
}
