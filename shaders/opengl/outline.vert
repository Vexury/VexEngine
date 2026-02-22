#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec3 aEmissive;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_outlineWidth;

void main()
{
    vec4 clipPos  = u_projection * u_view * vec4(aPos, 1.0);
    vec4 clipNorm = u_projection * u_view * vec4(aNormal, 0.0);
    // Offset in clip space so the outline is u_outlineWidth NDC units thick
    // regardless of camera distance (multiply by w to cancel perspective divide).
    clipPos.xy += normalize(clipNorm.xy) * u_outlineWidth * clipPos.w;
    gl_Position = clipPos;
}
