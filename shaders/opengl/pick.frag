#version 430 core
out vec4 FragColor;

in vec2 vUV;

uniform int u_objectID;
uniform sampler2D u_diffuseMap;
uniform bool u_alphaClip;

void main()
{
    if (u_alphaClip && texture(u_diffuseMap, vUV).a < 0.5)
        discard;

    // Encode object ID + 1 across RGB channels (24-bit, supports ~16M objects)
    int encoded = u_objectID + 1;
    FragColor = vec4(float((encoded >> 16) & 0xFF) / 255.0,
                     float((encoded >>  8) & 0xFF) / 255.0,
                     float( encoded        & 0xFF) / 255.0,
                     1.0);
}
