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

    // Encode object ID + 1 into the red channel (0 = background)
    float id = float(u_objectID + 1) / 255.0;
    FragColor = vec4(id, 0.0, 0.0, 1.0);
}
