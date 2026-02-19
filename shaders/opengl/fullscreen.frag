#version 430 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D u_diffuseMap;
uniform bool u_flipV;   // true for sources that store pixels top-to-bottom (e.g. CPU raytracer)

void main()
{
    vec2 uv = u_flipV ? vec2(TexCoords.x, 1.0 - TexCoords.y) : TexCoords;
    FragColor = texture(u_diffuseMap, uv);
}
