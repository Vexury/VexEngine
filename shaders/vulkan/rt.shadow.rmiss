#version 460
#extension GL_EXT_ray_tracing         : require
#extension GL_GOOGLE_include_directive : require

// Shadow payload: 1 = occluded (set by rgen before trace), 0 = clear (set here on miss)
layout(location = 1) rayPayloadInEXT uint g_shadowed;

void main() {
    g_shadowed = 0u;
}
