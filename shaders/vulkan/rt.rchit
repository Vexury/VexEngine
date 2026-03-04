#version 460
#include "rt.common.glsl"

layout(location = 0) rayPayloadInEXT HitPayload g_payload;
hitAttributeEXT vec2 g_barycentrics;

void main() {
    g_payload.hit    = 1u;
    g_payload.t      = gl_HitTEXT;
    g_payload.triIdx = instanceOffsets[gl_InstanceCustomIndexEXT] + uint(gl_PrimitiveID);
    g_payload.bary_u = g_barycentrics.x;
    g_payload.bary_v = g_barycentrics.y;
}
