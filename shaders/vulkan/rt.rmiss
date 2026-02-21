#version 460
#include "rt.common.glsl"

layout(location = 0) rayPayloadInEXT HitPayload g_payload;

void main() {
    g_payload.hit = 0u;
}
