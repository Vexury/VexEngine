#version 460
#include "rt.common.glsl"

layout(location = 0) rayPayloadInEXT HitPayload g_payload;
hitAttributeEXT vec2 g_barycentrics;

void main() {
    // Global triangle index: per-BLAS base + local primitive index
    uint triIdx = instanceOffsets[gl_InstanceCustomIndexEXT] + uint(gl_PrimitiveID);

    float u = g_barycentrics.x;
    float v = g_barycentrics.y;
    float w = 1.0 - u - v;

    // Interpolated UV
    vec2 uv = w * triUV0(triIdx) + u * triUV1(triIdx) + v * triUV2(triIdx);

    // World-space hit position from the ray equation (correct regardless of instance transform)
    vec3 hitPos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;

    // Geometric & interpolated shading normals
    vec3 geoN = triGeoNormal(triIdx);
    vec3 shadingN = (u_uniforms.flatShading != 0u)
        ? geoN
        : normalize(w * triN0(triIdx) + u * triN1(triIdx) + v * triN2(triIdx));

    // Fill payload
    g_payload.hit                   = 1u;
    g_payload.t                     = gl_HitTEXT;
    g_payload.position              = hitPos;
    g_payload.normal                = shadingN;
    g_payload.geometricNormal       = geoN;
    g_payload.uv                    = uv;
    g_payload.triangleIndex         = triIdx;
    g_payload.color                 = triColor(triIdx);
    g_payload.emissive              = triEmissive(triIdx);
    g_payload.area                  = triArea(triIdx);
    g_payload.roughness             = triRoughness(triIdx);
    g_payload.metallic              = triMetallic(triIdx);
    g_payload.ior                   = triIOR(triIdx);
    g_payload.tangent               = triTangent(triIdx);
    g_payload.bitangentSign         = triBitangentSign(triIdx);
    g_payload.textureIndex          = triTexIdx(triIdx);
    g_payload.emissiveTextureIndex  = triEmissiveTexIdx(triIdx);
    g_payload.normalMapTextureIndex = triNormalMapTexIdx(triIdx);
    g_payload.roughnessTextureIndex = triRoughnessTexIdx(triIdx);
    g_payload.metallicTextureIndex  = triMetallicTexIdx(triIdx);
    g_payload.materialType          = triMaterialType(triIdx);
}
