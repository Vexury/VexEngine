// rt.common.glsl — included by all RT shader stages
// Contains: binding declarations, payload struct, helpers (RNG, BRDF, tri accessors,
// texture/env sampling, light sampling). Each stage declares its own rayPayload.

#extension GL_EXT_ray_tracing                  : require
#extension GL_EXT_nonuniform_qualifier         : enable
#extension GL_GOOGLE_include_directive         : require

// ── Payload struct ──────────────────────────────────────────────────────────
struct HitPayload {
    vec3  position;           // world-space hit point
    float t;                  // hit distance
    vec3  normal;             // interpolated shading normal
    float roughness;
    vec3  geometricNormal;    // flat geometric normal
    float metallic;
    vec3  tangent;
    float bitangentSign;
    vec3  color;              // base color (unmodulated)
    float ior;
    vec3  emissive;
    float area;               // triangle surface area
    vec2  uv;                 // interpolated UV
    uint  triangleIndex;      // global flat tri index (for NEE)
    uint  hit;                // 0 = miss, 1 = hit
    int   textureIndex;
    int   emissiveTextureIndex;
    int   normalMapTextureIndex;
    int   roughnessTextureIndex;
    int   metallicTextureIndex;
    int   materialType;       // 0=opaque, 1=mirror, 2=dielectric
    float _pad[2];
};

// ── Descriptor set 0 ────────────────────────────────────────────────────────
layout(set = 0, binding = 0) uniform accelerationStructureEXT u_tlas;

layout(set = 0, binding = 1, rgba32f) uniform image2D u_image;

layout(set = 0, binding = 2) uniform Uniforms {
    mat4  inverseVP;         // offset   0
    vec3  cameraOrigin;      // offset  64
    float aperture;          // offset  76
    vec3  cameraRight;       // offset  80
    float focusDistance;     // offset  92
    vec3  cameraUp;          // offset  96
    uint  sampleCount;       // offset 108
    uint  width;             // offset 112
    uint  height;            // offset 116
    int   maxDepth;          // offset 120
    float rayEps;            // offset 124
    uint  enableNEE;         // offset 128
    uint  enableAA;
    uint  enableFireflyClamping;
    uint  enableEnvLighting;
    float envLightMultiplier;// offset 144
    uint  flatShading;
    uint  enableNormalMapping;
    uint  enableEmissive;
    uint  enableRR;          // offset 160
    uint  _pad0a;            // offset 164
    uint  _pad0b;            // offset 168
    uint  _pad0c;            // offset 172
    vec3  pointLightPos;     // offset 176
    uint  pointLightEnabled;
    vec3  pointLightColor;   // offset 192
    float _pad1;
    vec3  sunDir;            // offset 208
    float sunAngularRadius;
    vec3  sunColor;          // offset 224
    uint  sunEnabled;
    vec3  envColor;          // offset 240
    uint  hasEnvMap;
    int   envMapWidth;       // offset 256
    int   envMapHeight;
    uint  hasEnvCDF;
    float totalLightArea;
    uint  lightCount;        // offset 272
    uint  _pad2a;            // offset 276
    uint  _pad2b;            // offset 280
    uint  _pad2c;            // offset 284
} u_uniforms;

// TriShading: 13 vec4s per triangle (see below for layout)
layout(std430, set = 0, binding = 3) readonly buffer TriShading  { vec4  triShading[];   };

// Lights: header (lightCount, totalLightArea, pad, pad) + uint lightRawData[]
layout(std430, set = 0, binding = 4) readonly buffer Lights {
    uint  l_lightCount;
    float l_totalLightArea;
    uint  l_pad0;
    uint  l_pad1;
    uint  lightRawData[];
};

// TexData: [texCount][offset,w,h,pad per tex...][packed RGBA pixels as uint...]
layout(std430, set = 0, binding = 5) readonly buffer TexData     { uint  texHeader[];    };

// EnvMap: flat float RGB pixels (3 floats per pixel)
layout(std430, set = 0, binding = 6) readonly buffer EnvMap      { float envPixels[];    };

// EnvCDF: [marginal H floats][conditional W*H floats][totalIntegral 1 float]
layout(std430, set = 0, binding = 7) readonly buffer EnvCDF      { float envCdfData[];   };

// InstanceOffsets: first global triangle index per BLAS (indexed by gl_InstanceCustomIndexEXT)
layout(std430, set = 0, binding = 8) readonly buffer InstanceOff { uint  instanceOffsets[]; };

// ── Constants ────────────────────────────────────────────────────────────────
const float PI      = 3.14159265358979323846;
const float FLT_MAX = 3.402823466e+38;

// ── RNG (PCG) ─────────────────────────────────────────────────────────────
uint g_rngState;

uint rngHash(uint x) {
    x ^= x >> 16u; x *= 0x45d9f3bu;
    x ^= x >> 16u; x *= 0x45d9f3bu;
    x ^= x >> 16u;
    return x;
}

float rngNext() {
    g_rngState = g_rngState * 747796405u + 2891336453u;
    uint w = ((g_rngState >> ((g_rngState >> 28u) + 4u)) ^ g_rngState) * 277803737u;
    w = (w >> 22u) ^ w;
    return float(w) / 4294967296.0;
}

// ── Triangle shading accessors (13 vec4s per tri) ────────────────────────
// [0]  n0.xyz + roughnessTexIdx
// [1]  n1.xyz + metallicTexIdx
// [2]  n2.xyz + pad
// [3]  uv0.xy + uv1.zw
// [4]  uv2.xy + roughness + metallic
// [5]  color.xyz + texIdx
// [6]  emissive.xyz + area
// [7]  geoNormal.xyz + normalMapTexIdx
// [8]  alphaClip + materialType + ior + emissiveTexIdx
// [9]  tangent.xyz + bitangentSign
// [10] v0.xyz + pad
// [11] v1.xyz + pad
// [12] v2.xyz + pad
vec3  triN0(uint i)              { return triShading[i * 13u + 0u].xyz; }
int   triRoughnessTexIdx(uint i) { return floatBitsToInt(triShading[i * 13u + 0u].w); }
vec3  triN1(uint i)              { return triShading[i * 13u + 1u].xyz; }
int   triMetallicTexIdx(uint i)  { return floatBitsToInt(triShading[i * 13u + 1u].w); }
vec3  triN2(uint i)              { return triShading[i * 13u + 2u].xyz; }
vec2  triUV0(uint i)             { return triShading[i * 13u + 3u].xy; }
vec2  triUV1(uint i)             { return triShading[i * 13u + 3u].zw; }
vec2  triUV2(uint i)             { return triShading[i * 13u + 4u].xy; }
float triRoughness(uint i)       { return triShading[i * 13u + 4u].z; }
float triMetallic(uint i)        { return triShading[i * 13u + 4u].w; }
vec3  triColor(uint i)           { return triShading[i * 13u + 5u].xyz; }
int   triTexIdx(uint i)          { return floatBitsToInt(triShading[i * 13u + 5u].w); }
vec3  triEmissive(uint i)        { return triShading[i * 13u + 6u].xyz; }
float triArea(uint i)            { return triShading[i * 13u + 6u].w; }
vec3  triGeoNormal(uint i)       { return triShading[i * 13u + 7u].xyz; }
int   triNormalMapTexIdx(uint i) { return floatBitsToInt(triShading[i * 13u + 7u].w); }
bool  triAlphaClip(uint i)       { return triShading[i * 13u + 8u].x > 0.5; }
int   triMaterialType(uint i)    { return int(triShading[i * 13u + 8u].y); }
float triIOR(uint i)             { return triShading[i * 13u + 8u].z; }
int   triEmissiveTexIdx(uint i)  { return floatBitsToInt(triShading[i * 13u + 8u].w); }
vec3  triTangent(uint i)         { return triShading[i * 13u + 9u].xyz; }
float triBitangentSign(uint i)   { return triShading[i * 13u + 9u].w; }
vec3  triV0(uint i)              { return triShading[i * 13u + 10u].xyz; }
vec3  triV1(uint i)              { return triShading[i * 13u + 11u].xyz; }
vec3  triV2(uint i)              { return triShading[i * 13u + 12u].xyz; }

// ── Light accessors ──────────────────────────────────────────────────────
uint  getLightIndex(uint i) { return lightRawData[i]; }
float getLightCDF(uint i)   { return uintBitsToFloat(lightRawData[l_lightCount + i]); }

// ── Texture sampling ─────────────────────────────────────────────────────
vec4 fetchTexel(uint pixelOffset, int tw, int th, int px, int py) {
    px = clamp(px, 0, tw - 1);
    py = clamp(py, 0, th - 1);
    uint word = texHeader[pixelOffset + uint(py * tw + px)];
    return vec4(float((word >>  0u) & 0xFFu),
                float((word >>  8u) & 0xFFu),
                float((word >> 16u) & 0xFFu),
                float((word >> 24u) & 0xFFu)) / 255.0;
}

vec4 sampleTexture(int texIndex, vec2 uv) {
    if (texIndex < 0) return vec4(1.0);
    uint texCount = texHeader[0];
    if (uint(texIndex) >= texCount) return vec4(1.0);
    uint headerBase  = 1u + uint(texIndex) * 4u;
    uint pixelOffset = texHeader[headerBase + 0u];
    int  tw          = int(texHeader[headerBase + 1u]);
    int  th          = int(texHeader[headerBase + 2u]);

    // Wrap UVs and flip V
    float u = uv.x - floor(uv.x);
    float v = 1.0 - (uv.y - floor(uv.y));

    // Bilinear filtering: map to texel-center space
    float fu = u * float(tw) - 0.5;
    float fv = v * float(th) - 0.5;
    int   x0 = int(floor(fu));
    int   y0 = int(floor(fv));
    float fx = fu - float(x0);
    float fy = fv - float(y0);

    vec4 c00 = fetchTexel(pixelOffset, tw, th, x0,     y0    );
    vec4 c10 = fetchTexel(pixelOffset, tw, th, x0 + 1, y0    );
    vec4 c01 = fetchTexel(pixelOffset, tw, th, x0,     y0 + 1);
    vec4 c11 = fetchTexel(pixelOffset, tw, th, x0 + 1, y0 + 1);
    return mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
}

// ── Environment sampling ─────────────────────────────────────────────────
vec3 fetchEnvTexel(int px, int py) {
    px = clamp(px, 0, u_uniforms.envMapWidth  - 1);
    py = clamp(py, 0, u_uniforms.envMapHeight - 1);
    int idx = (py * u_uniforms.envMapWidth + px) * 3;
    return vec3(envPixels[idx], envPixels[idx+1], envPixels[idx+2]);
}

vec3 sampleEnvironment(vec3 dir) {
    if (u_uniforms.hasEnvMap != 0u && u_uniforms.envMapWidth > 0) {
        float u = 0.5 + atan(dir.z, dir.x) / (2.0 * PI);
        float v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI;
        int W = u_uniforms.envMapWidth;
        int H = u_uniforms.envMapHeight;

        float fu = u * float(W) - 0.5;
        float fv = v * float(H) - 0.5;
        int   x0 = int(floor(fu));
        int   y0 = int(floor(fv));
        float fx = fu - float(x0);
        float fy = fv - float(y0);

        vec3 c00 = fetchEnvTexel(x0,     y0    );
        vec3 c10 = fetchEnvTexel(x0 + 1, y0    );
        vec3 c01 = fetchEnvTexel(x0,     y0 + 1);
        vec3 c11 = fetchEnvTexel(x0 + 1, y0 + 1);
        return mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
    }
    return u_uniforms.envColor;
}

int binarySearchCDF(int offset, int count, float u) {
    int lo = 0, hi = count - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (envCdfData[offset + mid] < u) lo = mid + 1;
        else                              hi = mid;
    }
    return lo;
}

vec3 sampleEnvMapDirection(out vec3 outRadiance, out float outPdf) {
    int W = u_uniforms.envMapWidth;
    int H = u_uniforms.envMapHeight;
    int row = binarySearchCDF(0,           H, rngNext());
    int col = binarySearchCDF(H + row * W, W, rngNext());
    float texU = (float(col) + 0.5) / float(W);
    float texV = (float(row) + 0.5) / float(H);
    float phi   = (texU - 0.5) * 2.0 * PI;
    float theta = texV * PI;
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);
    vec3 dir = vec3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));
    int idx = (row * W + col) * 3;
    outRadiance = vec3(envPixels[idx], envPixels[idx+1], envPixels[idx+2]);
    float lum  = 0.2126*outRadiance.r + 0.7152*outRadiance.g + 0.0722*outRadiance.b;
    float totalIntegral = envCdfData[H + W * H];
    if (sinTheta < 1e-8 || totalIntegral < 1e-8 || lum < 1e-8) { outPdf = 0.0; return dir; }
    outPdf = (lum * float(W) * float(H)) / (2.0 * PI * PI * sinTheta * totalIntegral);
    return dir;
}

float envMapPdf(vec3 dir) {
    int W = u_uniforms.envMapWidth;
    int H = u_uniforms.envMapHeight;
    float u = 0.5 + atan(dir.z, dir.x) / (2.0 * PI);
    float v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI;
    int px = clamp(int(u * float(W)), 0, W - 1);
    int py = clamp(int(v * float(H)), 0, H - 1);
    int idx = (py * W + px) * 3;
    float lum = 0.2126*envPixels[idx] + 0.7152*envPixels[idx+1] + 0.0722*envPixels[idx+2];
    float sinTheta    = sin(PI * (float(py) + 0.5) / float(H));
    float totalIntegral = envCdfData[H + W * H];
    if (sinTheta < 1e-8 || totalIntegral < 1e-8) return 0.0;
    return (lum * float(W) * float(H)) / (2.0 * PI * PI * sinTheta * totalIntegral);
}

// ── ONB ──────────────────────────────────────────────────────────────────
void buildONB(vec3 n, out vec3 t, out vec3 b) {
    vec3 a = (abs(n.x) > 0.9) ? vec3(0,1,0) : vec3(1,0,0);
    t = normalize(cross(n, a));
    b = cross(n, t);
}

// ── Cook-Torrance GGX ────────────────────────────────────────────────────
float D_GGX(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}
float G1_Smith(float NdotX, float alpha) {
    float a2 = alpha * alpha;
    return 2.0 * NdotX / (NdotX + sqrt(a2 + (1.0 - a2) * NdotX * NdotX));
}
vec3 F_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 sampleVNDF(vec3 Ve, float alpha, float u1, float u2) {
    vec3  Vh = normalize(vec3(alpha * Ve.x, alpha * Ve.y, Ve.z));
    float lensq = Vh.x*Vh.x + Vh.y*Vh.y;
    vec3  T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : vec3(1,0,0);
    vec3  T2 = cross(Vh, T1);
    float r  = sqrt(u1);
    float phi = 2.0 * PI * u2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s  = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1*t1)) + s * t2;
    vec3 Nh = t1*T1 + t2*T2 + sqrt(max(0.0, 1.0 - t1*t1 - t2*t2)) * Vh;
    return normalize(vec3(alpha*Nh.x, alpha*Nh.y, max(0.0, Nh.z)));
}

vec3 ctEvaluate(vec3 N, vec3 V, vec3 L, vec3 baseColor, float alpha, float metallic, float ior) {
    float NdotL = dot(N, L);
    if (NdotL <= 0.0) return vec3(0.0);
    float NdotV = max(dot(N, V), 1e-4);
    vec3 H = normalize(V + L);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    float f = (ior - 1.0) / (ior + 1.0); f = f * f;
    vec3 F0 = mix(vec3(f), baseColor, metallic);
    float D = D_GGX(NdotH, alpha);
    float G = G1_Smith(NdotV, alpha) * G1_Smith(NdotL, alpha);
    vec3  F = F_Schlick(VdotH, F0);
    vec3 spec = D * G * F / max(4.0 * NdotV * NdotL, 1e-8);
    vec3 diff = (1.0 - F) * (1.0 - metallic) * baseColor / PI;
    return diff + spec;
}

float ctPdf(vec3 N, vec3 V, vec3 L, float alpha, float metallic) {
    float NdotL = dot(N, L);
    if (NdotL <= 0.0) return 0.0;
    float NdotV     = max(dot(N, V), 1e-4);
    float specWeight = 0.5 * (1.0 + metallic);
    vec3  H    = normalize(V + L);
    float NdotH = max(dot(N, H), 0.0);
    float specPdf = D_GGX(NdotH, alpha) * G1_Smith(NdotV, alpha) / (4.0 * NdotV);
    float diffPdf = NdotL / PI;
    return specWeight * specPdf + (1.0 - specWeight) * diffPdf;
}

vec2 sampleConcentricDisk(float u1, float u2) {
    float a = 2.0*u1 - 1.0, b = 2.0*u2 - 1.0;
    if (a == 0.0 && b == 0.0) return vec2(0.0);
    float r, phi;
    if (abs(a) > abs(b)) { r = a; phi = (PI/4.0)*(b/a); }
    else                  { r = b; phi = (PI/2.0) - (PI/4.0)*(a/b); }
    return vec2(r * cos(phi), r * sin(phi));
}

// ── Light point sampling ─────────────────────────────────────────────────
vec3 sampleLightPoint(out uint outTriIndex) {
    float u = rngNext();
    uint lo = 0u, hi = l_lightCount - 1u;
    while (lo < hi) {
        uint mid = (lo + hi) / 2u;
        if (getLightCDF(mid) < u) lo = mid + 1u; else hi = mid;
    }
    outTriIndex = getLightIndex(lo);
    vec3 v0 = triV0(outTriIndex);
    vec3 v1 = triV1(outTriIndex);
    vec3 v2 = triV2(outTriIndex);
    float su0 = sqrt(rngNext());
    float u2  = rngNext();
    return v0 * (1.0 - su0) + v1 * (su0 * (1.0 - u2)) + v2 * (su0 * u2);
}
