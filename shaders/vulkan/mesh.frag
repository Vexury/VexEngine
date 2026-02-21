#version 450

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec3 vColor;
layout(location = 3) in vec3 vEmissive;
layout(location = 4) in vec2 vUV;
layout(location = 5) in vec4 vTangent;

layout(set = 0, binding = 0) uniform UBO {
    mat4 view;
    mat4 projection;
    vec3 cameraPos;
    float _pad0;
    vec3 lightPos;
    float _pad1;
    vec3 lightColor;
    float _pad2;
    vec3 sunDirection;
    float _pad3;
    vec3 sunColor;
    float _pad4;
    vec3  envColor;
    float _pad5;
    uint  enableEnvLighting;
    float envLightMultiplier;
    uint  hasEnvMap;
    float _pad6;
};

layout(set = 1, binding = 0) uniform sampler2D u_diffuseMap;
layout(set = 2, binding = 0) uniform sampler2D u_normalMap;
layout(set = 3, binding = 0) uniform sampler2D u_roughnessMap;
layout(set = 4, binding = 0) uniform sampler2D u_metallicMap;
layout(set = 5, binding = 0) uniform sampler2D u_emissiveMap;
layout(set = 6, binding = 0) uniform sampler2D u_envMap;

layout(push_constant) uniform PC {
    uint  alphaClip;
    int   debugMode;
    float nearPlane;
    float farPlane;
    int   materialType;
    float roughness;
    float metallic;
    uint  hasNormalMap;
    uint  hasRoughnessMap;
    uint  hasMetallicMap;
    uint  flipV;          // offset 40 — unused here, keeps offsets aligned
    float sampleCount;    // offset 44 — unused here
    float exposure;       // offset 48 — unused here
    float gamma;          // offset 52 — unused here
    uint  enableACES;     // offset 56 — unused here
    uint  hasEmissiveMap; // offset 60
} pc;

layout(location = 0) out vec4 FragColor;

// --- Cook-Torrance GGX helpers ---
const float PI = 3.14159265358979323846;

float D_GGX(float NdotH, float alpha)
{
    float a2 = alpha * alpha;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float G1_Smith(float NdotX, float alpha)
{
    float a2 = alpha * alpha;
    return 2.0 * NdotX / (NdotX + sqrt(a2 + (1.0 - a2) * NdotX * NdotX));
}

float G_Smith(float NdotV, float NdotL, float alpha)
{
    return G1_Smith(NdotV, alpha) * G1_Smith(NdotL, alpha);
}

vec3 F_Schlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 cookTorranceBRDF(vec3 N, vec3 V, vec3 L, vec3 baseColor, float alpha, float metallic)
{
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.001);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    if (NdotL <= 0.0) return vec3(0.0);

    vec3 F0 = mix(vec3(0.04), baseColor, metallic);

    float D = D_GGX(NdotH, alpha);
    float G = G_Smith(NdotV, NdotL, alpha);
    vec3  F = F_Schlick(VdotH, F0);

    vec3 spec = D * G * F / (4.0 * NdotV * NdotL);
    vec3 diff = (1.0 - F) * (1.0 - metallic) * baseColor / PI;

    return (diff + spec) * NdotL;
}

void main()
{
    vec4 texColor = texture(u_diffuseMap, vUV);
    if (pc.alphaClip != 0u && texColor.a < 0.5)
        discard;

    // --- Debug modes (early-out) ---
    if (pc.debugMode == 1) // Wireframe: solid white
    {
        FragColor = vec4(1.0);
        return;
    }
    if (pc.debugMode == 2) // Depth: linearized grayscale (Vulkan NDC [0,1])
    {
        float z = gl_FragCoord.z;
        float linear = (pc.nearPlane * pc.farPlane) / (pc.farPlane - z * (pc.farPlane - pc.nearPlane));
        float d = (linear - pc.nearPlane) / (pc.farPlane - pc.nearPlane);
        FragColor = vec4(vec3(d), 1.0);
        return;
    }

    vec3 N = normalize(vNormal);

    if (pc.debugMode == 3) // Normals (geometric, not perturbed)
    {
        FragColor = vec4(N * 0.5 + 0.5, 1.0);
        return;
    }
    if (pc.debugMode == 4) // UVs
    {
        FragColor = vec4(vUV, 0.0, 1.0);
        return;
    }

    // Normal map perturbation (after debug normals, before shading)
    if (pc.hasNormalMap != 0u)
    {
        vec3 T = normalize(vTangent.xyz);
        T = normalize(T - dot(T, N) * N);  // re-orthogonalize
        vec3 B = cross(N, T) * vTangent.w;
        mat3 TBN = mat3(T, B, N);
        vec3 mapN = texture(u_normalMap, vUV).rgb * 2.0 - 1.0;
        N = normalize(TBN * mapN);
    }

    vec3 baseColor = vColor * texColor.rgb;

    if (pc.debugMode == 5) // Albedo (unlit)
    {
        FragColor = vec4(baseColor, 1.0);
        return;
    }
    if (pc.debugMode == 6) // Emission
    {
        vec3 em = vEmissive;
        if (pc.hasEmissiveMap != 0u) em += texture(u_emissiveMap, vUV).rgb;
        FragColor = vec4(em, 1.0);
        return;
    }
    if (pc.debugMode == 7) // Material ID
    {
        vec3 matColor;
        if (pc.materialType == 1)
            matColor = vec3(0.8, 0.8, 0.9); // Mirror: silver
        else if (pc.materialType == 2)
            matColor = vec3(0.2, 0.8, 0.7); // Dielectric: cyan
        else
            matColor = vec3(0.2, 0.4, 0.8); // Diffuse: blue
        FragColor = vec4(matColor, 1.0);
        return;
    }

    // --- Cook-Torrance GGX shading ---

    // Emissive surfaces glow directly
    if (length(vEmissive) > 0.001)
    {
        FragColor = vec4(vEmissive, 1.0);
        return;
    }

    vec3 V = normalize(cameraPos - vWorldPos);
    float roughness = (pc.hasRoughnessMap != 0u) ? texture(u_roughnessMap, vUV).r : pc.roughness;
    float metallic  = (pc.hasMetallicMap  != 0u) ? texture(u_metallicMap,  vUV).r : pc.metallic;
    float alpha = roughness * roughness;

    // Ambient (env-driven or fallback)
    vec3 envAmbient = (enableEnvLighting != 0u) ? envColor * envLightMultiplier : vec3(0.03);
    vec3 ambient = envAmbient * (1.0 - metallic) * baseColor;

    // Env map specular IBL
    if (enableEnvLighting != 0u && hasEnvMap != 0u)
    {
        vec3 R = reflect(-V, N);
        float eu = atan(R.z, R.x) / (2.0 * PI) + 0.5;
        float ev = asin(clamp(R.y, -1.0, 1.0)) / PI + 0.5;
        vec3 F0 = mix(vec3(0.04), baseColor, metallic);
        vec3 F  = F_Schlick(max(dot(N, V), 0.0), F0);
        ambient += F * texture(u_envMap, vec2(eu, ev)).rgb * envLightMultiplier;
    }

    // Point light
    vec3 L = normalize(lightPos - vWorldPos);
    float dist = length(lightPos - vWorldPos);
    float attenuation = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist);

    vec3 pointContrib = cookTorranceBRDF(N, V, L, baseColor, alpha, metallic) * lightColor;

    // Directional (sun) light
    vec3 sunL = normalize(-sunDirection);
    vec3 sunContrib = cookTorranceBRDF(N, V, sunL, baseColor, alpha, metallic) * sunColor;

    vec3 result = ambient
                + pointContrib * attenuation
                + sunContrib;
    if (pc.hasEmissiveMap != 0u)
        result += texture(u_emissiveMap, vUV).rgb;
    FragColor = vec4(result, 1.0);
}
