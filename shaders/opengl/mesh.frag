#version 430 core
out vec4 FragColor;

in vec3 vWorldPos;
in vec3 vNormal;
in vec3 vColor;
in vec3 vEmissive;
in vec2 vUV;
in vec4 vTangent;

uniform vec3 u_cameraPos;
uniform vec3 u_lightPos;
uniform vec3 u_lightColor;
uniform vec3 u_sunDirection;
uniform vec3 u_sunColor;
uniform sampler2D u_diffuseMap;
uniform sampler2D u_normalMap;
uniform sampler2D u_roughnessMap;
uniform sampler2D u_metallicMap;
uniform sampler2D u_emissiveMap;
uniform sampler2D u_envMap;
uniform bool u_alphaClip;
uniform bool u_hasNormalMap;
uniform bool u_hasRoughnessMap;
uniform bool u_hasMetallicMap;
uniform bool u_hasEmissiveMap;
uniform bool u_hasEnvMap;
uniform bool u_enableEnvLighting;
uniform vec3  u_envColor;
uniform float u_envLightMultiplier;

uniform mat4            u_shadowViewProj;
uniform sampler2DShadow u_shadowMap;
uniform bool            u_enableShadows;
uniform float           u_shadowNormalBias;

uniform int   u_debugMode;
uniform float u_nearPlane;
uniform float u_farPlane;
uniform int   u_materialType;
uniform float u_roughness;
uniform float u_metallic;

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
    if (u_alphaClip && texColor.a < 0.5)
        discard;

    // --- Debug modes (early-out) ---
    if (u_debugMode == 1) // Wireframe: solid white (polygon mode set on CPU)
    {
        FragColor = vec4(1.0);
        return;
    }
    if (u_debugMode == 2) // Depth: linearized grayscale
    {
        float z = gl_FragCoord.z * 2.0 - 1.0;
        float linear = (2.0 * u_nearPlane * u_farPlane) / (u_farPlane + u_nearPlane - z * (u_farPlane - u_nearPlane));
        float d = (linear - u_nearPlane) / (u_farPlane - u_nearPlane);
        FragColor = vec4(vec3(d), 1.0);
        return;
    }

    vec3 N = normalize(vNormal);

    if (u_debugMode == 3) // Normals (geometric, not perturbed)
    {
        FragColor = vec4(N * 0.5 + 0.5, 1.0);
        return;
    }
    if (u_debugMode == 4) // UVs
    {
        FragColor = vec4(vUV, 0.0, 1.0);
        return;
    }

    // Normal map perturbation (after debug normals, before shading)
    if (u_hasNormalMap)
    {
        vec3 T = normalize(vTangent.xyz);
        T = normalize(T - dot(T, N) * N);  // re-orthogonalize
        vec3 B = cross(N, T) * vTangent.w;
        mat3 TBN = mat3(T, B, N);
        vec3 mapN = texture(u_normalMap, vUV).rgb * 2.0 - 1.0;
        N = normalize(TBN * mapN);
    }

    vec3 baseColor = vColor * texColor.rgb;

    if (u_debugMode == 5) // Albedo (unlit)
    {
        FragColor = vec4(baseColor, 1.0);
        return;
    }
    if (u_debugMode == 6) // Emission
    {
        vec3 em = vEmissive;
        if (u_hasEmissiveMap) em += texture(u_emissiveMap, vUV).rgb;
        FragColor = vec4(em, 1.0);
        return;
    }
    if (u_debugMode == 7) // Material ID
    {
        vec3 matColor;
        if (u_materialType == 1)
            matColor = vec3(0.8, 0.8, 0.9); // Mirror: silver
        else if (u_materialType == 2)
            matColor = vec3(0.2, 0.8, 0.7); // Dielectric: cyan
        else
            matColor = vec3(0.2, 0.4, 0.8); // Diffuse: blue
        FragColor = vec4(matColor, 1.0);
        return;
    }

    // --- Cook-Torrance GGX shading ---

    vec3 V = normalize(u_cameraPos - vWorldPos);
    float roughness = u_hasRoughnessMap ? texture(u_roughnessMap, vUV).r : u_roughness;
    float metallic  = u_hasMetallicMap  ? texture(u_metallicMap,  vUV).r : u_metallic;
    float alpha = roughness * roughness;

    // Ambient (env-driven or fallback)
    vec3 envAmbient = u_enableEnvLighting ? u_envColor * u_envLightMultiplier : vec3(0.03);
    vec3 ambient = envAmbient * (1.0 - metallic) * baseColor;

    if (u_enableEnvLighting && u_hasEnvMap)
    {
        vec3 R = reflect(-V, N);
        float eu = atan(R.z, R.x) / (2.0 * PI) + 0.5;
        float ev = asin(clamp(R.y, -1.0, 1.0)) / PI + 0.5;
        vec3 F0 = mix(vec3(0.04), baseColor, metallic);
        vec3 F  = F_Schlick(max(dot(N, V), 0.0), F0);
        ambient += F * texture(u_envMap, vec2(eu, ev)).rgb * u_envLightMultiplier;
    }

    // Point light
    vec3 L = normalize(u_lightPos - vWorldPos);
    float dist = length(u_lightPos - vWorldPos);
    float attenuation = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist);

    vec3 pointContrib = cookTorranceBRDF(N, V, L, baseColor, alpha, metallic) * u_lightColor;

    // Directional (sun) light
    vec3 sunL = normalize(-u_sunDirection);
    vec3 sunContrib = cookTorranceBRDF(N, V, sunL, baseColor, alpha, metallic) * u_sunColor;

    // Shadow map (PCF 3x3)
    // Shadow map (PCF 3x3) with normal offset bias.
    // Shifts the lookup point along the surface normal by sin(theta) * shadowNormalBias,
    // where sin(theta) = sqrt(1 - NdotL^2). At grazing angles (NdotL→0) the offset is
    // maximum; when facing the light (NdotL=1) it is zero. shadowNormalBias is pre-scaled
    // to world-space texel size on the CPU, making it resolution and frustum invariant.
    // PCF 3x3 with normal offset + receiver plane depth bias (RPDB). See VK shader for details.
    if (u_enableShadows)
    {
        float nDotL    = max(dot(N, sunL), 0.0);
        float sinTheta = sqrt(max(0.0, 1.0 - nDotL * nDotL));
        vec3  biasedPos = vWorldPos + N * (u_shadowNormalBias * sinTheta);

        vec4 shadowClip = u_shadowViewProj * vec4(biasedPos, 1.0);
        vec3 shadowCoord = shadowClip.xyz / shadowClip.w;
        shadowCoord.xy = shadowCoord.xy * 0.5 + 0.5;
        shadowCoord.z  = shadowCoord.z  * 0.5 + 0.5;

        vec3  dsc_dx = dFdx(shadowCoord);
        vec3  dsc_dy = dFdy(shadowCoord);
        float det    = dsc_dx.x * dsc_dy.y - dsc_dy.x * dsc_dx.y;
        vec2  dzdUV  = vec2(0.0);
        if (abs(det) > 1e-10)
        {
            dzdUV.x = dsc_dy.y * dsc_dx.z - dsc_dx.y * dsc_dy.z;
            dzdUV.y = dsc_dx.x * dsc_dy.z - dsc_dy.x * dsc_dx.z;
            dzdUV  /= det;
        }

        float shadow    = 0.0;
        vec2  texelSize = 1.0 / vec2(textureSize(u_shadowMap, 0));
        for (int x = -1; x <= 1; ++x)
            for (int y = -1; y <= 1; ++y)
            {
                vec2  tapOffset = vec2(x, y) * texelSize;
                float tapZ = shadowCoord.z + clamp(dot(tapOffset, dzdUV), -0.005, 0.005);
                shadow += texture(u_shadowMap, vec3(shadowCoord.xy + tapOffset, tapZ));
            }
        shadow /= 9.0;
        sunContrib *= shadow;
    }

    vec3 result = ambient
                + pointContrib * attenuation
                + sunContrib;
    result += vEmissive;
    if (u_hasEmissiveMap)
        result += texture(u_emissiveMap, vUV).rgb;
    FragColor = vec4(result, 1.0);
}
