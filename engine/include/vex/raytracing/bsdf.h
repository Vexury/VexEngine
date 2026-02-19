#pragma once

#include <glm/glm.hpp>

#include <cmath>

namespace vex
{

constexpr float PI = 3.14159265358979323846f;

struct BSDFSample
{
    glm::vec3 direction;
    glm::vec3 throughput; // BRDF * cos_theta / pdf
    float pdf;
};

inline void buildONB(const glm::vec3& n, glm::vec3& t, glm::vec3& b)
{
    glm::vec3 a = (std::abs(n.x) > 0.9f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    t = glm::normalize(glm::cross(n, a));
    b = glm::cross(n, t);
}

struct MirrorBSDF
{
    glm::vec3 color; // albedo tint

    BSDFSample sample(const glm::vec3& normal, const glm::vec3& wo) const
    {
        glm::vec3 wi = glm::reflect(-wo, normal);
        return { wi, color, 1.0f }; // pdf = 1.0 (delta sentinel)
    }
};

struct DielectricBSDF
{
    glm::vec3 color; // transmission tint
    float ior;

    BSDFSample sample(const glm::vec3& normal, const glm::vec3& wo,
                      bool frontFace, float u_rand) const
    {
        float etaI = frontFace ? 1.0f : ior;
        float etaT = frontFace ? ior : 1.0f;
        float eta = etaI / etaT;

        float cosI = glm::dot(normal, wo);
        if (cosI < 0.0f) cosI = 0.0f;

        // Schlick approximation
        float F0 = (etaI - etaT) / (etaI + etaT);
        F0 = F0 * F0;
        float F = F0 + (1.0f - F0) * std::pow(1.0f - cosI, 5.0f);

        // Total internal reflection check
        float sinTSq = eta * eta * (1.0f - cosI * cosI);
        if (sinTSq > 1.0f)
            F = 1.0f;

        if (u_rand < F)
        {
            // Reflect
            glm::vec3 wi = glm::reflect(-wo, normal);
            return { wi, glm::vec3(1.0f), 1.0f };
        }
        else
        {
            // Refract
            glm::vec3 wi = glm::refract(-wo, normal, eta);
            if (glm::length(wi) < 0.001f)
            {
                // Fallback to reflection if refract returns zero
                wi = glm::reflect(-wo, normal);
                return { wi, glm::vec3(1.0f), 1.0f };
            }
            return { wi, color, 1.0f };
        }
    }
};

struct CookTorranceBSDF
{
    glm::vec3 baseColor;
    float roughness;
    float metallic;
    float ior;

    static float D_GGX(float NdotH, float alpha)
    {
        float a2 = alpha * alpha;
        float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
        return a2 / (PI * denom * denom);
    }

    static float G1_Smith(float NdotX, float alpha)
    {
        float a2 = alpha * alpha;
        return 2.0f * NdotX / (NdotX + std::sqrt(a2 + (1.0f - a2) * NdotX * NdotX));
    }

    static float G_Smith(float NdotV, float NdotL, float alpha)
    {
        return G1_Smith(NdotV, alpha) * G1_Smith(NdotL, alpha);
    }

    static glm::vec3 F_Schlick(float cosTheta, const glm::vec3& F0)
    {
        return F0 + (1.0f - F0) * std::pow(1.0f - cosTheta, 5.0f);
    }

    float getAlpha() const
    {
        float r = std::max(roughness, 0.01f);
        return r * r;
    }

    glm::vec3 getF0() const
    {
        float f = (ior - 1.0f) / (ior + 1.0f);
        f = f * f;
        return glm::mix(glm::vec3(f), baseColor, metallic);
    }

    // VNDF sampling (Heitz 2018): samples half-vectors visible from V,
    // eliminating most below-hemisphere reflections at grazing angles.
    static glm::vec3 sampleVNDF(const glm::vec3& Ve, float alpha, float u1, float u2)
    {
        // Stretch V into the hemisphere configuration
        glm::vec3 Vh = glm::normalize(glm::vec3(alpha * Ve.x, alpha * Ve.y, Ve.z));

        // Orthonormal basis around Vh
        float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
        glm::vec3 T1 = lensq > 0.0f
            ? glm::vec3(-Vh.y, Vh.x, 0.0f) / std::sqrt(lensq)
            : glm::vec3(1.0f, 0.0f, 0.0f);
        glm::vec3 T2 = glm::cross(Vh, T1);

        // Sample projected area
        float r = std::sqrt(u1);
        float phi = 2.0f * PI * u2;
        float t1 = r * std::cos(phi);
        float t2 = r * std::sin(phi);
        float s = 0.5f * (1.0f + Vh.z);
        t2 = (1.0f - s) * std::sqrt(std::max(0.0f, 1.0f - t1 * t1)) + s * t2;

        // Reproject onto hemisphere
        glm::vec3 Nh = t1 * T1 + t2 * T2
            + std::sqrt(std::max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

        // Unstretch
        return glm::normalize(glm::vec3(alpha * Nh.x, alpha * Nh.y, std::max(0.0f, Nh.z)));
    }

    // Ng = geometric (face) normal, used to constrain diffuse sampling to the
    // real surface hemisphere — prevents black pixels at grazing angles where
    // the interpolated shading normal N diverges from Ng.
    BSDFSample sample(const glm::vec3& N, const glm::vec3& Ng, const glm::vec3& V,
                      float u1, float u2, float u_lobe) const
    {
        float alpha = getAlpha();
        glm::vec3 F0 = getF0();
        float specWeight = 0.5f * (1.0f + metallic);

        glm::vec3 t, b;
        buildONB(N, t, b);  // ONB around shading normal (for specular)

        glm::vec3 L;
        if (u_lobe < specWeight)
        {
            // VNDF importance sampling (Heitz 2018) — around shading normal N
            glm::vec3 Vlocal(glm::dot(V, t), glm::dot(V, b), glm::dot(V, N));
            glm::vec3 Hlocal = sampleVNDF(Vlocal, alpha, u1, u2);
            glm::vec3 H = t * Hlocal.x + b * Hlocal.y + N * Hlocal.z;
            H = glm::normalize(H);

            L = glm::reflect(-V, H);

            // At grazing angles VNDF can reflect L below the geometric surface.
            // Fall back to a diffuse sample around Ng so the path always continues.
            // u1/u2 are reused here; the correlation is acceptable for this rare case.
            if (glm::dot(L, Ng) <= 0.0f)
            {
                float phi = 2.0f * PI * u1;
                float cosTheta = std::sqrt(1.0f - u2);
                float sinTheta = std::sqrt(u2);
                glm::vec3 localDir(std::cos(phi) * sinTheta,
                                   std::sin(phi) * sinTheta,
                                   cosTheta);
                glm::vec3 tg, bg;
                buildONB(Ng, tg, bg);
                L = tg * localDir.x + bg * localDir.y + Ng * localDir.z;
            }
        }
        else
        {
            // Cosine-weighted hemisphere around geometric normal Ng.
            // Sampling around Ng (not N) guarantees the bounce direction is always
            // above the actual surface, eliminating black pixels at grazing angles.
            float phi = 2.0f * PI * u1;
            float cosTheta = std::sqrt(1.0f - u2);
            float sinTheta = std::sqrt(u2);

            glm::vec3 localDir(std::cos(phi) * sinTheta,
                               std::sin(phi) * sinTheta,
                               cosTheta);
            glm::vec3 tg, bg;
            buildONB(Ng, tg, bg);
            L = tg * localDir.x + bg * localDir.y + Ng * localDir.z;
        }

        // Reject directions below the geometric surface
        if (glm::dot(L, Ng) <= 0.0f)
            return { L, glm::vec3(0.0f), 0.0f };

        float NdotL = glm::dot(N, L);
        if (NdotL <= 0.0f)
            return { L, glm::vec3(0.0f), 0.0f };

        glm::vec3 brdf = evaluate(N, V, L);
        float p = pdf(N, V, L);

        if (p < 1e-8f)
            return { L, glm::vec3(0.0f), 0.0f };

        glm::vec3 throughput = brdf * NdotL / p;
        return { L, throughput, p };
    }

    glm::vec3 evaluate(const glm::vec3& N, const glm::vec3& V,
                       const glm::vec3& L) const
    {
        float NdotL = glm::dot(N, L);
        if (NdotL <= 0.0f)
            return glm::vec3(0.0f);
        float NdotV = std::max(glm::dot(N, V), 1e-4f); // clamp — grazing V can go slightly negative

        float alpha = getAlpha();
        glm::vec3 F0 = getF0();
        glm::vec3 H = glm::normalize(V + L);
        float NdotH = std::max(glm::dot(N, H), 0.0f);
        float VdotH = std::max(glm::dot(V, H), 0.0f);

        float D = D_GGX(NdotH, alpha);
        float G = G_Smith(NdotV, NdotL, alpha);
        glm::vec3 F = F_Schlick(VdotH, F0);

        glm::vec3 spec = D * G * F / std::max(4.0f * NdotV * NdotL, 1e-8f);
        glm::vec3 diff = (1.0f - F) * (1.0f - metallic) * baseColor / PI;

        return diff + spec;
    }

    float pdf(const glm::vec3& N, const glm::vec3& V,
              const glm::vec3& L) const
    {
        float NdotL = glm::dot(N, L);
        if (NdotL <= 0.0f)
            return 0.0f;

        float NdotV = std::max(glm::dot(N, V), 1e-4f);
        float alpha = getAlpha();
        float specWeight = 0.5f * (1.0f + metallic);

        // VNDF specular PDF: D * G1(NdotV) / (4 * NdotV)
        glm::vec3 H = glm::normalize(V + L);
        float NdotH = std::max(glm::dot(N, H), 0.0f);
        float specPdf = D_GGX(NdotH, alpha) * G1_Smith(NdotV, alpha) / (4.0f * NdotV);

        // Diffuse PDF: cosine-weighted
        float diffPdf = NdotL / PI;

        return specWeight * specPdf + (1.0f - specWeight) * diffPdf;
    }
};

} // namespace vex
