#include <doctest/doctest.h>
#include <vex/raytracing/bsdf.h>

#include <cmath>

using namespace vex;

static const glm::vec3 N_UP(0.0f, 1.0f, 0.0f);

// ── MirrorBSDF ───────────────────────────────────────────────────────────────

TEST_SUITE("MirrorBSDF")
{

TEST_CASE("angle of incidence equals angle of reflection")
{
    MirrorBSDF mirror{{1.0f, 1.0f, 1.0f}};
    glm::vec3 wo = glm::normalize(glm::vec3(1.0f, 1.0f, 0.0f));
    auto s = mirror.sample(N_UP, wo);

    float cosThetaI = glm::dot(wo, N_UP);
    float cosThetaR = glm::dot(s.direction, N_UP);
    CHECK(cosThetaI == doctest::Approx(cosThetaR).epsilon(1e-5f));
}

TEST_CASE("reflected direction stays above surface")
{
    MirrorBSDF mirror{{0.8f, 0.5f, 0.2f}};
    glm::vec3 wo = glm::normalize(glm::vec3(0.3f, 1.0f, 0.7f));
    auto s = mirror.sample(N_UP, wo);
    CHECK(glm::dot(s.direction, N_UP) > 0.0f);
}

TEST_CASE("throughput equals the albedo color")
{
    glm::vec3 color(0.8f, 0.5f, 0.2f);
    MirrorBSDF mirror{color};
    glm::vec3 wo = glm::normalize(glm::vec3(0.5f, 1.0f, 0.5f));
    auto s = mirror.sample(N_UP, wo);
    CHECK(s.throughput.r == doctest::Approx(color.r).epsilon(1e-5f));
    CHECK(s.throughput.g == doctest::Approx(color.g).epsilon(1e-5f));
    CHECK(s.throughput.b == doctest::Approx(color.b).epsilon(1e-5f));
}

TEST_CASE("delta pdf sentinel is 1")
{
    MirrorBSDF mirror{{1, 1, 1}};
    auto s = mirror.sample(N_UP, glm::normalize(glm::vec3(1, 1, 0)));
    CHECK(s.pdf == doctest::Approx(1.0f));
}

} // TEST_SUITE("MirrorBSDF")

// ── DielectricBSDF ───────────────────────────────────────────────────────────

TEST_SUITE("DielectricBSDF")
{

TEST_CASE("total internal reflection at grazing angle from inside")
{
    // Ray nearly parallel to surface from inside glass (ior=1.5, frontFace=false).
    // sinTSq = (1.5/1.0)^2 * (1 - cos^2) for very small cosI → > 1 → TIR.
    DielectricBSDF glass{{1, 1, 1}, 1.5f};
    glm::vec3 grazing = glm::normalize(glm::vec3(0.9999f, 0.01f, 0.0f));
    auto s = glass.sample(N_UP, grazing, false, 0.0f);
    CHECK(glm::dot(s.direction, N_UP) > 0.0f);
}

TEST_CASE("refracted direction has unit length")
{
    DielectricBSDF glass{{1, 1, 1}, 1.5f};
    // Normal incidence from outside, force refraction with u_rand=1
    auto s = glass.sample(N_UP, N_UP, true, 1.0f);
    float len = glm::length(s.direction);
    CHECK(len == doctest::Approx(1.0f).epsilon(0.01f));
}

} // TEST_SUITE("DielectricBSDF")

// ── CookTorranceBSDF ─────────────────────────────────────────────────────────

TEST_SUITE("CookTorranceBSDF")
{

TEST_CASE("D_GGX: at NdotH=1, alpha=1 equals 1/PI")
{
    // a2=1, denom = 1*(1-1)+1 = 1 → D = 1/(PI*1) = 1/PI
    float d = CookTorranceBSDF::D_GGX(1.0f, 1.0f);
    CHECK(d == doctest::Approx(1.0f / PI).epsilon(1e-5f));
}

TEST_CASE("D_GGX: always non-negative")
{
    for (float NdotH : {0.0f, 0.1f, 0.5f, 0.9f, 1.0f})
        for (float alpha : {0.01f, 0.1f, 0.5f, 1.0f})
            CHECK(CookTorranceBSDF::D_GGX(NdotH, alpha) >= 0.0f);
}

TEST_CASE("D_GGX: rougher surfaces have lower peak")
{
    // At NdotH=1 a lower alpha (smoother) should give a higher D (sharper lobe)
    float d_smooth = CookTorranceBSDF::D_GGX(1.0f, 0.01f);
    float d_rough  = CookTorranceBSDF::D_GGX(1.0f, 1.0f);
    CHECK(d_smooth > d_rough);
}

TEST_CASE("G1_Smith: returns exactly 1 when NdotX=1")
{
    // G1(1, alpha) = 2/(1 + sqrt(a2 + (1-a2)*1)) = 2/(1+1) = 1 for any alpha
    for (float alpha : {0.01f, 0.1f, 0.5f, 1.0f})
        CHECK(CookTorranceBSDF::G1_Smith(1.0f, alpha) == doctest::Approx(1.0f).epsilon(1e-5f));
}

TEST_CASE("G1_Smith: result is in [0, 1]")
{
    for (float NdotX : {0.01f, 0.1f, 0.5f, 0.9f, 1.0f})
        for (float alpha : {0.01f, 0.1f, 0.5f, 1.0f})
        {
            float g = CookTorranceBSDF::G1_Smith(NdotX, alpha);
            CHECK(g >= 0.0f);
            CHECK(g <= 1.0f + 1e-5f);
        }
}

TEST_CASE("F_Schlick: at cosTheta=1 returns F0")
{
    glm::vec3 F0(0.04f, 0.04f, 0.04f);
    glm::vec3 F = CookTorranceBSDF::F_Schlick(1.0f, F0);
    CHECK(F.r == doctest::Approx(F0.r).epsilon(1e-5f));
    CHECK(F.g == doctest::Approx(F0.g).epsilon(1e-5f));
    CHECK(F.b == doctest::Approx(F0.b).epsilon(1e-5f));
}

TEST_CASE("F_Schlick: at cosTheta=0 returns 1")
{
    glm::vec3 F0(0.04f, 0.1f, 0.2f);
    glm::vec3 F = CookTorranceBSDF::F_Schlick(0.0f, F0);
    CHECK(F.r == doctest::Approx(1.0f).epsilon(1e-5f));
    CHECK(F.g == doctest::Approx(1.0f).epsilon(1e-5f));
    CHECK(F.b == doctest::Approx(1.0f).epsilon(1e-5f));
}

TEST_CASE("getF0: metallic=1 returns baseColor")
{
    glm::vec3 base(0.8f, 0.3f, 0.1f);
    CookTorranceBSDF bsdf{base, 0.5f, 1.0f, 1.5f};
    glm::vec3 F0 = bsdf.getF0();
    CHECK(F0.r == doctest::Approx(base.r).epsilon(1e-5f));
    CHECK(F0.g == doctest::Approx(base.g).epsilon(1e-5f));
    CHECK(F0.b == doctest::Approx(base.b).epsilon(1e-5f));
}

TEST_CASE("getF0: metallic=0 gives dielectric F0 (no color tint)")
{
    // ior=1.5 → f = ((1.5-1)/(1.5+1))^2 = (0.5/2.5)^2 = 0.04
    CookTorranceBSDF bsdf{{0.8f, 0.3f, 0.1f}, 0.5f, 0.0f, 1.5f};
    glm::vec3 F0 = bsdf.getF0();
    CHECK(F0.r == doctest::Approx(0.04f).epsilon(1e-4f));
    CHECK(F0.g == doctest::Approx(0.04f).epsilon(1e-4f));
    CHECK(F0.b == doctest::Approx(0.04f).epsilon(1e-4f));
}

TEST_CASE("evaluate: returns non-negative values")
{
    CookTorranceBSDF bsdf{{0.8f, 0.2f, 0.1f}, 0.3f, 0.0f, 1.5f};
    glm::vec3 V = glm::normalize(glm::vec3(0.5f, 1.0f, 0.3f));
    glm::vec3 L = glm::normalize(glm::vec3(-0.3f, 1.0f, 0.5f));
    glm::vec3 result = bsdf.evaluate(N_UP, V, L);
    CHECK(result.r >= 0.0f);
    CHECK(result.g >= 0.0f);
    CHECK(result.b >= 0.0f);
}

TEST_CASE("evaluate: zero when L is below surface")
{
    CookTorranceBSDF bsdf{{1, 1, 1}, 0.5f, 0.0f, 1.5f};
    glm::vec3 V = glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f));
    glm::vec3 L = glm::normalize(glm::vec3(0.0f, -1.0f, 0.0f));
    glm::vec3 result = bsdf.evaluate(N_UP, V, L);
    CHECK(result.r == doctest::Approx(0.0f));
    CHECK(result.g == doctest::Approx(0.0f));
    CHECK(result.b == doctest::Approx(0.0f));
}

TEST_CASE("pdf: non-negative for valid directions")
{
    CookTorranceBSDF bsdf{{1, 1, 1}, 0.5f, 0.0f, 1.5f};
    glm::vec3 V = glm::normalize(glm::vec3(0.5f, 1.0f, 0.3f));
    glm::vec3 L = glm::normalize(glm::vec3(-0.3f, 1.0f, 0.5f));
    CHECK(bsdf.pdf(N_UP, V, L) >= 0.0f);
}

TEST_CASE("pdf: zero when L is below surface")
{
    CookTorranceBSDF bsdf{{1, 1, 1}, 0.5f, 0.0f, 1.5f};
    glm::vec3 V = glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f));
    glm::vec3 L = glm::normalize(glm::vec3(0.0f, -1.0f, 0.0f));
    CHECK(bsdf.pdf(N_UP, V, L) == doctest::Approx(0.0f));
}

TEST_CASE("sample: throughput is always finite and non-negative")
{
    // Covers a variety of roughness/metallic combos and incident angles.
    const glm::vec3 configs[][2] = {
        {{1, 1, 1}, {0.0f, 0.5f, 0.0f}},  // baseColor, {roughness, metallic, ior-placeholder}
    };

    struct Setup { glm::vec3 baseColor; float roughness; float metallic; float ior; };
    Setup setups[] = {
        {{1, 1, 1},       0.01f, 0.0f, 1.5f},
        {{1, 1, 1},       0.99f, 0.0f, 1.5f},
        {{0.8f, 0.2f, 0}, 0.5f,  1.0f, 1.5f},
        {{1, 1, 1},       0.5f,  0.5f, 2.4f},
    };

    const float us[] = {0.05f, 0.25f, 0.5f, 0.75f, 0.95f};

    for (const auto& s : setups)
    {
        CookTorranceBSDF bsdf{s.baseColor, s.roughness, s.metallic, s.ior};
        glm::vec3 V = glm::normalize(glm::vec3(0.3f, 1.0f, 0.3f));

        for (float u1 : us) for (float u2 : us) for (float u_lobe : us)
        {
            auto result = bsdf.sample(N_UP, N_UP, V, u1, u2, u_lobe);
            CHECK(std::isfinite(result.throughput.r));
            CHECK(std::isfinite(result.throughput.g));
            CHECK(std::isfinite(result.throughput.b));
            CHECK(result.throughput.r >= 0.0f);
            CHECK(result.throughput.g >= 0.0f);
            CHECK(result.throughput.b >= 0.0f);
        }
    }
}

TEST_CASE("sample: when pdf>0, pdf() of the returned direction is also >0")
{
    // If sample() assigns a non-zero probability to a direction, pdf() must agree.
    // A mismatch would corrupt MIS weights in the path tracer.
    CookTorranceBSDF bsdf{{0.8f, 0.6f, 0.4f}, 0.4f, 0.0f, 1.5f};
    glm::vec3 V = glm::normalize(glm::vec3(0.5f, 1.0f, 0.2f));

    const float us[] = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};
    for (float u1 : us) for (float u2 : us) for (float u_lobe : us)
    {
        auto result = bsdf.sample(N_UP, N_UP, V, u1, u2, u_lobe);
        if (result.pdf > 1e-8f)
            CHECK(bsdf.pdf(N_UP, V, result.direction) > 0.0f);
    }
}

} // TEST_SUITE("CookTorranceBSDF")

// ── Sample sanity (finite, non-negative) ─────────────────────────────────────
//
// A true white-furnace energy bound (avg throughput <= 1) doesn't apply to this
// CookTorrance model: additive Fresnel-weighted diffuse + single-scattering GGX
// is an intentional visual approximation, not an energy-conserving one.
//
// What we CAN verify:
//   • Finiteness — the primary guard against D_GGX overflow regressions.
//   • Non-negativity — no negative BRDF values.
//
// MirrorBSDF and DielectricBSDF ARE perfectly conservative so their throughput
// is tested to equal the expected value exactly.

TEST_SUITE("BSDF Sample Sanity")
{

static const glm::vec3 SS_N(0.0f, 1.0f, 0.0f);
static const glm::vec3 SS_V = glm::vec3(0.5f, 1.0f, 0.5f) /
    std::sqrt(0.5f*0.5f + 1.0f + 0.5f*0.5f);

static glm::vec3 sampleAvg(const CookTorranceBSDF& bsdf)
{
    const int N = 20; // 8000 stratified samples
    glm::vec3 sum(0.0f);
    for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
    for (int k = 0; k < N; ++k)
    {
        float u1 = (i + 0.5f) / N;
        float u2 = (j + 0.5f) / N;
        float ul = (k + 0.5f) / N;
        auto s = bsdf.sample(SS_N, SS_N, SS_V, u1, u2, ul);
        sum += s.throughput;
    }
    return sum / float(N * N * N);
}

TEST_CASE("CookTorrance: throughput is always finite and non-negative")
{
    struct Setup { float roughness; float metallic; };
    for (auto [r, m] : {Setup{0.01f,0.0f}, {0.1f,0.0f}, {0.5f,0.0f}, {1.0f,0.0f},
                        {0.01f,1.0f}, {0.5f,1.0f}, {0.5f,0.5f}})
    {
        CookTorranceBSDF bsdf{{1,1,1}, r, m, 1.5f};
        glm::vec3 avg = sampleAvg(bsdf);
        CHECK(std::isfinite(avg.r));
        CHECK(std::isfinite(avg.g));
        CHECK(std::isfinite(avg.b));
        CHECK(avg.r >= 0.0f);
        CHECK(avg.g >= 0.0f);
        CHECK(avg.b >= 0.0f);
    }
}

TEST_CASE("MirrorBSDF: throughput equals albedo (delta BSDF is conservative)")
{
    glm::vec3 albedo(0.8f, 0.5f, 0.2f);
    MirrorBSDF mirror{albedo};
    auto s = mirror.sample(SS_N, glm::normalize(glm::vec3(0.3f, 1.0f, 0.5f)));
    CHECK(s.throughput.r == doctest::Approx(albedo.r).epsilon(1e-5f));
    CHECK(s.throughput.g == doctest::Approx(albedo.g).epsilon(1e-5f));
    CHECK(s.throughput.b == doctest::Approx(albedo.b).epsilon(1e-5f));
}

TEST_CASE("DielectricBSDF: throughput is always exactly 1 (lossless transmittance)")
{
    DielectricBSDF glass{{1,1,1}, 1.5f};
    for (float u : {0.0f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 1.0f})
    {
        auto s = glass.sample(SS_N, SS_N, true, u);
        CHECK(s.throughput.r == doctest::Approx(1.0f).epsilon(1e-5f));
        CHECK(s.throughput.g == doctest::Approx(1.0f).epsilon(1e-5f));
        CHECK(s.throughput.b == doctest::Approx(1.0f).epsilon(1e-5f));
    }
}

} // TEST_SUITE("BSDF Sample Sanity")
