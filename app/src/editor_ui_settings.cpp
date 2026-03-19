#include "editor_ui.h"
#include "scene_renderer.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <iterator>

void EditorUI::renderSettings(SceneRenderer& renderer)
{
    ImGui::Begin("Settings");

    // 2x2 render mode tile picker
    {
#ifdef VEX_BACKEND_VULKAN
        const char* tileLabels[] = {
            "Rasterization",
            "CPU Path Tracing",
            "GPU Path Tracing (HW RT)",
            "GPU Path Tracing (Compute)",
        };
        const int tileModes[] = { 0, 1, 2, 3 };
        const int tileCount = 4;
#else
        const char* tileLabels[] = {
            "Rasterization",
            "CPU Path Tracing",
            "GPU Path Tracing (Compute)",
        };
        const int tileModes[] = { 0, 1, 2 };
        const int tileCount = 3;
#endif
        const float tileW = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        const float tileH = 48.0f;
        for (int i = 0; i < tileCount; ++i)
        {
            if (i % 2 != 0) ImGui::SameLine();
            bool active = (m_renderModeIndex == tileModes[i]);
            if (active)
            {
                ImGui::PushStyleColor(ImGuiCol_Button,        ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
            }
            ImGui::PushID(i);
            if (ImGui::Button(tileLabels[i], ImVec2(tileW, tileH)))
                m_renderModeIndex = tileModes[i];
            ImGui::PopID();
            if (active)
                ImGui::PopStyleColor(2);
        }
    }

    if (m_renderModeIndex == 0)
    {
        const char* debugModes[] = { "None", "Wireframe", "Depth", "Normals",
                                      "UVs", "Albedo", "Emission", "Material Type",
                                      "Roughness", "Metallic", "AO" };
        ImGui::Combo("Debug Mode", &m_debugModeIndex, debugModes, static_cast<int>(std::size(debugModes)));

        ImGui::Checkbox("Normal Mapping", &renderer.getRasterSettings().enableNormalMapping);

        ImGui::SeparatorText("Shadows");
        ImGui::Checkbox("Shadow Mapping", &renderer.getRasterSettings().enableShadows);
        ImGui::SliderFloat("Shadow Strength##raster", &renderer.getRasterSettings().shadowStrength, 0.0f, 1.0f, "%.2f");
        ImGui::ColorEdit3("Shadow Color##raster", &renderer.getRasterSettings().shadowColor.x);

        ImGui::SeparatorText("Environment");
        ImGui::Checkbox("Environment Lighting##raster", &renderer.getRasterSettings().enableEnvLighting);
        ImGui::SliderFloat("Env Multiplier##raster", &renderer.getRasterSettings().envLightMultiplier, 0.0f, 5.0f, "%.2f");

        ImGui::SeparatorText("Post Processing");
        ImGui::SliderFloat("Exposure##raster", &renderer.getRasterSettings().exposure, -5.0f, 5.0f, "%.1f");
        ImGui::Checkbox("ACES Tonemapping##raster", &renderer.getRasterSettings().enableACES);
        ImGui::SliderFloat("Gamma##raster", &renderer.getRasterSettings().gamma, 1.0f, 3.0f, "%.2f");

        ImGui::SeparatorText("Bloom");
        ImGui::Checkbox("Bloom##raster", &renderer.getBloomSettings().enabled);
        ImGui::BeginDisabled(!renderer.getBloomSettings().enabled);
        ImGui::SliderFloat("Threshold##bloom", &renderer.getBloomSettings().threshold, 0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Intensity##bloom", &renderer.getBloomSettings().intensity, 0.0f, 1.0f, "%.3f");
        ImGui::SliderInt("Blur Passes##bloom", &renderer.getBloomSettings().blurPasses, 1, 10);
        ImGui::EndDisabled();

    }

    if (m_renderModeIndex == 1)
    {
        // ── Accumulation ─────────────────────────────────────────────────────
        {
            uint32_t samples = renderer.getRaytraceSampleCount();
            uint32_t maxSamp = renderer.getMaxSamples();
            if (maxSamp > 0 && samples >= maxSamp)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
                ImGui::Text("Samples: %u / %u  (converged)", samples, maxSamp);
                ImGui::PopStyleColor();
            }
            else if (maxSamp > 0)
                ImGui::Text("Samples: %u / %u", samples, maxSamp);
            else
                ImGui::Text("Samples: %u", samples);
        }
        {
            int v = static_cast<int>(renderer.getMaxSamples());
            if (ImGui::DragInt("Max Samples##cpu", &v, 8.0f, 0, 1 << 20))
                renderer.setMaxSamples(static_cast<uint32_t>(std::max(0, v)));
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("0 = unlimited");
        }
        {
            uint32_t sampleCount = renderer.getRaytraceSampleCount();
            bool canDenoise = renderer.isDenoiserReady() && (sampleCount > 0);
            if (!canDenoise) ImGui::BeginDisabled();
            if (ImGui::Button("Denoise##cpu"))  renderer.triggerDenoise();
            ImGui::SameLine();
            if (ImGui::Button("Denoise+##cpu")) renderer.triggerDenoiseAux();
            if (!canDenoise) ImGui::EndDisabled();
            if (renderer.getShowDenoisedResult())
            {
                ImGui::SameLine();
                ImGui::TextDisabled("(showing denoised)");
            }
        }

        // ── Path Tracing ──────────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Path Tracing##cpu", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::SliderInt("Max Depth", &renderer.getCPURTSettings().maxDepth, 1, 16);
            ImGui::Checkbox("Next Event Estimation", &renderer.getCPURTSettings().enableNEE);
            ImGui::Checkbox("Russian Roulette", &renderer.getCPURTSettings().enableRR);
            ImGui::Checkbox("Anti-Aliasing", &renderer.getCPURTSettings().enableAA);
            ImGui::Checkbox("Firefly Clamping", &renderer.getCPURTSettings().enableFireflyClamping);

            bool lumCDF = renderer.getUseLuminanceCDF();
            if (ImGui::Checkbox("Luminance CDF", &lumCDF))
                renderer.setUseLuminanceCDF(lumCDF);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Weight emissive triangle sampling by luminance x area\ninstead of area alone. Improves convergence for scenes\nwith bright emitters of varying color/intensity.");

        }

        // ── Lighting ──────────────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Lighting##cpu", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("Environment Lighting", &renderer.getCPURTSettings().enableEnvLighting);
            ImGui::SliderFloat("Env Multiplier", &renderer.getCPURTSettings().envLightMultiplier, 0.0f, 2.0f, "%.2f");
        }

        // ── Shading ───────────────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Shading##cpu"))
        {
            ImGui::Checkbox("Flat Shading", &renderer.getCPURTSettings().flatShading);
            ImGui::Checkbox("Normal Mapping", &renderer.getCPURTSettings().enableNormalMapping);
            ImGui::Checkbox("Emissive Materials", &renderer.getCPURTSettings().enableEmissive);
        }

        // ── Post Processing ───────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Post Processing##cpu", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::SliderFloat("Exposure", &renderer.getCPURTSettings().exposure, -5.0f, 5.0f, "%.1f");
            ImGui::Checkbox("ACES Tonemapping", &renderer.getCPURTSettings().enableACES);
            ImGui::SliderFloat("Gamma", &renderer.getCPURTSettings().gamma, 1.0f, 3.0f, "%.2f");

            ImGui::SeparatorText("Bloom");
            ImGui::Checkbox("Bloom##cpu", &renderer.getBloomSettings().enabled);
            ImGui::BeginDisabled(!renderer.getBloomSettings().enabled);
            ImGui::SliderFloat("Threshold##bloomcpu", &renderer.getBloomSettings().threshold, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("Intensity##bloomcpu", &renderer.getBloomSettings().intensity, 0.0f, 1.0f, "%.3f");
            ImGui::SliderInt("Blur Passes##bloomcpu", &renderer.getBloomSettings().blurPasses, 1, 10);
            ImGui::EndDisabled();
        }

        // ── Diagnostics ───────────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Diagnostics##cpu"))
        {
            float& rayEps = renderer.getCPURTSettings().rayEps;
            int expVal = static_cast<int>(std::round(std::log10(rayEps)));
            expVal = std::clamp(expVal, -5, -1);
            if (ImGui::SliderInt("Ray EPS (10^n)", &expVal, -5, -1))
                rayEps = std::pow(10.0f, static_cast<float>(expVal));
            ImGui::SameLine();
            ImGui::TextDisabled("= %.0e", rayEps);
        }
    }
    else if (m_renderModeIndex == 2)
    {
        {
            uint32_t samples = renderer.getRaytraceSampleCount();
            uint32_t maxSamp = renderer.getMaxSamples();
            if (maxSamp > 0 && samples >= maxSamp)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
                ImGui::Text("Samples: %u / %u  (converged)", samples, maxSamp);
                ImGui::PopStyleColor();
            }
            else if (maxSamp > 0)
                ImGui::Text("Samples: %u / %u", samples, maxSamp);
            else
                ImGui::Text("Samples: %u", samples);

        }
        {
            int v = static_cast<int>(renderer.getMaxSamples());
            if (ImGui::DragInt("Max Samples##gpu", &v, 8.0f, 0, 1 << 20))
                renderer.setMaxSamples(static_cast<uint32_t>(std::max(0, v)));
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("0 = unlimited");
        }

        {
            uint32_t sampleCount = renderer.getRaytraceSampleCount();
            bool canDenoise = renderer.isDenoiserReady() && (sampleCount > 0);
            if (!canDenoise) ImGui::BeginDisabled();
            if (ImGui::Button("Denoise##gpu"))  renderer.triggerDenoise();
            ImGui::SameLine();
            if (ImGui::Button("Denoise+##gpu")) renderer.triggerDenoiseAux();
            if (!canDenoise) ImGui::EndDisabled();
            if (renderer.getShowDenoisedResult())
            {
                ImGui::SameLine();
                ImGui::TextDisabled("(showing denoised)");
            }
        }

        // ── Path Tracing ──────────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Path Tracing##gpu", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::SliderInt("Max Depth", &renderer.getGPURTSettings().maxDepth, 1, 16);
            ImGui::Checkbox("Next Event Estimation", &renderer.getGPURTSettings().enableNEE);
            ImGui::Checkbox("Russian Roulette", &renderer.getGPURTSettings().enableRR);
            ImGui::Checkbox("Anti-Aliasing", &renderer.getGPURTSettings().enableAA);
            ImGui::Checkbox("Firefly Clamping", &renderer.getGPURTSettings().enableFireflyClamping);

            bool lumCDF = renderer.getUseLuminanceCDF();
            if (ImGui::Checkbox("Luminance CDF##hwrt", &lumCDF))
                renderer.setUseLuminanceCDF(lumCDF);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Weight emissive triangle sampling by luminance x area\ninstead of area alone. Improves convergence for scenes\nwith bright emitters of varying color/intensity.");

            const char* samplerItems[] = { "PCG (Default)", "Halton", "Blue Noise (IGN)" };
            ImGui::Combo("Sampler", &renderer.getGPURTSettings().samplerType, samplerItems, 3);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("PCG: pseudo-random\nHalton: low-discrepancy, faster convergence\nBlue Noise: spatially decorrelated, pleasant noise pattern");
        }

        // ── Lighting ──────────────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Lighting##gpu", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("Environment Lighting", &renderer.getGPURTSettings().enableEnvLighting);
            ImGui::SliderFloat("Env Multiplier", &renderer.getGPURTSettings().envLightMultiplier, 0.0f, 2.0f, "%.2f");
        }

        // ── Shading ───────────────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Shading##gpu"))
        {
            ImGui::Checkbox("Flat Shading", &renderer.getGPURTSettings().flatShading);
            ImGui::Checkbox("Normal Mapping", &renderer.getGPURTSettings().enableNormalMapping);
            ImGui::Checkbox("Emissive Materials", &renderer.getGPURTSettings().enableEmissive);
            ImGui::Checkbox("Bilinear Filtering", &renderer.getGPURTSettings().bilinearFiltering);
        }

        // ── Post Processing ───────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Post Processing##gpu", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::SliderFloat("Exposure", &renderer.getGPURTSettings().exposure, -5.0f, 5.0f, "%.1f");
            ImGui::Checkbox("ACES Tonemapping", &renderer.getGPURTSettings().enableACES);
            ImGui::SliderFloat("Gamma", &renderer.getGPURTSettings().gamma, 1.0f, 3.0f, "%.2f");

            ImGui::SeparatorText("Bloom");
            ImGui::Checkbox("Bloom##gpu", &renderer.getBloomSettings().enabled);
            ImGui::BeginDisabled(!renderer.getBloomSettings().enabled);
            ImGui::SliderFloat("Threshold##bloomgpu", &renderer.getBloomSettings().threshold, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("Intensity##bloomgpu", &renderer.getBloomSettings().intensity, 0.0f, 1.0f, "%.3f");
            ImGui::SliderInt("Blur Passes##bloomgpu", &renderer.getBloomSettings().blurPasses, 1, 10);
            ImGui::EndDisabled();
        }

        // ── Diagnostics ───────────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Diagnostics##gpu"))
        {
            if (ImGui::SmallButton("Reload Shader (F5)"))
                renderer.reloadGPUShader();

            float& rayEps = renderer.getGPURTSettings().rayEps;
            int expVal = static_cast<int>(std::round(std::log10(rayEps)));
            expVal = std::clamp(expVal, -5, -1);
            if (ImGui::SliderInt("Ray EPS (10^n)", &expVal, -5, -1))
                rayEps = std::pow(10.0f, static_cast<float>(expVal));
            ImGui::SameLine();
            ImGui::TextDisabled("= %.0e", rayEps);
        }
    }
#ifdef VEX_BACKEND_VULKAN
    else if (m_renderModeIndex == 3)
    {
        {
            uint32_t samples = renderer.getRaytraceSampleCount();
            uint32_t maxSamp = renderer.getMaxSamples();
            if (maxSamp > 0 && samples >= maxSamp)
            {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
                ImGui::Text("Samples: %u / %u  (converged)", samples, maxSamp);
                ImGui::PopStyleColor();
            }
            else if (maxSamp > 0)
                ImGui::Text("Samples: %u / %u", samples, maxSamp);
            else
                ImGui::Text("Samples: %u", samples);
        }
        {
            int v = static_cast<int>(renderer.getMaxSamples());
            if (ImGui::DragInt("Max Samples##compute", &v, 8.0f, 0, 1 << 20))
                renderer.setMaxSamples(static_cast<uint32_t>(std::max(0, v)));
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("0 = unlimited");
        }

        {
            uint32_t sampleCount = renderer.getRaytraceSampleCount();
            bool canDenoise = renderer.isDenoiserReady() && (sampleCount > 0);
            if (!canDenoise) ImGui::BeginDisabled();
            if (ImGui::Button("Denoise##compute"))  renderer.triggerDenoise();
            ImGui::SameLine();
            if (ImGui::Button("Denoise+##compute")) renderer.triggerDenoiseAux();
            if (!canDenoise) ImGui::EndDisabled();
            if (renderer.getShowDenoisedResult())
            {
                ImGui::SameLine();
                ImGui::TextDisabled("(showing denoised)");
            }
        }

        if (ImGui::CollapsingHeader("Path Tracing##compute", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::SliderInt("Max Depth##compute", &renderer.getGPURTSettings().maxDepth, 1, 16);
            ImGui::Checkbox("Next Event Estimation##compute", &renderer.getGPURTSettings().enableNEE);
            ImGui::Checkbox("Russian Roulette##compute", &renderer.getGPURTSettings().enableRR);
            ImGui::Checkbox("Anti-Aliasing##compute", &renderer.getGPURTSettings().enableAA);
            ImGui::Checkbox("Firefly Clamping##compute", &renderer.getGPURTSettings().enableFireflyClamping);

            bool lumCDF = renderer.getUseLuminanceCDF();
            if (ImGui::Checkbox("Luminance CDF##compute", &lumCDF))
                renderer.setUseLuminanceCDF(lumCDF);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Weight emissive triangle sampling by luminance x area\ninstead of area alone. Improves convergence for scenes\nwith bright emitters of varying color/intensity.");

            const char* samplerItems[] = { "PCG (Default)", "Halton", "Blue Noise (IGN)" };
            ImGui::Combo("Sampler##compute", &renderer.getGPURTSettings().samplerType, samplerItems, 3);
        }

        if (ImGui::CollapsingHeader("Lighting##compute", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("Environment Lighting##compute", &renderer.getGPURTSettings().enableEnvLighting);
            ImGui::SliderFloat("Env Multiplier##compute", &renderer.getGPURTSettings().envLightMultiplier, 0.0f, 2.0f, "%.2f");
        }

        if (ImGui::CollapsingHeader("Shading##compute"))
        {
            ImGui::Checkbox("Flat Shading##compute", &renderer.getGPURTSettings().flatShading);
            ImGui::Checkbox("Normal Mapping##compute", &renderer.getGPURTSettings().enableNormalMapping);
            ImGui::Checkbox("Emissive Materials##compute", &renderer.getGPURTSettings().enableEmissive);
            ImGui::Checkbox("Bilinear Filtering##compute", &renderer.getGPURTSettings().bilinearFiltering);
        }

        if (ImGui::CollapsingHeader("Post Processing##compute", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::SliderFloat("Exposure##compute", &renderer.getGPURTSettings().exposure, -5.0f, 5.0f, "%.1f");
            ImGui::Checkbox("ACES Tonemapping##compute", &renderer.getGPURTSettings().enableACES);
            ImGui::SliderFloat("Gamma##compute", &renderer.getGPURTSettings().gamma, 1.0f, 3.0f, "%.2f");

            ImGui::SeparatorText("Bloom");
            ImGui::Checkbox("Bloom##compute", &renderer.getBloomSettings().enabled);
            ImGui::BeginDisabled(!renderer.getBloomSettings().enabled);
            ImGui::SliderFloat("Threshold##bloomcompute", &renderer.getBloomSettings().threshold, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("Intensity##bloomcompute", &renderer.getBloomSettings().intensity, 0.0f, 1.0f, "%.3f");
            ImGui::SliderInt("Blur Passes##bloomcompute", &renderer.getBloomSettings().blurPasses, 1, 10);
            ImGui::EndDisabled();
        }

        if (ImGui::CollapsingHeader("Diagnostics##compute"))
        {
            float& rayEps = renderer.getGPURTSettings().rayEps;
            int expVal = static_cast<int>(std::round(std::log10(rayEps)));
            expVal = std::clamp(expVal, -5, -1);
            if (ImGui::SliderInt("Ray EPS (10^n)##compute", &expVal, -5, -1))
                rayEps = std::pow(10.0f, static_cast<float>(expVal));
            ImGui::SameLine();
            ImGui::TextDisabled("= %.0e", rayEps);

            float sps = renderer.getVKComputeSamplesPerSec();
            if (sps > 0.0f)
                ImGui::Text("Samples/sec: %.1f", sps);
        }
    }
#endif
    ImGui::End();
}
