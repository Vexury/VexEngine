#include "editor_ui.h"
#include "scene.h"
#include "scene_renderer.h"
#include "file_dialog.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <vex/core/log.h>
#include <vex/graphics/framebuffer.h>
#include <vex/graphics/graphics_context.h>

#include <imgui.h>

#include <iterator>
#include <string>

bool EditorUI::consumePickRequest(int& outX, int& outY)
{
    if (!m_pickRequested)
        return false;

    m_pickRequested = false;
    outX = m_pickX;
    outY = m_pickY;
    return true;
}

void EditorUI::renderViewport(SceneRenderer& renderer)
{
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Viewport");
    ImGui::PopStyleVar();

    m_viewportHovered = ImGui::IsWindowHovered();

    ImVec2 size = ImGui::GetContentRegionAvail();
    uint32_t w = static_cast<uint32_t>(size.x);
    uint32_t h = static_cast<uint32_t>(size.y);

    auto* fb = renderer.getFramebuffer();

    if (w > 0 && h > 0)
    {
        const auto& spec = fb->getSpec();
        if (spec.width != w || spec.height != h)
            fb->resize(w, h);

        // Detect left-click in viewport for picking
        ImVec2 cursor = ImGui::GetCursorScreenPos();

        if (fb->flipsUV())
        {
            ImGui::Image(
                static_cast<ImTextureID>(fb->getColorAttachmentHandle()),
                size,
                ImVec2(0, 1), ImVec2(1, 0) // flip Y for OpenGL
            );
        }
        else
        {
            ImGui::Image(
                static_cast<ImTextureID>(fb->getColorAttachmentHandle()),
                size
            );
        }

        if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        {
            ImVec2 mouse = ImGui::GetMousePos();
            m_pickX = static_cast<int>(mouse.x - cursor.x);
            m_pickY = static_cast<int>(mouse.y - cursor.y);
            m_pickRequested = true;
        }
    }

    ImGui::End();
}

void EditorUI::renderHierarchy(Scene& scene, SceneRenderer& renderer)
{
    ImGui::Begin("Hierarchy");

    ImGui::TextUnformatted("Scene");
    ImGui::Indent();

    // Mesh groups
    for (int gi = 0; gi < static_cast<int>(scene.meshGroups.size()); ++gi)
    {
        bool groupSelected = (m_selectionType == Selection::Mesh && m_selectionIndex == gi);

        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow
                                 | ImGuiTreeNodeFlags_DefaultOpen;
        if (groupSelected && m_submeshIndex == -1)
            flags |= ImGuiTreeNodeFlags_Selected;

        bool open = ImGui::TreeNodeEx(scene.meshGroups[gi].name.c_str(), flags);

        if (ImGui::IsItemClicked())
        {
            m_selectionType  = Selection::Mesh;
            m_selectionIndex = gi;
            m_submeshIndex   = -1;
        }

        if (open)
        {
            for (int si = 0; si < static_cast<int>(scene.meshGroups[gi].submeshes.size()); ++si)
            {
                const auto& sub = scene.meshGroups[gi].submeshes[si];
                bool subSelected = groupSelected && m_submeshIndex == si;
                ImGui::Indent();
                if (ImGui::Selectable(sub.name.c_str(), subSelected))
                {
                    m_selectionType  = Selection::Mesh;
                    m_selectionIndex = gi;
                    m_submeshIndex   = si;
                }
                ImGui::Unindent();
            }
            ImGui::TreePop();
        }
    }

    // Skybox
    if (scene.showSkybox)
    {
        bool selected = (m_selectionType == Selection::Skybox);
        if (ImGui::Selectable("Skybox", selected))
            m_selectionType = Selection::Skybox;
    }

    // Light
    {
        bool selected = (m_selectionType == Selection::Light);
        std::string lightLabel = scene.showLight ? "Light" : "Light (disabled)";
        if (ImGui::Selectable(lightLabel.c_str(), selected))
            m_selectionType = Selection::Light;
    }

    // Sun
    {
        bool selected = (m_selectionType == Selection::Sun);
        std::string sunLabel = scene.showSun ? "Sun" : "Sun (disabled)";
        if (ImGui::Selectable(sunLabel.c_str(), selected))
            m_selectionType = Selection::Sun;
    }

    // Camera
    {
        bool selected = (m_selectionType == Selection::Camera);
        if (ImGui::Selectable("Camera", selected))
            m_selectionType = Selection::Camera;
    }

    ImGui::Unindent();
    ImGui::Separator();

    // Import button
    if (ImGui::Button("Import OBJ..."))
    {
        vex::Log::info("File dialog opened");
        std::string path = openObjFileDialog();
        vex::Log::info(path.empty() ? "File dialog cancelled" : "File dialog closed: " + path);
        if (!path.empty())
        {
            // Extract filename without extension for display name
            std::string baseName = path;
            auto slash = baseName.find_last_of("/\\");
            if (slash != std::string::npos)
                baseName = baseName.substr(slash + 1);
            auto dot = baseName.rfind('.');
            if (dot != std::string::npos)
                baseName = baseName.substr(0, dot);

            if (scene.importOBJ(path, baseName))
                vex::Log::info("Imported: " + baseName);
            else
                vex::Log::error("Failed to load: " + path);
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("Save Image..."))
    {
        std::string path = saveImageFileDialog();
        if (!path.empty())
        {
            if (renderer.saveImage(path))
                vex::Log::info("Saved: " + path);
            else
                vex::Log::error("Failed to save: " + path);
        }
    }

    ImGui::End();
}

void EditorUI::renderInspector(Scene& scene, SceneRenderer& renderer)
{
    ImGui::Begin("Inspector");

    switch (m_selectionType)
    {
        case Selection::Mesh:
            if (m_selectionIndex >= 0 && m_selectionIndex < static_cast<int>(scene.meshGroups.size()))
            {
                auto& group = scene.meshGroups[m_selectionIndex];

                if (m_submeshIndex >= 0 && m_submeshIndex < static_cast<int>(group.submeshes.size()))
                {
                    // --- Submesh selected ---
                    auto& sm = group.submeshes[m_submeshIndex];

                    ImGui::Text("%s > %s", group.name.c_str(), sm.name.c_str());
                    ImGui::Separator();
                    ImGui::Text("Vertices:  %u", sm.vertexCount);
                    ImGui::Text("Triangles: %u", sm.indexCount / 3);

                    ImGui::SeparatorText("Material");

                    const char* matTypes[] = { "Microfacet (GGX)", "Mirror", "Dielectric" };
                    if (ImGui::Combo("Type", &sm.meshData.materialType, matTypes, 3))
                        scene.materialDirty = true;

                    if (sm.meshData.materialType == 0)
                    {
                        ImGui::DragFloat("Roughness", &sm.meshData.roughness, 0.01f, 0.0f, 1.0f, "%.2f");
                        if (ImGui::IsItemDeactivatedAfterEdit()) scene.materialDirty = true;
                        ImGui::DragFloat("Metallic", &sm.meshData.metallic, 0.01f, 0.0f, 1.0f, "%.2f");
                        if (ImGui::IsItemDeactivatedAfterEdit()) scene.materialDirty = true;
                    }
                    else if (sm.meshData.materialType == 2)
                    {
                        ImGui::DragFloat("IOR", &sm.meshData.ior, 0.01f, 1.0f, 3.0f, "%.2f");
                        if (ImGui::IsItemDeactivatedAfterEdit()) scene.materialDirty = true;
                    }

                    ImGui::SeparatorText("Textures");

                    auto texName = [](const std::string& path) -> const char* {
                        if (path.empty()) return "none";
                        auto s = path.find_last_of("/\\");
                        return path.c_str() + (s != std::string::npos ? s + 1 : 0);
                    };
                    ImGui::Text("Diffuse:   %s", texName(sm.meshData.diffuseTexturePath));
                    ImGui::Text("Normal:    %s", texName(sm.meshData.normalTexturePath));
                    ImGui::Text("Roughness: %s", texName(sm.meshData.roughnessTexturePath));
                    ImGui::Text("Metallic:  %s", texName(sm.meshData.metallicTexturePath));
                }
                else
                {
                    // --- Group selected ---
                    ImGui::TextUnformatted(group.name.c_str());
                    ImGui::Separator();
                    ImGui::Text("Submeshes: %d", static_cast<int>(group.submeshes.size()));

                    uint32_t bvhNodes = renderer.getBVHNodeCount();
                    if (bvhNodes > 0)
                    {
                        ImGui::SeparatorText("BVH");
                        ImGui::Text("Nodes: %u", bvhNodes);

                        vex::AABB root = renderer.getBVHRootAABB();
                        glm::vec3 sz = root.max - root.min;
                        ImGui::Text("Root AABB: %.2f x %.2f x %.2f", sz.x, sz.y, sz.z);
                        ImGui::Text("  Min: (%.2f, %.2f, %.2f)", root.min.x, root.min.y, root.min.z);
                        ImGui::Text("  Max: (%.2f, %.2f, %.2f)", root.max.x, root.max.y, root.max.z);

                        ImGui::Text("SAH Cost: %.1f", renderer.getBVHSAHCost());

                        size_t mem = renderer.getBVHMemoryBytes();
                        if (mem < 1024)
                            ImGui::Text("Memory: %zu B", mem);
                        else
                            ImGui::Text("Memory: %.1f KB", static_cast<float>(mem) / 1024.0f);
                    }
                }
            }
            break;

        case Selection::Skybox:
            ImGui::TextUnformatted("Skybox");
            ImGui::Separator();
            {
                const char* envItems[] = { "Solid Color", "sky", "warehouse", "Custom HDR..." };
                if (ImGui::Combo("Background", &scene.currentEnvmap, envItems, Scene::EnvmapCount))
                {
                    if (scene.currentEnvmap == Scene::CustomHDR)
                    {
                        std::string hdrPath = openHdrFileDialog();
                        if (!hdrPath.empty())
                        {
                            scene.customEnvmapPath = hdrPath;
                        }
                        else
                        {
                            scene.currentEnvmap = m_prevEnvmapForRevert;
                        }
                    }
                    else
                    {
                        scene.customEnvmapPath.clear();
                    }

                    if (scene.currentEnvmap >= Scene::Sky && scene.currentEnvmap <= Scene::Warehouse)
                    {
                        std::string path = std::string("assets/textures/envmaps/")
                                         + scene.envmapNames[scene.currentEnvmap] + "/"
                                         + scene.envmapNames[scene.currentEnvmap] + ".jpg";
                        if (scene.skybox)
                            scene.skybox->load(path);
                    }

                    m_prevEnvmapForRevert = scene.currentEnvmap;
                }
            }
            if (scene.currentEnvmap == Scene::SolidColor)
                ImGui::ColorEdit3("Color", &scene.skyboxColor.x);
            if (scene.currentEnvmap == Scene::CustomHDR && !scene.customEnvmapPath.empty())
            {
                ImGui::TextWrapped("Path: %s", scene.customEnvmapPath.c_str());
            }
            break;

        case Selection::Light:
            ImGui::TextUnformatted("Light");
            ImGui::Separator();
            ImGui::Checkbox("Enabled", &scene.showLight);
            ImGui::ColorEdit3("Color", &scene.lightColor.x);
            ImGui::DragFloat("Intensity", &scene.lightIntensity, 0.05f, 0.0f, 100.0f, "%.2f");
            ImGui::DragFloat3("Position", &scene.lightPos.x, 0.05f);
            break;

        case Selection::Sun:
            ImGui::TextUnformatted("Sun (Directional Light)");
            ImGui::Separator();
            ImGui::Checkbox("Enabled", &scene.showSun);
            ImGui::SliderFloat("Azimuth", &scene.sunAzimuth, 0.0f, 360.0f, "%.1f deg");
            ImGui::SliderFloat("Elevation", &scene.sunElevation, 0.0f, 90.0f, "%.1f deg");
            ImGui::ColorEdit3("Color", &scene.sunColor.x);
            ImGui::DragFloat("Intensity", &scene.sunIntensity, 0.1f, 0.0f, 100.0f, "%.2f");
            {
                float angDeg = glm::degrees(scene.sunAngularRadius);
                if (ImGui::SliderFloat("Angular Radius", &angDeg, 0.1f, 5.0f, "%.2f deg"))
                    scene.sunAngularRadius = glm::radians(angDeg);
            }
            ImGui::SeparatorText("Shadow Map");
            {
                float bias = renderer.getShadowNormalBiasTexels();
                if (ImGui::SliderFloat("Normal Bias (texels)", &bias, 0.0f, 6.0f, "%.2f"))
                    renderer.setShadowNormalBiasTexels(bias);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "World-space normal offset per shadow texel.\n"
                        "Increase to reduce acne; decrease to tighten shadow edges.\n"
                        "Scales automatically with scene/frustum size.");

                uintptr_t shadowHandle = renderer.getShadowMapDisplayHandle();
                if (shadowHandle != 0)
                {
                    float avail = ImGui::GetContentRegionAvail().x;
                    ImTextureID texID = static_cast<ImTextureID>(shadowHandle);
                    if (renderer.shadowMapFlipsUV())
                        ImGui::Image(texID, ImVec2(avail, avail), ImVec2(0, 1), ImVec2(1, 0));
                    else
                        ImGui::Image(texID, ImVec2(avail, avail));
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Shadow depth map (4096x4096)\nVulkan: red channel = depth");

                    if (ImGui::Button("Save Shadow Map..."))
                    {
                        std::string savePath = saveImageFileDialog();
                        if (!savePath.empty())
                        {
                            if (renderer.saveShadowMap(savePath))
                                vex::Log::info("Saved shadow map: " + savePath);
                            else
                                vex::Log::error("Failed to save shadow map: " + savePath);
                        }
                    }
                }
                else
                {
                    ImGui::TextDisabled("(enable Sun to see shadow map)");
                }
            }
            break;

        case Selection::Camera:
            ImGui::TextUnformatted("Camera");
            ImGui::Separator();
            ImGui::SliderFloat("FOV", &scene.camera.fov, 10.0f, 120.0f);
            ImGui::DragFloat("Near Plane", &scene.camera.nearPlane, 0.001f, 0.001f, 10.0f, "%.3f");
            ImGui::DragFloat("Far Plane", &scene.camera.farPlane, 1.0f, 1.0f, 10000.0f, "%.0f");
            ImGui::DragFloat3("Target", &scene.camera.getTarget().x, 0.05f);
            ImGui::DragFloat("Distance", &scene.camera.getDistance(), 0.05f, 0.1f, 100.0f);
            {
                glm::vec3 pos = scene.camera.getPosition();
                ImGui::Text("Position: %.2f, %.2f, %.2f", pos.x, pos.y, pos.z);
            }
            ImGui::Separator();
            ImGui::TextUnformatted("Depth of Field (Path Trace only)");
            ImGui::SliderFloat("Aperture",      &scene.camera.aperture,      0.0f, 0.1f, "%.4f");
            ImGui::DragFloat ("Focus Distance", &scene.camera.focusDistance, 0.1f, 0.1f, 1000.0f, "%.2f");
            break;

        case Selection::None:
        default:
            ImGui::TextDisabled("Select an object in the Hierarchy");
            break;
    }

    ImGui::End();
}

void EditorUI::renderSettings(SceneRenderer& renderer)
{
    ImGui::Begin("Settings");

    const char* renderModes[] = { "Rasterization", "CPU Raytracing", "GPU Raytracing" };
    ImGui::Combo("Render Mode", &m_renderModeIndex, renderModes, static_cast<int>(std::size(renderModes)));

    if (m_renderModeIndex == 0)
    {
        const char* debugModes[] = { "None", "Wireframe", "Depth", "Normals",
                                      "UVs", "Albedo", "Emission", "Material ID" };
        ImGui::Combo("Debug Mode", &m_debugModeIndex, debugModes, static_cast<int>(std::size(debugModes)));

        bool normalMap = renderer.getEnableNormalMapping();
        if (ImGui::Checkbox("Normal Mapping", &normalMap))
            renderer.setEnableNormalMapping(normalMap);

        ImGui::SeparatorText("Shadows");
        bool shadows = renderer.getRasterEnableShadows();
        if (ImGui::Checkbox("Shadow Mapping", &shadows))
            renderer.setRasterEnableShadows(shadows);

        ImGui::SeparatorText("Environment");
        bool rEnv = renderer.getRasterEnableEnvLighting();
        if (ImGui::Checkbox("Environment Lighting##raster", &rEnv))
            renderer.setRasterEnableEnvLighting(rEnv);
        float rMult = renderer.getRasterEnvLightMultiplier();
        if (ImGui::SliderFloat("Env Multiplier##raster", &rMult, 0.0f, 2.0f, "%.2f"))
            renderer.setRasterEnvLightMultiplier(rMult);

        ImGui::SeparatorText("Post Processing");
        float rExposure = renderer.getRasterExposure();
        if (ImGui::SliderFloat("Exposure##raster", &rExposure, -5.0f, 5.0f, "%.1f"))
            renderer.setRasterExposure(rExposure);
        bool rACES = renderer.getRasterEnableACES();
        if (ImGui::Checkbox("ACES Tonemapping##raster", &rACES))
            renderer.setRasterEnableACES(rACES);
        float rGamma = renderer.getRasterGamma();
        if (ImGui::SliderFloat("Gamma##raster", &rGamma, 1.0f, 3.0f, "%.2f"))
            renderer.setRasterGamma(rGamma);

    }

    if (m_renderModeIndex == 1)
    {
        {
            uint32_t samples = renderer.getRaytraceSampleCount();
            uint32_t maxSamp = renderer.getCPUMaxSamples();
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
            int v = static_cast<int>(renderer.getCPUMaxSamples());
            if (ImGui::DragInt("Max Samples##cpu", &v, 8.0f, 0, 1 << 20))
                renderer.setCPUMaxSamples(static_cast<uint32_t>(std::max(0, v)));
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("0 = unlimited");
        }

        int maxDepth = renderer.getMaxDepth();
        if (ImGui::SliderInt("Max Depth", &maxDepth, 1, 16))
            renderer.setMaxDepth(maxDepth);

        bool nee = renderer.getEnableNEE();
        if (ImGui::Checkbox("Next Event Estimation", &nee))
            renderer.setEnableNEE(nee);

        bool rr = renderer.getEnableRR();
        if (ImGui::Checkbox("Russian Roulette", &rr))
            renderer.setEnableRR(rr);

        bool aa = renderer.getEnableAA();
        if (ImGui::Checkbox("Anti-Aliasing", &aa))
            renderer.setEnableAA(aa);

        bool firefly = renderer.getEnableFireflyClamping();
        if (ImGui::Checkbox("Firefly Clamping", &firefly))
            renderer.setEnableFireflyClamping(firefly);

        bool env = renderer.getEnableEnvironment();
        if (ImGui::Checkbox("Environment Lighting", &env))
            renderer.setEnableEnvironment(env);

        float envMult = renderer.getEnvLightMultiplier();
        if (ImGui::SliderFloat("Env Multiplier", &envMult, 0.0f, 2.0f, "%.2f"))
            renderer.setEnvLightMultiplier(envMult);

        bool flatShade = renderer.getFlatShading();
        if (ImGui::Checkbox("Flat Shading", &flatShade))
            renderer.setFlatShading(flatShade);

        bool normalMap = renderer.getEnableNormalMapping();
        if (ImGui::Checkbox("Normal Mapping", &normalMap))
            renderer.setEnableNormalMapping(normalMap);

        bool emissive = renderer.getEnableEmissive();
        if (ImGui::Checkbox("Emissive Materials", &emissive))
            renderer.setEnableEmissive(emissive);

        ImGui::SeparatorText("Diagnostics");

        {
            int expVal = static_cast<int>(std::round(std::log10(renderer.getRayEps())));
            expVal = std::clamp(expVal, -5, -1);
            if (ImGui::SliderInt("Ray EPS (10^n)", &expVal, -5, -1))
                renderer.setRayEps(std::pow(10.0f, static_cast<float>(expVal)));
            ImGui::SameLine();
            ImGui::TextDisabled("= %.0e", renderer.getRayEps());
        }

        ImGui::SeparatorText("Post Processing");

        float exposure = renderer.getExposure();
        if (ImGui::SliderFloat("Exposure", &exposure, -5.0f, 5.0f, "%.1f"))
            renderer.setExposure(exposure);

        bool aces = renderer.getEnableACES();
        if (ImGui::Checkbox("ACES Tonemapping", &aces))
            renderer.setEnableACES(aces);

        float gamma = renderer.getGamma();
        if (ImGui::SliderFloat("Gamma", &gamma, 1.0f, 3.0f, "%.2f"))
            renderer.setGamma(gamma);
    }
    else if (m_renderModeIndex == 2)
    {
        {
            uint32_t samples = renderer.getRaytraceSampleCount();
            uint32_t maxSamp = renderer.getGPUMaxSamples();
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
            int v = static_cast<int>(renderer.getGPUMaxSamples());
            if (ImGui::DragInt("Max Samples##gpu", &v, 8.0f, 0, 1 << 20))
                renderer.setGPUMaxSamples(static_cast<uint32_t>(std::max(0, v)));
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("0 = unlimited");
        }
        if (ImGui::SmallButton("Reload Shader (F5)"))
            renderer.reloadGPUShader();

        int maxDepth = renderer.getGPUMaxDepth();
        if (ImGui::SliderInt("Max Depth", &maxDepth, 1, 16))
            renderer.setGPUMaxDepth(maxDepth);

        bool nee = renderer.getGPUEnableNEE();
        if (ImGui::Checkbox("Next Event Estimation", &nee))
            renderer.setGPUEnableNEE(nee);

        bool rr = renderer.getGPUEnableRR();
        if (ImGui::Checkbox("Russian Roulette", &rr))
            renderer.setGPUEnableRR(rr);

        bool aa = renderer.getGPUEnableAA();
        if (ImGui::Checkbox("Anti-Aliasing", &aa))
            renderer.setGPUEnableAA(aa);

        bool firefly = renderer.getGPUEnableFireflyClamping();
        if (ImGui::Checkbox("Firefly Clamping", &firefly))
            renderer.setGPUEnableFireflyClamping(firefly);

        bool env = renderer.getGPUEnableEnvironment();
        if (ImGui::Checkbox("Environment Lighting", &env))
            renderer.setGPUEnableEnvironment(env);

        float envMult = renderer.getGPUEnvLightMultiplier();
        if (ImGui::SliderFloat("Env Multiplier", &envMult, 0.0f, 2.0f, "%.2f"))
            renderer.setGPUEnvLightMultiplier(envMult);

        bool flatShade = renderer.getGPUFlatShading();
        if (ImGui::Checkbox("Flat Shading", &flatShade))
            renderer.setGPUFlatShading(flatShade);

        bool normalMap = renderer.getGPUEnableNormalMapping();
        if (ImGui::Checkbox("Normal Mapping", &normalMap))
            renderer.setGPUEnableNormalMapping(normalMap);

        bool emissive = renderer.getGPUEnableEmissive();
        if (ImGui::Checkbox("Emissive Materials", &emissive))
            renderer.setGPUEnableEmissive(emissive);

        ImGui::SeparatorText("Diagnostics");

        {
            int expVal = static_cast<int>(std::round(std::log10(renderer.getGPURayEps())));
            expVal = std::clamp(expVal, -5, -1);
            if (ImGui::SliderInt("Ray EPS (10^n)", &expVal, -5, -1))
                renderer.setGPURayEps(std::pow(10.0f, static_cast<float>(expVal)));
            ImGui::SameLine();
            ImGui::TextDisabled("= %.0e", renderer.getGPURayEps());
        }

        ImGui::SeparatorText("Post Processing");

        float exposure = renderer.getGPUExposure();
        if (ImGui::SliderFloat("Exposure", &exposure, -5.0f, 5.0f, "%.1f"))
            renderer.setGPUExposure(exposure);

        bool aces = renderer.getGPUEnableACES();
        if (ImGui::Checkbox("ACES Tonemapping", &aces))
            renderer.setGPUEnableACES(aces);

        float gamma = renderer.getGPUGamma();
        if (ImGui::SliderFloat("Gamma", &gamma, 1.0f, 3.0f, "%.2f"))
            renderer.setGPUGamma(gamma);
    }
    ImGui::End();
}

void EditorUI::renderConsole()
{
    ImGui::Begin("Console");

    if (ImGui::Button("Clear"))
        vex::Log::clear();

    ImGui::Separator();

    ImGui::BeginChild("LogScroll", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

    for (const auto& entry : vex::Log::getEntries())
    {
        ImVec4 color;
        const char* prefix;
        switch (entry.level)
        {
            case vex::Log::Level::Warn:
                color = { 1.0f, 0.8f, 0.0f, 1.0f };
                prefix = "[WARN] ";
                break;
            case vex::Log::Level::Error:
                color = { 1.0f, 0.3f, 0.3f, 1.0f };
                prefix = "[ERROR] ";
                break;
            default:
                color = { 0.8f, 0.8f, 0.8f, 1.0f };
                prefix = "[INFO] ";
                break;
        }
        // Dim timestamp
        char tsBuf[16];
        std::snprintf(tsBuf, sizeof(tsBuf), "[%7.3fs] ", entry.timestamp);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f, 0.45f, 0.45f, 1.0f));
        ImGui::TextUnformatted(tsBuf);
        ImGui::PopStyleColor();
        ImGui::SameLine(0.0f, 0.0f);

        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::TextUnformatted(prefix);
        ImGui::SameLine(0.0f, 0.0f);
        ImGui::TextUnformatted(entry.message.c_str());
        ImGui::PopStyleColor();
    }

    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
        ImGui::SetScrollHereY(1.0f);

    ImGui::EndChild();

    ImGui::End();
}

void EditorUI::renderStats(SceneRenderer& renderer, Scene& scene, vex::GraphicsContext& ctx)
{
    ImGui::Begin("Stats");

    // --- Performance ---
    const ImGuiIO& io = ImGui::GetIO();
    ImGui::SeparatorText("Performance");
    ImGui::Text("FPS:        %.1f", io.Framerate);
    ImGui::Text("Frame time: %.2f ms", 1000.0f / io.Framerate);
    ImGui::Text("Draw calls: %d", renderer.getDrawCalls());

    // --- Scene ---
    ImGui::SeparatorText("Scene");

    uint32_t totalVerts   = 0;
    uint32_t totalIndices = 0;
    int      totalSubs    = 0;
    int      totalTex     = 0;

    for (const auto& group : scene.meshGroups)
    {
        for (const auto& sm : group.submeshes)
        {
            totalVerts   += sm.vertexCount;
            totalIndices += sm.indexCount;
            ++totalSubs;
            if (sm.diffuseTexture)
                ++totalTex;
        }
    }

    ImGui::Text("Mesh groups:  %d", static_cast<int>(scene.meshGroups.size()));
    ImGui::Text("Submeshes:    %d", totalSubs);
    ImGui::Text("Vertices:     %u", totalVerts);
    ImGui::Text("Triangles:    %u", totalIndices / 3);
    ImGui::Text("Textures:     %d", totalTex);

    // --- Viewport ---
    ImGui::SeparatorText("Viewport");
    const auto& spec = renderer.getFramebuffer()->getSpec();
    ImGui::Text("Resolution: %u x %u", spec.width, spec.height);

    // --- GPU Memory ---
    vex::MemoryStats mem = ctx.getMemoryStats();
    if (mem.available)
    {
        ImGui::SeparatorText("GPU Memory");
        ImGui::Text("VRAM used:   %.1f MB", mem.usedMB);
        ImGui::Text("VRAM budget: %.1f MB", mem.budgetMB);
    }

    // --- BVH ---
    size_t bvhMem = renderer.getBVHMemoryBytes();
    if (bvhMem > 0)
    {
        ImGui::SeparatorText("BVH");
        ImGui::Text("Nodes:  %u", renderer.getBVHNodeCount());
        ImGui::Text("Memory: %.1f KB", static_cast<float>(bvhMem) / 1024.0f);
    }

    // --- Backend ---
    ImGui::SeparatorText("Backend");
    ImGui::TextUnformatted(std::string(ctx.backendName()).c_str());

    ImGui::End();
}
