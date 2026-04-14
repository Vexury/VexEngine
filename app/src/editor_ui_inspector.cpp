#include "editor_ui.h"
#include "scene.h"
#include "scene_renderer.h"
#include "file_dialog.h"

#include <imgui.h>

#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

#include <vex/core/log.h>

void EditorUI::renderInspector(Scene& scene, SceneRenderer& renderer)
{
    ImGui::Begin("Inspector");

    switch (m_selection->type)
    {
        case Selection::Mesh:
        {
            if (m_selection->index >= 0 && m_selection->index < static_cast<int>(scene.nodes.size()))
            {
                auto& node = scene.nodes[m_selection->index];

                const char* matTypes[] = { "Microfacet (GGX)", "Mirror", "Dielectric", "Thin Glass" };
                bool isRasterize = (renderer.getRenderMode() == RenderMode::Rasterize);

                auto drawSubmeshMaterial = [&](auto& sm)
                {
                    if (ImGui::ColorEdit3("Base Color", &sm.meshData.baseColor.x))
                        scene.materialDirty = true;

                    if (ImGui::Combo("Type", &sm.meshData.materialType, matTypes, 4))
                        scene.materialDirty = true;

                    if (sm.meshData.materialType == 0)
                    {
                        if (sm.meshData.roughnessTexturePath.empty())
                        {
                            ImGui::DragFloat("Roughness", &sm.meshData.roughness, 0.01f, 0.f, 1.f, "%.2f");
                            if (ImGui::IsItemDeactivatedAfterEdit()) scene.materialDirty = true;
                        }
                        else ImGui::TextDisabled("Roughness  (texture)");

                        if (sm.meshData.metallicTexturePath.empty())
                        {
                            ImGui::DragFloat("Metallic", &sm.meshData.metallic, 0.01f, 0.f, 1.f, "%.2f");
                            if (ImGui::IsItemDeactivatedAfterEdit()) scene.materialDirty = true;
                        }
                        else ImGui::TextDisabled("Metallic   (texture)");
                    }
                    else if (sm.meshData.materialType == 2 || sm.meshData.materialType == 3)
                    {
                        ImGui::BeginDisabled(isRasterize);
                        ImGui::DragFloat("IOR", &sm.meshData.ior, 0.01f, 1.f, 3.f, "%.2f");
                        if (ImGui::IsItemDeactivatedAfterEdit()) scene.materialDirty = true;

                        ImGui::SameLine();
                        if (ImGui::SmallButton("..."))
                            ImGui::OpenPopup("ior_presets");
                        if (ImGui::BeginPopup("ior_presets"))
                        {
                            struct IORPreset { const char* name; float ior; };
                            static constexpr IORPreset presets[] = {
                                {"Water (1.33)",   1.33f},
                                {"Glass (1.50)",   1.50f},
                                {"Crystal (1.70)", 1.70f},
                                {"Diamond (2.42)", 2.42f},
                            };
                            for (const auto& p : presets)
                            {
                                if (ImGui::MenuItem(p.name))
                                {
                                    sm.meshData.ior = p.ior;
                                    scene.materialDirty = true;
                                }
                            }
                            ImGui::EndPopup();
                        }
                        if (sm.meshData.materialType == 3)
                        {
                            ImGui::DragFloat("Tint", &sm.meshData.metallic, 0.01f, 0.f, 1.f, "%.2f");
                            if (ImGui::IsItemDeactivatedAfterEdit()) scene.materialDirty = true;
                        }
                        ImGui::EndDisabled();
                    }

                    if (ImGui::Checkbox("Alpha Clip", &sm.meshData.alphaClip))
                        scene.materialDirty = true;

                    if (ImGui::ColorEdit3("Emissive Color", &sm.meshData.emissiveColor.x))
                        scene.materialDirty = true;
                    ImGui::DragFloat("Emissive Strength", &sm.meshData.emissiveStrength, 0.05f, 0.f, 100.f, "%.2f");
                    if (ImGui::IsItemDeactivatedAfterEdit()) scene.materialDirty = true;

                    if (isRasterize && sm.meshData.materialType != 0)
                        ImGui::TextDisabled("Rendered as Microfacet in rasterizer.\nSwitch to a path tracer to see this material.");

                    if (!sm.meshData.name.empty())
                    {
                        ImGui::Spacing();
                        ImGui::TextDisabled("Material: %s", sm.meshData.name.c_str());
                        ImGui::SameLine();
                        if (ImGui::SmallButton("Apply to all"))
                        {
                            const std::string& matName = sm.meshData.name;
                            for (auto& n : scene.nodes)
                                for (auto& other : n.submeshes)
                                    if (other.meshData.name == matName && &other != &sm)
                                    {
                                        other.meshData.materialType     = sm.meshData.materialType;
                                        other.meshData.roughness        = sm.meshData.roughness;
                                        other.meshData.metallic         = sm.meshData.metallic;
                                        other.meshData.ior              = sm.meshData.ior;
                                        other.meshData.alphaClip        = sm.meshData.alphaClip;
                                        other.meshData.emissiveStrength = sm.meshData.emissiveStrength;
                                        other.meshData.emissiveColor    = sm.meshData.emissiveColor;
                                        other.meshData.baseColor        = sm.meshData.baseColor;
                                    }
                            scene.materialDirty = true;
                        }
                    }

                    ImGui::SeparatorText("Textures");

                    auto texRow = [](const char* label,
                                     const std::string& path,
                                     const std::shared_ptr<vex::Texture2D>& tex)
                    {
                        constexpr float kThumbSize = 18.0f;
                        constexpr float kPreviewSize = 192.0f;

                        if (tex)
                        {
                            ImTextureID tid = static_cast<ImTextureID>(tex->getNativeHandle());
                            ImGui::Image(tid, {kThumbSize, kThumbSize}, {0.f, 1.f}, {1.f, 0.f});

                            if (ImGui::IsItemHovered())
                            {
                                ImGui::BeginTooltip();
                                float aspect = (tex->getHeight() > 0)
                                    ? static_cast<float>(tex->getWidth()) / tex->getHeight()
                                    : 1.0f;
                                float pw = kPreviewSize * aspect;
                                float ph = kPreviewSize;
                                if (pw > kPreviewSize) { ph = kPreviewSize / aspect; pw = kPreviewSize; }
                                ImGui::Image(tid, {pw, ph}, {0.f, 1.f}, {1.f, 0.f});
                                ImGui::Text("%ux%u", tex->getWidth(), tex->getHeight());
                                ImGui::EndTooltip();
                            }
                        }
                        else
                        {
                            ImGui::Dummy({kThumbSize, kThumbSize});
                        }

                        ImGui::SameLine();
                        auto s = path.find_last_of("/\\");
                        const char* name = path.empty() ? "none"
                                          : path.c_str() + (s != std::string::npos ? s + 1 : 0);
                        ImGui::Text("%s: %s", label, name);
                    };

                    texRow("Diffuse",   sm.meshData.diffuseTexturePath,    sm.diffuseTexture);
                    texRow("Alpha",     sm.meshData.alphaTexturePath,      sm.alphaTexture);
                    texRow("Normal",    sm.meshData.normalTexturePath,     sm.normalTexture);
                    texRow("AO",        sm.meshData.aoTexturePath,         sm.aoTexture);
                    texRow("Roughness", sm.meshData.roughnessTexturePath,  sm.roughnessTexture);
                    texRow("Metallic",  sm.meshData.metallicTexturePath,   sm.metallicTexture);
                    texRow("Emissive",  sm.meshData.emissiveTexturePath,   sm.emissiveTexture);
                };

                ImGui::TextUnformatted(node.name.c_str());
                ImGui::Separator();

                ImGui::SeparatorText("Transform");

                glm::vec3 decompScale, decompTranslation, decompSkew;
                glm::vec4 decompPerspective;
                glm::quat decompRotation;
                glm::decompose(node.localMatrix, decompScale, decompRotation,
                               decompTranslation, decompSkew, decompPerspective);
                glm::vec3 eulerDeg = glm::degrees(glm::eulerAngles(glm::conjugate(decompRotation)));

                bool needRecompose = false;
                bool released      = false;

                if (ImGui::DragFloat3("Translation", &decompTranslation.x, 0.01f, 0.f, 0.f, "%.3f"))
                    needRecompose = true;
                released |= ImGui::IsItemDeactivatedAfterEdit();
                if (ImGui::DragFloat3("Rotation",    &eulerDeg.x,          0.5f,  0.f, 0.f, "%.1f"))
                    needRecompose = true;
                released |= ImGui::IsItemDeactivatedAfterEdit();
                if (ImGui::DragFloat3("Scale",       &decompScale.x,       0.01f, 0.001f, 0.f, "%.3f"))
                    needRecompose = true;
                released |= ImGui::IsItemDeactivatedAfterEdit();

                if (needRecompose)
                {
                    decompScale = glm::max(decompScale, glm::vec3(0.001f));
                    node.localMatrix = glm::translate(glm::mat4(1.f), decompTranslation)
                                     * glm::mat4_cast(glm::quat(glm::radians(eulerDeg)))
                                     * glm::scale(glm::mat4(1.f), decompScale);
                }
                if (released && renderer.getRenderMode() != RenderMode::Rasterize)
                    scene.geometryDirty = true;

                if (ImGui::Button("Reset Transform"))
                {
                    node.localMatrix = glm::mat4(1.f);
                    if (renderer.getRenderMode() != RenderMode::Rasterize)
                        scene.geometryDirty = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Duplicate"))
                    m_pendingDuplicate = true;

                // Materials — inline for single-submesh nodes, collapsibles for multi
                if (node.submeshes.size() == 1)
                {
                    ImGui::SeparatorText("Material");
                    drawSubmeshMaterial(node.submeshes[0]);
                }
                else
                {
                    ImGui::SeparatorText("Materials");
                    for (int si = 0; si < static_cast<int>(node.submeshes.size()); ++si)
                    {
                        ImGui::PushID(si);
                        if (ImGui::CollapsingHeader(node.submeshes[si].name.c_str()))
                        {
                            ImGui::Text("Vertices:  %u", node.submeshes[si].vertexCount);
                            ImGui::Text("Triangles: %u", node.submeshes[si].indexCount / 3);
                            drawSubmeshMaterial(node.submeshes[si]);
                        }
                        ImGui::PopID();
                    }
                }

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
            break;
        }

        case Selection::Skybox:
            ImGui::TextUnformatted("Skybox");
            ImGui::Separator();
            {
                int comboSel = (scene.currentEnvmap < Scene::CustomHDR) ? scene.currentEnvmap : 0;
                if (ImGui::Combo("Background", &comboSel, Scene::envmapNames, Scene::CustomHDR))
                {
                    scene.currentEnvmap = comboSel;
                    scene.customEnvmapPath.clear();
                    if (comboSel > Scene::SolidColor)
                        m_pendingEnvLoadPath = Scene::envmapPaths[comboSel];
                    m_prevEnvmapForRevert = scene.currentEnvmap;
                }
            }
            if (ImGui::Button("Load from file..."))
            {
                std::string hdrPath = openHdrFileDialog();
                if (!hdrPath.empty())
                {
                    scene.customEnvmapPath = hdrPath;
                    scene.currentEnvmap = Scene::CustomHDR;
                    m_pendingEnvLoadPath = hdrPath;
                    m_prevEnvmapForRevert = scene.currentEnvmap;
                }
            }
            if (scene.currentEnvmap == Scene::SolidColor)
                ImGui::ColorEdit3("Color", &scene.skyboxColor.x);
            if (scene.currentEnvmap == Scene::CustomHDR && !scene.customEnvmapPath.empty())
                ImGui::TextWrapped("Path: %s", scene.customEnvmapPath.c_str());
            if (scene.currentEnvmap != Scene::SolidColor)
            {
                float deg = glm::degrees(scene.envRotation);
                if (ImGui::SliderFloat("Rotation", &deg, -180.0f, 180.0f, "%.1f°"))
                    scene.envRotation = glm::radians(deg);
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
                ImGui::SliderFloat("Normal Bias (texels)", &renderer.getRasterSettings().shadowBiasTexels, 0.0f, 6.0f, "%.2f");
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

        case Selection::Volume:
            if (m_selection->index >= 0 && m_selection->index < static_cast<int>(scene.volumes.size()))
            {
                auto& vol = scene.volumes[m_selection->index];

                // Name field
                char nameBuf[256];
                std::snprintf(nameBuf, sizeof(nameBuf), "%s", vol.name.c_str());
                if (ImGui::InputText("Name", nameBuf, sizeof(nameBuf)))
                    vol.name = nameBuf;

                ImGui::Separator();
                ImGui::Checkbox("Enabled", &vol.enabled);
                ImGui::Checkbox("Infinite (global fog)", &vol.infinite);
                ImGui::Separator();

                if (!vol.infinite)
                {
                    ImGui::DragFloat3("Center",   &vol.center.x,   0.05f);
                    ImGui::DragFloat3("Half Size", &vol.halfSize.x, 0.05f, 0.01f, 1000.0f);
                    ImGui::Separator();
                }

                ImGui::SliderFloat("Density (σt)",  &vol.density, 0.001f, 10.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                ImGui::ColorEdit3("Scatter Color",  &vol.albedo.x);
                ImGui::SliderFloat("Anisotropy (g)", &vol.aniso,  -1.0f,  1.0f,  "%.2f");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "Henyey-Greenstein phase function:\n"
                        "  0 = isotropic (fog, uniform scatter)\n"
                        " +1 = forward (haze, glowing halos)\n"
                        " -1 = backward (some dust types)");

                ImGui::Separator();
                if (ImGui::Button("Remove Volume"))
                {
                    // Routed through App DEL handler (same as pressing Delete key)
                    // so it goes through the undo stack. We simulate a DEL by setting
                    // the selection and deferring to the Delete key path isn't available
                    // here — just erase directly as a fallback (no undo for inspector button).
                    scene.volumes.erase(scene.volumes.begin() + m_selection->index);
                    m_selection->type = Selection::None;
                }
            }
            break;

        case Selection::None:
        default:
            ImGui::TextDisabled("Select an object in the Hierarchy");
            break;
    }

    ImGui::End();
}
