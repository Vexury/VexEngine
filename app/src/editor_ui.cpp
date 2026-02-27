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

#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

#include <iterator>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Gizmo helpers (file-scope)
// ─────────────────────────────────────────────────────────────────────────────

// Project world pos → viewport screen pos (returns {-99999,-99999} if behind camera)
static ImVec2 gizmoProject(const glm::vec3& p, const glm::mat4& vp, ImVec2 org, ImVec2 sz)
{
    glm::vec4 c = vp * glm::vec4(p, 1.f);
    if (c.w < 1e-5f) return { -99999.f, -99999.f };
    float iw = 1.f / c.w;
    return { org.x + (c.x * iw * .5f + .5f) * sz.x,
             org.y + (1.f - (c.y * iw * .5f + .5f)) * sz.y };
}

// 2-D distance from point P to segment AB
static float gizmoSegDist(ImVec2 p, ImVec2 a, ImVec2 b)
{
    float dx = b.x - a.x, dy = b.y - a.y, d2 = dx*dx + dy*dy;
    if (d2 < 1.f) return hypotf(p.x - a.x, p.y - a.y);
    float t = glm::clamp(((p.x - a.x)*dx + (p.y - a.y)*dy) / d2, 0.f, 1.f);
    float ex = a.x + t*dx - p.x, ey = a.y + t*dy - p.y;
    return sqrtf(ex*ex + ey*ey);
}

// Draw arrow: line + filled triangle arrowhead
static void gizmoArrow(ImDrawList* dl, ImVec2 root, ImVec2 tip, ImU32 col)
{
    dl->AddLine(root, tip, col, 2.5f);
    float dx = tip.x - root.x, dy = tip.y - root.y;
    float len = sqrtf(dx*dx + dy*dy); if (len < 1.f) return;
    float nx = dx/len, ny = dy/len, px = -ny, py = nx;
    ImVec2 base = { tip.x - nx*14.f, tip.y - ny*14.f };
    dl->AddTriangleFilled(tip,
        { base.x + px*5.f, base.y + py*5.f },
        { base.x - px*5.f, base.y - py*5.f }, col);
}

bool EditorUI::consumePickRequest(int& outX, int& outY)
{
    if (!m_pickRequested)
        return false;

    m_pickRequested = false;
    outX = m_pickX;
    outY = m_pickY;
    return true;
}

bool EditorUI::consumePendingImport(std::string& outPath, std::string& outName)
{
    if (m_pendingImportPath.empty()) return false;
    outPath = std::move(m_pendingImportPath);
    outName = std::move(m_pendingImportName);
    m_pendingImportPath.clear();
    return true;
}

void EditorUI::setLoadingState(const std::string& stage, float progress)
{
    m_loadingStage    = stage;
    m_loadingProgress = progress;
}

void EditorUI::clearLoadingState()
{
    m_loadingStage.clear();
    m_loadingProgress = 0.f;
}

void EditorUI::renderLoadingOverlay()
{
    ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos({0.f, 0.f});
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::SetNextWindowBgAlpha(0.82f);
    ImGui::Begin("##loading_overlay", nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
        ImGuiWindowFlags_NoNav         | ImGuiWindowFlags_NoMove  |
        ImGuiWindowFlags_NoSavedSettings);

    float cx = io.DisplaySize.x * 0.5f;
    float cy = io.DisplaySize.y * 0.5f;
    float barW = io.DisplaySize.x * 0.38f;

    // Stage label
    ImVec2 textSz = ImGui::CalcTextSize(m_loadingStage.c_str());
    ImGui::SetCursorPos({cx - textSz.x * 0.5f, cy - 24.f});
    ImGui::TextUnformatted(m_loadingStage.c_str());

    // Progress bar
    ImGui::SetCursorPos({cx - barW * 0.5f, cy});
    ImGui::ProgressBar(m_loadingProgress, {barW, 18.f}, "");

    ImGui::End();
}

void EditorUI::renderViewport(SceneRenderer& renderer, Scene& scene)
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

        bool gizmoActive = false;
        if (m_selectionType == Selection::Mesh &&
            renderer.getRenderMode() == RenderMode::Rasterize)
            gizmoActive = drawGizmo(scene, ImGui::GetWindowDrawList(), cursor, size);

        if (!gizmoActive &&
            ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        {
            ImVec2 mouse = ImGui::GetMousePos();
            m_pickX = static_cast<int>(mouse.x - cursor.x);
            m_pickY = static_cast<int>(mouse.y - cursor.y);
            m_pickRequested = true;
        }
    }

    ImGui::End();
}

bool EditorUI::drawGizmo(Scene& scene, ImDrawList* dl, ImVec2 vpOrigin, ImVec2 vpSize)
{
    if (m_selectionIndex < 0 || m_selectionIndex >= static_cast<int>(scene.meshGroups.size()))
        return false;

    auto& group = scene.meshGroups[m_selectionIndex];

    float aspect = vpSize.x / vpSize.y;
    glm::mat4 view = scene.camera.getViewMatrix();
    glm::mat4 proj = scene.camera.getProjectionMatrix(aspect);
    glm::mat4 vp   = proj * view;

    glm::vec3 camPos = scene.camera.getPosition();
    glm::vec3 pivot  = glm::vec3(group.modelMatrix[3]);

    float camDist  = glm::length(camPos - pivot);
    float gizmoLen = camDist * 0.18f;

    const ImU32 colX  = IM_COL32(220,  50,  50, 255);
    const ImU32 colY  = IM_COL32( 50, 220,  50, 255);
    const ImU32 colZ  = IM_COL32( 50,  50, 220, 255);
    const ImU32 colXH = IM_COL32(255, 150, 150, 255);
    const ImU32 colYH = IM_COL32(150, 255, 150, 255);
    const ImU32 colZH = IM_COL32(150, 150, 255, 255);
    const ImU32 axisColors[3]  = { colX,  colY,  colZ  };
    const ImU32 axisColorsH[3] = { colXH, colYH, colZH };

    const glm::vec3 axisDirs[3] = {
        glm::vec3(1, 0, 0),
        glm::vec3(0, 1, 0),
        glm::vec3(0, 0, 1)
    };

    ImVec2 origin2D = gizmoProject(pivot, vp, vpOrigin, vpSize);
    if (origin2D.x < -9000.f) return false;

    ImVec2 mouse         = ImGui::GetMousePos();
    bool   lmbDown       = ImGui::IsMouseDown(ImGuiMouseButton_Left);
    bool   lmbJustPressed= ImGui::IsMouseClicked(ImGuiMouseButton_Left);
    bool   lmbReleased   = ImGui::IsMouseReleased(ImGuiMouseButton_Left);

    int hoveredAxis = m_gizmoDragging ? m_gizmoAxis : -1;

    if (m_gizmoMode == 0 || m_gizmoMode == 2) // Translate or Scale
    {
        ImVec2 tips[3];
        for (int a = 0; a < 3; ++a)
            tips[a] = gizmoProject(pivot + axisDirs[a] * gizmoLen, vp, vpOrigin, vpSize);

        if (!m_gizmoDragging)
        {
            float bestDist = 8.f;
            hoveredAxis = -1;
            for (int a = 0; a < 3; ++a)
            {
                if (tips[a].x < -9000.f) continue;
                float d = gizmoSegDist(mouse, origin2D, tips[a]);
                if (d < bestDist) { bestDist = d; hoveredAxis = a; }
            }
        }

        if (!m_gizmoDragging && lmbJustPressed && hoveredAxis >= 0)
        {
            m_gizmoDragging  = true;
            m_gizmoAxis      = hoveredAxis;
            m_gizmoDragStart = mouse;
            m_gizmoMatStart  = group.modelMatrix;
        }

        if (m_gizmoDragging && lmbDown)
        {
            int a = m_gizmoAxis;
            glm::vec3 startPivot = glm::vec3(m_gizmoMatStart[3]);
            ImVec2 startOrigin2D = gizmoProject(startPivot, vp, vpOrigin, vpSize);
            ImVec2 startTip2D    = gizmoProject(startPivot + axisDirs[a] * gizmoLen, vp, vpOrigin, vpSize);
            float dx = startTip2D.x - startOrigin2D.x;
            float dy = startTip2D.y - startOrigin2D.y;
            float screenLen = sqrtf(dx*dx + dy*dy);
            if (screenLen > 1.f)
            {
                float nx = dx / screenLen, ny = dy / screenLen;
                float pixelDist = (mouse.x - m_gizmoDragStart.x) * nx
                                + (mouse.y - m_gizmoDragStart.y) * ny;
                float worldDist = pixelDist * gizmoLen / screenLen;

                if (m_gizmoMode == 0)
                {
                    group.modelMatrix = glm::translate(m_gizmoMatStart, axisDirs[a] * worldDist);
                }
                else
                {
                    glm::vec3 sp = glm::vec3(m_gizmoMatStart[3]);
                    glm::mat4 T  = glm::translate(glm::mat4(1.f),  sp);
                    glm::mat4 iT = glm::translate(glm::mat4(1.f), -sp);
                    float sf = glm::max(0.01f, 1.0f + worldDist / gizmoLen);
                    glm::vec3 sv(1.f); sv[a] = sf;
                    group.modelMatrix = T * glm::scale(iT * m_gizmoMatStart, sv);
                }
            }
        }

        if (m_gizmoDragging && lmbReleased)
        {
            m_gizmoDragging = false;
            m_gizmoAxis     = -1;
        }

        for (int a = 0; a < 3; ++a)
        {
            if (tips[a].x < -9000.f) continue;
            bool    active = (hoveredAxis == a);
            ImU32   col    = active ? axisColorsH[a] : axisColors[a];
            if (m_gizmoMode == 0)
            {
                gizmoArrow(dl, origin2D, tips[a], col);
            }
            else
            {
                dl->AddLine(origin2D, tips[a], col, 2.5f);
                float hSize = 6.f;
                dl->AddRectFilled(
                    { tips[a].x - hSize, tips[a].y - hSize },
                    { tips[a].x + hSize, tips[a].y + hSize },
                    col);
            }
        }
    }
    else if (m_gizmoMode == 1) // Rotate
    {
        float ringRadius = camDist * 0.16f;
        const int SEGS = 64;

        if (!m_gizmoDragging)
        {
            float bestDist = 10.f;
            hoveredAxis = -1;
            for (int a = 0; a < 3; ++a)
            {
                const glm::vec3& axis = axisDirs[a];
                glm::vec3 up = (fabsf(axis.z) < 0.9f) ? glm::vec3(0,0,1) : glm::vec3(0,1,0);
                glm::vec3 t1 = glm::normalize(glm::cross(axis, up));
                glm::vec3 t2 = glm::cross(axis, t1);
                for (int s = 0; s < SEGS; ++s)
                {
                    float ang0 = static_cast<float>(s)   / SEGS * 2.f * 3.14159265f;
                    float ang1 = static_cast<float>(s+1) / SEGS * 2.f * 3.14159265f;
                    glm::vec3 p0 = pivot + (t1 * cosf(ang0) + t2 * sinf(ang0)) * ringRadius;
                    glm::vec3 p1 = pivot + (t1 * cosf(ang1) + t2 * sinf(ang1)) * ringRadius;
                    ImVec2 s0 = gizmoProject(p0, vp, vpOrigin, vpSize);
                    ImVec2 s1 = gizmoProject(p1, vp, vpOrigin, vpSize);
                    if (s0.x < -9000.f || s1.x < -9000.f) continue;
                    float d = gizmoSegDist(mouse, s0, s1);
                    if (d < bestDist) { bestDist = d; hoveredAxis = a; }
                }
            }
        }

        if (!m_gizmoDragging && lmbJustPressed && hoveredAxis >= 0)
        {
            m_gizmoDragging  = true;
            m_gizmoAxis      = hoveredAxis;
            m_gizmoDragStart = mouse;
            m_gizmoMatStart  = group.modelMatrix;
            m_gizmoRotRefSet = false;
        }

        if (m_gizmoDragging && lmbDown)
        {
            int a = m_gizmoAxis;
            const glm::vec3& axis = axisDirs[a];

            float ndcX = (mouse.x - vpOrigin.x) / vpSize.x * 2.f - 1.f;
            float ndcY = 1.f - (mouse.y - vpOrigin.y) / vpSize.y * 2.f;
            glm::mat4 invVP = glm::inverse(vp);
            glm::vec4 worldFar = invVP * glm::vec4(ndcX, ndcY, 1.f, 1.f);
            glm::vec3 rayDir = glm::normalize(glm::vec3(worldFar) / worldFar.w - camPos);

            float denom = glm::dot(rayDir, axis);
            if (fabsf(denom) > 1e-5f)
            {
                float t = glm::dot(pivot - camPos, axis) / denom;
                if (t > 0.f)
                {
                    glm::vec3 hitPoint = camPos + rayDir * t;
                    glm::vec3 hitVec   = hitPoint - pivot;
                    float hitLen = glm::length(hitVec);
                    if (hitLen > 1e-5f)
                    {
                        hitVec /= hitLen;
                        if (!m_gizmoRotRefSet)
                        {
                            m_gizmoRotRef    = hitVec;
                            m_gizmoRotRefSet = true;
                        }
                        else
                        {
                            float cosA  = glm::clamp(glm::dot(m_gizmoRotRef, hitVec), -1.f, 1.f);
                            float sinA  = glm::dot(glm::cross(m_gizmoRotRef, hitVec), axis);
                            float angle = atan2f(sinA, cosA);
                            glm::mat4 T  = glm::translate(glm::mat4(1.f),  pivot);
                            glm::mat4 iT = glm::translate(glm::mat4(1.f), -pivot);
                            glm::mat4 R  = glm::rotate(glm::mat4(1.f), angle, axis);
                            group.modelMatrix = T * R * iT * m_gizmoMatStart;
                        }
                    }
                }
            }
        }

        if (m_gizmoDragging && lmbReleased)
        {
            m_gizmoDragging  = false;
            m_gizmoAxis      = -1;
            m_gizmoRotRefSet = false;
        }

        for (int a = 0; a < 3; ++a)
        {
            const glm::vec3& axis = axisDirs[a];
            glm::vec3 up = (fabsf(axis.z) < 0.9f) ? glm::vec3(0,0,1) : glm::vec3(0,1,0);
            glm::vec3 t1 = glm::normalize(glm::cross(axis, up));
            glm::vec3 t2 = glm::cross(axis, t1);
            bool  active = (hoveredAxis == a);
            ImU32 col    = active ? axisColorsH[a] : axisColors[a];
            for (int s = 0; s < SEGS; ++s)
            {
                float ang0 = static_cast<float>(s)   / SEGS * 2.f * 3.14159265f;
                float ang1 = static_cast<float>(s+1) / SEGS * 2.f * 3.14159265f;
                glm::vec3 p0 = pivot + (t1 * cosf(ang0) + t2 * sinf(ang0)) * ringRadius;
                glm::vec3 p1 = pivot + (t1 * cosf(ang1) + t2 * sinf(ang1)) * ringRadius;
                ImVec2 s0 = gizmoProject(p0, vp, vpOrigin, vpSize);
                ImVec2 s1 = gizmoProject(p1, vp, vpOrigin, vpSize);
                if (s0.x < -9000.f || s1.x < -9000.f) continue;
                dl->AddLine(s0, s1, col, 2.f);
            }
        }
    }

    // Center dot
    dl->AddCircleFilled(origin2D, 4.f, IM_COL32(255, 255, 255, 200));

    return (hoveredAxis >= 0) || m_gizmoDragging;
}

void EditorUI::renderHierarchy(Scene& scene, SceneRenderer& renderer)
{
    ImGui::Begin("Hierarchy");

    ImGui::TextUnformatted("Scene");
    ImGui::Indent();

    // Camera
    {
        bool selected = (m_selectionType == Selection::Camera);
        if (ImGui::Selectable("Camera", selected))
            m_selectionType = Selection::Camera;
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

    // Skybox
    if (scene.showSkybox)
    {
        bool selected = (m_selectionType == Selection::Skybox);
        if (ImGui::Selectable("Skybox", selected))
            m_selectionType = Selection::Skybox;
    }

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
            m_selectionType      = Selection::Mesh;
            m_selectionIndex     = gi;
            m_submeshIndex       = -1;
            m_selectedObjectName.clear();
        }

        if (open)
        {
            const auto& submeshes = scene.meshGroups[gi].submeshes;
            std::string lastObj;
            for (int si = 0; si < static_cast<int>(submeshes.size()); ++si)
            {
                const std::string& objName = submeshes[si].meshData.objectName;
                // One entry per unique objectName (submeshes from the same shape
                // are contiguous — skip duplicates).
                if (objName == lastObj) continue;
                lastObj = objName;

                const std::string& displayName = objName.empty()
                    ? submeshes[si].name : objName;
                bool objSelected = groupSelected
                    && m_selectedObjectName == objName;

                if (ImGui::Selectable(displayName.c_str(), objSelected))
                {
                    m_selectionType         = Selection::Mesh;
                    m_selectionIndex        = gi;
                    m_selectedObjectName    = objName;
                    m_submeshIndex          = -1;
                }
            }
            ImGui::TreePop();
        }
    }

    ImGui::Unindent();
    ImGui::Separator();

    // Import button
    if (ImGui::Button("Import OBJ..."))
    {
        void* hwnd = ImGui::GetMainViewport()->PlatformHandleRaw;
        vex::Log::info("File dialog opened");
        std::string path = openObjFileDialog(hwnd);
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

            // Defer the actual import to between frames so we can pump a loading overlay.
            m_pendingImportPath = path;
            m_pendingImportName = baseName;
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("Save Image..."))
    {
        void* hwnd = ImGui::GetMainViewport()->PlatformHandleRaw;
        std::string path = saveImageFileDialog(hwnd);
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
                    bool isRasterize = (renderer.getRenderMode() == RenderMode::Rasterize);
                    if (ImGui::Combo("Type", &sm.meshData.materialType, matTypes, 3))
                        scene.materialDirty = true;

                    if (sm.meshData.materialType == 0)
                    {
                        bool hasRoughTex = !sm.meshData.roughnessTexturePath.empty();
                        bool hasMetalTex = !sm.meshData.metallicTexturePath.empty();

                        if (hasRoughTex)
                            ImGui::TextDisabled("Roughness  (texture)");
                        else
                        {
                            ImGui::DragFloat("Roughness", &sm.meshData.roughness, 0.01f, 0.0f, 1.0f, "%.2f");
                            if (ImGui::IsItemDeactivatedAfterEdit()) scene.materialDirty = true;
                        }

                        if (hasMetalTex)
                            ImGui::TextDisabled("Metallic   (texture)");
                        else
                        {
                            ImGui::DragFloat("Metallic", &sm.meshData.metallic, 0.01f, 0.0f, 1.0f, "%.2f");
                            if (ImGui::IsItemDeactivatedAfterEdit()) scene.materialDirty = true;
                        }
                    }
                    else if (sm.meshData.materialType == 2)
                    {
                        ImGui::BeginDisabled(isRasterize);
                        ImGui::DragFloat("IOR", &sm.meshData.ior, 0.01f, 1.0f, 3.0f, "%.2f");
                        if (ImGui::IsItemDeactivatedAfterEdit()) scene.materialDirty = true;
                        ImGui::EndDisabled();
                    }

                    if (isRasterize && sm.meshData.materialType != 0)
                        ImGui::TextDisabled("Rendered as Microfacet in rasterizer.\nSwitch to a path tracer to see this material.");

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
                else if (!m_selectedObjectName.empty())
                {
                    // --- Object (shape) selected — Unity-style: Transform + Materials ---
                    const std::string& displayName = m_selectedObjectName.empty()
                        ? group.name : m_selectedObjectName;
                    ImGui::TextUnformatted(displayName.c_str());
                    ImGui::Separator();

                    // Transform (shared across all objects in the group via modelMatrix)
                    ImGui::SeparatorText("Transform");

                    glm::vec3 decompScale, decompTranslation, decompSkew;
                    glm::vec4 decompPerspective;
                    glm::quat decompRotation;
                    glm::decompose(group.modelMatrix, decompScale, decompRotation,
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
                        group.modelMatrix = glm::translate(glm::mat4(1.f), decompTranslation)
                                          * glm::mat4_cast(glm::quat(glm::radians(eulerDeg)))
                                          * glm::scale(glm::mat4(1.f), decompScale);
                    }
                    if (released && renderer.getRenderMode() != RenderMode::Rasterize)
                        scene.geometryDirty = true;

                    if (ImGui::Button("Reset Transform"))
                    {
                        group.modelMatrix = glm::mat4(1.f);
                        if (renderer.getRenderMode() != RenderMode::Rasterize)
                            scene.geometryDirty = true;
                    }

                    // Materials — one collapsible slot per submesh belonging to this object
                    ImGui::SeparatorText("Materials");

                    const char* matTypes[] = { "Microfacet (GGX)", "Mirror", "Dielectric" };
                    bool isRasterize = (renderer.getRenderMode() == RenderMode::Rasterize);
                    auto texName = [](const std::string& path) -> const char* {
                        if (path.empty()) return "none";
                        auto s = path.find_last_of("/\\");
                        return path.c_str() + (s != std::string::npos ? s + 1 : 0);
                    };

                    for (int si = 0; si < static_cast<int>(group.submeshes.size()); ++si)
                    {
                        auto& sm = group.submeshes[si];
                        if (sm.meshData.objectName != m_selectedObjectName) continue;

                        ImGui::PushID(si);
                        if (ImGui::CollapsingHeader(sm.name.c_str()))
                        {
                            ImGui::Text("Vertices:  %u", sm.vertexCount);
                            ImGui::Text("Triangles: %u", sm.indexCount / 3);

                            if (ImGui::Combo("Type", &sm.meshData.materialType, matTypes, 3))
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
                            else if (sm.meshData.materialType == 2)
                            {
                                ImGui::BeginDisabled(isRasterize);
                                ImGui::DragFloat("IOR", &sm.meshData.ior, 0.01f, 1.f, 3.f, "%.2f");
                                if (ImGui::IsItemDeactivatedAfterEdit()) scene.materialDirty = true;
                                ImGui::EndDisabled();
                            }

                            if (isRasterize && sm.meshData.materialType != 0)
                                ImGui::TextDisabled("Rendered as Microfacet in rasterizer.\nSwitch to a path tracer to see this material.");

                            ImGui::Text("Diffuse:   %s", texName(sm.meshData.diffuseTexturePath));
                            ImGui::Text("Normal:    %s", texName(sm.meshData.normalTexturePath));
                            ImGui::Text("Roughness: %s", texName(sm.meshData.roughnessTexturePath));
                            ImGui::Text("Metallic:  %s", texName(sm.meshData.metallicTexturePath));
                        }
                        ImGui::PopID();
                    }
                }
                else
                {
                    // --- Group selected (top-level click) ---
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
                int comboSel = (scene.currentEnvmap < Scene::CustomHDR) ? scene.currentEnvmap : 0;
                if (ImGui::Combo("Background", &comboSel, Scene::envmapNames, Scene::CustomHDR))
                {
                    scene.currentEnvmap = comboSel;
                    scene.customEnvmapPath.clear();
                    if (comboSel > Scene::SolidColor && scene.skybox)
                        scene.skybox->load(Scene::envmapPaths[comboSel]);
                    m_prevEnvmapForRevert = scene.currentEnvmap;
                }
            }
            if (ImGui::Button("Load from file..."))
            {
                std::string hdrPath = openHdrFileDialog(ImGui::GetMainViewport()->PlatformHandleRaw);
                if (!hdrPath.empty())
                {
                    scene.customEnvmapPath = hdrPath;
                    scene.currentEnvmap = Scene::CustomHDR;
                    if (scene.skybox)
                        scene.skybox->load(hdrPath);
                    m_prevEnvmapForRevert = scene.currentEnvmap;
                }
            }
            if (scene.currentEnvmap == Scene::SolidColor)
                ImGui::ColorEdit3("Color", &scene.skyboxColor.x);
            if (scene.currentEnvmap == Scene::CustomHDR && !scene.customEnvmapPath.empty())
                ImGui::TextWrapped("Path: %s", scene.customEnvmapPath.c_str());
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
                        std::string savePath = saveImageFileDialog(ImGui::GetMainViewport()->PlatformHandleRaw);
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

        bool bilinear = renderer.getGPUBilinearFiltering();
        if (ImGui::Checkbox("Bilinear Filtering", &bilinear))
            renderer.setGPUBilinearFiltering(bilinear);

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
