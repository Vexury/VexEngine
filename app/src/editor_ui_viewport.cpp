#include "editor_ui.h"
#include "scene.h"
#include "scene_renderer.h"

#include <imgui.h>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// Gizmo helpers (file-scope)
// ─────────────────────────────────────────────────────────────────────────────

// Project world pos to viewport screen pos (returns {-99999,-99999} if behind camera)
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

        bool imageHovered = ImGui::IsItemHovered();

        bool gizmoActive = false;
        if (m_selection->type == Selection::Mesh &&
            renderer.getRenderMode() == RenderMode::Rasterize)
            gizmoActive = drawGizmo(scene, ImGui::GetWindowDrawList(), cursor, size);

        ImDrawList* dl2 = ImGui::GetWindowDrawList();

        // Local/Global space indicator — top-left corner, pill background
        if (renderer.getRenderMode() == RenderMode::Rasterize)
        {
            const char* label = m_gizmoLocal ? "LOCAL" : "GLOBAL";
            constexpr float padX = 6.f, padY = 3.f, rounding = 4.f;
            constexpr float margin = 8.f;
            ImVec2 textSz = ImGui::CalcTextSize(label);
            ImVec2 tl = { cursor.x + margin, cursor.y + margin };
            ImVec2 br = { tl.x + textSz.x + padX * 2.f, tl.y + textSz.y + padY * 2.f };
            dl2->AddRectFilled(tl, br, IM_COL32(200, 200, 200, 200), rounding);
            dl2->AddText({ tl.x + padX, tl.y + padY }, IM_COL32(20, 20, 20, 255), label);
        }

        // Help button — top-right corner
        bool helpHovered = false;
        {
            constexpr float btnR   = 14.f;
            constexpr float margin = 12.f;
            ImVec2 center = { cursor.x + size.x - btnR - margin,
                              cursor.y + btnR + margin };

            ImGui::SetCursorScreenPos({ center.x - btnR, center.y - btnR });
            ImGui::InvisibleButton("##help_btn", { btnR * 2.f, btnR * 2.f });
            helpHovered = ImGui::IsItemHovered();

            ImU32 bgCol   = helpHovered ? IM_COL32(230, 230, 230, 230) : IM_COL32(180, 180, 180, 160);
            ImU32 textCol = helpHovered ? IM_COL32( 20,  20,  20, 255) : IM_COL32( 30,  30,  30, 230);
            dl2->AddCircleFilled(center, btnR, bgCol);
            dl2->AddText({ center.x - 3.5f, center.y - 6.5f }, textCol, "?");

            if (helpHovered)
            {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted("Controls");
                ImGui::Separator();

                auto row = [](const char* key, const char* desc)
                {
                    ImGui::TextDisabled("%s", key);
                    ImGui::SameLine(130);
                    ImGui::TextUnformatted(desc);
                };

                ImGui::Spacing();
                ImGui::TextDisabled("Navigation");
                row("RMB drag",   "Orbit camera");
                row("MMB drag",   "Pan camera");
                row("Scroll",     "Zoom");

                ImGui::Spacing();
                ImGui::TextDisabled("Selection");
                row("LMB",        "Pick object");
                row("F",          "Focus selected");
                row("Del",        "Delete selected");

                ImGui::Spacing();
                ImGui::TextDisabled("Transform (Only in Rasterization mode)");
                row("W",          "Move");
                row("E",          "Rotate");
                row("R",          "Scale  (center knob = uniform)");
                row("G",          m_gizmoLocal ? "Local / Global  \xc2\xbb LOCAL"
                                               : "Local / Global  \xc2\xbb GLOBAL");

                ImGui::Spacing();
                ImGui::TextDisabled("Scene");
                row("Ctrl+D",     "Duplicate selected");
                row("Ctrl+Z / Y", "Undo / Redo");
                row("F5",         "Reload GPU shader  (RT mode)");

                ImGui::EndTooltip();
            }
        }

        if (!gizmoActive && !helpHovered && imageHovered &&
            ImGui::IsMouseClicked(ImGuiMouseButton_Left))
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
    if (m_selection->index < 0 || m_selection->index >= static_cast<int>(scene.nodes.size()))
        return false;

    auto& node = scene.nodes[m_selection->index];

    // Returns the current world-space matrix of the selected node.
    auto currentWorldMat = [&]() -> glm::mat4 {
        return scene.getWorldMatrix(m_selection->index);
    };

    // Writes a new world-space matrix back to the node's localMatrix.
    auto writeBack = [&](const glm::mat4& newWorld) {
        if (node.parentIndex >= 0)
            node.localMatrix = glm::inverse(scene.getWorldMatrix(node.parentIndex)) * newWorld;
        else
            node.localMatrix = newWorld;
        scene.geometryDirty = true;
    };

    float aspect = vpSize.x / vpSize.y;
    glm::mat4 view = scene.camera.getViewMatrix();
    glm::mat4 proj = scene.camera.getProjectionMatrix(aspect);
    glm::mat4 vp   = proj * view;

    glm::vec3 camPos = scene.camera.getPosition();

    // Compute the visual pivot: AABB center of all submeshes in node-local space
    // (applying each sm.modelMatrix to get node-local coordinates).
    glm::vec3 localCenter(0.f);
    {
        glm::vec3 bmin(FLT_MAX), bmax(-FLT_MAX);
        bool any = false;
        for (const auto& sm : node.submeshes)
        {
            for (const auto& v : sm.meshData.vertices)
            {
                glm::vec3 p = glm::vec3(sm.modelMatrix * glm::vec4(v.position, 1.0f));
                bmin = glm::min(bmin, p); bmax = glm::max(bmax, p); any = true;
            }
        }
        if (any) localCenter = (bmin + bmax) * 0.5f;
    }

    // Current world-space pivot: AABB center of the selection transformed to world space
    glm::vec3 pivot = glm::vec3(currentWorldMat() * glm::vec4(localCenter, 1.0f));

    float camDist  = glm::length(camPos - pivot);
    float gizmoLen = camDist * 0.144f;

    const ImU32 colX  = IM_COL32(220,  50,  50, 255);
    const ImU32 colY  = IM_COL32( 50, 220,  50, 255);
    const ImU32 colZ  = IM_COL32( 50,  50, 220, 255);
    const ImU32 colXH = IM_COL32(255, 150, 150, 255);
    const ImU32 colYH = IM_COL32(150, 255, 150, 255);
    const ImU32 colZH = IM_COL32(150, 150, 255, 255);
    const ImU32 axisColors[3]  = { colX,  colY,  colZ  };
    const ImU32 axisColorsH[3] = { colXH, colYH, colZH };

    // Display axes: local columns of the current world matrix (local mode)
    // or world unit axes (global mode). Used for drawing AND hover detection.
    glm::mat4 curMat = currentWorldMat();
    glm::vec3 displayAxes[3];
    if (m_gizmoLocal)
    {
        displayAxes[0] = glm::normalize(glm::vec3(curMat[0]));
        displayAxes[1] = glm::normalize(glm::vec3(curMat[1]));
        displayAxes[2] = glm::normalize(glm::vec3(curMat[2]));
    }
    else
    {
        displayAxes[0] = {1, 0, 0};
        displayAxes[1] = {0, 1, 0};
        displayAxes[2] = {0, 0, 1};
    }

    // Drag-start axes (frozen when drag begins, used for consistent worldDist projection).
    auto getStartAxis = [&](int a) -> glm::vec3 {
        if (m_gizmoLocal)
            return glm::normalize(glm::vec3(m_gizmoMatStart[a]));
        return displayAxes[a];
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
            tips[a] = gizmoProject(pivot + displayAxes[a] * gizmoLen, vp, vpOrigin, vpSize);

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
            // Center knob for uniform scale — takes priority over axes
            if (m_gizmoMode == 2)
            {
                float dx = mouse.x - origin2D.x, dy = mouse.y - origin2D.y;
                if (sqrtf(dx*dx + dy*dy) < 10.f) hoveredAxis = 3;
            }
        }

        if (!m_gizmoDragging && lmbJustPressed && hoveredAxis >= 0)
        {
            m_gizmoDragging  = true;
            m_gizmoAxis      = hoveredAxis;
            m_gizmoDragStart = mouse;
            m_gizmoMatStart  = currentWorldMat();
            m_gizmoLocalStart = node.localMatrix;
            m_gizmoPivot     = pivot;
        }

        if (m_gizmoDragging && lmbDown)
        {
            int a = m_gizmoAxis;
            if (m_gizmoMode == 2 && a == 3)
            {
                // Uniform scale: drag up = grow, drag down = shrink
                float pixelDist = m_gizmoDragStart.y - mouse.y;
                float sf = glm::max(0.01f, 1.0f + pixelDist / 100.f);
                glm::mat4 newWorld = glm::scale(m_gizmoMatStart, glm::vec3(sf));
                // Adjust translation so localCenter stays at m_gizmoPivot
                newWorld[3] = glm::vec4(m_gizmoPivot - glm::vec3(newWorld * glm::vec4(localCenter, 0.f)), 1.f);
                writeBack(newWorld);
            }
            else
            {
                glm::vec3 startAxis  = getStartAxis(a);
                ImVec2 startOrigin2D = gizmoProject(m_gizmoPivot, vp, vpOrigin, vpSize);
                ImVec2 startTip2D    = gizmoProject(m_gizmoPivot + startAxis * gizmoLen, vp, vpOrigin, vpSize);
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
                        // Translate along startAxis in world space
                        glm::mat4 newWorld = m_gizmoMatStart;
                        newWorld[3] += glm::vec4(startAxis * worldDist, 0.0f);
                        writeBack(newWorld);
                    }
                    else
                    {
                        // Scale along local axis a, keeping AABB pivot fixed
                        float sf = glm::max(0.01f, 1.0f + worldDist / gizmoLen);
                        glm::vec3 sv(1.f); sv[a] = sf;
                        glm::mat4 newWorld = glm::scale(m_gizmoMatStart, sv);
                        // Adjust translation so localCenter stays at m_gizmoPivot
                        newWorld[3] = glm::vec4(m_gizmoPivot - glm::vec3(newWorld * glm::vec4(localCenter, 0.f)), 1.f);
                        writeBack(newWorld);
                    }
                }
            }
        }

        if (m_gizmoDragging && lmbReleased)
        {
            m_transformCommit      = { m_selection->index, m_gizmoLocalStart, node.localMatrix };
            m_transformCommitReady = true;
            m_gizmoDragging        = false;
            m_gizmoAxis            = -1;
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

        // Center knob — uniform scale (drawn on top of axis lines)
        if (m_gizmoMode == 2)
        {
            bool  knobActive = (hoveredAxis == 3);
            ImU32 knobFill   = knobActive ? IM_COL32(255, 255, 255, 255) : IM_COL32(210, 210, 210, 220);
            dl->AddCircleFilled(origin2D, 8.f, knobFill);
            dl->AddCircle(origin2D, 8.f, IM_COL32(0, 0, 0, 140), 0, 1.5f);
        }
    }
    else if (m_gizmoMode == 1) // Rotate
    {
        float ringRadius = camDist * 0.128f;
        const int SEGS = 64;

        if (!m_gizmoDragging)
        {
            float bestDist = 10.f;
            hoveredAxis = -1;
            for (int a = 0; a < 3; ++a)
            {
                const glm::vec3 axis = displayAxes[a];
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
            m_gizmoDragging   = true;
            m_gizmoAxis       = hoveredAxis;
            m_gizmoDragStart  = mouse;
            m_gizmoMatStart   = currentWorldMat();
            m_gizmoLocalStart = node.localMatrix;
            m_gizmoPivot      = pivot;
            m_gizmoRotRefSet  = false;
        }

        if (m_gizmoDragging && lmbDown)
        {
            int a = m_gizmoAxis;
            // Rotation axis: local axis at drag start (local mode) or world axis (global mode)
            const glm::vec3 rotAxis = getStartAxis(a);

            float ndcX = (mouse.x - vpOrigin.x) / vpSize.x * 2.f - 1.f;
            float ndcY = 1.f - (mouse.y - vpOrigin.y) / vpSize.y * 2.f;
            glm::mat4 invVP = glm::inverse(vp);
            glm::vec4 worldFar = invVP * glm::vec4(ndcX, ndcY, 1.f, 1.f);
            glm::vec3 rayDir = glm::normalize(glm::vec3(worldFar) / worldFar.w - camPos);

            float denom = glm::dot(rayDir, rotAxis);
            if (fabsf(denom) > 1e-5f)
            {
                float t = glm::dot(m_gizmoPivot - camPos, rotAxis) / denom;
                if (t > 0.f)
                {
                    glm::vec3 hitPoint = camPos + rayDir * t;
                    glm::vec3 hitVec   = hitPoint - m_gizmoPivot;
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
                            float sinA  = glm::dot(glm::cross(m_gizmoRotRef, hitVec), rotAxis);
                            float angle = atan2f(sinA, cosA);
                            glm::mat4 T  = glm::translate(glm::mat4(1.f),  m_gizmoPivot);
                            glm::mat4 iT = glm::translate(glm::mat4(1.f), -m_gizmoPivot);
                            glm::mat4 R  = glm::rotate(glm::mat4(1.f), angle, rotAxis);
                            writeBack(T * R * iT * m_gizmoMatStart);
                        }
                    }
                }
            }
        }

        if (m_gizmoDragging && lmbReleased)
        {
            m_transformCommit      = { m_selection->index, m_gizmoLocalStart, node.localMatrix };
            m_transformCommitReady = true;
            m_gizmoDragging        = false;
            m_gizmoAxis            = -1;
            m_gizmoRotRefSet       = false;
        }

        for (int a = 0; a < 3; ++a)
        {
            const glm::vec3 axis = displayAxes[a];
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

    // Center dot (hidden in scale mode — the uniform-scale knob takes its place)
    if (m_gizmoMode != 2)
        dl->AddCircleFilled(origin2D, 4.f, IM_COL32(255, 255, 255, 200));

    return (hoveredAxis >= 0) || m_gizmoDragging;
}
