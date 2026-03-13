#include "app.h"

#include <vex/core/window.h>
#include <vex/core/log.h>
#include <vex/graphics/graphics_context.h>
#include <vex/scene/primitives.h>
#include <vex/raytracing/bvh.h>

#include <imgui.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>

#include <nfd.h>

static constexpr float ORBIT_SENSITIVITY = 0.005f;
static constexpr float PAN_SENSITIVITY   = 0.002f;

// ── Primitive helpers ─────────────────────────────────────────────────────────

static vex::MeshData generatePrimitive(EditorUI::PrimitiveType type)
{
    switch (type)
    {
        case EditorUI::PrimitiveType::Plane:    return vex::Primitives::makePlane();
        case EditorUI::PrimitiveType::Cube:     return vex::Primitives::makeCube();
        case EditorUI::PrimitiveType::Sphere:   return vex::Primitives::makeUVSphere();
        case EditorUI::PrimitiveType::Cylinder: return vex::Primitives::makeCylinder();
        default: return {};
    }
}

static const char* primitiveTypeName(EditorUI::PrimitiveType type)
{
    switch (type)
    {
        case EditorUI::PrimitiveType::Plane:    return "Plane";
        case EditorUI::PrimitiveType::Cube:     return "Cube";
        case EditorUI::PrimitiveType::Sphere:   return "Sphere";
        case EditorUI::PrimitiveType::Cylinder: return "Cylinder";
        default: return "Primitive";
    }
}

bool App::init(const vex::EngineConfig& config)
{
    NFD_Init();

    if (!m_engine.init(config, [] { return vex::GraphicsContext::create(); }))
        return false;

    if (config.headless)
        return true;

    if (!m_scene.importOBJ("VexAssetsCC0/Scenes/ChessSet/ChessSet.obj", "Chess Set"))
        return false;

    m_scene.skybox = vex::Skybox::create();

    if (!m_renderer.init(m_scene))
        return false;

    m_scene.camera.fov = 45.0f;

    // Focus camera on the loaded scene
    if (!m_scene.nodes.empty())
    {
        glm::vec3 sceneCenter{0.0f};
        for (const auto& n : m_scene.nodes)
            sceneCenter += n.center;
        sceneCenter /= static_cast<float>(m_scene.nodes.size());

        float sceneRadius = 0.0f;
        for (const auto& n : m_scene.nodes)
            sceneRadius = std::max(sceneRadius, glm::length(n.center - sceneCenter) + n.radius);

        m_scene.camera.setOrbit(sceneCenter, sceneRadius * 2.5f, 0.0f, 0.15f);
        m_scene.camera.farPlane = std::max(100.0f, sceneRadius * 4.5f);
    }
    else
    {
        m_scene.camera.setOrbit(glm::vec3(0.0f, 1.0f, 0.0f), 4.5f, 0.0f, 0.15f);
    }

    m_engine.getWindow().setScrollCallback([this](double yoffset)
    {
        if (m_ui.isViewportHovered())
            m_scene.camera.zoom(static_cast<float>(yoffset));
    });

    return true;
}

NodeSave App::saveNode(int idx) const
{
    const auto& node = m_scene.nodes[idx];
    NodeSave save;
    save.name         = node.name;
    save.center       = node.center;
    save.radius       = node.radius;
    save.localMatrix  = node.localMatrix;
    save.parentIndex  = node.parentIndex;
    save.childIndices = node.childIndices;
    for (const auto& sm : node.submeshes)
    {
        SubmeshSave ss;
        ss.name        = sm.name;
        ss.meshData    = sm.meshData;
        ss.modelMatrix = sm.modelMatrix;
        save.submeshes.push_back(std::move(ss));
    }
    return save;
}

void App::duplicateSelected()
{
    int idx = m_ui.getSelectedMeshGroup();
    if (idx < 0 || idx >= static_cast<int>(m_scene.nodes.size())) return;

    const auto& node = m_scene.nodes[idx];

    NodeSave save = saveNode(idx);
    save.name += " (Copy)";
    save.localMatrix = glm::translate(save.localMatrix, {0.3f, 0.f, 0.f});
    save.childIndices.clear();  // copy is childless

    int newIdx = static_cast<int>(m_scene.nodes.size());
    auto cmd = std::make_unique<CmdAddNode>(std::move(save), newIdx, node.parentIndex, -1);
    m_cmdStack.execute(std::move(cmd), m_scene, m_renderer, m_ui);
}

void App::handleInput()
{
    auto* win = m_engine.getWindow().getNativeWindow();

    double mx, my;
    glfwGetCursorPos(win, &mx, &my);

    if (m_ui.isViewportHovered() && glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        if (!m_dragging)
        {
            m_dragging = true;
            m_lastMouseX = mx;
            m_lastMouseY = my;
        }

        float dx = static_cast<float>(mx - m_lastMouseX) * ORBIT_SENSITIVITY;
        float dy = static_cast<float>(my - m_lastMouseY) * ORBIT_SENSITIVITY;
        m_scene.camera.rotate(-dx, dy);
        m_lastMouseX = mx;
        m_lastMouseY = my;
    }
    else
    {
        m_dragging = false;
    }

    // Middle mouse panning
    if (m_ui.isViewportHovered() && glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS)
    {
        if (!m_panning)
        {
            m_panning = true;
            m_lastMouseX = mx;
            m_lastMouseY = my;
        }

        float dx = static_cast<float>(mx - m_lastMouseX);
        float dy = static_cast<float>(my - m_lastMouseY);

        glm::mat4 view = m_scene.camera.getViewMatrix();
        glm::vec3 right = glm::vec3(view[0][0], view[1][0], view[2][0]);
        glm::vec3 up    = glm::vec3(view[0][1], view[1][1], view[2][1]);

        float panSpeed = m_scene.camera.getDistance() * PAN_SENSITIVITY;
        m_scene.camera.getTarget() -= right * dx * panSpeed;
        m_scene.camera.getTarget() += up    * dy * panSpeed;

        m_lastMouseX = mx;
        m_lastMouseY = my;
    }
    else
    {
        m_panning = false;
    }

    // Undo / Redo
    {
        bool ctrl = ImGui::GetIO().KeyCtrl;
        if (ctrl && ImGui::IsKeyPressed(ImGuiKey_Z))
            m_cmdStack.undo(m_scene, m_renderer, m_ui);
        if (ctrl && ImGui::IsKeyPressed(ImGuiKey_Y))
            m_cmdStack.redo(m_scene, m_renderer, m_ui);
        if (ctrl && ImGui::IsKeyPressed(ImGuiKey_D))
            duplicateSelected();
    }

    // Gizmo transform commit -> push undo command
    {
        EditorUI::TransformCommit tc;
        if (m_ui.consumeTransformCommit(tc))
        {
            // Only push if the transform actually changed
            bool changed = false;
            for (int i = 0; i < 4 && !changed; ++i)
                for (int j = 0; j < 4 && !changed; ++j)
                    changed = (tc.before[i][j] != tc.after[i][j]);
            if (changed)
            {
                auto cmd = std::make_unique<CmdSetTransform>(tc.nodeIdx, tc.before, tc.after);
                m_cmdStack.pushUndoOnly(std::move(cmd));
            }
        }
    }

    // DEL key deletion — routed through command stack for undo support
    if (ImGui::IsKeyPressed(ImGuiKey_Delete))
    {
        switch (m_ui.getSelectionType())
        {
            case Selection::Mesh:
            {
                int idx = m_ui.getSelectionIndex();
                if (idx >= 0 && idx < static_cast<int>(m_scene.nodes.size()))
                {
                    auto cmd = std::make_unique<CmdDeleteNode>(m_scene, idx);
                    m_cmdStack.execute(std::move(cmd), m_scene, m_renderer, m_ui);
                }
                else
                {
                    m_ui.clearSelection();
                }
                break;
            }
            case Selection::Volume:
            {
                int idx = m_ui.getSelectionIndex();
                if (idx >= 0 && idx < static_cast<int>(m_scene.volumes.size()))
                {
                    auto cmd = std::make_unique<CmdDeleteVolume>(m_scene.volumes[idx], idx);
                    m_cmdStack.execute(std::move(cmd), m_scene, m_renderer, m_ui);
                }
                else
                {
                    m_ui.clearSelection();
                }
                break;
            }
            default:
                break;
        }
    }

    // F5: reload GPU path tracer compute shader from disk
    if (ImGui::IsKeyPressed(ImGuiKey_F5) &&
        m_renderer.getRenderMode() == RenderMode::GPURaytrace)
    {
        m_renderer.reloadGPUShader();
    }

    // W/E/R: gizmo mode shortcuts (Translate / Rotate / Scale)
    // G: toggle local/global transform space
    if (m_ui.isViewportHovered())
    {
        if (ImGui::IsKeyPressed(ImGuiKey_W)) m_ui.setGizmoMode(0);
        if (ImGui::IsKeyPressed(ImGuiKey_E)) m_ui.setGizmoMode(1);
        if (ImGui::IsKeyPressed(ImGuiKey_R)) m_ui.setGizmoMode(2);
        if (ImGui::IsKeyPressed(ImGuiKey_G)) m_ui.toggleGizmoLocal();
    }

    // F key: focus camera on selected object
    if (ImGui::IsKeyPressed(ImGuiKey_F))
    {
        switch (m_ui.getSelectionType())
        {
            case Selection::Mesh:
            {
                int idx = m_ui.getSelectionIndex();
                if (idx >= 0 && idx < static_cast<int>(m_scene.nodes.size()))
                {
                    const auto& node = m_scene.nodes[idx];
                    // World-space center of the node
                    glm::vec3 center = glm::vec3(m_scene.getWorldMatrix(idx) * glm::vec4(node.center, 1.0f));
                    float radius = node.radius;

                    m_scene.camera.getTarget() = center;
                    m_scene.camera.getDistance() = radius * 2.5f;
                    float needed = m_scene.camera.getDistance() + radius * 2.0f;
                    m_scene.camera.farPlane = std::max(100.0f, needed);
                }
                break;
            }
            case Selection::Light:
                m_scene.camera.getTarget() = m_scene.lightPos;
                break;
            default:
                break;
        }
    }
}

void App::processPicking()
{
    int pickX, pickY;
    if (!m_ui.consumePickRequest(pickX, pickY))
        return;

    auto [nodeIdx, submeshIdx] = m_renderer.pick(m_scene, pickX, pickY);
    if (nodeIdx >= 0)
        m_ui.setSelection(Selection::Mesh, nodeIdx, -1);
    else
        m_ui.clearSelection();
}

void App::runImport(const std::string& path, const std::string& name)
{
    auto t_import_total = std::chrono::steady_clock::now();

    // Pump a single loading frame: update the overlay and present.
    auto pumpFrame = [&](const std::string& stage, float progress)
    {
        m_ui.setLoadingState(stage, progress);
        m_engine.beginFrame();
        m_ui.renderLoadingOverlay();
        m_engine.endFrame();
    };

    pumpFrame("Parsing OBJ...", 0.05f);

    if (!m_scene.importOBJ(path, name, pumpFrame))
    {
        vex::Log::error("Failed to load: " + path);
        m_ui.clearLoadingState();
        return;
    }
    vex::Log::info("Imported: " + name);

    // Build BLAS/TLAS/BVH with progress frames shown before each stage.
    m_renderer.buildGeometry(m_scene, pumpFrame);

    m_ui.clearLoadingState();

    float t_import_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - t_import_total).count();
    char buf[128];
    std::snprintf(buf, sizeof(buf),
        "Import complete: %.1f s total", t_import_ms / 1000.0f);
    vex::Log::info(buf);
}

void App::run()
{
    while (m_engine.isRunning())
    {
        // Handle deferred primitive creation
        {
            EditorUI::PrimitiveType primType;
            if (m_ui.consumePendingPrimitive(primType))
            {
                vex::MeshData md = generatePrimitive(primType);
                const char*   nm = primitiveTypeName(primType);

                NodeSave save;
                save.name        = nm;
                save.center      = {0.f, 0.f, 0.f};
                save.radius      = 1.0f;
                save.localMatrix = glm::mat4(1.f);
                save.parentIndex = -1;
                SubmeshSave ss;
                ss.name     = nm;
                ss.meshData = std::move(md);
                save.submeshes.push_back(std::move(ss));

                int newIdx = static_cast<int>(m_scene.nodes.size());
                auto cmd = std::make_unique<CmdAddNode>(std::move(save), newIdx);
                m_cmdStack.execute(std::move(cmd), m_scene, m_renderer, m_ui);
            }
        }

        // Handle deferred volume add
        if (m_ui.consumePendingAddVolume())
        {
            SceneVolume v;
            v.name = "Volume";
            int newIdx = static_cast<int>(m_scene.volumes.size());
            auto cmd = std::make_unique<CmdAddVolume>(v, newIdx);
            m_cmdStack.execute(std::move(cmd), m_scene, m_renderer, m_ui);
        }

        // Handle deferred duplicate
        if (m_ui.consumePendingDuplicate())
            duplicateSelected();

        // Handle deferred reparent (from hierarchy drag-and-drop)
        {
            EditorUI::PendingReparent pr;
            if (m_ui.consumePendingReparent(pr))
            {
                int nodeIdx = pr.nodeIdx;
                if (nodeIdx >= 0 && nodeIdx < (int)m_scene.nodes.size())
                {
                    auto& node = m_scene.nodes[nodeIdx];
                    int oldParent = node.parentIndex;

                    // Find old sibling position
                    int oldSibPos = 0;
                    if (oldParent >= 0 && oldParent < (int)m_scene.nodes.size())
                    {
                        const auto& pc = m_scene.nodes[oldParent].childIndices;
                        for (int i = 0; i < (int)pc.size(); ++i)
                            if (pc[i] == nodeIdx) { oldSibPos = i; break; }
                    }

                    glm::mat4 oldLocal = node.localMatrix;
                    glm::mat4 oldWorld = m_scene.getWorldMatrix(nodeIdx);

                    // Compute new local matrix that preserves world position
                    glm::mat4 newLocal;
                    if (pr.newParentIdx >= 0 && pr.newParentIdx < (int)m_scene.nodes.size())
                        newLocal = glm::inverse(m_scene.getWorldMatrix(pr.newParentIdx)) * oldWorld;
                    else
                        newLocal = oldWorld;

                    auto cmd = std::make_unique<CmdReparent>(
                        nodeIdx, oldParent, pr.newParentIdx, oldSibPos, oldLocal, newLocal);
                    m_cmdStack.execute(std::move(cmd), m_scene, m_renderer, m_ui);
                }
            }
        }

        // Handle any deferred import between frames so we can pump the loading overlay.
        std::string importPath, importName;
        if (m_ui.consumePendingImport(importPath, importName))
        {
            int prevCount = static_cast<int>(m_scene.nodes.size());
            runImport(importPath, importName);
            if (static_cast<int>(m_scene.nodes.size()) > prevCount)
            {
                // Root was inserted at prevCount; CmdImportUndo captures the full
                // subtree (root + any objectName children) for correct undo/redo.
                m_cmdStack.pushUndoOnly(
                    std::make_unique<CmdImportUndo>(CmdDeleteNode(m_scene, prevCount)));
                m_ui.setSelection(Selection::Mesh, prevCount);
            }
        }

        m_engine.beginFrame();
        handleInput();
        m_renderer.setRenderMode(static_cast<RenderMode>(m_ui.getRenderModeIndex()));
        m_renderer.setDebugMode(static_cast<DebugMode>(m_ui.getDebugModeIndex()));
        m_renderer.renderScene(m_scene, m_ui.getSelectedMeshGroup(), m_ui.getSelectedSubmesh());
        m_ui.renderViewport(m_renderer, m_scene);
        processPicking();
        m_ui.renderHierarchy(m_scene, m_renderer);
        m_ui.renderInspector(m_scene, m_renderer);
        m_ui.renderSettings(m_renderer);
        m_ui.renderConsole();
        m_ui.renderStats(m_renderer, m_scene, m_engine.getGraphicsContext());
        m_engine.endFrame();
    }
}

void App::shutdown()
{
    m_engine.getGraphicsContext().waitIdle();
    m_renderer.shutdown();
    m_scene.nodes.clear();
    m_scene.skybox.reset();
    m_engine.shutdown();
    NFD_Quit();
}
