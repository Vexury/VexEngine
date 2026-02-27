#include "app.h"

#include <vex/core/window.h>
#include <vex/core/log.h>
#include <vex/graphics/graphics_context.h>

#include <imgui.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <objbase.h>
#endif

static constexpr float ORBIT_SENSITIVITY = 0.005f;
static constexpr float PAN_SENSITIVITY   = 0.002f;

bool App::init(const vex::EngineConfig& config)
{
#ifdef _WIN32
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
#endif

    if (!m_engine.init(config))
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
    if (!m_scene.meshGroups.empty())
    {
        glm::vec3 sceneCenter{0.0f};
        for (const auto& g : m_scene.meshGroups)
            sceneCenter += g.center;
        sceneCenter /= static_cast<float>(m_scene.meshGroups.size());

        float sceneRadius = 0.0f;
        for (const auto& g : m_scene.meshGroups)
            sceneRadius = std::max(sceneRadius, glm::length(g.center - sceneCenter) + g.radius);

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

    // DEL key deletion (works from any focused window)
    if (ImGui::IsKeyPressed(ImGuiKey_Delete))
    {
        switch (m_ui.getSelectionType())
        {
            case Selection::Mesh:
            {
                int idx = m_ui.getSelectionIndex();
                if (idx >= 0 && idx < static_cast<int>(m_scene.meshGroups.size()))
                {
                    vex::Log::info("Deleted: " + m_scene.meshGroups[idx].name);
                    m_renderer.waitIdle();
                    m_scene.meshGroups.erase(m_scene.meshGroups.begin() + idx);
                    m_scene.geometryDirty = true;
                }
                m_ui.clearSelection();
                break;
            }
            default:
                break;
        }
    }

    // F12: save screenshot with timestamped filename
    if (ImGui::IsKeyPressed(ImGuiKey_F12))
    {
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm tm = {};
#ifdef _WIN32
        localtime_s(&tm, &now);
#else
        localtime_r(&now, &tm);
#endif
        char buf[64];
        std::snprintf(buf, sizeof(buf), "render_%04d%02d%02d_%02d%02d%02d.png",
                      tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                      tm.tm_hour, tm.tm_min, tm.tm_sec);
        std::string path(buf);

        if (m_renderer.saveImage(path))
            vex::Log::info("Saved screenshot: " + path);
        else
            vex::Log::error("Failed to save screenshot: " + path);
    }

    // F5: reload GPU path tracer compute shader from disk
    if (ImGui::IsKeyPressed(ImGuiKey_F5) &&
        m_renderer.getRenderMode() == RenderMode::GPURaytrace)
    {
        m_renderer.reloadGPUShader();
    }

    // W/E/R: gizmo mode shortcuts (Translate / Rotate / Scale)
    if (m_ui.isViewportHovered())
    {
        if (ImGui::IsKeyPressed(ImGuiKey_W)) m_ui.setGizmoMode(0);
        if (ImGui::IsKeyPressed(ImGuiKey_E)) m_ui.setGizmoMode(1);
        if (ImGui::IsKeyPressed(ImGuiKey_R)) m_ui.setGizmoMode(2);
    }

    // F key: focus camera on selected object
    if (ImGui::IsKeyPressed(ImGuiKey_F))
    {
        switch (m_ui.getSelectionType())
        {
            case Selection::Mesh:
            {
                int idx = m_ui.getSelectionIndex();
                if (idx >= 0 && idx < static_cast<int>(m_scene.meshGroups.size()))
                {
                    auto& g = m_scene.meshGroups[idx];
                    m_scene.camera.getTarget() = g.center;
                    m_scene.camera.getDistance() = g.radius * 2.5f;
                    float needed = m_scene.camera.getDistance() + g.radius * 2.0f;
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

    auto [groupIdx, submeshIdx] = m_renderer.pick(m_scene, pickX, pickY);
    if (groupIdx >= 0)
    {
        // Select at object level (not submesh level) so the hierarchy highlight
        // and the inspector view match what you'd get by clicking in the hierarchy.
        const auto& groups = m_scene.meshGroups;
        std::string objName;
        if (submeshIdx >= 0 && submeshIdx < (int)groups[groupIdx].submeshes.size())
            objName = groups[groupIdx].submeshes[submeshIdx].meshData.objectName;
        m_ui.setSelection(Selection::Mesh, groupIdx, -1);
        m_ui.setSelectedObjectName(objName);
    }
    else
        m_ui.clearSelection();
}

void App::runImport(const std::string& path, const std::string& name)
{
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
}

void App::run()
{
    while (m_engine.isRunning())
    {
        // Handle any deferred import between frames so we can pump the loading overlay.
        std::string importPath, importName;
        if (m_ui.consumePendingImport(importPath, importName))
            runImport(importPath, importName);

        m_engine.beginFrame();
        handleInput();
        m_renderer.setRenderMode(static_cast<RenderMode>(m_ui.getRenderModeIndex()));
        m_renderer.setDebugMode(static_cast<DebugMode>(m_ui.getDebugModeIndex()));
        m_renderer.renderScene(m_scene, m_ui.getSelectedMeshGroup(), m_ui.getSelectedSubmesh(),
                               m_ui.getSelectedObjectName());
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
    m_scene.meshGroups.clear();
    m_scene.skybox.reset();
    m_engine.shutdown();
#ifdef _WIN32
    CoUninitialize();
#endif
}
