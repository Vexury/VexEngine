#pragma once

#include "scene.h"
#include "scene_renderer.h"
#include "editor_ui.h"

#include <vex/core/engine.h>

struct App
{
    bool init(const vex::EngineConfig& config);
    void run();
    void shutdown();

private:
    void handleInput();
    void processPicking();
    void runImport(const std::string& path, const std::string& name);

    vex::Engine   m_engine;
    Scene         m_scene;
    SceneRenderer m_renderer;
    EditorUI      m_ui;

    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;
    bool   m_dragging   = false;
    bool   m_panning    = false;
};
