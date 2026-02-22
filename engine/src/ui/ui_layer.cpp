#include <vex/ui/ui_layer.h>
#include <vex/core/window.h>
#include <vex/graphics/graphics_context.h>

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>

#include <GLFW/glfw3.h>

namespace vex
{

bool UILayer::init(Window& window, GraphicsContext& context)
{
    m_context = &context;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
#ifdef VEX_BACKEND_OPENGL
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
#endif

    // Dark theme
    auto& colors = ImGui::GetStyle().Colors;
    colors[ImGuiCol_WindowBg]             = { 0.1f,  0.1f,  0.1f,  1.0f };
    colors[ImGuiCol_Header]               = { 0.2f,  0.2f,  0.2f,  1.0f };
    colors[ImGuiCol_HeaderHovered]        = { 0.3f,  0.3f,  0.3f,  1.0f };
    colors[ImGuiCol_HeaderActive]         = { 0.15f, 0.15f, 0.15f, 1.0f };
    colors[ImGuiCol_Button]               = { 0.2f,  0.2f,  0.2f,  1.0f };
    colors[ImGuiCol_ButtonHovered]        = { 0.3f,  0.3f,  0.3f,  1.0f };
    colors[ImGuiCol_ButtonActive]         = { 0.15f, 0.15f, 0.15f, 1.0f };
    colors[ImGuiCol_FrameBg]              = { 0.2f,  0.2f,  0.2f,  1.0f };
    colors[ImGuiCol_FrameBgHovered]       = { 0.3f,  0.3f,  0.3f,  1.0f };
    colors[ImGuiCol_FrameBgActive]        = { 0.15f, 0.15f, 0.15f, 1.0f };
    colors[ImGuiCol_Tab]                  = { 0.15f, 0.15f, 0.15f, 1.0f };
    colors[ImGuiCol_TabHovered]           = { 0.38f, 0.38f, 0.38f, 1.0f };
    colors[ImGuiCol_TabActive]            = { 0.28f, 0.28f, 0.28f, 1.0f };
    colors[ImGuiCol_TabUnfocused]         = { 0.15f, 0.15f, 0.15f, 1.0f };
    colors[ImGuiCol_TabUnfocusedActive]   = { 0.2f,  0.2f,  0.2f,  1.0f };
    colors[ImGuiCol_TitleBg]              = { 0.15f, 0.15f, 0.15f, 1.0f };
    colors[ImGuiCol_TitleBgActive]        = { 0.15f, 0.15f, 0.15f, 1.0f };
    colors[ImGuiCol_TitleBgCollapsed]     = { 0.15f, 0.15f, 0.15f, 1.0f };

    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    io.FontGlobalScale = 1.0f;

    context.imguiInit(window.getNativeWindow());

    return true;
}

void UILayer::shutdown()
{
    if (m_context)
        m_context->imguiShutdown();

    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    m_context = nullptr;
}

void UILayer::beginFrame()
{
    m_context->imguiNewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Fullscreen dockspace
    ImGuiWindowFlags windowFlags =
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground;

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    ImGui::Begin("DockSpaceWindow", nullptr, windowFlags);
    ImGui::PopStyleVar(3);

    ImGuiID dockSpaceId = ImGui::GetID("VexDockSpace");
    ImGui::DockSpace(dockSpaceId, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);

    if (m_firstFrame)
    {
        m_firstFrame = false;

        ImGui::DockBuilderRemoveNode(dockSpaceId);
        ImGui::DockBuilderAddNode(dockSpaceId, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dockSpaceId, viewport->Size);

        ImGuiID dockMain = dockSpaceId;

        // Carve out the four outer regions first
        ImGuiID dockBottom = ImGui::DockBuilderSplitNode(dockMain,   ImGuiDir_Down,  0.30f, nullptr, &dockMain);
        ImGuiID dockLeft   = ImGui::DockBuilderSplitNode(dockMain,   ImGuiDir_Left,  0.20f, nullptr, &dockMain);
        ImGuiID dockRight  = ImGui::DockBuilderSplitNode(dockMain,   ImGuiDir_Right, 0.35f, nullptr, &dockMain);

        // Right panel: Inspector on top, Settings below
        ImGuiID dockRightBottom;
        ImGui::DockBuilderSplitNode(dockRight, ImGuiDir_Down, 0.60f, &dockRightBottom, &dockRight);

        // Bottom panel: Console on the left, Stats on the right
        ImGuiID dockBottomRight;
        ImGui::DockBuilderSplitNode(dockBottom, ImGuiDir_Right, 0.28f, &dockBottomRight, &dockBottom);

        ImGui::DockBuilderDockWindow("Viewport",   dockMain);
        ImGui::DockBuilderDockWindow("Hierarchy",  dockLeft);
        ImGui::DockBuilderDockWindow("Inspector",  dockRight);
        ImGui::DockBuilderDockWindow("Settings",   dockRightBottom);
        ImGui::DockBuilderDockWindow("Console",    dockBottom);
        ImGui::DockBuilderDockWindow("Stats",      dockBottomRight);

        ImGui::DockBuilderFinish(dockSpaceId);
    }

    ImGui::End();
}

void UILayer::endFrame()
{
    ImGui::Render();
    m_context->imguiRenderDrawData();
}

} // namespace vex
