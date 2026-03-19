#include "editor_ui.h"
#include "scene.h"
#include "scene_renderer.h"

#include <cstdio>
#include <string>

#include <vex/core/log.h>
#include <vex/graphics/framebuffer.h>
#include <vex/graphics/graphics_context.h>

#include <imgui.h>

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

bool EditorUI::consumePendingGltfImport(std::string& outPath, std::string& outName)
{
    if (m_pendingGltfImportPath.empty()) return false;
    outPath = std::move(m_pendingGltfImportPath);
    outName = std::move(m_pendingGltfImportName);
    m_pendingGltfImportPath.clear();
    return true;
}

bool EditorUI::consumePendingPrimitive(PrimitiveType& outType)
{
    if (m_pendingPrimitive == PrimitiveType::None) return false;
    outType = m_pendingPrimitive;
    m_pendingPrimitive = PrimitiveType::None;
    return true;
}

bool EditorUI::consumePendingAddVolume()
{
    if (!m_pendingAddVolume) return false;
    m_pendingAddVolume = false;
    return true;
}

bool EditorUI::consumePendingDuplicate()
{
    if (!m_pendingDuplicate) return false;
    m_pendingDuplicate = false;
    return true;
}

bool EditorUI::consumeTransformCommit(TransformCommit& out)
{
    if (!m_transformCommitReady) return false;
    out = m_transformCommit;
    m_transformCommitReady = false;
    return true;
}

bool EditorUI::consumePendingReparent(PendingReparent& out)
{
    if (!m_pendingReparentReady) return false;
    out = m_pendingReparent;
    m_pendingReparentReady = false;
    return true;
}

bool EditorUI::isAncestorOf(const Scene& scene, int potentialAncestor, int node) const
{
    int cur = node;
    while (cur >= 0 && cur < (int)scene.nodes.size())
    {
        if (cur == potentialAncestor) return true;
        cur = scene.nodes[cur].parentIndex;
    }
    return false;
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
    {
        float sps = renderer.getSamplesPerSec();
        if (sps > 0.0f)
            ImGui::Text("Samples/sec: %.1f", sps);
    }

    bool vsync = ctx.getVSync();
    if (ImGui::Checkbox("VSync", &vsync))
        ctx.setVSync(vsync);

    // --- Scene ---
    ImGui::SeparatorText("Scene");

    uint32_t totalVerts   = 0;
    uint32_t totalIndices = 0;
    int      totalSubs    = 0;
    int      totalTex     = 0;

    for (const auto& node : scene.nodes)
    {
        for (const auto& sm : node.submeshes)
        {
            totalVerts   += sm.vertexCount;
            totalIndices += sm.indexCount;
            ++totalSubs;
            if (sm.diffuseTexture)
                ++totalTex;
        }
    }

    ImGui::Text("Nodes:        %d", static_cast<int>(scene.nodes.size()));
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
