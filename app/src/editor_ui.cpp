#include "editor_ui.h"
#include "scene.h"
#include "scene_renderer.h"

#include <cstdio>
#include <string>
#include <unordered_set>

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

    const RenderMode mode = renderer.getRenderMode();
    const bool isRT = (mode != RenderMode::Rasterize);

    // --- Performance ---
    const ImGuiIO& io = ImGui::GetIO();
    ImGui::SeparatorText("Performance");

    const char* modeName = [mode]() -> const char* {
        switch (mode)
        {
            case RenderMode::Rasterize:      return "Rasterizer";
            case RenderMode::CPURaytrace:    return "CPU Path Tracer";
            case RenderMode::GPURaytrace:    return "GPU Path Tracer";
            case RenderMode::ComputeRaytrace: return "Compute Path Tracer";
            default:                          return "Unknown";
        }
    }();
    ImGui::Text("Mode:       %s", modeName);
    ImGui::Text("FPS:        %.1f", io.Framerate);
    ImGui::Text("Frame time: %.2f ms", 1000.0f / io.Framerate);

    if (!isRT)
        ImGui::Text("Draw calls: %d", renderer.getDrawCalls());

    bool vsync = ctx.getVSync();
    if (ImGui::Checkbox("VSync", &vsync))
        ctx.setVSync(vsync);

    // --- Path Tracer ---
    if (isRT)
    {
        ImGui::SeparatorText("Path Tracer");

        uint32_t sampleCount = renderer.getRaytraceSampleCount();
        uint32_t maxSamples  = renderer.getMaxSamples();
        if (maxSamples > 0)
            ImGui::Text("Sample:      %u / %u", sampleCount, maxSamples);
        else
            ImGui::Text("Sample:      %u", sampleCount);

        float sps = renderer.getSamplesPerSec();
        if (sps >= 1000.0f)
            ImGui::Text("Samples/sec: %.1f k", sps / 1000.0f);
        else if (sps > 0.0f)
            ImGui::Text("Samples/sec: %.1f", sps);

        size_t lightTris = renderer.getLightTriangleCount();
        if (lightTris > 0)
        {
            ImGui::Text("Light tris:  %zu", lightTris);
            ImGui::Text("Light area:  %.2f m\xc2\xb2", renderer.getTotalLightArea());
        }
    }

    // --- Scene ---
    ImGui::SeparatorText("Scene");

    // Recompute per-submesh stats only when node count changes
    {
        int nodeCount = static_cast<int>(scene.nodes.size());

        if (nodeCount != m_sceneStats.cachedNodeCount)
        {
            m_sceneStats = {};
            m_sceneStats.cachedNodeCount = nodeCount;

            std::unordered_set<vex::Texture2D*> uniqueTextures;
            for (const auto& node : scene.nodes)
            {
                for (const auto& sm : node.submeshes)
                {
                    m_sceneStats.totalVerts   += sm.vertexCount;
                    m_sceneStats.totalIndices += sm.indexCount;
                    ++m_sceneStats.totalSubs;
                    if (sm.diffuseTexture)   uniqueTextures.insert(sm.diffuseTexture.get());
                    if (sm.normalTexture)    uniqueTextures.insert(sm.normalTexture.get());
                    if (sm.roughnessTexture) uniqueTextures.insert(sm.roughnessTexture.get());
                    if (sm.metallicTexture)  uniqueTextures.insert(sm.metallicTexture.get());
                    if (sm.emissiveTexture)  uniqueTextures.insert(sm.emissiveTexture.get());
                    if (sm.aoTexture)        uniqueTextures.insert(sm.aoTexture.get());

                    bool hasEmissive = sm.meshData.emissiveStrength > 0.0f &&
                                       (sm.emissiveTexture != nullptr ||
                                        sm.meshData.emissiveColor != glm::vec3(0.0f));
                    if (hasEmissive) ++m_sceneStats.emissiveMeshCount;
                }
            }
            m_sceneStats.uniqueTextureCount = static_cast<int>(uniqueTextures.size());
        }
    }

    ImGui::Text("Nodes:      %d", static_cast<int>(scene.nodes.size()));
    ImGui::Text("Submeshes:  %d", m_sceneStats.totalSubs);
    ImGui::Text("Triangles:  %s", [this]() {
        static char buf[32];
        uint32_t t = m_sceneStats.totalIndices / 3;
        if (t >= 1000000)
            snprintf(buf, sizeof(buf), "%.2f M", t / 1000000.0f);
        else if (t >= 1000)
            snprintf(buf, sizeof(buf), "%.1f k", t / 1000.0f);
        else
            snprintf(buf, sizeof(buf), "%u", t);
        return buf;
    }());
    ImGui::Text("Vertices:   %u", m_sceneStats.totalVerts);
    ImGui::Text("Textures:   %d", m_sceneStats.uniqueTextureCount);
    ImGui::Text("Lights");
    ImGui::Text("  Directional: %d", scene.showSun ? 1 : 0);
    ImGui::Text("  Point:       %d", scene.showLight ? 1 : 0);
    ImGui::Text("  Emissive:    %d", m_sceneStats.emissiveMeshCount);

    if (!scene.volumes.empty())
    {
        int activeVols = 0;
        for (const auto& v : scene.volumes)
            if (v.enabled) ++activeVols;
        ImGui::Text("Volumes:    %d / %d", activeVols, static_cast<int>(scene.volumes.size()));
    }

    if (!isRT)
        ImGui::Text("Shadows:    %s", scene.showSun ? "On" : "Off");

    // --- BVH ---
    size_t bvhMem = renderer.getBVHMemoryBytes();
    if (bvhMem > 0)
    {
        ImGui::SeparatorText("BVH");
        ImGui::Text("Nodes:    %u", renderer.getBVHNodeCount());
        float sahCost = renderer.getBVHSAHCost();
        if (sahCost > 0.0f)
            ImGui::Text("SAH cost: %.1f", sahCost);
        ImGui::Text("Memory:   %.1f KB", static_cast<float>(bvhMem) / 1024.0f);
    }

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

        // Per-category breakdown (Vulkan only — tracked via VKMemoryTracker)
        float tracked = mem.texturesMB + mem.geometryMB + mem.framebuffersMB + mem.rayTracingMB;
        if (tracked > 0.0f)
        {
            if (mem.texturesMB     > 0.0f) ImGui::Text("  Textures:     %.1f MB", mem.texturesMB);
            if (mem.geometryMB     > 0.0f) ImGui::Text("  Geometry:     %.1f MB", mem.geometryMB);
            if (mem.framebuffersMB > 0.0f) ImGui::Text("  Framebuffers: %.1f MB", mem.framebuffersMB);
            if (mem.rayTracingMB   > 0.0f) ImGui::Text("  Ray Tracing:  %.1f MB", mem.rayTracingMB);
            float other = mem.usedMB - tracked;
            if (other > 0.1f)              ImGui::Text("  Other:        %.1f MB", other);
        }
    }

    // --- Backend ---
    ImGui::SeparatorText("Backend");
    ImGui::TextUnformatted(std::string(ctx.backendName()).c_str());

    ImGui::End();
}
