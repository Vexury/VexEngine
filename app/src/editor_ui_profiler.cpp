#include "editor_ui.h"

#include <vex/core/profiler.h>
#include <vex/graphics/graphics_context.h>

#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <string>

void EditorUI::renderProfiler(vex::GraphicsContext& ctx)
{
    // Not part of the app's one-time DockBuilder layout (that lives in ui_layer.cpp and
    // is out of scope here), so on a first-ever launch this window would otherwise
    // auto-fit to a near-zero size before any content has been measured. A FirstUseEver
    // size hint only applies when imgui.ini has no saved entry for this window yet, so it
    // never overrides a size the user picked (by resizing or docking it) on a later run.
    ImGui::SetNextWindowSize(ImVec2(420.0f, 480.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Profiler");

    auto&       prof    = vex::Profiler::get();
    const auto& results = prof.results();

    const float gpuMs = prof.frameGpuMs();
    const float cpuMs = prof.frameCpuMs();

    if (!m_profPaused && gpuMs >= 0.0f)
    {
        m_profHistory[m_profHistoryPos] = gpuMs;
        m_profHistoryPos = (m_profHistoryPos + 1) % IM_ARRAYSIZE(m_profHistory);
    }

    ImGui::TextDisabled("%s", ctx.deviceName().c_str());

    if (gpuMs >= 0.0f)
        ImGui::Text("GPU %.2f ms", gpuMs);
    else
        ImGui::TextDisabled("GPU timing unavailable on this device");
    ImGui::SameLine();
    ImGui::Text("  CPU %.2f ms", cpuMs < 0.0f ? 0.0f : cpuMs);
    ImGui::SameLine();
    ImGui::Text("  %.0f FPS", ImGui::GetIO().Framerate);

    ImGui::Checkbox("Pause", &m_profPaused);
    ImGui::SameLine();
    if (ImGui::Button("Reset peaks"))
        m_profAccum.clear();

    float scaleMax = 1.0f;
    for (float v : m_profHistory) scaleMax = std::max(scaleMax, v);
    ImGui::PlotLines("##frametime", m_profHistory, IM_ARRAYSIZE(m_profHistory),
                     m_profHistoryPos, nullptr, 0.0f, scaleMax * 1.1f,
                     ImVec2(-1.0f, 60.0f));

    ImGui::Separator();

    const float frameGpu = (gpuMs > 0.0f) ? gpuMs : 1.0f;

    if (ImGui::BeginTable("passes", 5,
                          ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg))
    {
        ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthStretch, 2.0f);
        ImGui::TableSetupColumn("last");
        ImGui::TableSetupColumn("avg");
        ImGui::TableSetupColumn("max");
        ImGui::TableSetupColumn("%");
        ImGui::TableHeadersRow();

        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Child zones may sum to more than their parent: "
                              "begin timestamps use TOP_OF_PIPE and end "
                              "timestamps use BOTTOM_OF_PIPE, so a zone can "
                              "absorb time from work still draining ahead of it.");

        for (const auto& r : results)
        {
            const float value = (r.gpuMs >= 0.0f) ? r.gpuMs : r.cpuMs;
            if (value < 0.0f) continue;

            auto& acc = m_profAccum[r.name ? r.name : "?"];
            if (!m_profPaused)
            {
                acc.ema  = (acc.ema < 0.0f) ? value : (acc.ema * 0.95f + value * 0.05f);
                acc.peak = std::max(acc.peak, value);
            }

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%*s%s", r.depth * 2, "", r.name ? r.name : "?");
            ImGui::TableNextColumn(); ImGui::Text("%6.2f", value);
            ImGui::TableNextColumn(); ImGui::Text("%6.2f", acc.ema);
            ImGui::TableNextColumn(); ImGui::Text("%6.2f", acc.peak);
            ImGui::TableNextColumn(); ImGui::Text("%5.0f", 100.0f * value / frameGpu);
        }
        ImGui::EndTable();
    }

    ImGui::Separator();

    if (ImGui::CollapsingHeader("One-shot costs"))
    {
        for (const auto& s : prof.oneShots())
            ImGui::Text("  %-20s %8.1f ms", s.name.c_str(), s.ms);
        if (prof.oneShots().empty())
            ImGui::TextDisabled("  none recorded yet");
        if (ImGui::Button("Clear one-shots"))
            prof.clearOneShots();
    }

    if (ImGui::Button("Copy as Markdown"))
    {
        std::string md = "| Pass | last (ms) | avg (ms) | max (ms) | % |\n";
        md += "|---|---|---|---|---|\n";
        char line[256];
        for (const auto& r : results)
        {
            const float value = (r.gpuMs >= 0.0f) ? r.gpuMs : r.cpuMs;
            if (value < 0.0f) continue;
            const auto& acc = m_profAccum[r.name ? r.name : "?"];
            std::snprintf(line, sizeof(line), "| %*s%s | %.2f | %.2f | %.2f | %.0f |\n",
                          r.depth * 2, "", r.name ? r.name : "?",
                          value, acc.ema, acc.peak, 100.0f * value / frameGpu);
            md += line;
        }
        ImGui::SetClipboardText(md.c_str());
    }

    ImGui::End();
}
