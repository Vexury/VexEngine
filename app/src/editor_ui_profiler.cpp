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
    ImGui::SetNextWindowSize(ImVec2(660.0f, 480.0f), ImGuiCond_FirstUseEver);
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

    // No fallback denominator: a fabricated 1.0ms denominator is how a near-zero GPU
    // frame time (e.g. CPU-bound render modes) used to produce percentages in the
    // tens of thousands. Percent is only meaningful when the frame's GPU time is a
    // real, measured value.
    const bool haveFrameGpu = gpuMs > 0.0f;

    // Columns are grouped by unit, GPU then CPU, and every aggregate column names
    // the unit it holds. Nothing here ever falls back from one unit to the other.
    if (ImGui::BeginTable("passes", 8,
                          ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg))
    {
        ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthStretch, 2.4f);
        ImGui::TableSetupColumn("GPU");
        ImGui::TableSetupColumn("avg GPU");
        ImGui::TableSetupColumn("max GPU");
        ImGui::TableSetupColumn("CPU");
        ImGui::TableSetupColumn("avg CPU");
        ImGui::TableSetupColumn("max CPU");
        ImGui::TableSetupColumn("% GPU");
        ImGui::TableHeadersRow();

        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("All times are milliseconds. The three GPU columns "
                              "hold GPU time only and the three CPU columns hold "
                              "CPU time only; a dash means that side was never "
                              "measured for this zone, which is not the same as a "
                              "measured zero. Child zones may sum to more than "
                              "their parent: begin timestamps use TOP_OF_PIPE and "
                              "end timestamps use BOTTOM_OF_PIPE, so a zone can "
                              "absorb time from work still draining ahead of it. "
                              "% GPU is a share of GPU frame time only, so a row "
                              "with real CPU cost and no GPU cost correctly reads "
                              "0, see the CPU columns for that cost.");

        // Prints a measured value, or a dimmed dash when the sentinel says the
        // series was never measured. Used for every numeric cell so an absent
        // measurement can never render as 0.00.
        auto cell = [](float v)
        {
            ImGui::TableNextColumn();
            if (v < 0.0f) ImGui::TextDisabled("%7s", "-");
            else          ImGui::Text("%7.2f", v);
        };

        for (const auto& r : results)
        {
            // Skip only when neither timestamp was recorded. A zone with a real,
            // near-zero gpuMs (the "cheap on GPU, expensive to submit" case) must
            // still show its cpuMs, not be discarded by an ">= 0.0f" test that a
            // literal 0.0000 always wins.
            if (r.gpuMs < 0.0f && r.cpuMs < 0.0f) continue;

            auto& acc = m_profAccum[r.name ? r.name : "?"];
            if (!m_profPaused)
            {
                // Each side is folded in only on the frames where it was actually
                // measured, so an unresolved GPU query never lets CPU time leak
                // into the GPU average and vice versa.
                if (r.gpuMs >= 0.0f)
                {
                    acc.gpuEma  = (acc.gpuEma < 0.0f)
                                ? r.gpuMs : (acc.gpuEma * 0.95f + r.gpuMs * 0.05f);
                    acc.gpuPeak = std::max(acc.gpuPeak, r.gpuMs);
                }
                if (r.cpuMs >= 0.0f)
                {
                    acc.cpuEma  = (acc.cpuEma < 0.0f)
                                ? r.cpuMs : (acc.cpuEma * 0.95f + r.cpuMs * 0.05f);
                    acc.cpuPeak = std::max(acc.cpuPeak, r.cpuMs);
                }
            }

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%*s%s", r.depth * 2, "", r.name ? r.name : "?");
            cell(r.gpuMs);
            cell(acc.gpuEma);
            cell(acc.gpuPeak);
            cell(r.cpuMs);
            cell(acc.cpuEma);
            cell(acc.cpuPeak);
            ImGui::TableNextColumn();
            if (haveFrameGpu && r.gpuMs >= 0.0f)
                ImGui::Text("%5.0f", 100.0f * r.gpuMs / gpuMs);
            else
                ImGui::TextDisabled("%5s", "-");
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
        // Headers carry the same unit qualifiers as the table. This export is
        // what gets pasted into the README, so a column named plain "avg" whose
        // contents could be either GPU or CPU milliseconds is exactly the way an
        // unqualified number reaches a published document.
        std::string md = "| Pass | GPU (ms) | avg GPU (ms) | max GPU (ms) "
                         "| CPU (ms) | avg CPU (ms) | max CPU (ms) | % GPU |\n";
        md += "|---|---|---|---|---|---|---|---|\n";

        // Same dash-or-value rule as the table, so the two paths cannot disagree
        // about what an unmeasured series looks like.
        auto cellText = [](float v)
        {
            if (v < 0.0f) return std::string("-");
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%.2f", v);
            return std::string(buf);
        };

        char line[384];
        char pctCell[16];
        for (const auto& r : results)
        {
            if (r.gpuMs < 0.0f && r.cpuMs < 0.0f) continue;

            const auto& acc = m_profAccum[r.name ? r.name : "?"];

            if (haveFrameGpu && r.gpuMs >= 0.0f)
                std::snprintf(pctCell, sizeof(pctCell), "%.0f", 100.0f * r.gpuMs / gpuMs);
            else
                std::snprintf(pctCell, sizeof(pctCell), "-");

            std::snprintf(line, sizeof(line),
                          "| %*s%s | %s | %s | %s | %s | %s | %s | %s |\n",
                          r.depth * 2, "", r.name ? r.name : "?",
                          cellText(r.gpuMs).c_str(),
                          cellText(acc.gpuEma).c_str(),
                          cellText(acc.gpuPeak).c_str(),
                          cellText(r.cpuMs).c_str(),
                          cellText(acc.cpuEma).c_str(),
                          cellText(acc.cpuPeak).c_str(),
                          pctCell);
            md += line;
        }
        ImGui::SetClipboardText(md.c_str());
    }

    ImGui::End();
}
