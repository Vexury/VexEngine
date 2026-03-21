#include "editor_ui.h"
#include "scene.h"
#include "scene_renderer.h"
#include "file_dialog.h"

#include <imgui.h>

#include <vex/core/log.h>

void EditorUI::drawHierarchyNode(int nodeIdx, Scene& scene)
{
    SceneNode& node = scene.nodes[nodeIdx];
    bool isSelected = (m_selection->type == Selection::Mesh && m_selection->index == nodeIdx);

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow
                             | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (node.childIndices.empty())
        flags |= ImGuiTreeNodeFlags_Leaf;
    if (isSelected)
        flags |= ImGuiTreeNodeFlags_Selected;

    ImGui::PushID(nodeIdx);
    bool open = ImGui::TreeNodeEx(node.name.c_str(), flags);

    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
    {
        m_selection->type       = Selection::Mesh;
        m_selection->index      = nodeIdx;
        m_selection->submeshIdx = -1;
        m_selection->objectName.clear();
    }

    // Drag source
    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
    {
        ImGui::SetDragDropPayload("SCENE_NODE", &nodeIdx, sizeof(int));
        ImGui::Text("Move: %s", node.name.c_str());
        ImGui::EndDragDropSource();
    }

    // Drop target — reparent dragged node onto this node
    if (ImGui::BeginDragDropTarget())
    {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SCENE_NODE"))
        {
            int draggedIdx = *static_cast<const int*>(payload->Data);
            if (draggedIdx != nodeIdx && !isAncestorOf(scene, draggedIdx, nodeIdx))
                m_pendingReparent = { draggedIdx, nodeIdx }, m_pendingReparentReady = true;
        }
        ImGui::EndDragDropTarget();
    }

    if (open)
    {
        for (int childIdx : node.childIndices)
            drawHierarchyNode(childIdx, scene);
        ImGui::TreePop();
    }

    ImGui::PopID();
}

void EditorUI::renderHierarchy(Scene& scene, SceneRenderer& renderer)
{
    ImGui::Begin("Hierarchy");

    // ── Toolbar ───────────────────────────────────────────────────────────────
    if (ImGui::Button("Create..."))
        ImGui::OpenPopup("##create_prim");
    if (ImGui::BeginPopup("##create_prim"))
    {
        if (ImGui::MenuItem("Plane"))    m_pendingPrimitive = PrimitiveType::Plane;
        if (ImGui::MenuItem("Cube"))     m_pendingPrimitive = PrimitiveType::Cube;
        if (ImGui::MenuItem("Sphere"))   m_pendingPrimitive = PrimitiveType::Sphere;
        if (ImGui::MenuItem("Cylinder")) m_pendingPrimitive = PrimitiveType::Cylinder;
        ImGui::EndPopup();
    }

    ImGui::SameLine();
    if (ImGui::Button("Import..."))
        ImGui::OpenPopup("##import_menu");
    if (ImGui::BeginPopup("##import_menu"))
    {
        if (ImGui::MenuItem("OBJ..."))
        {
            ImGui::CloseCurrentPopup();
            vex::Log::info("File dialog opened");
            std::string path = openObjFileDialog();
            vex::Log::info(path.empty() ? "File dialog cancelled" : "File dialog closed: " + path);
            if (!path.empty())
            {
                std::string baseName = path;
                auto slash = baseName.find_last_of("/\\");
                if (slash != std::string::npos) baseName = baseName.substr(slash + 1);
                auto dot = baseName.rfind('.');
                if (dot != std::string::npos)   baseName = baseName.substr(0, dot);
                m_pendingImportPath = path;
                m_pendingImportName = baseName;
            }
        }
        if (ImGui::MenuItem("GLTF..."))
        {
            ImGui::CloseCurrentPopup();
            vex::Log::info("File dialog opened");
            std::string path = openGltfFileDialog();
            vex::Log::info(path.empty() ? "File dialog cancelled" : "File dialog closed: " + path);
            if (!path.empty())
            {
                std::string baseName = path;
                auto slash = baseName.find_last_of("/\\");
                if (slash != std::string::npos) baseName = baseName.substr(slash + 1);
                auto dot = baseName.rfind('.');
                if (dot != std::string::npos)   baseName = baseName.substr(0, dot);
                m_pendingGltfImportPath = path;
                m_pendingGltfImportName = baseName;
            }
        }
        ImGui::EndPopup();
    }

    ImGui::SameLine();
    if (ImGui::Button("Save Image..."))
    {
        std::string path = saveImageFileDialog();
        if (!path.empty())
        {
            if (renderer.saveImage(path))
                vex::Log::info("Saved: " + path);
            else
                vex::Log::error("Failed to save: " + path);
        }
    }

    ImGui::Separator();
    // ─────────────────────────────────────────────────────────────────────────

    ImGui::TextUnformatted("Scene");
    ImGui::Indent();

    // Camera
    {
        bool selected = (m_selection->type == Selection::Camera);
        if (ImGui::Selectable("Camera", selected))
            m_selection->type = Selection::Camera;
    }

    // Light
    {
        bool selected = (m_selection->type == Selection::Light);
        std::string lightLabel = scene.showLight ? "Light" : "Light (disabled)";
        if (ImGui::Selectable(lightLabel.c_str(), selected))
            m_selection->type = Selection::Light;
    }

    // Sun
    {
        bool selected = (m_selection->type == Selection::Sun);
        std::string sunLabel = scene.showSun ? "Sun" : "Sun (disabled)";
        if (ImGui::Selectable(sunLabel.c_str(), selected))
            m_selection->type = Selection::Sun;
    }

    // Skybox
    if (scene.showSkybox)
    {
        bool selected = (m_selection->type == Selection::Skybox);
        if (ImGui::Selectable("Skybox", selected))
            m_selection->type = Selection::Skybox;
    }

    // Volumes
    for (int vi = 0; vi < static_cast<int>(scene.volumes.size()); ++vi)
    {
        ImGui::PushID(vi);
        const auto& vol = scene.volumes[vi];
        bool selected = (m_selection->type == Selection::Volume && m_selection->index == vi);
        std::string label = vol.name + (vol.enabled ? "" : " (disabled)");
        if (ImGui::Selectable(label.c_str(), selected))
        {
            m_selection->type  = Selection::Volume;
            m_selection->index = vi;
        }
        ImGui::PopID();
    }

    if (ImGui::Button("Add Volume"))
        m_pendingAddVolume = true;

    // Scene nodes (recursive tree)
    for (int ni = 0; ni < static_cast<int>(scene.nodes.size()); ++ni)
        if (scene.nodes[ni].parentIndex == -1)
            drawHierarchyNode(ni, scene);

    // Drop-to-root invisible target at bottom of node list
    {
        ImGui::Dummy(ImVec2(ImGui::GetContentRegionAvail().x, 8.f));
        if (ImGui::BeginDragDropTarget())
        {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SCENE_NODE"))
            {
                int draggedIdx = *static_cast<const int*>(payload->Data);
                if (scene.nodes[draggedIdx].parentIndex != -1)
                    m_pendingReparent = { draggedIdx, -1 }, m_pendingReparentReady = true;
            }
            ImGui::EndDragDropTarget();
        }
    }

    ImGui::Unindent();

    ImGui::End();
}
