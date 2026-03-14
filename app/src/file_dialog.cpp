#include "file_dialog.h"

#include <nfd.h>

std::string openObjFileDialog()
{
    nfdu8char_t* outPath = nullptr;
    nfdu8filteritem_t filter = { "OBJ Files", "obj" };
    nfdresult_t result = NFD_OpenDialogU8(&outPath, &filter, 1, nullptr);
    if (result == NFD_OKAY)
    {
        std::string path(outPath);
        NFD_FreePathU8(outPath);
        return path;
    }
    return {};
}

std::string openGltfFileDialog()
{
    nfdu8char_t* outPath = nullptr;
    nfdu8filteritem_t filter = { "GLTF Files", "gltf" };
    nfdresult_t result = NFD_OpenDialogU8(&outPath, &filter, 1, nullptr);
    if (result == NFD_OKAY)
    {
        std::string path(outPath);
        NFD_FreePathU8(outPath);
        return path;
    }
    return {};
}

std::string openHdrFileDialog()
{
    nfdu8char_t* outPath = nullptr;
    nfdu8filteritem_t filter = { "Image Files", "hdr,jpg,png" };
    nfdresult_t result = NFD_OpenDialogU8(&outPath, &filter, 1, nullptr);
    if (result == NFD_OKAY)
    {
        std::string path(outPath);
        NFD_FreePathU8(outPath);
        return path;
    }
    return {};
}

std::string saveImageFileDialog()
{
    nfdu8char_t* outPath = nullptr;
    nfdu8filteritem_t filter = { "PNG Image", "png" };
    nfdresult_t result = NFD_SaveDialogU8(&outPath, &filter, 1, nullptr, nullptr);
    if (result == NFD_OKAY)
    {
        std::string path(outPath);
        NFD_FreePathU8(outPath);
        return path;
    }
    return {};
}
