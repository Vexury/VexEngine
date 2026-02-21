#include "file_dialog.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shobjidl.h>   // IFileOpenDialog / IFileSaveDialog
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "ole32.lib")

// Helper: open IFileOpenDialog with a given filter, returns selected path or "".
static std::string showOpenDialog(const wchar_t* filterName, const wchar_t* filterSpec)
{
    std::string result;

    IFileOpenDialog* pfd = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr,
                                   CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pfd));
    if (FAILED(hr)) return result;

    COMDLG_FILTERSPEC filter{ filterName, filterSpec };
    pfd->SetFileTypes(1, &filter);
    pfd->SetFileTypeIndex(1);

    DWORD flags = 0;
    pfd->GetOptions(&flags);
    pfd->SetOptions(flags | FOS_FILEMUSTEXIST | FOS_PATHMUSTEXIST | FOS_NOCHANGEDIR);

    hr = pfd->Show(nullptr);
    if (SUCCEEDED(hr))
    {
        IShellItem* psi = nullptr;
        if (SUCCEEDED(pfd->GetResult(&psi)))
        {
            wchar_t* pszPath = nullptr;
            if (SUCCEEDED(psi->GetDisplayName(SIGDN_FILESYSPATH, &pszPath)))
            {
                // Convert wide string to narrow
                int len = WideCharToMultiByte(CP_UTF8, 0, pszPath, -1,
                                              nullptr, 0, nullptr, nullptr);
                if (len > 0)
                {
                    result.resize(len - 1);
                    WideCharToMultiByte(CP_UTF8, 0, pszPath, -1,
                                        result.data(), len, nullptr, nullptr);
                }
                CoTaskMemFree(pszPath);
            }
            psi->Release();
        }
    }
    pfd->Release();
    return result;
}

static std::string showSaveDialog(const wchar_t* filterName,
                                   const wchar_t* filterSpec,
                                   const wchar_t* defaultExt)
{
    std::string result;

    IFileSaveDialog* pfd = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_FileSaveDialog, nullptr,
                                   CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pfd));
    if (FAILED(hr)) return result;

    COMDLG_FILTERSPEC filter{ filterName, filterSpec };
    pfd->SetFileTypes(1, &filter);
    pfd->SetFileTypeIndex(1);
    pfd->SetDefaultExtension(defaultExt);

    DWORD flags = 0;
    pfd->GetOptions(&flags);
    pfd->SetOptions(flags | FOS_OVERWRITEPROMPT | FOS_NOCHANGEDIR);

    hr = pfd->Show(nullptr);
    if (SUCCEEDED(hr))
    {
        IShellItem* psi = nullptr;
        if (SUCCEEDED(pfd->GetResult(&psi)))
        {
            wchar_t* pszPath = nullptr;
            if (SUCCEEDED(psi->GetDisplayName(SIGDN_FILESYSPATH, &pszPath)))
            {
                int len = WideCharToMultiByte(CP_UTF8, 0, pszPath, -1,
                                              nullptr, 0, nullptr, nullptr);
                if (len > 0)
                {
                    result.resize(len - 1);
                    WideCharToMultiByte(CP_UTF8, 0, pszPath, -1,
                                        result.data(), len, nullptr, nullptr);
                }
                CoTaskMemFree(pszPath);
            }
            psi->Release();
        }
    }
    pfd->Release();
    return result;
}
#endif

std::string openObjFileDialog()
{
#ifdef _WIN32
    return showOpenDialog(L"OBJ Files (*.obj)", L"*.obj");
#endif
    return {};
}

std::string openHdrFileDialog()
{
#ifdef _WIN32
    return showOpenDialog(L"HDR / Image Files", L"*.hdr;*.jpg;*.png");
#endif
    return {};
}

std::string saveImageFileDialog()
{
#ifdef _WIN32
    return showSaveDialog(L"PNG Image (*.png)", L"*.png", L"png");
#endif
    return {};
}
