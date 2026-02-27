#pragma once

#include <string>

// parentHwnd: platform window handle (HWND on Windows) cast to void*.
// Passing the app window makes the dialog modal to it and appear on the right monitor.
std::string openObjFileDialog(void* parentHwnd = nullptr);
std::string openHdrFileDialog(void* parentHwnd = nullptr);
std::string saveImageFileDialog(void* parentHwnd = nullptr);
