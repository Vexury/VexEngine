#pragma once
#include <cstdint>
#include <vector>

namespace vex
{

class Denoiser
{
public:
    ~Denoiser();
    bool init();
    bool isReady() const { return m_device != nullptr; }
    // In-place denoise. rgb: float RGB interleaved (3 floats/pixel), linear HDR.
    bool denoise(float* rgb, uint32_t width, uint32_t height);
    // Same as denoise() but also passes albedo and normal feature buffers (float RGB, 3 floats/pixel).
    bool denoiseAux(float* rgb, float* albedo, float* normal, uint32_t width, uint32_t height);

private:
    void* m_device = nullptr; // OIDNDevice (void* avoids header pollution when OIDN absent)
};

} // namespace vex
