#include "denoiser.h"

#ifdef VEX_HAS_OIDN
#include <OpenImageDenoise/oidn.hpp>
#endif

namespace vex
{

Denoiser::~Denoiser()
{
#ifdef VEX_HAS_OIDN
    if (m_device) oidnReleaseDevice(static_cast<OIDNDevice>(m_device));
#endif
}

bool Denoiser::init()
{
#ifdef VEX_HAS_OIDN
    OIDNDevice dev = oidnNewDevice(OIDN_DEVICE_TYPE_CPU);
    oidnCommitDevice(dev);
    if (oidnGetDeviceError(dev, nullptr) != OIDN_ERROR_NONE)
    {
        oidnReleaseDevice(dev);
        return false;
    }
    m_device = dev;
    return true;
#else
    return false;
#endif
}

bool Denoiser::denoise(float* rgb, uint32_t width, uint32_t height)
{
#ifdef VEX_HAS_OIDN
    if (!m_device) return false;
    OIDNDevice dev = static_cast<OIDNDevice>(m_device);

    OIDNFilter filter = oidnNewFilter(dev, "RT");
    oidnSetSharedFilterImage(filter, "color",  rgb, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetSharedFilterImage(filter, "output", rgb, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetFilterBool(filter, "hdr", true);
    oidnCommitFilter(filter);
    oidnExecuteFilter(filter);
    oidnReleaseFilter(filter);

    const char* err = nullptr;
    if (oidnGetDeviceError(dev, &err) != OIDN_ERROR_NONE)
        return false;
    return true;
#else
    (void)rgb; (void)width; (void)height;
    return false;
#endif
}

bool Denoiser::denoiseAux(float* rgb, float* albedo, float* normal, uint32_t width, uint32_t height)
{
#ifdef VEX_HAS_OIDN
    if (!m_device) return false;
    OIDNDevice dev = static_cast<OIDNDevice>(m_device);

    OIDNFilter filter = oidnNewFilter(dev, "RT");
    oidnSetSharedFilterImage(filter, "color",  rgb,    OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetSharedFilterImage(filter, "albedo", albedo, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetSharedFilterImage(filter, "normal", normal, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetSharedFilterImage(filter, "output", rgb,    OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetFilterBool(filter, "hdr", true);
    oidnCommitFilter(filter);
    oidnExecuteFilter(filter);
    oidnReleaseFilter(filter);

    const char* err = nullptr;
    if (oidnGetDeviceError(dev, &err) != OIDN_ERROR_NONE)
        return false;
    return true;
#else
    (void)rgb; (void)albedo; (void)normal; (void)width; (void)height;
    return false;
#endif
}

} // namespace vex
