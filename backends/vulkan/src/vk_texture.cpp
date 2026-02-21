#include <vex/vulkan/vk_texture.h>
#include <vex/vulkan/vk_context.h>
#include <vex/core/log.h>

#include <imgui_impl_vulkan.h>
#include <stb_image.h>

#include <cstring>

namespace vex
{

// Factories
std::unique_ptr<Texture2D> Texture2D::create(uint32_t width, uint32_t height, uint32_t channels)
{
    return std::make_unique<VKTexture2D>(width, height, channels);
}

std::unique_ptr<Texture2D> Texture2D::createFromFile(const std::string& path)
{
    int w, h, ch;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, 4);
    if (!data)
    {
        Log::error("Failed to load texture: " + path);
        return nullptr;
    }

    auto tex = std::make_unique<VKTexture2D>(static_cast<uint32_t>(w),
                                              static_cast<uint32_t>(h), 4);
    tex->setData(data, static_cast<uint32_t>(w), static_cast<uint32_t>(h), 4);
    stbi_image_free(data);
    return tex;
}

VKTexture2D::VKTexture2D(uint32_t width, uint32_t height, uint32_t channels)
    : m_width(width), m_height(height), m_channels(channels)
{
    createImage(width, height, VK_FORMAT_R8G8B8A8_UNORM);
}

VKTexture2D::~VKTexture2D()
{
    if (m_imguiDescriptorSet != VK_NULL_HANDLE)
    {
        ImGui_ImplVulkan_RemoveTexture(m_imguiDescriptorSet);
    }
    destroyImage();
}

void VKTexture2D::createImage(uint32_t width, uint32_t height, VkFormat format)
{
    auto& ctx = VKContext::get();
    auto device = ctx.getDevice();
    auto allocator = ctx.getAllocator();

    // Create image
    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = format;
    imgInfo.extent = { width, height, 1 };
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateImage(allocator, &imgInfo, &allocInfo, &m_image, &m_allocation, nullptr);

    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = m_image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;

    vkCreateImageView(device, &viewInfo, nullptr, &m_imageView);

    // Create sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    vkCreateSampler(device, &samplerInfo, nullptr, &m_sampler);
}

void VKTexture2D::destroyImage()
{
    auto device = VKContext::get().getDevice();
    auto allocator = VKContext::get().getAllocator();

    if (m_sampler)   vkDestroySampler(device, m_sampler, nullptr);
    if (m_imageView) vkDestroyImageView(device, m_imageView, nullptr);
    if (m_image)     vmaDestroyImage(allocator, m_image, m_allocation);

    m_sampler    = VK_NULL_HANDLE;
    m_imageView  = VK_NULL_HANDLE;
    m_image      = VK_NULL_HANDLE;
    m_allocation = VK_NULL_HANDLE;
}

void VKTexture2D::transitionLayout(VkCommandBuffer cmd, VkImage image,
                                    VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags srcStage, dstStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else
    {
        srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);
}

void VKTexture2D::setData(const void* data, uint32_t width, uint32_t height, uint32_t channels)
{
    auto& ctx = VKContext::get();
    auto allocator = ctx.getAllocator();

    m_width = width;
    m_height = height;
    m_channels = channels;

    VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * 4; // always RGBA

    // Create staging buffer
    VkBufferCreateInfo stagingInfo{};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = imageSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocInfo{};
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    VkBuffer stagingBuffer;
    VmaAllocation stagingAlloc;
    vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo,
                    &stagingBuffer, &stagingAlloc, nullptr);

    void* mapped;
    vmaMapMemory(allocator, stagingAlloc, &mapped);
    std::memcpy(mapped, data, static_cast<size_t>(imageSize));
    vmaUnmapMemory(allocator, stagingAlloc);

    // Transition + copy
    ctx.immediateSubmit([&](VkCommandBuffer cmd)
    {
        transitionLayout(cmd, m_image, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { width, height, 1 };

        vkCmdCopyBufferToImage(cmd, stagingBuffer, m_image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        transitionLayout(cmd, m_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    vmaDestroyBuffer(allocator, stagingBuffer, stagingAlloc);
}

void VKTexture2D::bind(uint32_t /*slot*/)
{
    // In Vulkan, binding is done via descriptor sets, not texture slots
}

void VKTexture2D::unbind()
{
    // No-op
}

uintptr_t VKTexture2D::getNativeHandle() const
{
    return reinterpret_cast<uintptr_t>(m_imguiDescriptorSet);
}

void VKTexture2D::createImGuiDescriptorSet()
{
    if (m_imguiDescriptorSet != VK_NULL_HANDLE)
    {
        ImGui_ImplVulkan_RemoveTexture(m_imguiDescriptorSet);
    }

    m_imguiDescriptorSet = ImGui_ImplVulkan_AddTexture(
        m_sampler, m_imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

} // namespace vex
