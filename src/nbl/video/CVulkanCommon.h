#ifndef __NBL_VIDEO_C_VULKAN_COMMON_H_INCLUDED__

#include <volk.h>

namespace nbl::video
{
static inline asset::E_FORMAT getFormatFromVkFormat(VkFormat in)
{
    switch(in)
    {
        case VK_FORMAT_D16_UNORM: return asset::E_FORMAT::EF_D16_UNORM;
        case VK_FORMAT_X8_D24_UNORM_PACK32: return asset::E_FORMAT::EF_X8_D24_UNORM_PACK32;
        case VK_FORMAT_D32_SFLOAT: return asset::E_FORMAT::EF_D32_SFLOAT;
        case VK_FORMAT_S8_UINT: return asset::E_FORMAT::EF_S8_UINT;
        case VK_FORMAT_D16_UNORM_S8_UINT: return asset::E_FORMAT::EF_D16_UNORM_S8_UINT;
        case VK_FORMAT_D24_UNORM_S8_UINT: return asset::E_FORMAT::EF_D24_UNORM_S8_UINT;
        case VK_FORMAT_D32_SFLOAT_S8_UINT: return asset::E_FORMAT::EF_D32_SFLOAT_S8_UINT;
        case VK_FORMAT_R4G4_UNORM_PACK8: return asset::E_FORMAT::EF_R4G4_UNORM_PACK8;
        case VK_FORMAT_R4G4B4A4_UNORM_PACK16: return asset::E_FORMAT::EF_R4G4B4A4_UNORM_PACK16;
        case VK_FORMAT_B4G4R4A4_UNORM_PACK16: return asset::E_FORMAT::EF_B4G4R4A4_UNORM_PACK16;
        case VK_FORMAT_R5G6B5_UNORM_PACK16: return asset::E_FORMAT::EF_R5G6B5_UNORM_PACK16;
        case VK_FORMAT_B5G6R5_UNORM_PACK16: return asset::E_FORMAT::EF_B5G6R5_UNORM_PACK16;
        case VK_FORMAT_R5G5B5A1_UNORM_PACK16: return asset::E_FORMAT::EF_R5G5B5A1_UNORM_PACK16;
        case VK_FORMAT_B5G5R5A1_UNORM_PACK16: return asset::E_FORMAT::EF_B5G5R5A1_UNORM_PACK16;
        case VK_FORMAT_A1R5G5B5_UNORM_PACK16: return asset::E_FORMAT::EF_A1R5G5B5_UNORM_PACK16;
        case VK_FORMAT_R8_UNORM: return asset::E_FORMAT::EF_R8_UNORM;
        case VK_FORMAT_R8_SNORM: return asset::E_FORMAT::EF_R8_SNORM;
        case VK_FORMAT_R8_USCALED: return asset::E_FORMAT::EF_R8_USCALED;
        case VK_FORMAT_R8_SSCALED: return asset::E_FORMAT::EF_R8_SSCALED;
        case VK_FORMAT_R8_UINT: return asset::E_FORMAT::EF_R8_UINT;
        case VK_FORMAT_R8_SINT: return asset::E_FORMAT::EF_R8_SINT;
        case VK_FORMAT_R8_SRGB: return asset::E_FORMAT::EF_R8_SRGB;
        case VK_FORMAT_R8G8_UNORM: return asset::E_FORMAT::EF_R8G8_UNORM;
        case VK_FORMAT_R8G8_SNORM: return asset::E_FORMAT::EF_R8G8_SNORM;
        case VK_FORMAT_R8G8_USCALED: return asset::E_FORMAT::EF_R8G8_USCALED;
        case VK_FORMAT_R8G8_SSCALED: return asset::E_FORMAT::EF_R8G8_SSCALED;
        case VK_FORMAT_R8G8_UINT: return asset::E_FORMAT::EF_R8G8_UINT;
        case VK_FORMAT_R8G8_SINT: return asset::E_FORMAT::EF_R8G8_SINT;
        case VK_FORMAT_R8G8_SRGB: return asset::E_FORMAT::EF_R8G8_SRGB;
        case VK_FORMAT_R8G8B8_UNORM: return asset::E_FORMAT::EF_R8G8B8_UNORM;
        case VK_FORMAT_R8G8B8_SNORM: return asset::E_FORMAT::EF_R8G8B8_SNORM;
        case VK_FORMAT_R8G8B8_USCALED: return asset::E_FORMAT::EF_R8G8B8_USCALED;
        case VK_FORMAT_R8G8B8_SSCALED: return asset::E_FORMAT::EF_R8G8B8_SSCALED;
        case VK_FORMAT_R8G8B8_UINT: return asset::E_FORMAT::EF_R8G8B8_UINT;
        case VK_FORMAT_R8G8B8_SINT: return asset::E_FORMAT::EF_R8G8B8_SINT;
        case VK_FORMAT_R8G8B8_SRGB: return asset::E_FORMAT::EF_R8G8B8_SRGB;
        case VK_FORMAT_B8G8R8_UNORM: return asset::E_FORMAT::EF_B8G8R8_UNORM;
        case VK_FORMAT_B8G8R8_SNORM: return asset::E_FORMAT::EF_B8G8R8_SNORM;
        case VK_FORMAT_B8G8R8_USCALED: return asset::E_FORMAT::EF_B8G8R8_USCALED;
        case VK_FORMAT_B8G8R8_SSCALED: return asset::E_FORMAT::EF_B8G8R8_SSCALED;
        case VK_FORMAT_B8G8R8_UINT: return asset::E_FORMAT::EF_B8G8R8_UINT;
        case VK_FORMAT_B8G8R8_SINT: return asset::E_FORMAT::EF_B8G8R8_SINT;
        case VK_FORMAT_B8G8R8_SRGB: return asset::E_FORMAT::EF_B8G8R8_SRGB;
        case VK_FORMAT_R8G8B8A8_UNORM: return asset::E_FORMAT::EF_R8G8B8A8_UNORM;
        case VK_FORMAT_R8G8B8A8_SNORM: return asset::E_FORMAT::EF_R8G8B8A8_SNORM;
        case VK_FORMAT_R8G8B8A8_USCALED: return asset::E_FORMAT::EF_R8G8B8A8_USCALED;
        case VK_FORMAT_R8G8B8A8_SSCALED: return asset::E_FORMAT::EF_R8G8B8A8_SSCALED;
        case VK_FORMAT_R8G8B8A8_UINT: return asset::E_FORMAT::EF_R8G8B8A8_UINT;
        case VK_FORMAT_R8G8B8A8_SINT: return asset::E_FORMAT::EF_R8G8B8A8_SINT;
        case VK_FORMAT_R8G8B8A8_SRGB: return asset::E_FORMAT::EF_R8G8B8A8_SRGB;
        case VK_FORMAT_B8G8R8A8_UNORM: return asset::E_FORMAT::EF_B8G8R8A8_UNORM;
        case VK_FORMAT_B8G8R8A8_SNORM: return asset::E_FORMAT::EF_B8G8R8A8_SNORM;
        case VK_FORMAT_B8G8R8A8_USCALED: return asset::E_FORMAT::EF_B8G8R8A8_USCALED;
        case VK_FORMAT_B8G8R8A8_SSCALED: return asset::E_FORMAT::EF_B8G8R8A8_SSCALED;
        case VK_FORMAT_B8G8R8A8_UINT: return asset::E_FORMAT::EF_B8G8R8A8_UINT;
        case VK_FORMAT_B8G8R8A8_SINT: return asset::E_FORMAT::EF_B8G8R8A8_SINT;
        case VK_FORMAT_B8G8R8A8_SRGB: return asset::E_FORMAT::EF_B8G8R8A8_SRGB;
        case VK_FORMAT_A8B8G8R8_UNORM_PACK32: return asset::E_FORMAT::EF_A8B8G8R8_UNORM_PACK32;
        case VK_FORMAT_A8B8G8R8_SNORM_PACK32: return asset::E_FORMAT::EF_A8B8G8R8_SNORM_PACK32;
        case VK_FORMAT_A8B8G8R8_USCALED_PACK32: return asset::E_FORMAT::EF_A8B8G8R8_USCALED_PACK32;
        case VK_FORMAT_A8B8G8R8_SSCALED_PACK32: return asset::E_FORMAT::EF_A8B8G8R8_SSCALED_PACK32;
        case VK_FORMAT_A8B8G8R8_UINT_PACK32: return asset::E_FORMAT::EF_A8B8G8R8_UINT_PACK32;
        case VK_FORMAT_A8B8G8R8_SINT_PACK32: return asset::E_FORMAT::EF_A8B8G8R8_SINT_PACK32;
        case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return asset::E_FORMAT::EF_A8B8G8R8_SRGB_PACK32;
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32: return asset::E_FORMAT::EF_A2R10G10B10_UNORM_PACK32;
        case VK_FORMAT_A2R10G10B10_SNORM_PACK32: return asset::E_FORMAT::EF_A2R10G10B10_SNORM_PACK32;
        case VK_FORMAT_A2R10G10B10_USCALED_PACK32: return asset::E_FORMAT::EF_A2R10G10B10_USCALED_PACK32;
        case VK_FORMAT_A2R10G10B10_SSCALED_PACK32: return asset::E_FORMAT::EF_A2R10G10B10_SSCALED_PACK32;
        case VK_FORMAT_A2R10G10B10_UINT_PACK32: return asset::E_FORMAT::EF_A2R10G10B10_UINT_PACK32;
        case VK_FORMAT_A2R10G10B10_SINT_PACK32: return asset::E_FORMAT::EF_A2R10G10B10_SINT_PACK32;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return asset::E_FORMAT::EF_A2B10G10R10_UNORM_PACK32;
        case VK_FORMAT_A2B10G10R10_SNORM_PACK32: return asset::E_FORMAT::EF_A2B10G10R10_SNORM_PACK32;
        case VK_FORMAT_A2B10G10R10_USCALED_PACK32: return asset::E_FORMAT::EF_A2B10G10R10_USCALED_PACK32;
        case VK_FORMAT_A2B10G10R10_SSCALED_PACK32: return asset::E_FORMAT::EF_A2B10G10R10_SSCALED_PACK32;
        case VK_FORMAT_A2B10G10R10_UINT_PACK32: return asset::E_FORMAT::EF_A2B10G10R10_UINT_PACK32;
        case VK_FORMAT_A2B10G10R10_SINT_PACK32: return asset::E_FORMAT::EF_A2B10G10R10_SINT_PACK32;
        case VK_FORMAT_R16_UNORM: return asset::E_FORMAT::EF_R16_UNORM;
        case VK_FORMAT_R16_SNORM: return asset::E_FORMAT::EF_R16_SNORM;
        case VK_FORMAT_R16_USCALED: return asset::E_FORMAT::EF_R16_USCALED;
        case VK_FORMAT_R16_SSCALED: return asset::E_FORMAT::EF_R16_SSCALED;
        case VK_FORMAT_R16_UINT: return asset::E_FORMAT::EF_R16_UINT;
        case VK_FORMAT_R16_SINT: return asset::E_FORMAT::EF_R16_SINT;
        case VK_FORMAT_R16_SFLOAT: return asset::E_FORMAT::EF_R16_SFLOAT;
        case VK_FORMAT_R16G16_UNORM: return asset::E_FORMAT::EF_R16G16_UNORM;
        case VK_FORMAT_R16G16_SNORM: return asset::E_FORMAT::EF_R16G16_SNORM;
        case VK_FORMAT_R16G16_USCALED: return asset::E_FORMAT::EF_R16G16_USCALED;
        case VK_FORMAT_R16G16_SSCALED: return asset::E_FORMAT::EF_R16G16_SSCALED;
        case VK_FORMAT_R16G16_UINT: return asset::E_FORMAT::EF_R16G16_UINT;
        case VK_FORMAT_R16G16_SINT: return asset::E_FORMAT::EF_R16G16_SINT;
        case VK_FORMAT_R16G16_SFLOAT: return asset::E_FORMAT::EF_R16G16_SFLOAT;
        case VK_FORMAT_R16G16B16_UNORM: return asset::E_FORMAT::EF_R16G16B16_UNORM;
        case VK_FORMAT_R16G16B16_SNORM: return asset::E_FORMAT::EF_R16G16B16_SNORM;
        case VK_FORMAT_R16G16B16_USCALED: return asset::E_FORMAT::EF_R16G16B16_USCALED;
        case VK_FORMAT_R16G16B16_SSCALED: return asset::E_FORMAT::EF_R16G16B16_SSCALED;
        case VK_FORMAT_R16G16B16_UINT: return asset::E_FORMAT::EF_R16G16B16_UINT;
        case VK_FORMAT_R16G16B16_SINT: return asset::E_FORMAT::EF_R16G16B16_SINT;
        case VK_FORMAT_R16G16B16_SFLOAT: return asset::E_FORMAT::EF_R16G16B16_SFLOAT;
        case VK_FORMAT_R16G16B16A16_UNORM: return asset::E_FORMAT::EF_R16G16B16A16_UNORM;
        case VK_FORMAT_R16G16B16A16_SNORM: return asset::E_FORMAT::EF_R16G16B16A16_SNORM;
        case VK_FORMAT_R16G16B16A16_USCALED: return asset::E_FORMAT::EF_R16G16B16A16_USCALED;
        case VK_FORMAT_R16G16B16A16_SSCALED: return asset::E_FORMAT::EF_R16G16B16A16_SSCALED;
        case VK_FORMAT_R16G16B16A16_UINT: return asset::E_FORMAT::EF_R16G16B16A16_UINT;
        case VK_FORMAT_R16G16B16A16_SINT: return asset::E_FORMAT::EF_R16G16B16A16_SINT;
        case VK_FORMAT_R16G16B16A16_SFLOAT: return asset::E_FORMAT::EF_R16G16B16A16_SFLOAT;
        case VK_FORMAT_R32_UINT: return asset::E_FORMAT::EF_R32_UINT;
        case VK_FORMAT_R32_SINT: return asset::E_FORMAT::EF_R32_SINT;
        case VK_FORMAT_R32_SFLOAT: return asset::E_FORMAT::EF_R32_SFLOAT;
        case VK_FORMAT_R32G32_UINT: return asset::E_FORMAT::EF_R32G32_UINT;
        case VK_FORMAT_R32G32_SINT: return asset::E_FORMAT::EF_R32G32_SINT;
        case VK_FORMAT_R32G32_SFLOAT: return asset::E_FORMAT::EF_R32G32_SFLOAT;
        case VK_FORMAT_R32G32B32_UINT: return asset::E_FORMAT::EF_R32G32B32_UINT;
        case VK_FORMAT_R32G32B32_SINT: return asset::E_FORMAT::EF_R32G32B32_SINT;
        case VK_FORMAT_R32G32B32_SFLOAT: return asset::E_FORMAT::EF_R32G32B32_SFLOAT;
        case VK_FORMAT_R32G32B32A32_UINT: return asset::E_FORMAT::EF_R32G32B32A32_UINT;
        case VK_FORMAT_R32G32B32A32_SINT: return asset::E_FORMAT::EF_R32G32B32A32_SINT;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return asset::E_FORMAT::EF_R32G32B32A32_SFLOAT;
        case VK_FORMAT_R64_UINT: return asset::E_FORMAT::EF_R64_UINT;
        case VK_FORMAT_R64_SINT: return asset::E_FORMAT::EF_R64_SINT;
        case VK_FORMAT_R64_SFLOAT: return asset::E_FORMAT::EF_R64_SFLOAT;
        case VK_FORMAT_R64G64_UINT: return asset::E_FORMAT::EF_R64G64_UINT;
        case VK_FORMAT_R64G64_SINT: return asset::E_FORMAT::EF_R64G64_SINT;
        case VK_FORMAT_R64G64_SFLOAT: return asset::E_FORMAT::EF_R64G64_SFLOAT;
        case VK_FORMAT_R64G64B64_UINT: return asset::E_FORMAT::EF_R64G64B64_UINT;
        case VK_FORMAT_R64G64B64_SINT: return asset::E_FORMAT::EF_R64G64B64_SINT;
        case VK_FORMAT_R64G64B64_SFLOAT: return asset::E_FORMAT::EF_R64G64B64_SFLOAT;
        case VK_FORMAT_R64G64B64A64_UINT: return asset::E_FORMAT::EF_R64G64B64A64_UINT;
        case VK_FORMAT_R64G64B64A64_SINT: return asset::E_FORMAT::EF_R64G64B64A64_SINT;
        case VK_FORMAT_R64G64B64A64_SFLOAT: return asset::E_FORMAT::EF_R64G64B64A64_SFLOAT;
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return asset::E_FORMAT::EF_B10G11R11_UFLOAT_PACK32;
        case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return asset::E_FORMAT::EF_E5B9G9R9_UFLOAT_PACK32;
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK: return asset::E_FORMAT::EF_BC1_RGB_UNORM_BLOCK;
        case VK_FORMAT_BC1_RGB_SRGB_BLOCK: return asset::E_FORMAT::EF_BC1_RGB_SRGB_BLOCK;
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: return asset::E_FORMAT::EF_BC1_RGBA_UNORM_BLOCK;
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: return asset::E_FORMAT::EF_BC1_RGBA_SRGB_BLOCK;
        case VK_FORMAT_BC2_UNORM_BLOCK: return asset::E_FORMAT::EF_BC2_UNORM_BLOCK;
        case VK_FORMAT_BC2_SRGB_BLOCK: return asset::E_FORMAT::EF_BC2_SRGB_BLOCK;
        case VK_FORMAT_BC3_UNORM_BLOCK: return asset::E_FORMAT::EF_BC3_UNORM_BLOCK;
        case VK_FORMAT_BC3_SRGB_BLOCK: return asset::E_FORMAT::EF_BC3_SRGB_BLOCK;
        case VK_FORMAT_BC4_UNORM_BLOCK: return asset::E_FORMAT::EF_BC4_UNORM_BLOCK;
        case VK_FORMAT_BC4_SNORM_BLOCK: return asset::E_FORMAT::EF_BC4_SNORM_BLOCK;
        case VK_FORMAT_BC5_UNORM_BLOCK: return asset::E_FORMAT::EF_BC5_UNORM_BLOCK;
        case VK_FORMAT_BC5_SNORM_BLOCK: return asset::E_FORMAT::EF_BC5_SNORM_BLOCK;
        case VK_FORMAT_BC6H_UFLOAT_BLOCK: return asset::E_FORMAT::EF_BC6H_UFLOAT_BLOCK;
        case VK_FORMAT_BC6H_SFLOAT_BLOCK: return asset::E_FORMAT::EF_BC6H_SFLOAT_BLOCK;
        case VK_FORMAT_BC7_UNORM_BLOCK: return asset::E_FORMAT::EF_BC7_UNORM_BLOCK;
        case VK_FORMAT_BC7_SRGB_BLOCK: return asset::E_FORMAT::EF_BC7_SRGB_BLOCK;
        case VK_FORMAT_ASTC_4x4_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_4x4_UNORM_BLOCK;
        case VK_FORMAT_ASTC_4x4_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_4x4_SRGB_BLOCK;
        case VK_FORMAT_ASTC_5x4_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_5x4_UNORM_BLOCK;
        case VK_FORMAT_ASTC_5x4_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_5x4_SRGB_BLOCK;
        case VK_FORMAT_ASTC_5x5_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_5x5_UNORM_BLOCK;
        case VK_FORMAT_ASTC_5x5_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_5x5_SRGB_BLOCK;
        case VK_FORMAT_ASTC_6x5_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_6x5_UNORM_BLOCK;
        case VK_FORMAT_ASTC_6x5_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_6x5_SRGB_BLOCK;
        case VK_FORMAT_ASTC_6x6_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_6x6_UNORM_BLOCK;
        case VK_FORMAT_ASTC_6x6_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_6x6_SRGB_BLOCK;
        case VK_FORMAT_ASTC_8x5_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_8x5_UNORM_BLOCK;
        case VK_FORMAT_ASTC_8x5_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_8x5_SRGB_BLOCK;
        case VK_FORMAT_ASTC_8x6_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_8x6_UNORM_BLOCK;
        case VK_FORMAT_ASTC_8x6_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_8x6_SRGB_BLOCK;
        case VK_FORMAT_ASTC_8x8_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_8x8_UNORM_BLOCK;
        case VK_FORMAT_ASTC_8x8_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_8x8_SRGB_BLOCK;
        case VK_FORMAT_ASTC_10x5_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_10x5_UNORM_BLOCK;
        case VK_FORMAT_ASTC_10x5_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_10x5_SRGB_BLOCK;
        case VK_FORMAT_ASTC_10x6_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_10x6_UNORM_BLOCK;
        case VK_FORMAT_ASTC_10x6_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_10x6_SRGB_BLOCK;
        case VK_FORMAT_ASTC_10x8_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_10x8_UNORM_BLOCK;
        case VK_FORMAT_ASTC_10x8_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_10x8_SRGB_BLOCK;
        case VK_FORMAT_ASTC_10x10_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_10x10_UNORM_BLOCK;
        case VK_FORMAT_ASTC_10x10_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_10x10_SRGB_BLOCK;
        case VK_FORMAT_ASTC_12x10_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_12x10_UNORM_BLOCK;
        case VK_FORMAT_ASTC_12x10_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_12x10_SRGB_BLOCK;
        case VK_FORMAT_ASTC_12x12_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_12x12_UNORM_BLOCK;
        case VK_FORMAT_ASTC_12x12_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_12x12_SRGB_BLOCK;
        case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK: return asset::E_FORMAT::EF_ETC2_R8G8B8_UNORM_BLOCK;
        case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK: return asset::E_FORMAT::EF_ETC2_R8G8B8_SRGB_BLOCK;
        case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK: return asset::E_FORMAT::EF_ETC2_R8G8B8A1_UNORM_BLOCK;
        case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK: return asset::E_FORMAT::EF_ETC2_R8G8B8A1_SRGB_BLOCK;
        case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK: return asset::E_FORMAT::EF_ETC2_R8G8B8A8_UNORM_BLOCK;
        case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK: return asset::E_FORMAT::EF_ETC2_R8G8B8A8_SRGB_BLOCK;
        case VK_FORMAT_EAC_R11_UNORM_BLOCK: return asset::E_FORMAT::EF_EAC_R11_UNORM_BLOCK;
        case VK_FORMAT_EAC_R11_SNORM_BLOCK: return asset::E_FORMAT::EF_EAC_R11_SNORM_BLOCK;
        case VK_FORMAT_EAC_R11G11_UNORM_BLOCK: return asset::E_FORMAT::EF_EAC_R11G11_UNORM_BLOCK;
        case VK_FORMAT_EAC_R11G11_SNORM_BLOCK: return asset::E_FORMAT::EF_EAC_R11G11_SNORM_BLOCK;
        case VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG: return asset::E_FORMAT::EF_PVRTC1_2BPP_UNORM_BLOCK_IMG;
        case VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG: return asset::E_FORMAT::EF_PVRTC1_4BPP_UNORM_BLOCK_IMG;
        case VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG: return asset::E_FORMAT::EF_PVRTC2_2BPP_UNORM_BLOCK_IMG;
        case VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG: return asset::E_FORMAT::EF_PVRTC2_4BPP_UNORM_BLOCK_IMG;
        case VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG: return asset::E_FORMAT::EF_PVRTC1_2BPP_SRGB_BLOCK_IMG;
        case VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG: return asset::E_FORMAT::EF_PVRTC1_4BPP_SRGB_BLOCK_IMG;
        case VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG: return asset::E_FORMAT::EF_PVRTC2_2BPP_SRGB_BLOCK_IMG;
        case VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG: return asset::E_FORMAT::EF_PVRTC2_4BPP_SRGB_BLOCK_IMG;
        case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM: return asset::E_FORMAT::EF_G8_B8_R8_3PLANE_420_UNORM;
        case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM: return asset::E_FORMAT::EF_G8_B8R8_2PLANE_420_UNORM;
        case VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM: return asset::E_FORMAT::EF_G8_B8_R8_3PLANE_422_UNORM;
        case VK_FORMAT_G8_B8R8_2PLANE_422_UNORM: return asset::E_FORMAT::EF_G8_B8R8_2PLANE_422_UNORM;
        case VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM: return asset::E_FORMAT::EF_G8_B8_R8_3PLANE_444_UNORM;
        default:
            return asset::E_FORMAT::EF_UNKNOWN;
    }
}

static inline ISurface::SColorSpace getColorSpaceFromVkColorSpaceKHR(VkColorSpaceKHR in)
{
    ISurface::SColorSpace result = {asset::ECP_COUNT, asset::EOTF_UNKNOWN};

    switch(in)
    {
        case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR: {
            result.primary = asset::ECP_SRGB;
            result.eotf = asset::EOTF_sRGB;
        }
        break;

        case VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT: {
            result.primary = asset::ECP_DISPLAY_P3;
            result.eotf = asset::EOTF_sRGB;  // spec says "sRGB-like"
        }
        break;

        case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT: {
            result.primary = asset::ECP_SRGB;
            result.eotf = asset::EOTF_IDENTITY;
        }
        break;

        case VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT: {
            result.primary = asset::ECP_DISPLAY_P3;
            result.eotf = asset::EOTF_IDENTITY;
        }
        break;

        case VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT: {
            result.primary = asset::ECP_DCI_P3;
            result.eotf = asset::EOTF_DCI_P3_XYZ;
        }
        break;

        case VK_COLOR_SPACE_BT709_LINEAR_EXT: {
            result.primary = asset::ECP_SRGB;
            result.eotf = asset::EOTF_IDENTITY;
        }
        break;

        case VK_COLOR_SPACE_BT709_NONLINEAR_EXT: {
            result.primary = asset::ECP_SRGB;
            result.eotf = asset::EOTF_SMPTE_170M;
        }
        break;

        case VK_COLOR_SPACE_BT2020_LINEAR_EXT: {
            result.primary = asset::ECP_BT2020;
            result.eotf = asset::EOTF_IDENTITY;
        }
        break;

        case VK_COLOR_SPACE_HDR10_ST2084_EXT: {
            result.primary = asset::ECP_BT2020;
            result.eotf = asset::EOTF_SMPTE_ST2084;
        }
        break;

        case VK_COLOR_SPACE_DOLBYVISION_EXT: {
            result.primary = asset::ECP_BT2020;
            result.eotf = asset::EOTF_SMPTE_ST2084;
        }
        break;

        case VK_COLOR_SPACE_HDR10_HLG_EXT: {
            result.primary = asset::ECP_BT2020;
            result.eotf = asset::EOTF_HDR10_HLG;
        }
        break;

        case VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT: {
            result.primary = asset::ECP_ADOBERGB;
            result.eotf = asset::EOTF_IDENTITY;
        }
        break;

        case VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT: {
            result.primary = asset::ECP_ADOBERGB;
            result.eotf = asset::EOTF_GAMMA_2_2;
        }
        break;

        case VK_COLOR_SPACE_PASS_THROUGH_EXT: {
            result.primary = asset::ECP_PASS_THROUGH;
            result.eotf = asset::EOTF_IDENTITY;
        }
        break;

        case VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT: {
            result.primary = asset::ECP_SRGB;
            result.eotf = asset::EOTF_sRGB;
        }
        break;

        case VK_COLOR_SPACE_DISPLAY_NATIVE_AMD:  // this one is completely bogus, I don't understand it at all
        {
            result.primary = asset::ECP_SRGB;
            result.eotf = asset::EOTF_UNKNOWN;
        }
        break;
    }

    return result;
}

static inline ISurface::E_PRESENT_MODE getPresentModeFromVkPresentModeKHR(VkPresentModeKHR in)
{
    switch(in)
    {
        case VK_PRESENT_MODE_IMMEDIATE_KHR:
            return ISurface::EPM_IMMEDIATE;
        case VK_PRESENT_MODE_MAILBOX_KHR:
            return ISurface::EPM_MAILBOX;
        case VK_PRESENT_MODE_FIFO_KHR:
            return ISurface::EPM_FIFO;
        case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
            return ISurface::EPM_FIFO_RELAXED;
        default:
            return ISurface::EPM_UNKNOWN;
    }
}

static inline VkFormat getVkFormatFromFormat(asset::E_FORMAT in)
{
    switch(in)
    {
        case asset::E_FORMAT::EF_D16_UNORM: return VK_FORMAT_D16_UNORM;
        case asset::E_FORMAT::EF_X8_D24_UNORM_PACK32: return VK_FORMAT_X8_D24_UNORM_PACK32;
        case asset::E_FORMAT::EF_D32_SFLOAT: return VK_FORMAT_D32_SFLOAT;
        case asset::E_FORMAT::EF_S8_UINT: return VK_FORMAT_S8_UINT;
        case asset::E_FORMAT::EF_D16_UNORM_S8_UINT: return VK_FORMAT_D16_UNORM_S8_UINT;
        case asset::E_FORMAT::EF_D24_UNORM_S8_UINT: return VK_FORMAT_D24_UNORM_S8_UINT;
        case asset::E_FORMAT::EF_D32_SFLOAT_S8_UINT: return VK_FORMAT_D32_SFLOAT_S8_UINT;
        case asset::E_FORMAT::EF_R4G4_UNORM_PACK8: return VK_FORMAT_R4G4_UNORM_PACK8;
        case asset::E_FORMAT::EF_R4G4B4A4_UNORM_PACK16: return VK_FORMAT_R4G4B4A4_UNORM_PACK16;
        case asset::E_FORMAT::EF_B4G4R4A4_UNORM_PACK16: return VK_FORMAT_B4G4R4A4_UNORM_PACK16;
        case asset::E_FORMAT::EF_R5G6B5_UNORM_PACK16: return VK_FORMAT_R5G6B5_UNORM_PACK16;
        case asset::E_FORMAT::EF_B5G6R5_UNORM_PACK16: return VK_FORMAT_B5G6R5_UNORM_PACK16;
        case asset::E_FORMAT::EF_R5G5B5A1_UNORM_PACK16: return VK_FORMAT_R5G5B5A1_UNORM_PACK16;
        case asset::E_FORMAT::EF_B5G5R5A1_UNORM_PACK16: return VK_FORMAT_B5G5R5A1_UNORM_PACK16;
        case asset::E_FORMAT::EF_A1R5G5B5_UNORM_PACK16: return VK_FORMAT_A1R5G5B5_UNORM_PACK16;
        case asset::E_FORMAT::EF_R8_UNORM: return VK_FORMAT_R8_UNORM;
        case asset::E_FORMAT::EF_R8_SNORM: return VK_FORMAT_R8_SNORM;
        case asset::E_FORMAT::EF_R8_USCALED: return VK_FORMAT_R8_USCALED;
        case asset::E_FORMAT::EF_R8_SSCALED: return VK_FORMAT_R8_SSCALED;
        case asset::E_FORMAT::EF_R8_UINT: return VK_FORMAT_R8_UINT;
        case asset::E_FORMAT::EF_R8_SINT: return VK_FORMAT_R8_SINT;
        case asset::E_FORMAT::EF_R8_SRGB: return VK_FORMAT_R8_SRGB;
        case asset::E_FORMAT::EF_R8G8_UNORM: return VK_FORMAT_R8G8_UNORM;
        case asset::E_FORMAT::EF_R8G8_SNORM: return VK_FORMAT_R8G8_SNORM;
        case asset::E_FORMAT::EF_R8G8_USCALED: return VK_FORMAT_R8G8_USCALED;
        case asset::E_FORMAT::EF_R8G8_SSCALED: return VK_FORMAT_R8G8_SSCALED;
        case asset::E_FORMAT::EF_R8G8_UINT: return VK_FORMAT_R8G8_UINT;
        case asset::E_FORMAT::EF_R8G8_SINT: return VK_FORMAT_R8G8_SINT;
        case asset::E_FORMAT::EF_R8G8_SRGB: return VK_FORMAT_R8G8_SRGB;
        case asset::E_FORMAT::EF_R8G8B8_UNORM: return VK_FORMAT_R8G8B8_UNORM;
        case asset::E_FORMAT::EF_R8G8B8_SNORM: return VK_FORMAT_R8G8B8_SNORM;
        case asset::E_FORMAT::EF_R8G8B8_USCALED: return VK_FORMAT_R8G8B8_USCALED;
        case asset::E_FORMAT::EF_R8G8B8_SSCALED: return VK_FORMAT_R8G8B8_SSCALED;
        case asset::E_FORMAT::EF_R8G8B8_UINT: return VK_FORMAT_R8G8B8_UINT;
        case asset::E_FORMAT::EF_R8G8B8_SINT: return VK_FORMAT_R8G8B8_SINT;
        case asset::E_FORMAT::EF_R8G8B8_SRGB: return VK_FORMAT_R8G8B8_SRGB;
        case asset::E_FORMAT::EF_B8G8R8_UNORM: return VK_FORMAT_B8G8R8_UNORM;
        case asset::E_FORMAT::EF_B8G8R8_SNORM: return VK_FORMAT_B8G8R8_SNORM;
        case asset::E_FORMAT::EF_B8G8R8_USCALED: return VK_FORMAT_B8G8R8_USCALED;
        case asset::E_FORMAT::EF_B8G8R8_SSCALED: return VK_FORMAT_B8G8R8_SSCALED;
        case asset::E_FORMAT::EF_B8G8R8_UINT: return VK_FORMAT_B8G8R8_UINT;
        case asset::E_FORMAT::EF_B8G8R8_SINT: return VK_FORMAT_B8G8R8_SINT;
        case asset::E_FORMAT::EF_B8G8R8_SRGB: return VK_FORMAT_B8G8R8_SRGB;
        case asset::E_FORMAT::EF_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
        case asset::E_FORMAT::EF_R8G8B8A8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
        case asset::E_FORMAT::EF_R8G8B8A8_USCALED: return VK_FORMAT_R8G8B8A8_USCALED;
        case asset::E_FORMAT::EF_R8G8B8A8_SSCALED: return VK_FORMAT_R8G8B8A8_SSCALED;
        case asset::E_FORMAT::EF_R8G8B8A8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
        case asset::E_FORMAT::EF_R8G8B8A8_SINT: return VK_FORMAT_R8G8B8A8_SINT;
        case asset::E_FORMAT::EF_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
        case asset::E_FORMAT::EF_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
        case asset::E_FORMAT::EF_B8G8R8A8_SNORM: return VK_FORMAT_B8G8R8A8_SNORM;
        case asset::E_FORMAT::EF_B8G8R8A8_USCALED: return VK_FORMAT_B8G8R8A8_USCALED;
        case asset::E_FORMAT::EF_B8G8R8A8_SSCALED: return VK_FORMAT_B8G8R8A8_SSCALED;
        case asset::E_FORMAT::EF_B8G8R8A8_UINT: return VK_FORMAT_B8G8R8A8_UINT;
        case asset::E_FORMAT::EF_B8G8R8A8_SINT: return VK_FORMAT_B8G8R8A8_SINT;
        case asset::E_FORMAT::EF_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;
        case asset::E_FORMAT::EF_A8B8G8R8_UNORM_PACK32: return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
        case asset::E_FORMAT::EF_A8B8G8R8_SNORM_PACK32: return VK_FORMAT_A8B8G8R8_SNORM_PACK32;
        case asset::E_FORMAT::EF_A8B8G8R8_USCALED_PACK32: return VK_FORMAT_A8B8G8R8_USCALED_PACK32;
        case asset::E_FORMAT::EF_A8B8G8R8_SSCALED_PACK32: return VK_FORMAT_A8B8G8R8_SSCALED_PACK32;
        case asset::E_FORMAT::EF_A8B8G8R8_UINT_PACK32: return VK_FORMAT_A8B8G8R8_UINT_PACK32;
        case asset::E_FORMAT::EF_A8B8G8R8_SINT_PACK32: return VK_FORMAT_A8B8G8R8_SINT_PACK32;
        case asset::E_FORMAT::EF_A8B8G8R8_SRGB_PACK32: return VK_FORMAT_A8B8G8R8_SRGB_PACK32;
        case asset::E_FORMAT::EF_A2R10G10B10_UNORM_PACK32: return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
        case asset::E_FORMAT::EF_A2R10G10B10_SNORM_PACK32: return VK_FORMAT_A2R10G10B10_SNORM_PACK32;
        case asset::E_FORMAT::EF_A2R10G10B10_USCALED_PACK32: return VK_FORMAT_A2R10G10B10_USCALED_PACK32;
        case asset::E_FORMAT::EF_A2R10G10B10_SSCALED_PACK32: return VK_FORMAT_A2R10G10B10_SSCALED_PACK32;
        case asset::E_FORMAT::EF_A2R10G10B10_UINT_PACK32: return VK_FORMAT_A2R10G10B10_UINT_PACK32;
        case asset::E_FORMAT::EF_A2R10G10B10_SINT_PACK32: return VK_FORMAT_A2R10G10B10_SINT_PACK32;
        case asset::E_FORMAT::EF_A2B10G10R10_UNORM_PACK32: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case asset::E_FORMAT::EF_A2B10G10R10_SNORM_PACK32: return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
        case asset::E_FORMAT::EF_A2B10G10R10_USCALED_PACK32: return VK_FORMAT_A2B10G10R10_USCALED_PACK32;
        case asset::E_FORMAT::EF_A2B10G10R10_SSCALED_PACK32: return VK_FORMAT_A2B10G10R10_SSCALED_PACK32;
        case asset::E_FORMAT::EF_A2B10G10R10_UINT_PACK32: return VK_FORMAT_A2B10G10R10_UINT_PACK32;
        case asset::E_FORMAT::EF_A2B10G10R10_SINT_PACK32: return VK_FORMAT_A2B10G10R10_SINT_PACK32;
        case asset::E_FORMAT::EF_R16_UNORM: return VK_FORMAT_R16_UNORM;
        case asset::E_FORMAT::EF_R16_SNORM: return VK_FORMAT_R16_SNORM;
        case asset::E_FORMAT::EF_R16_USCALED: return VK_FORMAT_R16_USCALED;
        case asset::E_FORMAT::EF_R16_SSCALED: return VK_FORMAT_R16_SSCALED;
        case asset::E_FORMAT::EF_R16_UINT: return VK_FORMAT_R16_UINT;
        case asset::E_FORMAT::EF_R16_SINT: return VK_FORMAT_R16_SINT;
        case asset::E_FORMAT::EF_R16_SFLOAT: return VK_FORMAT_R16_SFLOAT;
        case asset::E_FORMAT::EF_R16G16_UNORM: return VK_FORMAT_R16G16_UNORM;
        case asset::E_FORMAT::EF_R16G16_SNORM: return VK_FORMAT_R16G16_SNORM;
        case asset::E_FORMAT::EF_R16G16_USCALED: return VK_FORMAT_R16G16_USCALED;
        case asset::E_FORMAT::EF_R16G16_SSCALED: return VK_FORMAT_R16G16_SSCALED;
        case asset::E_FORMAT::EF_R16G16_UINT: return VK_FORMAT_R16G16_UINT;
        case asset::E_FORMAT::EF_R16G16_SINT: return VK_FORMAT_R16G16_SINT;
        case asset::E_FORMAT::EF_R16G16_SFLOAT: return VK_FORMAT_R16G16_SFLOAT;
        case asset::E_FORMAT::EF_R16G16B16_UNORM: return VK_FORMAT_R16G16B16_UNORM;
        case asset::E_FORMAT::EF_R16G16B16_SNORM: return VK_FORMAT_R16G16B16_SNORM;
        case asset::E_FORMAT::EF_R16G16B16_USCALED: return VK_FORMAT_R16G16B16_USCALED;
        case asset::E_FORMAT::EF_R16G16B16_SSCALED: return VK_FORMAT_R16G16B16_SSCALED;
        case asset::E_FORMAT::EF_R16G16B16_UINT: return VK_FORMAT_R16G16B16_UINT;
        case asset::E_FORMAT::EF_R16G16B16_SINT: return VK_FORMAT_R16G16B16_SINT;
        case asset::E_FORMAT::EF_R16G16B16_SFLOAT: return VK_FORMAT_R16G16B16_SFLOAT;
        case asset::E_FORMAT::EF_R16G16B16A16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
        case asset::E_FORMAT::EF_R16G16B16A16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
        case asset::E_FORMAT::EF_R16G16B16A16_USCALED: return VK_FORMAT_R16G16B16A16_USCALED;
        case asset::E_FORMAT::EF_R16G16B16A16_SSCALED: return VK_FORMAT_R16G16B16A16_SSCALED;
        case asset::E_FORMAT::EF_R16G16B16A16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
        case asset::E_FORMAT::EF_R16G16B16A16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
        case asset::E_FORMAT::EF_R16G16B16A16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case asset::E_FORMAT::EF_R32_UINT: return VK_FORMAT_R32_UINT;
        case asset::E_FORMAT::EF_R32_SINT: return VK_FORMAT_R32_SINT;
        case asset::E_FORMAT::EF_R32_SFLOAT: return VK_FORMAT_R32_SFLOAT;
        case asset::E_FORMAT::EF_R32G32_UINT: return VK_FORMAT_R32G32_UINT;
        case asset::E_FORMAT::EF_R32G32_SINT: return VK_FORMAT_R32G32_SINT;
        case asset::E_FORMAT::EF_R32G32_SFLOAT: return VK_FORMAT_R32G32_SFLOAT;
        case asset::E_FORMAT::EF_R32G32B32_UINT: return VK_FORMAT_R32G32B32_UINT;
        case asset::E_FORMAT::EF_R32G32B32_SINT: return VK_FORMAT_R32G32B32_SINT;
        case asset::E_FORMAT::EF_R32G32B32_SFLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
        case asset::E_FORMAT::EF_R32G32B32A32_UINT: return VK_FORMAT_R32G32B32A32_UINT;
        case asset::E_FORMAT::EF_R32G32B32A32_SINT: return VK_FORMAT_R32G32B32A32_SINT;
        case asset::E_FORMAT::EF_R32G32B32A32_SFLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case asset::E_FORMAT::EF_R64_UINT: return VK_FORMAT_R64_UINT;
        case asset::E_FORMAT::EF_R64_SINT: return VK_FORMAT_R64_SINT;
        case asset::E_FORMAT::EF_R64_SFLOAT: return VK_FORMAT_R64_SFLOAT;
        case asset::E_FORMAT::EF_R64G64_UINT: return VK_FORMAT_R64G64_UINT;
        case asset::E_FORMAT::EF_R64G64_SINT: return VK_FORMAT_R64G64_SINT;
        case asset::E_FORMAT::EF_R64G64_SFLOAT: return VK_FORMAT_R64G64_SFLOAT;
        case asset::E_FORMAT::EF_R64G64B64_UINT: return VK_FORMAT_R64G64B64_UINT;
        case asset::E_FORMAT::EF_R64G64B64_SINT: return VK_FORMAT_R64G64B64_SINT;
        case asset::E_FORMAT::EF_R64G64B64_SFLOAT: return VK_FORMAT_R64G64B64_SFLOAT;
        case asset::E_FORMAT::EF_R64G64B64A64_UINT: return VK_FORMAT_R64G64B64A64_UINT;
        case asset::E_FORMAT::EF_R64G64B64A64_SINT: return VK_FORMAT_R64G64B64A64_SINT;
        case asset::E_FORMAT::EF_R64G64B64A64_SFLOAT: return VK_FORMAT_R64G64B64A64_SFLOAT;
        case asset::E_FORMAT::EF_B10G11R11_UFLOAT_PACK32: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        case asset::E_FORMAT::EF_E5B9G9R9_UFLOAT_PACK32: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
        case asset::E_FORMAT::EF_BC1_RGB_UNORM_BLOCK: return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
        case asset::E_FORMAT::EF_BC1_RGB_SRGB_BLOCK: return VK_FORMAT_BC1_RGB_SRGB_BLOCK;
        case asset::E_FORMAT::EF_BC1_RGBA_UNORM_BLOCK: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case asset::E_FORMAT::EF_BC1_RGBA_SRGB_BLOCK: return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
        case asset::E_FORMAT::EF_BC2_UNORM_BLOCK: return VK_FORMAT_BC2_UNORM_BLOCK;
        case asset::E_FORMAT::EF_BC2_SRGB_BLOCK: return VK_FORMAT_BC2_SRGB_BLOCK;
        case asset::E_FORMAT::EF_BC3_UNORM_BLOCK: return VK_FORMAT_BC3_UNORM_BLOCK;
        case asset::E_FORMAT::EF_BC3_SRGB_BLOCK: return VK_FORMAT_BC3_SRGB_BLOCK;
        case asset::E_FORMAT::EF_BC4_UNORM_BLOCK: return VK_FORMAT_BC4_UNORM_BLOCK;
        case asset::E_FORMAT::EF_BC4_SNORM_BLOCK: return VK_FORMAT_BC4_SNORM_BLOCK;
        case asset::E_FORMAT::EF_BC5_UNORM_BLOCK: return VK_FORMAT_BC5_UNORM_BLOCK;
        case asset::E_FORMAT::EF_BC5_SNORM_BLOCK: return VK_FORMAT_BC5_SNORM_BLOCK;
        case asset::E_FORMAT::EF_BC6H_UFLOAT_BLOCK: return VK_FORMAT_BC6H_UFLOAT_BLOCK;
        case asset::E_FORMAT::EF_BC6H_SFLOAT_BLOCK: return VK_FORMAT_BC6H_SFLOAT_BLOCK;
        case asset::E_FORMAT::EF_BC7_UNORM_BLOCK: return VK_FORMAT_BC7_UNORM_BLOCK;
        case asset::E_FORMAT::EF_BC7_SRGB_BLOCK: return VK_FORMAT_BC7_SRGB_BLOCK;
        case asset::E_FORMAT::EF_ASTC_4x4_UNORM_BLOCK: return VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
        case asset::E_FORMAT::EF_ASTC_4x4_SRGB_BLOCK: return VK_FORMAT_ASTC_4x4_SRGB_BLOCK;
        case asset::E_FORMAT::EF_ASTC_5x4_UNORM_BLOCK: return VK_FORMAT_ASTC_5x4_UNORM_BLOCK;
        case asset::E_FORMAT::EF_ASTC_5x4_SRGB_BLOCK: return VK_FORMAT_ASTC_5x4_SRGB_BLOCK;
        case asset::E_FORMAT::EF_ASTC_5x5_UNORM_BLOCK: return VK_FORMAT_ASTC_5x5_UNORM_BLOCK;
        case asset::E_FORMAT::EF_ASTC_5x5_SRGB_BLOCK: return VK_FORMAT_ASTC_5x5_SRGB_BLOCK;
        case asset::E_FORMAT::EF_ASTC_6x5_UNORM_BLOCK: return VK_FORMAT_ASTC_6x5_UNORM_BLOCK;
        case asset::E_FORMAT::EF_ASTC_6x5_SRGB_BLOCK: return VK_FORMAT_ASTC_6x5_SRGB_BLOCK;
        case asset::E_FORMAT::EF_ASTC_6x6_UNORM_BLOCK: return VK_FORMAT_ASTC_6x6_UNORM_BLOCK;
        case asset::E_FORMAT::EF_ASTC_6x6_SRGB_BLOCK: return VK_FORMAT_ASTC_6x6_SRGB_BLOCK;
        case asset::E_FORMAT::EF_ASTC_8x5_UNORM_BLOCK: return VK_FORMAT_ASTC_8x5_UNORM_BLOCK;
        case asset::E_FORMAT::EF_ASTC_8x5_SRGB_BLOCK: return VK_FORMAT_ASTC_8x5_SRGB_BLOCK;
        case asset::E_FORMAT::EF_ASTC_8x6_UNORM_BLOCK: return VK_FORMAT_ASTC_8x6_UNORM_BLOCK;
        case asset::E_FORMAT::EF_ASTC_8x6_SRGB_BLOCK: return VK_FORMAT_ASTC_8x6_SRGB_BLOCK;
        case asset::E_FORMAT::EF_ASTC_8x8_UNORM_BLOCK: return VK_FORMAT_ASTC_8x8_UNORM_BLOCK;
        case asset::E_FORMAT::EF_ASTC_8x8_SRGB_BLOCK: return VK_FORMAT_ASTC_8x8_SRGB_BLOCK;
        case asset::E_FORMAT::EF_ASTC_10x5_UNORM_BLOCK: return VK_FORMAT_ASTC_10x5_UNORM_BLOCK;
        case asset::E_FORMAT::EF_ASTC_10x5_SRGB_BLOCK: return VK_FORMAT_ASTC_10x5_SRGB_BLOCK;
        case asset::E_FORMAT::EF_ASTC_10x6_UNORM_BLOCK: return VK_FORMAT_ASTC_10x6_UNORM_BLOCK;
        case asset::E_FORMAT::EF_ASTC_10x6_SRGB_BLOCK: return VK_FORMAT_ASTC_10x6_SRGB_BLOCK;
        case asset::E_FORMAT::EF_ASTC_10x8_UNORM_BLOCK: return VK_FORMAT_ASTC_10x8_UNORM_BLOCK;
        case asset::E_FORMAT::EF_ASTC_10x8_SRGB_BLOCK: return VK_FORMAT_ASTC_10x8_SRGB_BLOCK;
        case asset::E_FORMAT::EF_ASTC_10x10_UNORM_BLOCK: return VK_FORMAT_ASTC_10x10_UNORM_BLOCK;
        case asset::E_FORMAT::EF_ASTC_10x10_SRGB_BLOCK: return VK_FORMAT_ASTC_10x10_SRGB_BLOCK;
        case asset::E_FORMAT::EF_ASTC_12x10_UNORM_BLOCK: return VK_FORMAT_ASTC_12x10_UNORM_BLOCK;
        case asset::E_FORMAT::EF_ASTC_12x10_SRGB_BLOCK: return VK_FORMAT_ASTC_12x10_SRGB_BLOCK;
        case asset::E_FORMAT::EF_ASTC_12x12_UNORM_BLOCK: return VK_FORMAT_ASTC_12x12_UNORM_BLOCK;
        case asset::E_FORMAT::EF_ASTC_12x12_SRGB_BLOCK: return VK_FORMAT_ASTC_12x12_SRGB_BLOCK;
        case asset::E_FORMAT::EF_ETC2_R8G8B8_UNORM_BLOCK: return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
        case asset::E_FORMAT::EF_ETC2_R8G8B8_SRGB_BLOCK: return VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK;
        case asset::E_FORMAT::EF_ETC2_R8G8B8A1_UNORM_BLOCK: return VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK;
        case asset::E_FORMAT::EF_ETC2_R8G8B8A1_SRGB_BLOCK: return VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK;
        case asset::E_FORMAT::EF_ETC2_R8G8B8A8_UNORM_BLOCK: return VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;
        case asset::E_FORMAT::EF_ETC2_R8G8B8A8_SRGB_BLOCK: return VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;
        case asset::E_FORMAT::EF_EAC_R11_UNORM_BLOCK: return VK_FORMAT_EAC_R11_UNORM_BLOCK;
        case asset::E_FORMAT::EF_EAC_R11_SNORM_BLOCK: return VK_FORMAT_EAC_R11_SNORM_BLOCK;
        case asset::E_FORMAT::EF_EAC_R11G11_UNORM_BLOCK: return VK_FORMAT_EAC_R11G11_UNORM_BLOCK;
        case asset::E_FORMAT::EF_EAC_R11G11_SNORM_BLOCK: return VK_FORMAT_EAC_R11G11_SNORM_BLOCK;
        case asset::E_FORMAT::EF_PVRTC1_2BPP_UNORM_BLOCK_IMG: return VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;
        case asset::E_FORMAT::EF_PVRTC1_4BPP_UNORM_BLOCK_IMG: return VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG;
        case asset::E_FORMAT::EF_PVRTC2_2BPP_UNORM_BLOCK_IMG: return VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG;
        case asset::E_FORMAT::EF_PVRTC2_4BPP_UNORM_BLOCK_IMG: return VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG;
        case asset::E_FORMAT::EF_PVRTC1_2BPP_SRGB_BLOCK_IMG: return VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG;
        case asset::E_FORMAT::EF_PVRTC1_4BPP_SRGB_BLOCK_IMG: return VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG;
        case asset::E_FORMAT::EF_PVRTC2_2BPP_SRGB_BLOCK_IMG: return VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG;
        case asset::E_FORMAT::EF_PVRTC2_4BPP_SRGB_BLOCK_IMG: return VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG;
        case asset::E_FORMAT::EF_G8_B8_R8_3PLANE_420_UNORM: return VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
        case asset::E_FORMAT::EF_G8_B8R8_2PLANE_420_UNORM: return VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
        case asset::E_FORMAT::EF_G8_B8_R8_3PLANE_422_UNORM: return VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM;
        case asset::E_FORMAT::EF_G8_B8R8_2PLANE_422_UNORM: return VK_FORMAT_G8_B8R8_2PLANE_422_UNORM;
        case asset::E_FORMAT::EF_G8_B8_R8_3PLANE_444_UNORM: return VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM;
        default:
        case asset::E_FORMAT::EF_UNKNOWN:
            return VK_FORMAT_MAX_ENUM;
    }
#ifdef UNMAINTAINABLE_CODE
    if(in <= asset::EF_BC7_SRGB_BLOCK)
        return static_cast<VkFormat>(in);

    if(in >= asset::EF_ETC2_R8G8B8_UNORM_BLOCK && in <= asset::EF_EAC_R11G11_SNORM_BLOCK)
        return static_cast<VkFormat>(in - 28u);

    if(in >= asset::EF_ASTC_4x4_UNORM_BLOCK && in <= asset::EF_ASTC_12x12_SRGB_BLOCK)
        return static_cast<VkFormat>(in + 10u);

    if(in >= asset::EF_PVRTC1_2BPP_UNORM_BLOCK_IMG && in <= asset::EF_PVRTC2_4BPP_SRGB_BLOCK_IMG)
        return static_cast<VkFormat>(in + 1000053815u);

    if(in >= asset::EF_G8_B8_R8_3PLANE_420_UNORM && in <= asset::EF_G8_B8_R8_3PLANE_444_UNORM)
        return static_cast<VkFormat>(in + 1000155809);

    return VK_FORMAT_MAX_ENUM;
#endif
}

static inline VkColorSpaceKHR getVkColorSpaceKHRFromColorSpace(ISurface::SColorSpace in)
{
    if(in.primary == asset::ECP_SRGB && in.eotf == asset::EOTF_sRGB)
        return VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

    if(in.primary == asset::ECP_DISPLAY_P3 && in.eotf == asset::EOTF_sRGB)
        return VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT;

    if(in.primary == asset::ECP_SRGB && in.eotf == asset::EOTF_IDENTITY)
        return VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT;

    if(in.primary == asset::ECP_DISPLAY_P3 && in.eotf == asset::EOTF_IDENTITY)
        return VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT;

    if(in.primary == asset::ECP_DCI_P3 && in.eotf == asset::EOTF_DCI_P3_XYZ)
        return VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT;

    if(in.primary == asset::ECP_SRGB && in.eotf == asset::EOTF_IDENTITY)
        return VK_COLOR_SPACE_BT709_LINEAR_EXT;

    if(in.primary == asset::ECP_SRGB && in.eotf == asset::EOTF_SMPTE_170M)
        return VK_COLOR_SPACE_BT709_NONLINEAR_EXT;

    if(in.primary == asset::ECP_BT2020 && in.eotf == asset::EOTF_IDENTITY)
        return VK_COLOR_SPACE_BT2020_LINEAR_EXT;

    if(in.primary == asset::ECP_BT2020 && in.eotf == asset::EOTF_SMPTE_ST2084)
        return VK_COLOR_SPACE_HDR10_ST2084_EXT;

    if(in.primary == asset::ECP_BT2020 && in.eotf == asset::EOTF_SMPTE_ST2084)
        return VK_COLOR_SPACE_DOLBYVISION_EXT;

    if(in.primary == asset::ECP_BT2020 && in.eotf == asset::EOTF_HDR10_HLG)
        return VK_COLOR_SPACE_HDR10_HLG_EXT;

    if(in.primary == asset::ECP_ADOBERGB && in.eotf == asset::EOTF_IDENTITY)
        return VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT;

    if(in.primary == asset::ECP_ADOBERGB && in.eotf == asset::EOTF_GAMMA_2_2)
        return VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT;

    if(in.primary == asset::ECP_PASS_THROUGH && in.eotf == asset::EOTF_IDENTITY)
        return VK_COLOR_SPACE_PASS_THROUGH_EXT;

    if(in.primary == asset::ECP_SRGB && in.eotf == asset::EOTF_sRGB)
        return VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT;

    if(in.primary == asset::ECP_SRGB && in.eotf == asset::EOTF_UNKNOWN)
        return VK_COLOR_SPACE_DISPLAY_NATIVE_AMD;

    return VK_COLOR_SPACE_MAX_ENUM_KHR;
}

static inline VkSamplerAddressMode getVkAddressModeFromTexClamp(const asset::ISampler::E_TEXTURE_CLAMP in)
{
    switch(in)
    {
        case asset::ISampler::ETC_REPEAT:
            return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case asset::ISampler::ETC_CLAMP_TO_EDGE:
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case asset::ISampler::ETC_CLAMP_TO_BORDER:
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        case asset::ISampler::ETC_MIRROR:
            return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        default:
            assert(!"ADDRESS MODE NOT SUPPORTED!");
            return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    }
}

static inline std::pair<VkDebugUtilsMessageSeverityFlagsEXT, VkDebugUtilsMessageTypeFlagsEXT> getDebugCallbackFlagsFromLogLevelMask(const core::bitflag<system::ILogger::E_LOG_LEVEL> logLevelMask)
{
    std::pair<VkDebugUtilsMessageSeverityFlagsEXT, VkDebugUtilsMessageTypeFlagsEXT> result = {0, 0};
    auto& sev = result.first;
    auto& type = result.second;

    if((logLevelMask & system::ILogger::ELL_DEBUG).value)
    {
        type |= VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    }
    if((logLevelMask & system::ILogger::ELL_INFO).value)
    {
        sev |= (VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT);
        type |= VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
    }
    if((logLevelMask & system::ILogger::ELL_WARNING).value)
    {
        sev |= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
    }
    if((logLevelMask & system::ILogger::ELL_PERFORMANCE).value)
    {
        type |= VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    }
    if((logLevelMask & system::ILogger::ELL_ERROR).value)
    {
        sev |= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    }

    return result;
}

static inline VkBlendFactor getVkBlendFactorFromBlendFactor(const asset::E_BLEND_FACTOR in)
{
    return static_cast<VkBlendFactor>(in);
}

static inline VkBlendOp getVkBlendOpFromBlendOp(const asset::E_BLEND_OP in)
{
    return static_cast<VkBlendOp>(in);
}

static inline VkLogicOp getVkLogicOpFromLogicOp(const asset::E_LOGIC_OP in)
{
    return static_cast<VkLogicOp>(in);
}

static inline VkColorComponentFlags getVkColorComponentFlagsFromColorWriteMask(const uint64_t in)
{
    return static_cast<VkColorComponentFlags>(in);
}

}

#define __NBL_VIDEO_C_VULKAN_COMMON_H_INCLUDED__
#endif
