#ifndef __NBL_VIDEO_C_VULKAN_DESCRIPTOR_SET_H_INCLUDED__

#include "nbl/video/IGPUDescriptorSet.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;
class CVulkanDescriptorPool;

class CVulkanDescriptorSet : public IGPUDescriptorSet
{
public:
    CVulkanDescriptorSet(ILogicalDevice* dev,
        core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout,
        core::smart_refctd_ptr<const CVulkanDescriptorPool> parentPool, VkDescriptorSet descriptorSet)
        : IGPUDescriptorSet(dev, std::move(layout)), m_parentPool(parentPool),
        m_descriptorSet(descriptorSet)
    {}

    inline VkDescriptorSet getInternalObject() const { return m_descriptorSet; }

private:
    core::smart_refctd_ptr<const CVulkanDescriptorPool> m_parentPool = nullptr;
    VkDescriptorSet m_descriptorSet;
};

}

#define __NBL_VIDEO_C_VULKAN_DESCRIPTOR_SET_H_INCLUDED__
#endif