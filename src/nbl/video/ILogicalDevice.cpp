#include "nbl/video/IPhysicalDevice.h"

using namespace nbl;
using namespace nbl::video;


ILogicalDevice::ILogicalDevice(core::smart_refctd_ptr<IAPIConnection>&& api, IPhysicalDevice* physicalDevice, const SCreationParams& params)
    : m_api(api), m_physicalDevice(physicalDevice), m_enabledFeatures(params.featuresToEnable), m_compilerSet(params.compilerSet)
{
    uint32_t qcnt = 0u;
    uint32_t greatestFamNum = 0u;
    for (uint32_t i = 0u; i < params.queueParamsCount; ++i)
    {
        greatestFamNum = core::max(greatestFamNum,params.queueParams[i].familyIndex);
        qcnt += params.queueParams[i].count;
    }

    m_queues = core::make_refctd_dynamic_array<queues_array_t>(qcnt);
    m_queueFamilyInfos = core::make_refctd_dynamic_array<q_family_info_array_t>(greatestFamNum+1u);

    for (const auto& qci : core::SRange<const SQueueCreationParams>(params.queueParams,params.queueParams+params.queueParamsCount))
    {
        auto& info = const_cast<QueueFamilyInfo&>(m_queueFamilyInfos->operator[](qci.familyIndex));
        {
            using stage_flags_t = asset::PIPELINE_STAGE_FLAGS;
            info.supportedStages = stage_flags_t::HOST_BIT;

            const core::bitflag<stage_flags_t> transferStages = stage_flags_t::COPY_BIT|stage_flags_t::CLEAR_BIT|(m_enabledFeatures.accelerationStructure ? stage_flags_t::ACCELERATION_STRUCTURE_COPY_BIT:stage_flags_t::NONE)|stage_flags_t::RESOLVE_BIT|stage_flags_t::BLIT_BIT;
            const core::bitflag<stage_flags_t> computeAndGraphicsStages = (m_enabledFeatures.deviceGeneratedCommands ? stage_flags_t::COMMAND_PREPROCESS_BIT:stage_flags_t::NONE)|
                (m_enabledFeatures.conditionalRendering ? stage_flags_t::CONDITIONAL_RENDERING_BIT:stage_flags_t::NONE)|transferStages|stage_flags_t::DISPATCH_INDIRECT_COMMAND_BIT;

            const auto familyFlags = m_physicalDevice->getQueueFamilyProperties()[qci.familyIndex].queueFlags;
            if (familyFlags.hasFlags(IGPUQueue::FAMILY_FLAGS::COMPUTE_BIT))
            {
                info.supportedStages |= computeAndGraphicsStages|stage_flags_t::COMPUTE_SHADER_BIT;
                if (m_enabledFeatures.accelerationStructure)
                    info.supportedStages |= stage_flags_t::ACCELERATION_STRUCTURE_COPY_BIT|stage_flags_t::ACCELERATION_STRUCTURE_BUILD_BIT;
                if (m_enabledFeatures.rayTracingPipeline)
                    info.supportedStages |= stage_flags_t::RAY_TRACING_SHADER_BIT;
            }
            if (familyFlags.hasFlags(IGPUQueue::FAMILY_FLAGS::GRAPHICS_BIT))
            {
                info.supportedStages |= computeAndGraphicsStages|stage_flags_t::VERTEX_INPUT_BITS|stage_flags_t::VERTEX_SHADER_BIT;

                if (m_enabledFeatures.tessellationShader)
                    info.supportedStages |= stage_flags_t::TESSELLATION_CONTROL_SHADER_BIT|stage_flags_t::TESSELLATION_EVALUATION_SHADER_BIT;
                if (m_enabledFeatures.geometryShader)
                    info.supportedStages |= stage_flags_t::GEOMETRY_SHADER_BIT;
                // we don't do transform feedback
                //if (m_enabledFeatures.meshShader)
                //    info.supportedStages |= stage_flags_t::;
                //if (m_enabledFeatures.taskShader)
                //    info.supportedStages |= stage_flags_t::;
                if (m_enabledFeatures.fragmentDensityMap)
                    info.supportedStages |= stage_flags_t::FRAGMENT_DENSITY_PROCESS_BIT;
                //if (m_enabledFeatures.????)
                //    info.supportedStages |= stage_flags_t::SHADING_RATE_ATTACHMENT_BIT;

                info.supportedStages |= stage_flags_t::FRAMEBUFFER_SPACE_BITS;
            }
            if (familyFlags.hasFlags(IGPUQueue::FAMILY_FLAGS::TRANSFER_BIT))
                info.supportedStages |= transferStages;
        }
        {
            using access_flags_t = asset::ACCESS_FLAGS;
            info.supportedAccesses = access_flags_t::HOST_READ_BIT|access_flags_t::HOST_WRITE_BIT;
        }
        info.firstQueueIndex = qci.count;
    }
    // bothering with an `std::exclusive_scan` is a bit too cumbersome here
    uint32_t sum = 0u;
    for (auto i=0u; i<m_queueFamilyInfos->size(); i++)
    {
        auto& x = m_queueFamilyInfos->operator[](i).firstQueueIndex;
        auto tmp = sum+x;
        x = sum;
        sum = tmp;
    }
}

E_API_TYPE ILogicalDevice::getAPIType() const { return m_physicalDevice->getAPIType(); }

bool ILogicalDevice::supportsMask(const uint32_t queueFamilyIndex, core::bitflag<asset::PIPELINE_STAGE_FLAGS> stageMask) const
{
    if (queueFamilyIndex>m_queueFamilyInfos->size())
        return false;
    using q_family_flags_t = IGPUQueue::FAMILY_FLAGS;
    const auto& familyProps = m_physicalDevice->getQueueFamilyProperties()[queueFamilyIndex].queueFlags;
    // strip special values
    if (stageMask.hasFlags(asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS))
        return true;
    if (stageMask.hasFlags(asset::PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS) && bool(familyProps&(q_family_flags_t::COMPUTE_BIT|q_family_flags_t::GRAPHICS_BIT|q_family_flags_t::TRANSFER_BIT)))
        stageMask ^= asset::PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
    if (familyProps.hasFlags(q_family_flags_t::GRAPHICS_BIT))
    {
        if (stageMask.hasFlags(asset::PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS))
            stageMask ^= asset::PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS;
        if (stageMask.hasFlags(asset::PIPELINE_STAGE_FLAGS::PRE_RASTERIZATION_SHADERS_BITS))
            stageMask ^= asset::PIPELINE_STAGE_FLAGS::PRE_RASTERIZATION_SHADERS_BITS;
    }
    return getSupportedStageMask(queueFamilyIndex).hasFlags(stageMask);
}

bool ILogicalDevice::supportsMask(const uint32_t queueFamilyIndex, core::bitflag<asset::ACCESS_FLAGS> stageMask) const
{
    if (queueFamilyIndex>m_queueFamilyInfos->size())
        return false;
    using q_family_flags_t = IGPUQueue::FAMILY_FLAGS;
    const auto& familyProps = m_physicalDevice->getQueueFamilyProperties()[queueFamilyIndex].queueFlags;
    const bool shaderCapableFamily = bool(familyProps&(q_family_flags_t::COMPUTE_BIT|q_family_flags_t::GRAPHICS_BIT));
    // strip special values
    if (stageMask.hasFlags(asset::ACCESS_FLAGS::MEMORY_READ_BITS))
        stageMask ^= asset::ACCESS_FLAGS::MEMORY_READ_BITS;
    else if (stageMask.hasFlags(asset::ACCESS_FLAGS::SHADER_READ_BITS) && shaderCapableFamily)
        stageMask ^= asset::ACCESS_FLAGS::SHADER_READ_BITS;
    if (stageMask.hasFlags(asset::ACCESS_FLAGS::MEMORY_WRITE_BITS))
        stageMask ^= asset::ACCESS_FLAGS::MEMORY_WRITE_BITS;
    else if (stageMask.hasFlags(asset::ACCESS_FLAGS::SHADER_WRITE_BITS) && shaderCapableFamily)
        stageMask ^= asset::ACCESS_FLAGS::SHADER_WRITE_BITS;
    return getSupportedAccessMask(queueFamilyIndex).hasFlags(stageMask);
}

bool ILogicalDevice::validateMemoryBarrier(const uint32_t queueFamilyIndex, asset::SMemoryBarrier barrier) const
{
    if (!supportsMask(queueFamilyIndex,barrier.srcStageMask) || !supportsMask(queueFamilyIndex,barrier.dstStageMask))
        return false;
    if (!supportsMask(queueFamilyIndex,barrier.srcAccessMask) || !supportsMask(queueFamilyIndex,barrier.dstAccessMask))
        return false;

    using stage_flags_t = asset::PIPELINE_STAGE_FLAGS;
    const core::bitflag<stage_flags_t> supportedStageMask = getSupportedStageMask(queueFamilyIndex);
    using access_flags_t = asset::ACCESS_FLAGS;
    const core::bitflag<access_flags_t> supportedAccessMask = getSupportedAccessMask(queueFamilyIndex);
    auto validAccess = [supportedStageMask,supportedAccessMask](core::bitflag<stage_flags_t>& stageMask, core::bitflag<access_flags_t>& accessMask) -> bool
    {
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03916
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03917
        if (bool(accessMask&(access_flags_t::HOST_READ_BIT|access_flags_t::HOST_WRITE_BIT)) && !stageMask.hasFlags(stage_flags_t::HOST_BIT))
            return false;
        // this takes care of all stuff below
        if (stageMask.hasFlags(stage_flags_t::ALL_COMMANDS_BITS))
            return true;
        // first strip unsupported bits
        stageMask &= supportedStageMask;
        accessMask &= supportedAccessMask;
        // TODO: finish this stuff
        if (stageMask.hasFlags(stage_flags_t::ALL_GRAPHICS_BITS))
        {
            if (stageMask.hasFlags(stage_flags_t::ALL_TRANSFER_BITS))
            {
            }
            else
            {
            }
        }
        else
        {
            if (stageMask.hasFlags(stage_flags_t::ALL_TRANSFER_BITS))
            {
            }
            else
            {
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03914
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03915
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03927
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03928
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-06256
            }
            // this is basic valid usage stuff
            #ifdef _NBL_DEBUG
            // TODO:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03900
            if (accessMask.hasFlags(access_flags_t::INDIRECT_COMMAND_READ_BIT) && !bool(stageMask&(stage_flags_t::DISPATCH_INDIRECT_COMMAND_BIT|stage_flags_t::ACCELERATION_STRUCTURE_BUILD_BIT)))
                return false;
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03901
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03902
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03903
            //constexpr core::bitflag<stage_flags_t> ShaderStages = stage_flags_t::PRE_RASTERIZATION_SHADERS;
            //const bool noShaderStages = stageMask&ShaderStages;
            // TODO:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03904
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03905
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03906
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03907
            // IMPLICIT: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-07454
            // IMPLICIT: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03909
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-07272
            // TODO:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03910
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03911
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03912
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03913
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03918
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03919
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03924
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-03925
            #endif
        }
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkMemoryBarrier2-srcAccessMask-07457
        return true;
    };

    return true;
}


core::smart_refctd_ptr<IGPUDescriptorSetLayout> ILogicalDevice::createDescriptorSetLayout(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end)
{
    uint32_t dynamicSSBOCount=0u,dynamicUBOCount=0u;
    for (auto b=_begin; b!=_end; ++b)
    {
        if (b->type == asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC)
            dynamicSSBOCount++;
        else if (b->type == asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC)
            dynamicUBOCount++;
        else if (b->type == asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER && b->samplers)
        {
            auto* samplers = b->samplers;
            for (uint32_t i = 0u; i < b->count; ++i)
                if (!samplers[i]->wasCreatedBy(this))
                    return nullptr;
        }
    }
    const auto& limits = m_physicalDevice->getLimits();
    if (dynamicSSBOCount>limits.maxDescriptorSetDynamicOffsetSSBOs || dynamicUBOCount>limits.maxDescriptorSetDynamicOffsetUBOs)
        return nullptr;
    return createDescriptorSetLayout_impl(_begin,_end);
}

bool ILogicalDevice::updateDescriptorSets(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies)
{
    for (auto i = 0; i < descriptorWriteCount; ++i)
    {
        const auto& write = pDescriptorWrites[i];
        auto* ds = static_cast<IGPUDescriptorSet*>(write.dstSet);

        assert(ds->getLayout()->isCompatibleDevicewise(ds));

        if (!ds->validateWrite(write))
            return false;
    }

    for (auto i = 0; i < descriptorCopyCount; ++i)
    {
        const auto& copy = pDescriptorCopies[i];
        const auto* srcDS = static_cast<const IGPUDescriptorSet*>(copy.srcSet);
        const auto* dstDS = static_cast<IGPUDescriptorSet*>(copy.dstSet);

        if (!dstDS->isCompatibleDevicewise(srcDS))
            return false;

        if (!dstDS->validateCopy(copy))
            return false;
    }

    for (auto i = 0; i < descriptorWriteCount; ++i)
    {
        auto& write = pDescriptorWrites[i];
        auto* ds = static_cast<IGPUDescriptorSet*>(write.dstSet);
        ds->processWrite(write);
    }

    for (auto i = 0; i < descriptorCopyCount; ++i)
    {
        const auto& copy = pDescriptorCopies[i];
        auto* dstDS = static_cast<IGPUDescriptorSet*>(pDescriptorCopies[i].dstSet);
        dstDS->processCopy(copy);
    }

    updateDescriptorSets_impl(descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);

    return true;
}

void ILogicalDevice::addCommonShaderDefines(std::ostringstream& pool, const bool runningInRenderdoc)
{
    const auto& limits = m_physicalDevice->getProperties().limits;
    const auto& features = getEnabledFeatures();

    // TODO: NBL_GLSL needs to be renamed to something else when we support multiple shader languages

    // SPhysicalDeviceLimits
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_1D", limits.maxImageDimension1D);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_2D", limits.maxImageDimension2D);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_3D",limits.maxImageDimension3D);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_CUBE",limits.maxImageDimensionCube);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_IMAGE_ARRAY_LAYERS", limits.maxImageArrayLayers);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_BUFFER_VIEW_TEXELS", limits.maxBufferViewTexels);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_UBO_SIZE",limits.maxUBOSize);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SSBO_SIZE",limits.maxSSBOSize);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PUSH_CONSTANTS_SIZE", limits.maxPushConstantsSize);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_MEMORY_ALLOCATION_COUNT", limits.maxMemoryAllocationCount);
    // addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLER_ALLOCATION_COUNT",limits.maxSamplerAllocationCount); // shader doesn't need to know about that
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_BUFFER_IMAGE_GRANULARITY",core::min(limits.bufferImageGranularity, std::numeric_limits<int32_t>::max()));

    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_SAMPLERS", limits.maxPerStageDescriptorSamplers);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UBOS", limits.maxPerStageDescriptorUBOs);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_SSBOS",limits.maxPerStageDescriptorSSBOs);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_IMAGES", limits.maxPerStageDescriptorImages);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_STORAGE_IMAGES", limits.maxPerStageDescriptorStorageImages);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_INPUT_ATTACHMENTS",limits.maxPerStageDescriptorInputAttachments);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_RESOURCES",limits.maxPerStageResources);

    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_SAMPLERS",limits.maxDescriptorSetSamplers);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UBOS",limits.maxDescriptorSetUBOs);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_DYNAMIC_OFFSET_UBOS",limits.maxDescriptorSetDynamicOffsetUBOs);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_SSBOS",limits.maxDescriptorSetSSBOs);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_DYNAMIC_OFFSET_SSBOS",limits.maxDescriptorSetDynamicOffsetSSBOs);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_IMAGES",limits.maxDescriptorSetImages);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_STORAGE_IMAGES",limits.maxDescriptorSetStorageImages);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_INPUT_ATTACHMENTS",limits.maxDescriptorSetInputAttachments);

    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_GENERATION_LEVEL",limits.maxTessellationGenerationLevel);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_PATCH_SIZE",limits.maxTessellationPatchSize);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_CONTROL_PER_VERTEX_INPUT_COMPONENTS",limits.maxTessellationControlPerVertexInputComponents);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_CONTROL_PER_VERTEX_OUTPUT_COMPONENTS",limits.maxTessellationControlPerVertexOutputComponents);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_CONTROL_PER_PATCH_OUTPUT_COMPONENTS",limits.maxTessellationControlPerPatchOutputComponents);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_CONTROL_TOTAL_OUTPUT_COMPONENTS",limits.maxTessellationControlTotalOutputComponents);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_EVALUATION_INPUT_COMPONENTS",limits.maxTessellationEvaluationInputComponents);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_EVALUATION_OUTPUT_COMPONENTS",limits.maxTessellationEvaluationOutputComponents);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_SHADER_INVOCATIONS",limits.maxGeometryShaderInvocations);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_INPUT_COMPONENTS",limits.maxGeometryInputComponents);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_OUTPUT_COMPONENTS",limits.maxGeometryOutputComponents);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_OUTPUT_VERTICES",limits.maxGeometryOutputVertices);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS",limits.maxGeometryTotalOutputComponents);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAGMENT_INPUT_COMPONENTS",limits.maxFragmentInputComponents);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAGMENT_OUTPUT_ATTACHMENTS",limits.maxFragmentOutputAttachments);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAGMENT_COMBINED_OUTPUT_RESOURCES",limits.maxFragmentCombinedOutputResources);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAGMENT_DUAL_SRC_ATTACHMENTS",limits.maxFragmentDualSrcAttachments);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_SHARED_MEMORY_SIZE",limits.maxComputeSharedMemorySize);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_X",limits.maxComputeWorkGroupCount[0]);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Y",limits.maxComputeWorkGroupCount[1]);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Z",limits.maxComputeWorkGroupCount[2]);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_INVOCATIONS",limits.maxComputeWorkGroupInvocations);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_WORKGROUP_SIZE_X",limits.maxWorkgroupSize[0]);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_WORKGROUP_SIZE_Y",limits.maxWorkgroupSize[1]);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_WORKGROUP_SIZE_Z",limits.maxWorkgroupSize[2]);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SUB_PIXEL_PRECISION_BITS",limits.subPixelPrecisionBits);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DRAW_INDIRECT_COUNT",limits.maxDrawIndirectCount);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLER_LOD_BIAS",limits.maxSamplerLodBias);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLER_ANISOTROPY_LOG2",limits.maxSamplerAnisotropyLog2);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VIEWPORTS",limits.maxViewports);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VIEWPORT_DIMS_X",limits.maxViewportDims[0]);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VIEWPORT_DIMS_Y",limits.maxViewportDims[1]);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_VIEWPORT_BOUNDS_RANGE_BEGIN",limits.viewportBoundsRange[0]);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_VIEWPORT_BOUNDS_RANGE_END",limits.viewportBoundsRange[1]);
    
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_VIEWPORT_SUB_PIXEL_BITS",limits.viewportSubPixelBits);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_MEMORY_MAP_ALIGNMENT",core::min(limits.minMemoryMapAlignment, 1u << 30));
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_BUFFER_VIEW_ALIGNMENT",limits.bufferViewAlignment);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_UBO_ALIGNMENT",limits.minUBOAlignment);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_SSBO_ALIGNMENT",limits.minSSBOAlignment);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_TEXEL_OFFSET",limits.minTexelOffset);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TEXEL_OFFSET",limits.maxTexelOffset);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_TEXEL_GATHER_OFFSET",limits.minTexelGatherOffset);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TEXEL_GATHER_OFFSET",limits.maxTexelGatherOffset);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_INTERPOLATION_OFFSET",limits.minInterpolationOffset);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_INTERPOLATION_OFFSET",limits.maxInterpolationOffset);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAMEBUFFER_WIDTH",limits.maxFramebufferWidth);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAMEBUFFER_HEIGHT",limits.maxFramebufferHeight);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAMEBUFFER_LAYERS",limits.maxFramebufferLayers);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_FRAMEBUFFER_COLOR_SAMPLE_COUNTS",static_cast<uint32_t>(limits.framebufferColorSampleCounts.value));
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_FRAMEBUFFER_DEPTH_SAMPLE_COUNTS", static_cast<uint32_t>(limits.framebufferDepthSampleCounts.value));
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_FRAMEBUFFER_STENCIL_SAMPLE_COUNTS", static_cast<uint32_t>(limits.framebufferStencilSampleCounts.value));
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_FRAMEBUFFER_NO_ATTACHMENTS_SAMPLE_COUNTS", static_cast<uint32_t>(limits.framebufferNoAttachmentsSampleCounts.value));
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COLOR_ATTACHMENTS",limits.maxColorAttachments);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLED_IMAGE_COLOR_SAMPLE_COUNTS", static_cast<uint32_t>(limits.sampledImageColorSampleCounts.value));
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLED_IMAGE_INTEGER_SAMPLE_COUNTS", static_cast<uint32_t>(limits.sampledImageIntegerSampleCounts.value));
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLED_IMAGE_DEPTH_SAMPLE_COUNTS", static_cast<uint32_t>(limits.sampledImageDepthSampleCounts.value));
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLED_IMAGE_STENCIL_SAMPLE_COUNTS", static_cast<uint32_t>(limits.sampledImageStencilSampleCounts.value));
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_STORAGE_IMAGE_SAMPLE_COUNTS", static_cast<uint32_t>(limits.storageImageSampleCounts.value));
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLE_MASK_WORDS",limits.maxSampleMaskWords);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_TIMESTAMP_COMPUTE_AND_GRAPHICS",limits.timestampComputeAndGraphics);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_TIMESTAMP_PERIOD_IN_NANO_SECONDS",limits.timestampPeriodInNanoSeconds);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_CLIP_DISTANCES",limits.maxClipDistances);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_CULL_DISTANCES",limits.maxCullDistances);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMBINED_CLIP_AND_CULL_DISTANCES",limits.maxCombinedClipAndCullDistances);
    // addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_DISCRETE_QUEUE_PRIORITIES",limits.discreteQueuePriorities); // shader doesn't need to know about that

    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_POINT_SIZE",limits.pointSizeRange[0]);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_POINT_SIZE",limits.pointSizeRange[1]);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_LINE_WIDTH",limits.lineWidthRange[0]);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_LINE_WIDTH",limits.lineWidthRange[1]);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_POINT_SIZE_GRANULARITY",limits.pointSizeGranularity);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_LINE_WIDTH_GRANULARITY",limits.lineWidthGranularity);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_STRICT_LINES",limits.strictLines);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_STANDARD_SAMPLE_LOCATIONS",limits.standardSampleLocations);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_OPTIMAL_BUFFER_COPY_OFFSET_ALIGNMENT",core::min(limits.optimalBufferCopyOffsetAlignment, 1u << 30));
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_OPTIMAL_BUFFER_COPY_ROW_PITCH_ALIGNMENT",core::min(limits.optimalBufferCopyRowPitchAlignment, 1u << 30));
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_NON_COHERENT_ATOM_SIZE",core::min(limits.nonCoherentAtomSize, std::numeric_limits<int32_t>::max()));

    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VERTEX_OUTPUT_COMPONENTS",limits.maxVertexOutputComponents);
    
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SUBGROUP_SIZE",limits.subgroupSize);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SUBGROUP_OPS_SHADER_STAGES", static_cast<uint32_t>(limits.subgroupOpsShaderStages.value));
    if (limits.shaderSubgroupBasic) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_BASIC");
    if (limits.shaderSubgroupVote) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_VOTE");
    if (limits.shaderSubgroupArithmetic) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_ARITHMETIC");
    if (limits.shaderSubgroupBallot) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_BALLOT");
    if (limits.shaderSubgroupShuffle) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_SHUFFLE");
    if (limits.shaderSubgroupShuffleRelative) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_SHUFFLE_RELATIVE");
    if (limits.shaderSubgroupClustered) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_CLUSTERED");
    if (limits.shaderSubgroupQuad) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_QUAD");
    if (limits.shaderSubgroupQuadAllStages) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_QUAD_ALL_STAGES");

    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_POINT_CLIPPING_BEHAVIOR",(uint32_t)limits.pointClippingBehavior);
    
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_SET_DESCRIPTORS",limits.maxPerSetDescriptors);
    // addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_MEMORY_ALLOCATION_SIZE",limits.maxMemoryAllocationSize); // shader doesn't need to know about that

    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SIGNED_ZERO_INF_NAN_PRESERVE_FLOAT16",(uint32_t)limits.shaderSignedZeroInfNanPreserveFloat16);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SIGNED_ZERO_INF_NAN_PRESERVE_FLOAT32",(uint32_t)limits.shaderSignedZeroInfNanPreserveFloat32);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SIGNED_ZERO_INF_NAN_PRESERVE_FLOAT64",(uint32_t)limits.shaderSignedZeroInfNanPreserveFloat64);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_PRESERVE_FLOAT16",(uint32_t)limits.shaderDenormPreserveFloat16);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_PRESERVE_FLOAT32",(uint32_t)limits.shaderDenormPreserveFloat32);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_PRESERVE_FLOAT64",(uint32_t)limits.shaderDenormPreserveFloat64);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_FLUSH_TO_ZERO_FLOAT16",(uint32_t)limits.shaderDenormFlushToZeroFloat16);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_FLUSH_TO_ZERO_FLOAT32",(uint32_t)limits.shaderDenormFlushToZeroFloat32);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_FLUSH_TO_ZERO_FLOAT64",(uint32_t)limits.shaderDenormFlushToZeroFloat64);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTE_FLOAT16",(uint32_t)limits.shaderRoundingModeRTEFloat16);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTE_FLOAT32",(uint32_t)limits.shaderRoundingModeRTEFloat32);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTE_FLOAT64",(uint32_t)limits.shaderRoundingModeRTEFloat64);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTZ_FLOAT16",(uint32_t)limits.shaderRoundingModeRTZFloat16);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTZ_FLOAT32",(uint32_t)limits.shaderRoundingModeRTZFloat32);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTZ_FLOAT64",(uint32_t)limits.shaderRoundingModeRTZFloat64);

    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_SAMPLERS",limits.maxPerStageDescriptorUpdateAfterBindSamplers);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_UPDATE_AFTER_BIND_DESCRIPTORS_IN_ALL_POOLS",limits.maxUpdateAfterBindDescriptorsInAllPools);
    if (limits.shaderUniformBufferArrayNonUniformIndexingNative) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_UNIFORM_BUFFER_ARRAY_NON_UNIFORM_INDEXING_NATIVE");
    if (limits.shaderSampledImageArrayNonUniformIndexingNative) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING_NATIVE");
    if (limits.shaderStorageBufferArrayNonUniformIndexingNative) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING_NATIVE");
    if (limits.shaderStorageImageArrayNonUniformIndexingNative) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_STORAGE_IMAGE_ARRAY_NON_UNIFORM_INDEXING_NATIVE");
    if (limits.shaderInputAttachmentArrayNonUniformIndexingNative) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_INPUT_ATTACHMENT_ARRAY_NON_UNIFORM_INDEXING_NATIVE");
    if (limits.robustBufferAccessUpdateAfterBind) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_ROBUST_BUFFER_ACCESS_UPDATE_AFTER_BIND");
    if (limits.quadDivergentImplicitLod) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_QUAD_DIVERGENT_IMPLICIT_LOD");
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_SAMPLERS",limits.maxPerStageDescriptorUpdateAfterBindSamplers);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_UBOS",limits.maxPerStageDescriptorUpdateAfterBindUBOs);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_SSBOS",limits.maxPerStageDescriptorUpdateAfterBindSSBOs);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_IMAGES",limits.maxPerStageDescriptorUpdateAfterBindImages);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_STORAGE_IMAGES",limits.maxPerStageDescriptorUpdateAfterBindStorageImages);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_INPUT_ATTACHMENTS",limits.maxPerStageDescriptorUpdateAfterBindInputAttachments);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_UPDATE_AFTER_BIND_RESOURCES",limits.maxPerStageUpdateAfterBindResources);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_SAMPLERS",limits.maxDescriptorSetUpdateAfterBindSamplers);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_UBOS",limits.maxDescriptorSetUpdateAfterBindUBOs);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_DYNAMIC_OFFSET_UBOS",limits.maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_SSBOS",limits.maxDescriptorSetUpdateAfterBindSSBOs);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_DYNAMIC_OFFSET_SSBOS",limits.maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_IMAGES",limits.maxDescriptorSetUpdateAfterBindImages);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_STORAGE_IMAGES",limits.maxDescriptorSetUpdateAfterBindStorageImages);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_INPUT_ATTACHMENTS",limits.maxDescriptorSetUpdateAfterBindInputAttachments);

    if (limits.filterMinmaxSingleComponentFormats) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_FILTER_MINMAX_SINGLE_COMPONENT_FORMATS");
    if (limits.filterMinmaxImageComponentMapping) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_FILTER_MINMAX_IMAGE_COMPONENT_MAPPING");
    
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_SUBGROUP_SIZE",limits.minSubgroupSize);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SUBGROUP_SIZE",limits.maxSubgroupSize);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_SUBGROUPS",limits.maxComputeWorkgroupSubgroups);

    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_REQUIRED_SUBGROUP_SIZE_STAGES", static_cast<uint32_t>(limits.requiredSubgroupSizeStages.value));

    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_STORAGE_TEXEL_BUFFER_OFFSET_ALIGNMENT_BYTES",limits.minSubgroupSize);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_UNIFORM_TEXEL_BUFFER_OFFSET_ALIGNMENT_BYTES",limits.maxSubgroupSize);

    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_BUFFER_SIZE",core::min(limits.maxBufferSize, std::numeric_limits<int32_t>::max()));

    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_PRIMITIVE_OVERESTIMATION_SIZE", limits.primitiveOverestimationSize);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_EXTRA_PRIMITIVE_OVERESTIMATION_SIZE", limits.maxExtraPrimitiveOverestimationSize);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_EXTRA_PRIMITIVE_OVERESTIMATION_SIZE_GRANULARITY", limits.extraPrimitiveOverestimationSizeGranularity);
    if (limits.primitiveUnderestimation) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_PRIMITIVE_UNDERESTIMATION");
    if (limits.conservativePointAndLineRasterization) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_CONSERVATIVE_POINT_AND_LINE_RASTERIZATION");
    if (limits.degenerateTrianglesRasterized) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_DEGENERATE_TRIANGLES_RASTERIZED");
    if (limits.degenerateLinesRasterized) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_DEGENERATE_LINES_RASTERIZED");
    if (limits.fullyCoveredFragmentShaderInputVariable) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_FULLY_COVERED_FRAGMENT_SHADER_INPUT_VARIABLE");
    if (limits.conservativeRasterizationPostDepthCoverage) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_CONSERVATIVE_RASTERIZATION_POST_DEPTH_COVERAGE");

    // addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DISCARD_RECTANGLES",limits.maxDiscardRectangles); // shader doesn't need to know about
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_LINE_SUB_PIXEL_PRECISION_BITS",limits.lineSubPixelPrecisionBits);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VERTEX_ATTRIB_DIVISOR",limits.maxVertexAttribDivisor);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SUBPASS_SHADING_WORKGROUP_SIZE_ASPECT_RATIO",limits.maxSubpassShadingWorkgroupSizeAspectRatio);

    if (limits.integerDotProduct8BitUnsignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_8BIT_UNSIGNED_ACCELERATED");
    if (limits.integerDotProduct8BitSignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_8BIT_SIGNED_ACCELERATED");
    if (limits.integerDotProduct8BitMixedSignednessAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_8BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProduct4x8BitPackedUnsignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_4X_8BIT_PACKED_UNSIGNED_ACCELERATED");
    if (limits.integerDotProduct4x8BitPackedSignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_4X_8BIT_PACKED_SIGNED_ACCELERATED");
    if (limits.integerDotProduct4x8BitPackedMixedSignednessAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_4X_8BIT_PACKED_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProduct16BitUnsignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_16BIT_UNSIGNED_ACCELERATED");
    if (limits.integerDotProduct16BitSignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_16BIT_SIGNED_ACCELERATED");
    if (limits.integerDotProduct16BitMixedSignednessAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_16BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProduct32BitUnsignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_32BIT_UNSIGNED_ACCELERATED");
    if (limits.integerDotProduct32BitSignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_32BIT_SIGNED_ACCELERATED");
    if (limits.integerDotProduct32BitMixedSignednessAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_32BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProduct64BitUnsignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_64BIT_UNSIGNED_ACCELERATED");
    if (limits.integerDotProduct64BitSignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_64BIT_SIGNED_ACCELERATED");
    if (limits.integerDotProduct64BitMixedSignednessAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_64BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating8BitUnsignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_8BIT_UNSIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating8BitSignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_8BIT_SIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_8BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_4X_8BIT_PACKED_UNSIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_4X_8BIT_PACKED_SIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_4X_8BIT_PACKED_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating16BitUnsignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_16BIT_UNSIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating16BitSignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_16BIT_SIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_16BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating32BitUnsignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_32BIT_UNSIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating32BitSignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_32BIT_SIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_32BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating64BitUnsignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_64BIT_UNSIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating64BitSignedAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_64BIT_SIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_64BIT_MIXED_SIGNEDNESS_ACCELERATED");

    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_COUNT",core::min(limits.maxGeometryCount, std::numeric_limits<int32_t>::max()));
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_INSTANCE_COUNT",core::min(limits.maxInstanceCount, std::numeric_limits<int32_t>::max()));
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PRIMITIVE_COUNT",core::min(limits.maxPrimitiveCount, std::numeric_limits<int32_t>::max()));
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_ACCELERATION_STRUCTURES",limits.maxPerStageDescriptorAccelerationStructures);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_ACCELERATION_STRUCTURES",limits.maxPerStageDescriptorUpdateAfterBindAccelerationStructures);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_ACCELERATION_STRUCTURES",limits.maxDescriptorSetAccelerationStructures);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_ACCELERATION_STRUCTURES",limits.maxDescriptorSetUpdateAfterBindAccelerationStructures);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_ACCELERATION_STRUCTURE_SCRATCH_OFFSET_ALIGNMENT",limits.minAccelerationStructureScratchOffsetAlignment);

    if (limits.variableSampleLocations) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_VARIABLE_SAMPLE_LOCATIONS");
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLE_LOCATION_SUBPIXEL_BITS",limits.sampleLocationSubPixelBits);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLE_LOCATION_SAMPLE_COUNTS",static_cast<uint32_t>(limits.sampleLocationSampleCounts.value));
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLE_LOCATION_GRID_SIZE_X",limits.maxSampleLocationGridSize.width);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLE_LOCATION_GRID_SIZE_Y",limits.maxSampleLocationGridSize.height);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLE_LOCATION_COORDINATE_RANGE_X",limits.sampleLocationCoordinateRange[0]);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLE_LOCATION_COORDINATE_RANGE_Y",limits.sampleLocationCoordinateRange[1]);

    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_IMPORTED_HOST_POINTER_ALIGNMENT",core::min(limits.minImportedHostPointerAlignment, 1u << 30));

    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_FRAGMENT_DENSITY_TEXEL_SIZE_X",limits.minFragmentDensityTexelSize.width);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_FRAGMENT_DENSITY_TEXEL_SIZE_Y",limits.minFragmentDensityTexelSize.height);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAGMENT_DENSITY_TEXEL_SIZE_X",limits.maxFragmentDensityTexelSize.width);
    addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_MAX_FRAGMENT_DENSITY_TEXEL_SIZE_Y",limits.maxFragmentDensityTexelSize.height);
    if (limits.fragmentDensityInvocations) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_FRAGMENT_DENSITY_INVOCATIONS");

    if (limits.subsampledLoads) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SUBSAMPLED_LOADS");
    if (limits.subsampledCoarseReconstructionEarlyAccess) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SUBSAMPLED_COARSE_RECONSTRUCTION_EARLY_ACCESS");
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SUBSAMPLED_ARRAY_LAYERS",limits.maxSubsampledArrayLayers);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_SUBSAMPLED_SAMPLERS",limits.maxDescriptorSetSubsampledSamplers);

    // no need to know
    // addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_PCI_DOMAN",limits.pciDomain);
    // addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_PCI_BUS",limits.pciBus);
    // addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_PCI_DEVICE",limits.pciDevice);
    // addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_PCI_FUNCTION",limits.pciFunction);

    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_GROUP_HANDLE_SIZE",limits.shaderGroupHandleSize);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_RAY_RECURSION_DEPTH",limits.maxRayRecursionDepth);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SHADER_GROUP_STRIDE",limits.maxShaderGroupStride);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_GROUP_BASE_ALIGNMENT",limits.shaderGroupBaseAlignment);
    // addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_SIZE",limits.shaderGroupHandleCaptureReplaySize); // [DO NOT EXPOSE] for capture tools 
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_RAY_DISPATCH_INVOCATION_COUNT",limits.maxRayDispatchInvocationCount);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_GROUP_HANDLE_ALIGNMENT",limits.shaderGroupHandleAlignment);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_RAY_HIT_ATTRIBUTE_SIZE",limits.maxRayHitAttributeSize);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_COOPERATIVE_MATRIX_SUPPORTED_STAGES", static_cast<uint32_t>(limits.cooperativeMatrixSupportedStages.value));
  
    if (limits.shaderOutputViewportIndex) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_OUTPUT_VIEWPORT_INDEX");
    if (limits.shaderOutputLayer) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_OUTPUT_LAYER");
    if (limits.shaderIntegerFunctions2) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_INTEGER_FUNCTIONS_2");
    if (limits.shaderSubgroupClock) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_CLOCK");
    if (limits.imageFootprint) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_IMAGE_FOOTPRINT");
    // if (limits.texelBufferAlignment) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_TEXEL_BUFFER_ALIGNMENT"); // shader doesn't need to know about that
    if (limits.shaderSMBuiltins) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SM_BUILTINS");
    if (limits.shaderSubgroupPartitioned) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_PARTITIONED");
    if (limits.gcnShader) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_GCN_SHADER");
    if (limits.gpuShaderHalfFloat) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_GPU_SHADER_HALF_FLOAT");
    if (limits.shaderBallot) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_BALLOT");
    if (limits.shaderImageLoadStoreLod) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_IMAGE_LOAD_STORE_LOD");
    if (limits.shaderTrinaryMinmax) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_TRINARY_MINMAX");
    if (limits.postDepthCoverage) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_POST_DEPTH_COVERAGE");
    if (limits.shaderStencilExport) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_STENCIL_EXPORT");
    if (limits.decorateString) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_DECORATE_STRING");
    // if (limits.externalFence) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_EXTERNAL_FENCE"); // shader doesn't need to know about that
    // if (limits.externalMemory) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_EXTERNAL_MEMORY"); // shader doesn't need to know about that
    // if (limits.externalSemaphore) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_EXTERNAL_SEMAPHORE"); // shader doesn't need to know about that
    if (limits.shaderNonSemanticInfo) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_NON_SEMANTIC_INFO");
    if (limits.fragmentShaderBarycentric) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_FRAGMENT_SHADER_BARYCENTRIC");
    if (limits.geometryShaderPassthrough) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_GEOMETRY_SHADER_PASSTHROUGH");
    if (limits.viewportSwizzle) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_VIEWPORT_SWIZZLE");

    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_COMPUTE_UNITS",limits.computeUnits);
    if (limits.dispatchBase) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_DISPATCH_BASE");
    if (limits.allowCommandBufferQueryCopies) addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_ALLOW_COMMAND_BUFFER_QUERY_COPIES");
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_OPTIMALLY_RESIDENT_WORKGROUP_INVOCATIONS",limits.maxOptimallyResidentWorkgroupInvocations);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_RESIDENT_INVOCATIONS",limits.maxResidentInvocations);
    addShaderDefineToPool(pool,"NBL_GLSL_LIMIT_SPIRV_VERSION",(uint32_t)limits.spirvVersion);
    if (limits.vertexPipelineStoresAndAtomics) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_VERTEX_PIPELINE_STORES_AND_ATOMICS");
    if (limits.fragmentStoresAndAtomics) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_FRAGMENT_STORES_AND_ATOMICS");
    if (limits.shaderTessellationAndGeometryPointSize) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_TESSELLATION_AND_GEOMETRY_POINT_SIZE");
    if (limits.shaderImageGatherExtended) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_IMAGE_GATHER_EXTENDED");
    if (limits.shaderFloat64) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_FLOAT64");
    if (limits.shaderInt64) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_INT64");
    if (limits.shaderInt16) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_INT16");
    if (limits.storageBuffer16BitAccess) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_STORAGE_BUFFER_16BIT_ACCESS");
    if (limits.uniformAndStorageBuffer16BitAccess) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_UNIFORM_AND_STORAGE_BUFFER_16BIT_ACCESS");
    if (limits.storagePushConstant16) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_STORAGE_PUSH_CONSTANT_16");
    if (limits.storageInputOutput16) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_STORAGE_INPUT_OUTPUT_16");
    if (limits.variablePointers) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_VARIABLE_POINTERS");
    if (limits.storageBuffer8BitAccess) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_STORAGE_BUFFER_8BIT_ACCESS");
    if (limits.uniformAndStorageBuffer8BitAccess) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_UNIFORM_AND_STORAGE_BUFFER_8BIT_ACCESS");
    if (limits.storagePushConstant8) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_STORAGE_PUSH_CONSTANT_8");
    if (limits.shaderBufferInt64Atomics) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_BUFFER_INT64_ATOMICS");
    if (limits.shaderSharedInt64Atomics) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_SHARED_INT64_ATOMICS");
    if (limits.shaderFloat16) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_FLOAT16");
    if (limits.shaderInt8) addShaderDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_INT8");

    // SPhysicalDeviceFeatures
    if (features.robustBufferAccess) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_ROBUST_BUFFER_ACCESS");
    if (features.geometryShader) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_GEOMETRY_SHADER");
    if (features.tessellationShader) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_TESSELLATION_SHADER");
    if (features.dualSrcBlend) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DUAL_SRC_BLEND");
    if (features.logicOp) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_LOGIC_OP");
    if (features.fillModeNonSolid) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_FILL_MODE_NON_SOLID");
    if (features.depthBounds) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DEPTH_BOUNDS");
    if (features.wideLines) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_WIDE_LINES");
    if (features.largePoints) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_LARGE_POINTS");
    if (features.alphaToOne) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_ALPHA_TO_ONE");
    if (features.multiViewport) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_MULTI_VIEWPORT");
    // if (features.pipelineStatisticsQuery) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_PIPELINE_STATISTICS_QUERY"); // shader doesn't need to know about
    if (limits.shaderStorageImageMultisample) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_IMAGE_MULTISAMPLE");
    if (features.shaderStorageImageReadWithoutFormat) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_IMAGE_READ_WITHOUT_FORMAT");
    if (features.shaderStorageImageWriteWithoutFormat) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT");
    if (limits.shaderStorageImageArrayDynamicIndexing) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_IMAGE_ARRAY_DYNAMIC_INDEXING");
    if (features.shaderClipDistance) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_CLIP_DISTANCE");
    if (features.shaderCullDistance) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_CULL_DISTANCE");
    if (features.shaderResourceResidency) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_RESOURCE_RESIDENCY");
    if (features.shaderResourceMinLod) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_RESOURCE_MIN_LOD");
    if (features.variableMultisampleRate) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_VARIABLE_MULTISAMPLE_RATE");
    // if (features.inheritedQueries) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_INHERITED_QUERIES"); // shader doesn't need to know about
    if (features.shaderDrawParameters) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_DRAW_PARAMETERS");
    if (limits.drawIndirectCount) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DRAW_INDIRECT_COUNT");
    if (features.descriptorIndexing) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_INDEXING");
    if (features.shaderInputAttachmentArrayDynamicIndexing) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_INPUT_ATTACHMENT_ARRAY_DYNAMIC_INDEXING");
    if (features.shaderUniformTexelBufferArrayDynamicIndexing) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_UNIFORM_TEXEL_BUFFER_ARRAY_DYNAMIC_INDEXING");
    if (features.shaderStorageTexelBufferArrayDynamicIndexing) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_TEXEL_BUFFER_ARRAY_DYNAMIC_INDEXING");
    if (features.shaderUniformBufferArrayNonUniformIndexing) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_UNIFORM_BUFFER_ARRAY_NON_UNIFORM_INDEXING");
    if (features.shaderSampledImageArrayNonUniformIndexing) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING");
    if (features.shaderStorageBufferArrayNonUniformIndexing) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING");
    if (features.shaderStorageImageArrayNonUniformIndexing) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_IMAGE_ARRAY_NON_UNIFORM_INDEXING");
    if (features.shaderInputAttachmentArrayNonUniformIndexing) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_INPUT_ATTACHMENT_ARRAY_NON_UNIFORM_INDEXING");
    if (features.shaderUniformTexelBufferArrayNonUniformIndexing) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_UNIFORM_TEXEL_BUFFER_ARRAY_NON_UNIFORM_INDEXING");
    if (features.shaderStorageTexelBufferArrayNonUniformIndexing) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_TEXEL_BUFFER_ARRAY_NON_UNIFORM_INDEXING");
    if (features.descriptorBindingUniformBufferUpdateAfterBind) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_UNIFORM_BUFFER_UPDATE_AFTER_BIND");
    if (features.descriptorBindingSampledImageUpdateAfterBind) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_SAMPLED_IMAGE_UPDATE_AFTER_BIND");
    if (features.descriptorBindingStorageImageUpdateAfterBind) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_STORAGE_IMAGE_UPDATE_AFTER_BIND");
    if (features.descriptorBindingStorageBufferUpdateAfterBind) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_STORAGE_BUFFER_UPDATE_AFTER_BIND");
    if (features.descriptorBindingUniformTexelBufferUpdateAfterBind) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_UNIFORM_TEXEL_BUFFER_UPDATE_AFTER_BIND");
    if (features.descriptorBindingStorageTexelBufferUpdateAfterBind) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_STORAGE_TEXEL_BUFFER_UPDATE_AFTER_BIND");
    // if (features.descriptorBindingUpdateUnusedWhilePending) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING"); // shader doesn't need to know about
    if (features.descriptorBindingPartiallyBound) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_PARTIALLY_BOUND");
    if (features.descriptorBindingVariableDescriptorCount) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT");
    if (features.runtimeDescriptorArray) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_RUNTIME_DESCRIPTOR_ARRAY");
    if (features.samplerFilterMinmax) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SAMPLER_FILTER_MINMAX");
    if (features.bufferDeviceAddress) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_BUFFER_DEVICE_ADDRESS");
    if (features.bufferDeviceAddressMultiDevice) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_BUFFER_DEVICE_ADDRESS_MULTI_DEVICE");
    if (limits.vulkanMemoryModel) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_VULKAN_MEMORY_MODEL");
    if (limits.vulkanMemoryModelDeviceScope) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE");
    if (limits.vulkanMemoryModelAvailabilityVisibilityChains) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_VULKAN_MEMORY_MODEL_AVAILABILITY_VISIBILITY_CHAINS");
    if (features.shaderDemoteToHelperInvocation) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_DEMOTE_TO_HELPER_INVOCATION");
    if (features.shaderTerminateInvocation) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_TERMINATE_INVOCATION");
    if (features.subgroupSizeControl) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SUBGROUP_SIZE_CONTROL");
    if (features.computeFullSubgroups) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_COMPUTE_FULL_SUBGROUPS");
    if (features.shaderIntegerDotProduct) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_INTEGER_DOT_PRODUCT");
    if (features.rasterizationOrderColorAttachmentAccess) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_RASTERIZATION_ORDER_COLOR_ATTACHMENT_ACCESS");
    if (features.rasterizationOrderDepthAttachmentAccess) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_RASTERIZATION_ORDER_DEPTH_ATTACHMENT_ACCESS");
    if (features.rasterizationOrderStencilAttachmentAccess) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_RASTERIZATION_ORDER_STENCIL_ATTACHMENT_ACCESS");
    if (features.fragmentShaderSampleInterlock) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_SHADER_SAMPLE_INTERLOCK");
    if (features.fragmentShaderPixelInterlock) addShaderDefineToPool(pool, "NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK");
    if (features.fragmentShaderShadingRateInterlock) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_SHADER_SHADING_RATE_INTERLOCK");
    if (features.indexTypeUint8) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_INDEX_TYPE_UINT8");
    if (features.shaderBufferFloat32Atomics) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT32_ATOMICS");
    if (features.shaderBufferFloat32AtomicAdd) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT32_ATOMIC_ADD");
    if (features.shaderBufferFloat64Atomics) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT64_ATOMICS");
    if (features.shaderBufferFloat64AtomicAdd) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT64_ATOMIC_ADD");
    if (features.shaderSharedFloat32Atomics) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT32_ATOMICS");
    if (features.shaderSharedFloat32AtomicAdd) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT32_ATOMIC_ADD");
    if (features.shaderSharedFloat64Atomics) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT64_ATOMICS");
    if (features.shaderSharedFloat64AtomicAdd) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT64_ATOMIC_ADD");
    if (features.shaderImageFloat32Atomics) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_IMAGE_FLOAT32_ATOMICS");
    if (features.shaderImageFloat32AtomicAdd) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_IMAGE_FLOAT32_ATOMIC_ADD");
    if (features.sparseImageFloat32Atomics) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SPARSE_IMAGE_FLOAT32_ATOMICS");
    if (features.sparseImageFloat32AtomicAdd) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SPARSE_IMAGE_FLOAT32_ATOMIC_ADD");
    if (features.shaderBufferFloat16Atomics) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT16_ATOMICS");
    if (features.shaderBufferFloat16AtomicAdd) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT16_ATOMIC_ADD");
    if (features.shaderBufferFloat16AtomicMinMax) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT16_ATOMIC_MIN_MAX");
    if (features.shaderBufferFloat32AtomicMinMax) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT32_ATOMIC_MIN_MAX");
    if (features.shaderBufferFloat64AtomicMinMax) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT64_ATOMIC_MIN_MAX");
    if (features.shaderSharedFloat16Atomics) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT16_ATOMICS");
    if (features.shaderSharedFloat16AtomicAdd) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT16_ATOMIC_ADD");
    if (features.shaderSharedFloat16AtomicMinMax) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT16_ATOMIC_MIN_MAX");
    if (features.shaderSharedFloat32AtomicMinMax) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT32_ATOMIC_MIN_MAX");
    if (features.shaderSharedFloat64AtomicMinMax) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT64_ATOMIC_MIN_MAX");
    if (features.shaderImageFloat32AtomicMinMax) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_IMAGE_FLOAT32_ATOMIC_MIN_MAX");
    if (features.sparseImageFloat32AtomicMinMax) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SPARSE_IMAGE_FLOAT32_ATOMIC_MIN_MAX");
    if (features.shaderImageInt64Atomics) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_IMAGE_INT64_ATOMICS");
    if (features.sparseImageInt64Atomics) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SPARSE_IMAGE_INT64_ATOMICS");
    if (features.accelerationStructure) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_ACCELERATION_STRUCTURE");
    if (features.accelerationStructureIndirectBuild) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_ACCELERATION_STRUCTURE_INDIRECT_BUILD");
    if (features.accelerationStructureHostCommands) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_ACCELERATION_STRUCTURE_HOST_COMMANDS");
    // if (features.descriptorBindingAccelerationStructureUpdateAfterBind) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_ACCELERATION_STRUCTURE_UPDATE_AFTER_BIND"); // shader doesn't need to know about
    if (features.rayQuery) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_QUERY");
    if (features.rayTracingPipeline) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_TRACING_PIPELINE");
    if (features.rayTracingPipelineTraceRaysIndirect) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_TRACING_PIPELINE_TRACE_RAYS_INDIRECT");
    if (features.rayTraversalPrimitiveCulling) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_TRAVERSAL_PRIMITIVE_CULLING");
    if (features.shaderDeviceClock) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_DEVICE_CLOCK");
    if (features.shaderSubgroupUniformControlFlow) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW");
    if (features.workgroupMemoryExplicitLayout) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT");
    if (features.workgroupMemoryExplicitLayoutScalarBlockLayout) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_SCALAR_BLOCK_LAYOUT");
    if (features.workgroupMemoryExplicitLayout8BitAccess) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_8BIT_ACCESS");
    if (features.workgroupMemoryExplicitLayout16BitAccess) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_16BIT_ACCESS");
    if (features.computeDerivativeGroupQuads) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_COMPUTE_DERIVATIVE_GROUP_QUADS");
    if (features.computeDerivativeGroupLinear) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_COMPUTE_DERIVATIVE_GROUP_LINEAR");
    if (features.cooperativeMatrix) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_COOPERATIVE_MATRIX");
    if (features.cooperativeMatrixRobustBufferAccess) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_COOPERATIVE_MATRIX_ROBUST_BUFFER_ACCESS");
    if (features.rayTracingMotionBlur) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_TRACING_MOTION_BLUR");
    if (features.rayTracingMotionBlurPipelineTraceRaysIndirect) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_TRACING_MOTION_BLUR_PIPELINE_TRACE_RAYS_INDIRECT");
    if (features.coverageReductionMode) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_COVERAGE_REDUCTION_MODE");
    if (features.deviceGeneratedCommands) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DEVICE_GENERATED_COMMANDS");
    if (features.taskShader) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_TASK_SHADER");
    if (features.meshShader) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_MESH_SHADER");
    if (features.representativeFragmentTest) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_REPRESENTATIVE_FRAGMENT_TEST");
    if (features.mixedAttachmentSamples) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_MIXED_ATTACHMENT_SAMPLES");
    if (features.hdrMetadata) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_HDR_METADATA");
    // if (features.displayTiming) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DISPLAY_TIMING"); // shader doesn't need to know about
    if (features.rasterizationOrder) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_RASTERIZATION_ORDER");
    if (features.shaderExplicitVertexParameter) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_EXPLICIT_VERTEX_PARAMETER");
    if (features.shaderInfoAMD) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_INFO_AMD");
    // if (features.pipelineCreationCacheControl) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_PIPELINE_CREATION_CACHE_CONTROL"); // shader doesn't need to know about
    if (features.colorWriteEnable) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_COLOR_WRITE_ENABLE");
    if (features.conditionalRendering) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_CONDITIONAL_RENDERING");
    if (features.inheritedConditionalRendering) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_INHERITED_CONDITIONAL_RENDERING");
    // if (features.deviceMemoryReport) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DEVICE_MEMORY_REPORT"); // shader doesn't need to know about
    if (features.fragmentDensityMap) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_DENSITY_MAP");
    if (features.fragmentDensityMapDynamic) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_DENSITY_MAP_DYNAMIC");
    if (features.fragmentDensityMapNonSubsampledImages) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_DENSITY_MAP_NON_SUBSAMPLED_IMAGES");
    if (features.fragmentDensityMapDeferred) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_DENSITY_MAP_DEFERRED");
    if (features.robustImageAccess) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_ROBUST_IMAGE_ACCESS");
    if (features.inlineUniformBlock) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_INLINE_UNIFORM_BLOCK");
    // if (features.descriptorBindingInlineUniformBlockUpdateAfterBind) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_INLINE_UNIFORM_BLOCK_UPDATE_AFTER_BIND"); // shader doesn't need to know about
    if (features.rectangularLines) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_RECTANGULAR_LINES");
    if (features.bresenhamLines) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_BRESENHAM_LINES");
    if (features.smoothLines) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_SMOOTH_LINES");
    if (features.stippledRectangularLines) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_STIPPLED_RECTANGULAR_LINES");
    if (features.stippledBresenhamLines) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_STIPPLED_BRESENHAM_LINES");
    if (features.stippledSmoothLines) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_STIPPLED_SMOOTH_LINES");
    // if (features.memoryPriority) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_MEMORY_PRIORITY"); // shader doesn't need to know about
    if (features.robustBufferAccess2) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_ROBUST_BUFFER_ACCESS_2");
    if (features.robustImageAccess2) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_ROBUST_IMAGE_ACCESS_2");
    if (features.nullDescriptor) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_NULL_DESCRIPTOR");
    if (features.performanceCounterQueryPools) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_PERFORMANCE_COUNTER_QUERY_POOLS");
    if (features.performanceCounterMultipleQueryPools) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_PERFORMANCE_COUNTER_MULTIPLE_QUERY_POOLS");
    if (features.pipelineExecutableInfo) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_PIPELINE_EXECUTABLE_INFO");
    // if (features.maintenance4) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_MAINTENANCE_4"); // shader doesn't need to know about
    if (features.deviceCoherentMemory) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_DEVICE_COHERENT_MEMORY");
    // if (features.bufferMarkerAMD) addShaderDefineToPool(pool, "NBL_GLSL_FEATURE_BUFFER_MARKER_AMD"); // shader doesn't need to know about

    // TODO: @achal test examples 14 and 48 on all APIs and GPUs

    if (runningInRenderdoc)
        addShaderDefineToPool(pool,"NBL_GLSL_RUNNING_IN_RENDERDOC");
}
