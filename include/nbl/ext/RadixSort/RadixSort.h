#ifndef _NBL_EXT_RADIXSORT_INCLUDED_

#include "nbl/system/declarations.h"
#include "nabla.h"
#include "nbl/video/utilities/CScanner.h"

namespace nbl {
    namespace ext {
        namespace RadixSort {
            typedef uint32_t uint; // this is mandatory before inclusion of glsl file

#include "nbl/builtin/glsl/ext/RadixSort/parameters_struct.glsl"

class NBL_API RadixSort final : public core::IReferenceCounted {
public:
    static inline const uint32_t DEFAULT_WORKGROUP_SIZE = 256u;
    static inline const uint32_t BITS_PER_PASS = 4u;
    static inline const uint32_t PASS_COUNT = 32u / BITS_PER_PASS;
    static inline const uint32_t BUCKETS_COUNT = 1 << BITS_PER_PASS;

    enum E_SHADER_TYPE : uint8_t {
        ESHT_HISTOGRAM = 0,
        ESHT_SCATTER = 1
    };

    typedef nbl_glsl_ext_RadixSort_Parameters_t Parameters_t;

    struct DispatchInfo_t {
        uint32_t wg_count[3]; // 3, one for each radix sort stage
    };

    const uint32_t m_wg_size;
    const uint32_t m_element_count;

    RadixSort(video::ILogicalDevice *device,
              const uint32_t wg_size, const uint32_t element_count,
              core::smart_refctd_ptr <video::IGPUDescriptorSetLayout> &scanDSLayout,
              core::smart_refctd_ptr <video::IGPUComputePipeline> &scan_pipeline);

    static void sort(video::ILogicalDevice *device, video::IGPUCommandBuffer *cmdbuf, video::CScanner *scanner,
                     video::IGPUComputePipeline *histogram,
                     video::IGPUComputePipeline *scan,
                     video::IGPUComputePipeline *scatter,
                     core::smart_refctd_ptr <video::IGPUDescriptorSet> *ds_sort,
                     core::smart_refctd_ptr <video::IGPUDescriptorSet> *ds_scan,
                     Parameters_t *sort_push_constants,
                     DispatchInfo_t *sort_dispatch_info,
                     video::CScanner::DefaultPushConstants *scan_push_constants,
                     video::CScanner::DispatchInfo *scan_dispatch_info,
                     asset::SBufferRange <video::IGPUBuffer>& input_sort_range,
                     asset::SBufferRange <video::IGPUBuffer> &scratch_sort_range,
                     asset::SBufferRange <video::IGPUBuffer> &histogram_range,
                     asset::SBufferRange <video::IGPUBuffer> &scratch_scan_range,
                     asset::E_PIPELINE_STAGE_FLAGS start_mask, asset::E_PIPELINE_STAGE_FLAGS end_mask);

    inline auto getDefaultScanDescriptorSetLayout() const { return m_scan_ds_layout.get(); }

    inline auto getDefaultSortDescriptorSetLayout() const { return m_sort_ds_layout.get(); }

    inline auto getDefaultPipelineLayout() const { return m_pipeline_layout.get(); }

    inline auto getDefaultHistogramPipeline() const { return m_histogram_pipeline.get(); }

    inline auto getDefaultScanPipeline() const { return m_scan_pipeline.get(); }

    inline auto getDefaultScatterPipeline() const { return m_scatter_pipeline.get(); }

    static inline void
    dispatchHelper(video::IGPUCommandBuffer *cmdbuf, const video::IGPUPipelineLayout *pipeline_layout,
                   const Parameters_t &params, const DispatchInfo_t &dispatch_info,
                   const asset::E_PIPELINE_STAGE_FLAGS srcStageMask, const uint32_t srcBufferBarrierCount,
                   const video::IGPUCommandBuffer::SBufferMemoryBarrier *srcBufferBarriers,
                   const asset::E_PIPELINE_STAGE_FLAGS dstStageMask, const uint32_t dstBufferBarrierCount,
                   const video::IGPUCommandBuffer::SBufferMemoryBarrier *dstBufferBarriers) {
      // Since we're using a single pc range we need to update this for both the radix sort exclusive pipelines (histogram and scatter)
      cmdbuf->pushConstants(pipeline_layout, asset::IShader::ESS_COMPUTE, 0u, sizeof(Parameters_t), &params);
      if (srcStageMask != asset::E_PIPELINE_STAGE_FLAGS::EPSF_TOP_OF_PIPE_BIT && srcBufferBarrierCount)
        cmdbuf->pipelineBarrier(
            srcStageMask, asset::EPSF_COMPUTE_SHADER_BIT,
            asset::EDF_NONE, 0u, nullptr, srcBufferBarrierCount, srcBufferBarriers,
            0u, nullptr
        );
      cmdbuf->dispatch(dispatch_info.wg_count[0], 1, 1);
      if (dstStageMask != asset::E_PIPELINE_STAGE_FLAGS::EPSF_BOTTOM_OF_PIPE_BIT && dstBufferBarrierCount)
        cmdbuf->pipelineBarrier(
            asset::EPSF_COMPUTE_SHADER_BIT, dstStageMask,
            asset::EDF_NONE, 0u, nullptr, dstBufferBarrierCount, dstBufferBarriers, 0u, nullptr);
    }

    // Returns the total number of passes required by scan, since the total number of passes required by the radix sort
    // is always constant and is given by `PASS_COUNT` constant above
    static inline uint32_t buildParameters(const uint32_t in_count, const uint32_t wg_size,
                                           Parameters_t *sort_push_constants, DispatchInfo_t *sort_dispatch_info) {
      const uint32_t wg_count = (in_count + wg_size - 1) / wg_size;
      const uint32_t histogram_count = wg_count * BUCKETS_COUNT;

      sort_dispatch_info[0] = {{wg_count, 1, 1}};
      for (uint32_t pass = 0; pass < PASS_COUNT; ++pass) {
        sort_push_constants[pass].shift = BITS_PER_PASS * pass;
        sort_push_constants[pass].element_count_total = in_count;
      }

      return histogram_count;
    }

    static inline void updateDescriptorSetsPingPong(core::smart_refctd_ptr <video::IGPUDescriptorSet> *pingpong_sets,
                                                    const asset::SBufferRange <video::IGPUBuffer> &range_zero,
                                                    const asset::SBufferRange <video::IGPUBuffer> &range_one, video::ILogicalDevice *device) {
      const uint32_t count = 2u;
      asset::SBufferRange <video::IGPUBuffer> ranges[count];
      for (uint32_t i = 0; i < 2u; ++i) {
        if (i == 0) {
          ranges[0] = range_zero;
          ranges[1] = range_one;
        }
        else {
          ranges[0] = range_one;
          ranges[1] = range_zero;
        }
        updateDescriptorSet(device, pingpong_sets[i].get(), ranges, count);
      }
    }

    static inline void
    updateDescriptorSet(video::ILogicalDevice *device, video::IGPUDescriptorSet *ds, const asset::SBufferRange <video::IGPUBuffer> *descriptor_ranges,
                        const uint32_t count) {
      constexpr uint32_t MAX_DESCRIPTOR_COUNT = 2u;
      assert(count <= MAX_DESCRIPTOR_COUNT);

      video::IGPUDescriptorSet::SDescriptorInfo ds_info[MAX_DESCRIPTOR_COUNT];
      video::IGPUDescriptorSet::SWriteDescriptorSet writes[MAX_DESCRIPTOR_COUNT];

      for (uint32_t i = 0; i < count; ++i) {
        ds_info[i].desc = descriptor_ranges[i].buffer;
        ds_info[i].buffer = {descriptor_ranges[i].offset, descriptor_ranges[i].size};

        writes[i] = {ds, i, 0u, 1u, asset::EDT_STORAGE_BUFFER, ds_info + i};
      }

      device->updateDescriptorSets(count, writes, 0u, nullptr);
    }

    inline asset::ICPUShader *getDefaultShader(video::ILogicalDevice *device, E_SHADER_TYPE type) {
      core::smart_refctd_ptr <asset::ICPUShader> shader;
      if (type == E_SHADER_TYPE::ESHT_HISTOGRAM && !m_histogram_shader) {
        m_histogram_shader = createShader(device, type);
        shader = m_histogram_shader;
      }
      else if (type == E_SHADER_TYPE::ESHT_SCATTER && !m_scatter_shader) {
        m_scatter_shader = createShader(device, type);
        shader = m_scatter_shader;
      }

      shader.get()->setShaderStage(asset::IShader::ESS_COMPUTE);

      return shader.get();
    }

    //
    inline core::smart_refctd_ptr <video::IGPUSpecializedShader> getDefaultSpecializedShader(video::ILogicalDevice *device, E_SHADER_TYPE type) {
      if (type == E_SHADER_TYPE::ESHT_HISTOGRAM && !m_specialized_hist_shader) {
        auto cpuShader = core::smart_refctd_ptr<asset::ICPUShader>(getDefaultShader(device, type));
        auto gpushader = device->createShader(std::move(cpuShader));
        return device->createSpecializedShader(gpushader.get(), {nullptr, nullptr, "main"});
      }
      else if (type == E_SHADER_TYPE::ESHT_SCATTER && !m_specialized_scatter_shader) {
        auto cpuShader = core::smart_refctd_ptr<asset::ICPUShader>(getDefaultShader(device, type));
        auto gpushader = device->createShader(std::move(cpuShader));
        return device->createSpecializedShader(gpushader.get(), {nullptr, nullptr, "main"});
      }
    }

private:
    ~RadixSort() {}

    core::smart_refctd_ptr <video::IGPUDescriptorSetLayout> m_scan_ds_layout = nullptr;
    core::smart_refctd_ptr <video::IGPUDescriptorSetLayout> m_sort_ds_layout = nullptr;

    core::smart_refctd_ptr <video::IGPUPipelineLayout> m_pipeline_layout = nullptr;

    core::smart_refctd_ptr <video::IGPUComputePipeline> m_histogram_pipeline = nullptr;
    core::smart_refctd_ptr <video::IGPUComputePipeline> m_scan_pipeline = nullptr;
    core::smart_refctd_ptr <video::IGPUComputePipeline> m_scatter_pipeline = nullptr;

    core::smart_refctd_ptr <asset::ICPUShader> m_histogram_shader;
    core::smart_refctd_ptr <asset::ICPUShader> m_scatter_shader;
    core::smart_refctd_ptr <video::IGPUSpecializedShader> m_specialized_hist_shader;
    core::smart_refctd_ptr <video::IGPUSpecializedShader> m_specialized_scatter_shader;

    core::smart_refctd_ptr <asset::ICPUShader>
    createShader(video::ILogicalDevice *device, E_SHADER_TYPE type);
};

}
}
}

#define _NBL_EXT_RADIXSORT_INCLUDED_
#endif
