#include "nbl/ext/RadixSort/RadixSort.h"

namespace nbl {
    namespace ext {
        namespace RadixSort {

            RadixSort::RadixSort(video::ILogicalDevice *device,
                                 const uint32_t wg_size, const uint32_t element_count,
                                 core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> &scanDSLayout,
                                 core::smart_refctd_ptr<video::IGPUComputePipeline> &scan_pipeline)
                : m_wg_size(wg_size), m_element_count(element_count), m_scan_ds_layout(scanDSLayout), m_scan_pipeline(scan_pipeline) {

              assert(nbl::core::isPoT(m_wg_size));

              const asset::SPushConstantRange pc_range = {
                  asset::IShader::ESS_COMPUTE, 0u,
                  static_cast<uint32_t>(core::max(sizeof(Parameters_t), sizeof(video::CScanner::Parameters)))
              };

              {
                const uint32_t count = 2u;
                video::IGPUDescriptorSetLayout::SBinding binding[count];
                for (uint32_t i = 0; i < count; ++i)
                  binding[i] = {i, asset::EDT_STORAGE_BUFFER, 1u, asset::IShader::ESS_COMPUTE, nullptr};
                m_sort_ds_layout = device->createDescriptorSetLayout(binding, binding + count);
              }

              m_pipeline_layout = device->createPipelineLayout(&pc_range, &pc_range + 1,
                                                               core::smart_refctd_ptr(m_scan_ds_layout), core::smart_refctd_ptr(m_sort_ds_layout));

              m_histogram_pipeline = device->createComputePipeline(
                  nullptr, std::move(core::smart_refctd_ptr(m_pipeline_layout)),
                  std::move(getDefaultSpecializedShader(device, E_SHADER_TYPE::ESHT_HISTOGRAM))
              );

              m_scatter_pipeline = device->createComputePipeline(
                  nullptr, std::move(core::smart_refctd_ptr(m_pipeline_layout)),
                  std::move(getDefaultSpecializedShader(device, E_SHADER_TYPE::ESHT_SCATTER))
              );
            }

            void RadixSort::sort(video::ILogicalDevice *device, video::IGPUCommandBuffer *cmdbuf, video::CScanner *scanner,
                                 video::IGPUComputePipeline *histogram,
                                 video::IGPUComputePipeline *scan,
                                 video::IGPUComputePipeline *scatter,
                                 core::smart_refctd_ptr<video::IGPUDescriptorSet> *ds_sort,
                                 core::smart_refctd_ptr<video::IGPUDescriptorSet> *ds_scan,
                                 Parameters_t *sort_push_constants,
                                 DispatchInfo_t *sort_dispatch_info,
                                 video::CScanner::DefaultPushConstants *scan_push_constants,
                                 video::CScanner::DispatchInfo *scan_dispatch_info,
                                 asset::SBufferRange <video::IGPUBuffer> &input_sort_range,
                                 asset::SBufferRange <video::IGPUBuffer> &scratch_sort_range,
                                 asset::SBufferRange <video::IGPUBuffer> &histogram_range,
                                 asset::SBufferRange <video::IGPUBuffer> &scratch_scan_range,
                                 asset::E_PIPELINE_STAGE_FLAGS start_mask, asset::E_PIPELINE_STAGE_FLAGS end_mask) {
              // (Penta): This function must record all passes to the command buffer and the buffer itself must be submitted only once.
              // Due to the multi-pass nature of the algorithm, it is expected to take the results of the each pass into the next one
              // but this is avoided by interchanging the input and scratch buffers for each pass (there are 2 descriptor sets, one
              // where the input buffer is used for input and one where the scratch buffer is used for input). This way we don't need
              // to copy the scratch buffer results to the input for the next pass, which would require a fence and multiple submissions,
              // one for each pass.

              for (uint32_t pass = 0; pass < PASS_COUNT; ++pass) {
//                uint32_t pass = 0;
                cmdbuf->fillBuffer(histogram_range.buffer.get(), 0u, /*sizeof(uint32_t) + histogram_range.size / 2u*/histogram_range.size / 4u, 0u);
                cmdbuf->fillBuffer(scratch_scan_range.buffer.get(), 0u, /*sizeof(uint32_t) + scratch_scan_range.size / 2u*/scratch_scan_range.size/ 4u, 0u);

                video::IGPUCommandBuffer::SBufferMemoryBarrier scanDstBufBarrier;
                {
                  scanDstBufBarrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
                  scanDstBufBarrier.dstQueueFamilyIndex = scanDstBufBarrier.srcQueueFamilyIndex = cmdbuf->getQueueFamilyIndex();
                  scanDstBufBarrier.buffer = scratch_scan_range.buffer;
                  scanDstBufBarrier.offset = scratch_scan_range.offset;
                  scanDstBufBarrier.size = scratch_scan_range.size;
                }
                video::IGPUCommandBuffer::SBufferMemoryBarrier histogramDstBufBarrier;
                {
                  histogramDstBufBarrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
                  histogramDstBufBarrier.dstQueueFamilyIndex = histogramDstBufBarrier.srcQueueFamilyIndex = cmdbuf->getQueueFamilyIndex();
                  histogramDstBufBarrier.buffer = histogram_range.buffer;
                  histogramDstBufBarrier.offset = histogram_range.offset;
                  histogramDstBufBarrier.size = histogram_range.size;
                }
                video::IGPUCommandBuffer::SBufferMemoryBarrier input_barrier;
                {
                    //input_barrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
                    input_barrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
                    input_barrier.dstQueueFamilyIndex = input_barrier.srcQueueFamilyIndex = cmdbuf->getQueueFamilyIndex();
                    input_barrier.buffer = input_sort_range.buffer;
                    input_barrier.offset = input_sort_range.offset;
                    input_barrier.size = input_sort_range.size;
                }
                video::IGPUCommandBuffer::SBufferMemoryBarrier output_barrier;
                {
                    //output_barrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
                    output_barrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
                    output_barrier.dstQueueFamilyIndex = output_barrier.srcQueueFamilyIndex = cmdbuf->getQueueFamilyIndex();
                    output_barrier.buffer = scratch_sort_range.buffer;
                    output_barrier.offset = scratch_sort_range.offset;
                    output_barrier.size = scratch_sort_range.size;
                }


				video::IGPUCommandBuffer::SBufferMemoryBarrier barriers[4] = { input_barrier, output_barrier, scanDstBufBarrier, histogramDstBufBarrier };
                cmdbuf->pipelineBarrier(start_mask, static_cast<asset::E_PIPELINE_STAGE_FLAGS>(asset::EPSF_COMPUTE_SHADER_BIT |
                asset::EPSF_TRANSFER_BIT), asset::EDF_NONE, 0u, nullptr, 4, barriers,0u,nullptr);

                const video::IGPUDescriptorSet *descriptor_sets[2] = {ds_scan->get(), ds_sort[pass % 2].get()};

                // TODO (Penta): Probably needs barriers for each pass as well

                cmdbuf->bindComputePipeline(histogram);
                cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, histogram->getLayout(), 0u, 2u, descriptor_sets);

                auto histogramSrcMask = pass == 0
                                        ? start_mask
                                        : static_cast<asset::E_PIPELINE_STAGE_FLAGS>(asset::EPSF_COMPUTE_SHADER_BIT | asset::EPSF_TRANSFER_BIT);
                dispatchHelper(cmdbuf, histogram->getLayout(), sort_push_constants[pass], *sort_dispatch_info,
                               histogramSrcMask, 4, barriers,
                               static_cast<asset::E_PIPELINE_STAGE_FLAGS>(asset::EPSF_COMPUTE_SHADER_BIT | asset::EPSF_TRANSFER_BIT), 4, barriers
//                               end_mask, 0u, nullptr
                );

                cmdbuf->bindComputePipeline(scan);
                cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, scan->getLayout(), 0u, 1u, &ds_scan->get());
                scanner->dispatchHelper(cmdbuf, scan->getLayout(), *scan_push_constants, *scan_dispatch_info,
                                        static_cast<asset::E_PIPELINE_STAGE_FLAGS>(asset::EPSF_COMPUTE_SHADER_BIT | asset::EPSF_TRANSFER_BIT), 4, barriers,
                                        static_cast<asset::E_PIPELINE_STAGE_FLAGS>(asset::EPSF_COMPUTE_SHADER_BIT | asset::EPSF_TRANSFER_BIT), 4, barriers
                );

                cmdbuf->bindComputePipeline(scatter);
                cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, scatter->getLayout(), 0u, 2u, descriptor_sets);

                auto scatterDstMask = pass == (PASS_COUNT - 1)
                                      ? end_mask
                                      : static_cast<asset::E_PIPELINE_STAGE_FLAGS>(asset::EPSF_COMPUTE_SHADER_BIT | asset::EPSF_TRANSFER_BIT);
                dispatchHelper(cmdbuf, scatter->getLayout(), sort_push_constants[pass], *sort_dispatch_info,
                               static_cast<asset::E_PIPELINE_STAGE_FLAGS>(asset::EPSF_COMPUTE_SHADER_BIT | asset::EPSF_TRANSFER_BIT), 4, barriers,
                               scatterDstMask, 4, barriers
                );
              }
            }

            core::smart_refctd_ptr<asset::ICPUShader>
            RadixSort::createShader(video::ILogicalDevice *device, E_SHADER_TYPE type) {
              auto system = device->getPhysicalDevice()->getSystem();
              core::smart_refctd_ptr<const system::IFile> glsl = type == E_SHADER_TYPE::ESHT_HISTOGRAM
                                                                 ? system->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE(
                      "nbl/builtin/glsl/ext/RadixSort/default_histogram.comp") >()
                                                                 : system->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE(
                      "nbl/builtin/glsl/ext/RadixSort/default_scatter.comp") >();
              auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(glsl->getSize());
              memcpy(buffer->getPointer(), glsl->getMappedPointer(), glsl->getSize());
              auto cpushader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(buffer), asset::IShader::buffer_contains_glsl_t{},
                                                                              asset::IShader::ESS_COMPUTE, "????");

              core::smart_refctd_ptr<asset::ICPUShader> shader = asset::IGLSLCompiler::createOverridenCopy(
                  cpushader.get(),
                  "#define _NBL_GLSL_WORKGROUP_SIZE_ %d\n#define _NBL_GLSL_EXT_RADIXSORT_BUCKET_COUNT_ %d\n",
                  m_wg_size, BUCKETS_COUNT
              );
              shader->setFilePathHint(type == E_SHADER_TYPE::ESHT_HISTOGRAM
                                      ? "nbl/builtin/glsl/ext/RadixSort/default_histogram.comp"
                                      : "nbl/builtin/glsl/ext/RadixSort/default_scatter.comp");
              return shader;
            }

        }
    }
}
