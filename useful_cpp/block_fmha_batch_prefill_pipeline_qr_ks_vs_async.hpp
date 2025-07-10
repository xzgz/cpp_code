// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

// #define USED_VLLM_PAGE_TABLE_VERSION 0
// #define USED_VLLM_PAGE_TABLE_VERSION 1
// #define USED_VLLM_PAGE_TABLE_VERSION 2
#define USED_VLLM_PAGE_TABLE_VERSION 3

namespace ck_tile {

union DoubleIndext
{
    index_t idx2[2];
    long_index_t ldx;
};

template <typename OffsetVecType,
          typename CoordVecType,
          index_t kCoordAxis,
          index_t kPageBlockSize,
          index_t kPageShiftSize,
          index_t kLoopStart,
          index_t kLoopCount,
          index_t kLoopStride,
          bool kIsSglangLayout,
          bool kIsKcache>
CK_TILE_HOST_DEVICE void kv_offset_array_transform(const index_t* page_vec,
                                                   const index_t& stride_kv,
                                                   const CoordVecType& coord_vec,
                                                   OffsetVecType& kv_offset_vec)
{
    const index_t& thread_coord_start = coord_vec[kCoordAxis];
    // if(blockIdx.x + blockIdx.y + blockIdx.z + threadIdx.y + threadIdx.z == 0 && threadIdx.x == 0)
    // if(blockIdx.x + blockIdx.y + threadIdx.y + threadIdx.z == 0 && blockIdx.z == 3 &&
    //    threadIdx.x == 102)
    // {
    //     printf("kIsSglangLayout=%d\n", kIsSglangLayout);
    //     if constexpr(kIsKcache)
    //     {
    //         printf("k_id: blkz=%d, thr_idx=%d, thr_coord_st=%d\n",
    //                blockIdx.z,
    //                threadIdx.x,
    //                thread_coord_start);
    //     }
    //     else
    //     {
    //         printf("v_id: blkz=%d, thr_idx=%d, thr_coord_st=%d\n",
    //                blockIdx.z,
    //                threadIdx.x,
    //                thread_coord_start);
    //     }
    // }

    if constexpr(kIsSglangLayout)
    {
        static_for<0, kLoopCount, 1>{}([&](auto k0) {
            kv_offset_vec[k0] =
                page_vec[thread_coord_start + kLoopStart + kLoopStride * k0.value] * stride_kv;
        });
    }
    else
    {
#if USED_VLLM_PAGE_TABLE_VERSION == 3
        constexpr index_t kPageMask = (1 << kPageShiftSize) - 1;
        if constexpr(kIsKcache)
        {
            // for k_offset_vec
            constexpr index_t kPageStride = kLoopStride >> kPageShiftSize;
            // constexpr array<index_t, kLoopCount> kPageIdArray = []() {
            //     array<index_t, kLoopCount> arr;
            //     static_for<0, kLoopCount, 1>{}([&](auto k0) {
            //         constexpr index_t kPageId = kPageStride * k0.value;
            //         arr[k0]                   = kPageId;
            //     });
            //     return arr;
            // }();
            static_for<0, kLoopCount, 1>{}([&](auto k0) {
                // constexpr index_t kPageId = kPageIdArray[k0];
                constexpr index_t kPageId = kPageStride * k0.value;
                const index_t page_offset =
                    (thread_coord_start + kLoopStride * k0.value) & kPageMask;
                kv_offset_vec[k0] =
                    ((page_vec[kPageId] << kPageShiftSize) + page_offset) * stride_kv;
            });
        }
        else
        {
            // for v_offset_vec
            const index_t lane0_start   = __builtin_amdgcn_readfirstlane(thread_coord_start);
            const index_t lane0_page_id = (lane0_start + kLoopStart) >> kPageShiftSize;
            const index_t page_loc      = page_vec[lane0_page_id] << kPageShiftSize;
            static_for<0, kLoopCount, 1>{}([&](auto k0) {
                const index_t page_offset =
                    (thread_coord_start + kLoopStart + k0.value) & kPageMask;
                kv_offset_vec[k0] = (page_loc + page_offset) * stride_kv;
            });
        }

#else

        index_t i_page;
        index_t i_seq;
        static_for<0, kLoopCount, 1>{}([&](auto k0) {
#if USED_VLLM_PAGE_TABLE_VERSION == 0
            int32_t seqlen_v_idx_per_repeat =
                thread_coord_start + kLoopStart + kLoopStride * k0.value;
            i_page            = seqlen_v_idx_per_repeat / kPageBlockSize;
            i_seq             = seqlen_v_idx_per_repeat % kPageBlockSize;
            kv_offset_vec[k0] = (page_vec[i_page] * kPageBlockSize + i_seq) * stride_kv;

#elif USED_VLLM_PAGE_TABLE_VERSION == 1
            if constexpr(kIsKcache)
            {
                // for k_offset_vec
                constexpr index_t kItemOffset =
                    (kLoopStart + kLoopStride * k0.value) >> kPageShiftSize;
                i_page = (thread_coord_start >> kPageShiftSize) + kItemOffset;
                kv_offset_vec[k0] =
                    ((page_vec[i_page] << kPageShiftSize) + thread_coord_start) * stride_kv;
            }
            else
            {
                // for v_offset_vec
                constexpr index_t kPageMask = (1 << kPageShiftSize) - 1;
                int32_t seqlen_v_idx_per_repeat =
                    thread_coord_start + kLoopStart + kLoopStride * k0.value;
                i_page            = seqlen_v_idx_per_repeat >> kPageShiftSize;
                i_seq             = seqlen_v_idx_per_repeat & kPageMask;
                kv_offset_vec[k0] = ((page_vec[i_page] << kPageShiftSize) + i_seq) * stride_kv;
            }

#elif USED_VLLM_PAGE_TABLE_VERSION == 2
            if constexpr(kIsKcache)
            {
                // for k_offset_vec
                // index_t i_page;
                constexpr index_t kItemOffset = kLoopStride * k0.value >> kPageShiftSize;
                // i_page = (thread_coord_start >> kPageShiftSize) + kItemOffset;
                asm volatile("v_lshrrev_b32_e32 %[i_page], %[kPageShiftSize],
                                 % [thread_coord_start]\n\t " " v_add_u32_e32 % [i_page],
                             % [kItemOffset],
                             % [i_page]\n\t " : [i_page] " + v "(i_page) : [thread_coord_start]
                                                               "v"(thread_coord_start),
                             [kItemOffset] "i"(kItemOffset),
                             [kPageShiftSize] "i"(kPageShiftSize));
                kv_offset_vec[k0] =
                    ((page_vec[i_page] << kPageShiftSize) + thread_coord_start) * stride_kv;
            }
            else
            {
                constexpr index_t kPageMask   = (1 << kPageShiftSize) - 1;
                constexpr index_t kItemOffset = kLoopStart + kLoopStride * k0.value;
                // i_page = thread_coord_start + kItemOffset
                // i_seq = i_page & (kPageBlockSize - 1)
                // i_page = i_page >> log2(kPageBlockSize)
                asm volatile("v_add_u32_e32 %[i_page], %[kItemOffset],
                                     % [thread_coord_start]\n\t
                                     "
                                     "v_and_b32_e32 %[i_seq], %[kPageMask], %[i_page]\n\t"
                                     "v_lshrrev_b32_e32 %[i_page], %[kPageShiftSize],
                                     % [i_page]\n\t " : [i_page] " +
                                 v "(i_page), [i_seq] " = v "(i_seq)
                             : [thread_coord_start] "v"(thread_coord_start),
                               [kItemOffset] "i"(kItemOffset),
                               [kPageMask] "i"(kPageMask),
                               [kPageShiftSize] "i"(kPageShiftSize));
                kv_offset_vec[k0] = ((page_vec[i_page] << kPageShiftSize) + i_seq) * stride_kv;
            }
#endif
        });

#endif
    }
}

// a variation of qr/ks/vs, where we use async copy to load k (potentially v in the future)
template <typename Problem_,
          typename Policy_ = BlockFmhaBatchPrefillPipelineQRKSVSAsyncDefaultPolicy>
struct BlockFmhaBatchPrefillPipelineQRKSVSAsync
{
    using Problem               = remove_cvref_t<Problem_>;
    using Policy                = remove_cvref_t<Policy_>;
    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType          = remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType   = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using PDataType             = remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType          = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using AttentionVariant      = remove_cvref_t<typename Problem::AttentionVariant>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;

    using BlockFmhaShape             = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using VLayout                    = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static constexpr bool kQLoadOnce = true; // if q_tile load whole block length (hdim) at once
    static_assert(kQLoadOnce == Policy::QLoadOnce);

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0            = BlockFmhaShape::kM0;
    static constexpr index_t kN0            = BlockFmhaShape::kN0;
    static constexpr index_t kK0            = BlockFmhaShape::kK0;
    static constexpr index_t kN1            = BlockFmhaShape::kN1;
    static constexpr index_t kK1            = BlockFmhaShape::kK1;
    static constexpr index_t kQKHeaddim     = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim  = BlockFmhaShape::kSubQKHeaddim;
    static constexpr index_t kPageBlockSize = 16;
    // static constexpr index_t kPageBlockSize = 1;

    static constexpr index_t kPageShiftSize = 4;
    // static constexpr index_t kPageShiftSize = 0;

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};
    static constexpr auto I3 = number<3>{};

    static_assert(kSubQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");

    static constexpr bool kIsGroupMode = Problem::kIsGroupMode;
    // TODO: seq_q always support padding, hdim_q/v support multiple of vector(like 8x)
    //       only need special care about seq_k padding (oob need set -INF of p instead of zero)
    static_assert(Problem::kPadSeqLenQ == true && Problem::kPadHeadDimQ == true &&
                  Problem::kPadHeadDimV == true);
    static constexpr bool kPadSeqLenQ       = true;
    static constexpr bool kPadSeqLenK       = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = true; // support multiple of vector(like 8x)
    static constexpr bool kPadHeadDimV      = true; // support multiple of vector(like 8x)
    static constexpr bool kHasLogitsSoftCap = Problem::kHasLogitsSoftCap;
    static constexpr bool kIsSglangLayout   = Problem::kIsSglangLayout;
    // static constexpr bool kIsSglangLayout   = true;
    static constexpr bool kIsChunkedPrefill = Problem::kIsChunkedPrefill;
    static constexpr auto BiasEnum          = Problem::BiasEnum;
    static constexpr bool kStoreLSE         = Problem::kStoreLSE;
    static constexpr bool kHasDropout       = Problem::kHasDropout;

    static_assert((CK_TILE_FMHA_FWD_FAST_EXP2 &&
                   (kHasLogitsSoftCap && Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS ||
                    !kHasLogitsSoftCap)) ||
                  (!CK_TILE_FMHA_FWD_FAST_EXP2 && !kHasLogitsSoftCap));

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ = Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK = Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV = []() {
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            return Policy::template GetAlignmentV<Problem>();
        else
            return kPadSeqLenK ? 1 : Policy::template GetAlignmentV<Problem>();
    }();
    static constexpr index_t kAlignmentO = Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();

#if CK_TILE_FMHA_FWD_FAST_EXP2
    static constexpr auto R_LOG2E = 1.0 / log2e_v<SaccDataType>;
#endif

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            // minimize occupancy
            if constexpr(BiasEnum != BlockAttentionBiasEnum::NO_BIAS && kHasDropout)
            {
                return 1;
            }

            if constexpr(kQKHeaddim <= 32)
            {
                if constexpr(kPadSeqLenK && BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS &&
                             FmhaMask::IsMasking)
                    return 1;
                else
                    return 2;
            }
            else if constexpr(kQKHeaddim <= 64)
            {
                if constexpr(kPadSeqLenK && BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                    return 2;
                else
                    return 3;
            }
            else if constexpr(kQKHeaddim <= 128)
            {
                if constexpr(kPadSeqLenK && BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                    return 1;
                else
                    // return 1;
                    return 2;
            }
            else if constexpr(kQKHeaddim <= 192)
            {
                if constexpr(kPadSeqLenK && BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                    return 1;
                else
                    return 2;
            }
            else if constexpr(kQKHeaddim <= 256)
            {
                return 1;
            }
            else
            {
                return 1;
            };
        }
    }();

    static constexpr const char* name = "qr_async";

    using DropoutType = std::conditional_t<kHasDropout, BlockDropout, NullBlockDropout>;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename BiasElementFunction,
              typename LSEElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*K0 tile
               const KElementFunction& /*k_element_func*/,
               const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               const VElementFunction& v_element_func,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               const BiasElementFunction& bias_element_func,
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
               LSEDramBlockWindowTmp& lse_dram_window_tmp, // M0*1 tile
               const LSEElementFunction& lse_element_func,
               const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               const index_t* page_idx,
               const index_t stride_k,
               const index_t stride_v,
               DropoutType& dropout,
               const index_t page_block_size) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        constexpr auto LdsSeq = Policy::template GetLdsBufferSequence<Problem>();

        // K tile in LDS
        auto k_lds_ptr   = reinterpret_cast<KDataType*>(smem_ptr);
        auto k_lds_store = generate_tuple(
            [&](auto i_buf) {
                return make_tile_window(
                    make_tensor_view<address_space_enum::lds>(
                        k_lds_ptr, Policy::template MakeKLdsStoreBlockDescriptor<Problem>(i_buf)),
                    Policy::template MakeKLdsStoreBlockDescriptor<Problem>(i_buf).get_lengths(),
                    {0, 0, 0});
            },
            number<Policy::NumKVLdsBuffers>{});

        auto k_lds_Load_view = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsLoadBlockDescriptor<Problem>());

        auto k_lds_load =
            make_tile_window(k_lds_Load_view,
                             Policy::template MakeKLdsLoadBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});

        // V tile in LDS
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(smem_ptr),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_window = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetKVBlockGemm<Problem>();

        auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                              q_dram_block_window_tmp.get_window_lengths(),
                                              q_dram_block_window_tmp.get_window_origin(),
                                              Policy::template MakeQRegTileDistribution<Problem>());
        q_dram_window.init_raw();

        // TODO: we use async Copy for K, which is inline asm
        // a side effect is we have to use inline asm for q as well
        auto q = decltype(load_tile(q_dram_window)){};
        // TODO: start from rocm-6.2, compiler will have problem if manually set clear of q.
        // however, q would be cleared in the constructor of static distributed tensor
        // set_tile(q, number<0>{}); // use per-dword clear to avoid scratch
        load_tile_raw(q, q_dram_window);
        __builtin_amdgcn_sched_barrier(0);

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(s_acc));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        // init Oacc, M, L
        auto o_acc = OaccBlockTileType{};
        auto m     = MLBlockTileType{};
        auto l     = MLBlockTileType{};

        clear_tile(o_acc);
        set_tile(m, -numeric<SMPLComputeDataType>::infinity());
        clear_tile(l);

        __builtin_amdgcn_sched_barrier(0);
        const auto q_origin = q_dram_window.get_window_origin();
        const auto [seqlen_k_start, seqlen_k_end] =
            mask.GetTileRangeAlongX(q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});

        const auto num_total_loop = integer_divide_ceil(seqlen_k_end - seqlen_k_start, kN0);

        // check early exit if no work to do
        if constexpr(FmhaMask::IsMasking || kPadSeqLenK)
        {
            if(num_total_loop <= 0)
            {
                if constexpr(kStoreLSE)
                {
                    auto lse =
                        make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

                    set_tile(lse, -numeric<SMPLComputeDataType>::infinity());

                    store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse));
                }
                buffer_load_fence(0); // rocm-6.1, if whole tile is masked out, need to fence(0)
                                      // otherwise will have compute error(maybe compiler bug?)

                // Note: here occ are all cleard, return it
                return o_acc;
            }
            __builtin_amdgcn_sched_barrier(0); // make sure sched_barrier(0) for this check
        }

        auto k_dram_block_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_k_start, 0});

        auto k_dist               = Policy::template MakeKDramTileDistribution<Problem>();
        auto k_coord              = k_dist.calculate_index();
        using KDstrEncode         = typename decltype(k_dist)::DstrEncode;
        constexpr index_t NRepeat = KDstrEncode::hs_lengthss_[I0][I0];
        statically_indexed_array<index_t, NRepeat> k_offsets;

        kv_offset_array_transform<statically_indexed_array<index_t, NRepeat>,
                                  decltype(k_coord),
                                  0,
                                  kPageBlockSize,
                                  kPageShiftSize,
                                  0,
                                  NRepeat,
                                  kN0 / NRepeat,
                                  kIsSglangLayout,
                                  true>(page_idx, stride_k, k_coord, k_offsets);

        //         if constexpr(kIsSglangLayout)
        //         {
        //             static_for<0, NRepeat, 1>{}([&](auto n0) {
        //                 k_offsets[n0] = page_idx[k_coord[0] + kN0 / NRepeat * n0.value] *
        //                 stride_k;
        //             });
        //         }
        //         else
        //         {
        //             static_for<0, NRepeat, 1>{}([&](auto n0) {
        // #if USED_VLLM_PAGE_TABLE_VERSION == 0
        //                 int32_t seqlen_k_idx_per_repeat = k_coord[0] + kN0 / NRepeat * n0.value;
        //                 int32_t i_page                  = seqlen_k_idx_per_repeat /
        //                 kPageBlockSize; int32_t i_seq                   = seqlen_k_idx_per_repeat
        //                 % kPageBlockSize; k_offsets[n0] = (page_idx[i_page] * kPageBlockSize +
        //                 i_seq) * stride_k;
        // #elif USED_VLLM_PAGE_TABLE_VERSION == 1
        //                 // constexpr index_t kPageMask     = (1 << kPageShiftSize) - 1;
        //                 // int32_t seqlen_k_idx_per_repeat = k_coord[0] + n0.value * kN0 /
        //                 NRepeat;
        //                 // int32_t i_page                  = seqlen_k_idx_per_repeat >>
        //                 kPageShiftSize;
        //                 // int32_t i_seq                   = seqlen_k_idx_per_repeat & kPageMask;
        //                 // k_offsets[n0] = ((page_idx[i_page] << kPageShiftSize) + i_seq) *
        //                 stride_k;

        //                 constexpr index_t kItemOffset = n0.value * kN0 / NRepeat >>
        //                 kPageShiftSize; int32_t i_page                = (k_coord[0] >>
        //                 kPageShiftSize) + kItemOffset; k_offsets[n0] = ((page_idx[i_page] <<
        //                 kPageShiftSize) + k_coord[0]) * stride_k;
        // #elif USED_VLLM_PAGE_TABLE_VERSION == 2
        //                 constexpr index_t kPageMask   = (1 << kPageShiftSize) - 1;
        //                 constexpr index_t kItemOffset = n0.value * kN0 / NRepeat;
        //                 index_t i_page;
        //                 index_t i_seq;
        //                 // i_page = k_coord[0] + kItemOffset
        //                 // i_seq = i_page & (kPageBlockSize - 1)
        //                 // i_page = i_page >> log2(kPageBlockSize)
        //                 asm volatile("v_add_u32_e32 %[i_page], %[kItemOffset], %[k_coord_0]\n\t"
        //                              "v_and_b32_e32 %[i_seq], %[kPageMask], %[i_page]\n\t"
        //                              "v_lshrrev_b32_e32 %[i_page], %[kPageShiftSize],
        //                                      % [i_page]\n\t " : [i_page] " +
        //                                  v "(i_page), [i_seq] " = v "(i_seq)
        //                              : [k_coord_0] "v"(k_coord[0]),
        //                                [kItemOffset] "i"(kItemOffset),
        //                                [kPageMask] "i"(kPageMask),
        //                                [kPageShiftSize] "i"(kPageShiftSize));
        //                 k_offsets[n0] = ((page_idx[i_page] << kPageShiftSize) + i_seq) *
        //                 stride_k;
        // #endif
        //             });
        //         }

        auto k_dram_window = make_tile_scatter_gather(k_dram_block_window.get_bottom_tensor_view(),
                                                      k_dram_block_window.get_window_lengths(),
                                                      k_dram_block_window.get_window_origin(),
                                                      k_dist,
                                                      k_offsets); // K DRAM tile window for

        k_dram_window.init_raw();
        constexpr auto k_oob_ck = bool_constant<true>{};
        constexpr auto k_pre_np = [&]() {
            if constexpr(kPadSeqLenK &&
                         (BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                          (BiasEnum != BlockAttentionBiasEnum::NO_BIAS && kHasDropout)))
                return bool_constant<true>{};
            else
                return bool_constant<false>{};
        }();

        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), seqlen_k_start}, // M/N
                             Policy::template MakeBiasDramTileDistribution<decltype(gemm_0)>());

        auto randval_dram_window = dropout.template MakeRandvalDramWindow<decltype(gemm_0)>(
            randval_dram_block_window_tmp, seqlen_k_start);

        auto v_dist                 = Policy::template MakeVDramTileDistribution<Problem>();
        auto v_coord                = v_dist.calculate_index();
        const auto VPageIndexDim    = I1;
        using VDstrEncode           = typename decltype(v_dist)::DstrEncode;
        constexpr index_t V_KRepeat = VDstrEncode::hs_lengthss_[I1][I3];
        statically_indexed_array<index_t, V_KRepeat> v_offsets;

        kv_offset_array_transform<statically_indexed_array<index_t, V_KRepeat>,
                                  decltype(v_coord),
                                  VPageIndexDim,
                                  kPageBlockSize,
                                  kPageShiftSize,
                                  0,
                                  V_KRepeat,
                                  1,
                                  kIsSglangLayout,
                                  false>(page_idx, stride_v, v_coord, v_offsets);

        //         // (void)stride_k;
        //         if constexpr(kIsSglangLayout)
        //         {
        //             static_for<0, V_KRepeat, 1>{}([&](auto k0) {
        //                 v_offsets[k0] = page_idx[v_coord[VPageIndexDim] + k0.value] * stride_v;
        //             });
        //         }
        //         else
        //         {
        //             static_for<0, V_KRepeat, 1>{}([&](auto k0) {
        // #if USED_VLLM_PAGE_TABLE_VERSION == 0
        //                 int32_t seqlen_v_idx_per_repeat = v_coord[VPageIndexDim] + k0.value;
        //                 int32_t i_page                  = seqlen_v_idx_per_repeat /
        //                 kPageBlockSize; int32_t i_seq                   = seqlen_v_idx_per_repeat
        //                 % kPageBlockSize; v_offsets[k0] = (page_idx[i_page] * kPageBlockSize +
        //                 i_seq) * stride_v;
        // #elif USED_VLLM_PAGE_TABLE_VERSION == 1
        //                 constexpr index_t kPageMask     = (1 << kPageShiftSize) - 1;
        //                 int32_t seqlen_v_idx_per_repeat = v_coord[VPageIndexDim] + k0.value;
        //                 int32_t i_page                  = seqlen_v_idx_per_repeat >>
        //                 kPageShiftSize; int32_t i_seq                   = seqlen_v_idx_per_repeat
        //                 & kPageMask; v_offsets[k0] = ((page_idx[i_page] << kPageShiftSize) +
        //                 i_seq) * stride_v;
        // #elif USED_VLLM_PAGE_TABLE_VERSION == 2
        //                 constexpr index_t kPageMask   = (1 << kPageShiftSize) - 1;
        //                 constexpr index_t kItemOffset = k0.value;
        //                 index_t i_page;
        //                 index_t i_seq;
        //                 // i_page = v_coord_i + kItemOffset
        //                 // i_seq = i_page & (kPageBlockSize - 1)
        //                 // i_page = i_page >> log2(kPageBlockSize)
        //                 asm volatile("v_add_u32_e32 %[i_page], %[kItemOffset], %[v_coord_i]\n\t"
        //                              "v_and_b32_e32 %[i_seq], %[kPageMask], %[i_page]\n\t"
        //                              "v_lshrrev_b32_e32 %[i_page], %[kPageShiftSize],
        //                              %[i_page]\n\t" : [i_page] "+v"(i_page), [i_seq] "=v"(i_seq)
        //                              : [v_coord_i] "v"(v_coord[VPageIndexDim]),
        //                                [kItemOffset] "i"(kItemOffset),
        //                                [kPageMask] "i"(kPageMask),
        //                                [kPageShiftSize] "i"(kPageShiftSize));
        //                 v_offsets[k0] = ((page_idx[i_page] << kPageShiftSize) + i_seq) *
        //                 stride_v;
        // #endif
        //             });
        //         }

        auto v_dram_window =
            make_tile_scatter_gather(v_dram_block_window_tmp.get_bottom_tensor_view(),
                                     v_dram_block_window_tmp.get_window_lengths(),
                                     {0, seqlen_k_start}, // TODO: hdim split?
                                     v_dist,
                                     v_offsets,
                                     VPageIndexDim);

        // prefetch K tile
        async_load_tile_raw(
            k_lds_store(LdsSeq.at(number<0>{})), k_dram_window, number<-1>{}, k_oob_ck, k_pre_np);
        move_tile_window(k_dram_window, {0, kK0});
        __builtin_amdgcn_sched_barrier(0);

        buffer_load_fence(k_dram_window.get_num_of_access(), q.get_thread_buffer());
        (void)q_element_func; // ??? rocm-6.x if use q element func will have scratch on hdim=64/32
        // auto q_tile = q;      // tile_elementwise_in(q_element_func, q);

        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(1 <= k0_loops);
        static_assert(1 <= k1_loops);
        // main loop
        do
        {
            // STAGE 1, QK gemm
            clear_tile(s_acc); // initialize C
            if constexpr(k0_loops > 1)
            {
                static_for<0, k0_loops - 1, 1>{}([&](auto i_k0) {
                    async_load_tile_raw(k_lds_store(number<LdsSeq.at(number<i_k0 + 1>{})>{}),
                                        k_dram_window,
                                        number<-1>{},
                                        k_oob_ck,
                                        k_pre_np);
                    if constexpr(i_k0 < k0_loops - 1)
                        move_tile_window(k_dram_window, {0, kK0});

                    async_load_fence(k_dram_window.get_num_of_access());
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    gemm_0(s_acc,
                           get_slice_tile(
                               q, sequence<0, i_k0 * kK0>{}, sequence<kM0, (i_k0 + 1) * kK0>{}),
                           get_slice_tile(k_lds_load,
                                          sequence<(LdsSeq.at(number<i_k0>{})) * kN0, 0>{},
                                          sequence<(LdsSeq.at(number<i_k0>{}) + 1) * kN0, kK0>{}));
                });
            }

            // TODO: this to fix a bug when loop smaller than 2,
            // the following fence/barrier will be scheduled inside 1st loop
            if constexpr(k0_loops <= 2)
                __builtin_amdgcn_sched_barrier(0);

            async_load_fence();
            __builtin_amdgcn_s_barrier();

            const auto bias_tile = load_tile(bias_dram_window); // load bias tile
            auto v_buf           = load_tile(v_dram_window, number<-1>{}, bool_constant<false>{});

            kv_offset_array_transform<statically_indexed_array<index_t, V_KRepeat>,
                                      decltype(v_coord),
                                      VPageIndexDim,
                                      kPageBlockSize,
                                      kPageShiftSize,
                                      kK1,
                                      V_KRepeat,
                                      1,
                                      kIsSglangLayout,
                                      false>(page_idx, stride_v, v_coord, v_offsets);

            //             if constexpr(kIsSglangLayout)
            //             {
            //                 static_for<0, V_KRepeat, 1>{}([&](auto k0) {
            //                     v_offsets[k0] = page_idx[kK1 + v_coord[VPageIndexDim] + k0.value]
            //                     * stride_v;
            //                 });
            //             }
            //             else
            //             {
            //                 static_for<0, V_KRepeat, 1>{}([&](auto k0) {
            // #if USED_VLLM_PAGE_TABLE_VERSION == 0
            //                     int32_t seqlen_v_idx_per_repeat = kK1 + v_coord[VPageIndexDim] +
            //                     k0.value; int32_t i_page                  =
            //                     seqlen_v_idx_per_repeat / kPageBlockSize; int32_t i_seq =
            //                     seqlen_v_idx_per_repeat % kPageBlockSize; v_offsets[k0] =
            //                     (page_idx[i_page] * kPageBlockSize + i_seq) * stride_v;
            // #elif USED_VLLM_PAGE_TABLE_VERSION == 1
            //                     constexpr index_t kPageMask     = (1 << kPageShiftSize) - 1;
            //                     int32_t seqlen_v_idx_per_repeat = v_coord[VPageIndexDim] + kK1 +
            //                     k0.value; int32_t i_page                  =
            //                     seqlen_v_idx_per_repeat >> kPageShiftSize; int32_t i_seq =
            //                     seqlen_v_idx_per_repeat & kPageMask; v_offsets[k0] =
            //                     ((page_idx[i_page] << kPageShiftSize) + i_seq) * stride_v;
            // #elif USED_VLLM_PAGE_TABLE_VERSION == 2
            //                     constexpr index_t kPageMask   = (1 << kPageShiftSize) - 1;
            //                     constexpr index_t kItemOffset = kK1 + k0.value;
            //                     index_t i_page;
            //                     index_t i_seq;
            //                     // i_page = v_coord_i + kItemOffset
            //                     // i_seq = i_page & (kPageBlockSize - 1)
            //                     // i_page = i_page >> log2(kPageBlockSize)
            //                     asm volatile("v_add_u32_e32 %[i_page], %[kItemOffset],
            //                     %[v_coord_i]\n\t"
            //                                  "v_and_b32_e32 %[i_seq], %[kPageMask],
            //                                  %[i_page]\n\t" "v_lshrrev_b32_e32 %[i_page],
            //                                  %[kPageShiftSize], %[i_page]\n\t" : [i_page]
            //                                  "+v"(i_page), [i_seq] "=v"(i_seq) : [v_coord_i]
            //                                  "v"(v_coord[VPageIndexDim]),
            //                                    [kItemOffset] "i"(kItemOffset),
            //                                    [kPageMask] "i"(kPageMask),
            //                                    [kPageShiftSize] "i"(kPageShiftSize));
            //                     v_offsets[k0] = ((page_idx[i_page] << kPageShiftSize) + i_seq) *
            //                     stride_v;
            // #endif
            //                 });
            //             }

            v_dram_window.update_page_idx(v_offsets);
            __builtin_amdgcn_sched_barrier(0);
            { // tail
                gemm_0(
                    s_acc,
                    get_slice_tile(
                        q, sequence<0, (k0_loops - 1) * kK0>{}, sequence<kM0, k0_loops * kK0>{}),
                    get_slice_tile(k_lds_load,
                                   sequence<(LdsSeq.at(number<k0_loops - 1>{})) * kN0, 0>{},
                                   sequence<(LdsSeq.at(number<k0_loops - 1>{}) + 1) * kN0, kK0>{}));
            }
            __builtin_amdgcn_sched_barrier(1);

            // STAGE 2, scale_s, add bias, mask, softmax
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                s_acc = tile_elementwise_in(s_acc_element_func, s_acc);
                tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
                tile_elementwise_inout(
                    [&](auto& x, const auto& y) {
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                        x += type_convert<SaccDataType>(bias_element_func(y));
#else
                        x += log2e_v<SaccDataType> *
                             type_convert<SaccDataType>(bias_element_func(y));
#endif
                    },
                    s_acc,
                    bias_tile);
            }
            else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                const auto k_origin    = k_dram_block_window.get_window_origin();
                constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                s_acc                  = tile_elementwise_in(s_acc_element_func, s_acc);
                sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            s_acc.get_tile_distribution(), make_tuple(idx0, idx1));

                        const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        s_acc(i_j_idx) *= scale_s;
                        position_encoding.update(s_acc(i_j_idx), row, col);
                    });
                });
            }
            else
            {
                s_acc = tile_elementwise_in(s_acc_element_func, s_acc);
                if constexpr(kHasLogitsSoftCap)
                {
                    auto apply_logits_transform =
                        [&variant, &variant_params, &block_indices](auto& x) {
                            x = variant.LogitsTransform(variant_params,
                                                        variant.QueryTransform(variant_params, x),
                                                        block_indices.batch_idx,
                                                        block_indices.qo_head_idx,
                                                        block_indices.kv_head_idx);
                        };
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                    for(index_t i = 0; i < s_acc.thread_buf_.size(); ++i)
                    {
                        apply_logits_transform(s_acc.thread_buf_[i]);
                    }
#else
                    for(index_t i = 0; i < s_acc.thread_buf_.size(); ++i)
                    {
                        apply_logits_transform(s_acc.thread_buf_[i]);
                    }
#endif
                }
                else
                {
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                    tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
#endif
                }
            }
            move_tile_window(bias_dram_window, {0, kN0});
            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                const auto k_origin      = k_dram_block_window.get_window_origin();
                bool need_perpixel_check = mask.IsEdgeTile(q_origin.at(number<0>{}),
                                                           k_origin.at(number<0>{}),
                                                           number<kM0>{},
                                                           number<kN0>{});

                if(need_perpixel_check)
                {
                    set_tile_if(
                        s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                            const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                            const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                            return !variant.LogitsMask(variant_params,
                                                       block_indices.batch_idx,
                                                       row,
                                                       col,
                                                       block_indices.qo_head_idx,
                                                       block_indices.kv_head_idx);
                        });
                }
            }

            const auto s = cast_tile<SMPLComputeDataType>(s_acc); // S{j}
            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s,
                sequence<1>{},
                f_max,
                -numeric<SMPLComputeDataType>::infinity()); // m_local = rowmax(S{j})
            block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

            const auto m_old = m; // m{j-1}
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local); // m{j}

            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s.get_tile_distribution()); // Pcompute{j}

            __builtin_amdgcn_sched_barrier(0x7F);
            // store & prefetch next v, after the max reduction
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                    Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                shuffle_tile(v_shuffle_tmp, v_buf);

                auto v_lds_window_tmp =
                    get_slice_tile(v_lds_window,
                                   sequence<(LdsSeq.at(number<k0_loops>{})) * kN1, 0>{},
                                   sequence<(LdsSeq.at(number<k0_loops>{}) + 1) * kN1, kK1>{});

                store_tile(
                    v_lds_window_tmp,
                    tile_elementwise_in(v_element_func, v_shuffle_tmp)); // store the prefetch
            }
            else
            {
                auto v_lds_window_tmp =
                    get_slice_tile(v_lds_window,
                                   sequence<(LdsSeq.at(number<k0_loops>{})) * kN1, 0>{},
                                   sequence<(LdsSeq.at(number<k0_loops>{}) + 1) * kN1, kK1>{});
                store_tile(v_lds_window_tmp,
                           tile_elementwise_in(v_element_func, v_buf)); // store the prefetch
            }

            if constexpr(k1_loops > 1)
            {
                move_tile_window(
                    v_dram_window,
                    {0, kK1}); // will have scratch if move this right after load_tile(v_dram)...
                v_buf = load_tile(
                    v_dram_window, number<-1>{}, bool_constant<false>{}); // load next v_buf

                kv_offset_array_transform<statically_indexed_array<index_t, V_KRepeat>,
                                          decltype(v_coord),
                                          VPageIndexDim,
                                          kPageBlockSize,
                                          kPageShiftSize,
                                          2 * kK1,
                                          V_KRepeat,
                                          1,
                                          kIsSglangLayout,
                                          false>(page_idx, stride_v, v_coord, v_offsets);

                //                 if constexpr(kIsSglangLayout)
                //                 {
                //                     static_for<0, V_KRepeat, 1>{}([&](auto k0) {
                //                         v_offsets[k0] =
                //                             page_idx[kK1 * 2 + v_coord[VPageIndexDim] + k0.value]
                //                             * stride_v;
                //                     });
                //                 }
                //                 else
                //                 {
                //                     static_for<0, V_KRepeat, 1>{}([&](auto k0) {
                // #if USED_VLLM_PAGE_TABLE_VERSION == 0
                //                         int32_t seqlen_v_idx_per_repeat =
                //                             kK1 * 2 + v_coord[VPageIndexDim] + k0.value;
                //                         int32_t i_page = seqlen_v_idx_per_repeat /
                //                         kPageBlockSize; int32_t i_seq  = seqlen_v_idx_per_repeat
                //                         % kPageBlockSize; v_offsets[k0]  = (page_idx[i_page] *
                //                         kPageBlockSize + i_seq) * stride_k;
                // #elif USED_VLLM_PAGE_TABLE_VERSION == 1
                //                         constexpr index_t kPageMask = (1 << kPageShiftSize) - 1;
                //                         int32_t seqlen_v_idx_per_repeat =
                //                             v_coord[VPageIndexDim] + 2 * kK1 + k0.value;
                //                         int32_t i_page = seqlen_v_idx_per_repeat >>
                //                         kPageShiftSize; int32_t i_seq  = seqlen_v_idx_per_repeat
                //                         & kPageMask; v_offsets[k0]  = ((page_idx[i_page] <<
                //                         kPageShiftSize) + i_seq) * stride_v;
                // #elif USED_VLLM_PAGE_TABLE_VERSION == 2
                //                         constexpr index_t kPageMask   = (1 << kPageShiftSize) -
                //                         1; constexpr index_t kItemOffset = 2 * kK1 + k0.value;
                //                         index_t i_page;
                //                         index_t i_seq;
                //                         // i_page = v_coord_i + kItemOffset
                //                         // i_seq = i_page & (kPageBlockSize - 1)
                //                         // i_page = i_page >> log2(kPageBlockSize)
                //                         asm volatile(
                //                             "v_add_u32_e32 %[i_page], %[kItemOffset],
                //                             %[v_coord_i]\n\t" "v_and_b32_e32 %[i_seq],
                //                             %[kPageMask], %[i_page]\n\t" "v_lshrrev_b32_e32
                //                             %[i_page], %[kPageShiftSize], %[i_page]\n\t" :
                //                             [i_page] "+v"(i_page), [i_seq] "=v"(i_seq) :
                //                             [v_coord_i] "v"(v_coord[VPageIndexDim]),
                //                               [kItemOffset] "i"(kItemOffset),
                //                               [kPageMask] "i"(kPageMask),
                //                               [kPageShiftSize] "i"(kPageShiftSize));
                //                         v_offsets[k0] = ((page_idx[i_page] << kPageShiftSize) +
                //                         i_seq) * stride_v;
                // #endif
                //                     });
                //                 }

                v_dram_window.update_page_idx(v_offsets);
            }
            __builtin_amdgcn_sched_barrier(0);

            static const auto get_validated_m = [](SMPLComputeDataType raw_m) {
                /// NOTICE: bias might be materialized mask including -inf values, need
                /// consideration. alibi does not have this problem
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             FmhaMask::IsMasking)
                {
                    return raw_m == -numeric<SMPLComputeDataType>::infinity()
                               ? type_convert<SMPLComputeDataType>(0.f)
                               : raw_m;
                }
                else
                {
                    return raw_m;
                }
            };

            constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                auto row_max = scale_s * get_validated_m(m[i_idx]);
#endif
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        p_compute(i_j_idx) = exp2(s[i_j_idx] - get_validated_m(m[i_idx]));
                    }
                    else
                    {
                        if constexpr(kHasLogitsSoftCap)
                        {
                            p_compute(i_j_idx) = exp2(s[i_j_idx] - get_validated_m(m[i_idx]));
                        }
                        else
                        {
                            p_compute(i_j_idx) = exp2(scale_s * s[i_j_idx] - row_max);
                        }
                    }
#else
                    p_compute(i_j_idx)     = exp(s[i_j_idx] - get_validated_m(m[i_idx]));
#endif
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0}); // rowsum(Pcompute{j})

            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});
            // l{j}, Oacc{j}
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                const auto tmp = [&]() {
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        return exp2(m_old[i_idx] - get_validated_m(m[i_idx]));
                    }
                    else
                    {
                        if constexpr(kHasLogitsSoftCap)
                        {
                            return exp2(m_old[i_idx] - get_validated_m(m[i_idx]));
                        }
                        else
                        {
                            auto row_max = scale_s * get_validated_m(m[i_idx]);
                            return exp2(scale_s * m_old[i_idx] - row_max);
                        }
                    }
                }();
#else
                const auto tmp = exp(m_old[i_idx] - get_validated_m(m[i_idx]));
#endif
                l(i_idx) = tmp * l[i_idx] + rowsum_p[i_idx];
                sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    // FIXME: this use different equation from FA v2 paper,
                    // but produce correc result.
                    // Is the equation wrong?
                    o_acc(i_j_idx) *= tmp;
                });
            });

            if constexpr(kHasDropout)
            {
                auto randval_ptr =
                    reinterpret_cast<char*>(smem_ptr) + Policy::template GetSmemSizeKV<Problem>();
                dropout.template Run<decltype(gemm_0), SMPLComputeDataType, RandValOutputDataType>(
                    randval_ptr,
                    seqlen_k_start + i_total_loops * kN0,
                    p_compute,
                    randval_dram_window);
            }

            const auto p = [&]() {
                if constexpr(std::is_same_v<PDataType, fp16_t>)
                    return impl::cast_tile_pk_fp16_fp32<PDataType>(
                        tile_elementwise_in(p_compute_element_func, p_compute));
                else
                    return cast_tile<PDataType>(
                        tile_elementwise_in(p_compute_element_func, p_compute));
            }();

            // STAGE 3, KV gemm
            if constexpr(k1_loops > 1)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    if constexpr(i_k1 != 0 && i_k1 < k1_loops - 1)
                    {
                        v_buf = load_tile(
                            v_dram_window, number<-1>{}, bool_constant<false>{}); // load next v_buf

                        kv_offset_array_transform<statically_indexed_array<index_t, V_KRepeat>,
                                                  decltype(v_coord),
                                                  VPageIndexDim,
                                                  kPageBlockSize,
                                                  kPageShiftSize,
                                                  (2 + i_k1.value) * kK1,
                                                  V_KRepeat,
                                                  1,
                                                  kIsSglangLayout,
                                                  false>(page_idx, stride_v, v_coord, v_offsets);

                        //                         if constexpr(kIsSglangLayout)
                        //                         {
                        //                             static_for<0, V_KRepeat, 1>{}([&](auto k0) {
                        //                                 v_offsets[k0] = page_idx[kK1 * 2 +
                        //                                 i_k1.value * kK1 +
                        //                                                          v_coord[VPageIndexDim]
                        //                                                          + k0.value] *
                        //                                                 stride_v;
                        //                             });
                        //                         }
                        //                         else
                        //                         {
                        //                             static_for<0, V_KRepeat, 1>{}([&](auto k0) {
                        // #if USED_VLLM_PAGE_TABLE_VERSION == 0
                        //                                 int32_t seqlen_v_idx_per_repeat =
                        //                                     kK1 * 2 + i_k1.value * kK1 +
                        //                                     v_coord[VPageIndexDim] + k0.value;
                        //                                 int32_t i_page = seqlen_v_idx_per_repeat
                        //                                 / kPageBlockSize; int32_t i_seq  =
                        //                                 seqlen_v_idx_per_repeat % kPageBlockSize;
                        //                                 v_offsets[k0] =
                        //                                     (page_idx[i_page] * kPageBlockSize +
                        //                                     i_seq) * stride_v;
                        // #elif USED_VLLM_PAGE_TABLE_VERSION == 1
                        //                                 constexpr index_t kPageMask = (1 <<
                        //                                 kPageShiftSize) - 1; int32_t
                        //                                 seqlen_v_idx_per_repeat =
                        //                                     v_coord[VPageIndexDim] + 2 * kK1 +
                        //                                     i_k1.value * kK1 + k0.value;
                        //                                 int32_t i_page = seqlen_v_idx_per_repeat
                        //                                 >> kPageShiftSize; int32_t i_seq  =
                        //                                 seqlen_v_idx_per_repeat & kPageMask;
                        //                                 v_offsets[k0] =
                        //                                     ((page_idx[i_page] << kPageShiftSize)
                        //                                     + i_seq) * stride_v;
                        // #elif USED_VLLM_PAGE_TABLE_VERSION == 2
                        //                                 constexpr index_t kPageMask = (1 <<
                        //                                 kPageShiftSize) - 1; constexpr index_t
                        //                                 kItemOffset =
                        //                                     2 * kK1 + i_k1.value * kK1 +
                        //                                     k0.value;
                        //                                 index_t i_page;
                        //                                 index_t i_seq;
                        //                                 // i_page = v_coord_i + kItemOffset
                        //                                 // i_seq = i_page & (kPageBlockSize - 1)
                        //                                 // i_page = i_page >>
                        //                                 log2(kPageBlockSize) asm volatile(
                        //                                     "v_add_u32_e32 %[i_page],
                        //                                     %[kItemOffset], %[v_coord_i]\n\t"
                        //                                     "v_and_b32_e32 %[i_seq],
                        //                                     %[kPageMask], %[i_page]\n\t"
                        //                                     "v_lshrrev_b32_e32 %[i_page],
                        //                                     %[kPageShiftSize], %[i_page]\n\t" :
                        //                                     [i_page] "+v"(i_page), [i_seq]
                        //                                     "=v"(i_seq) : [v_coord_i]
                        //                                     "v"(v_coord[VPageIndexDim]),
                        //                                       [kItemOffset] "i"(kItemOffset),
                        //                                       [kPageMask] "i"(kPageMask),
                        //                                       [kPageShiftSize]
                        //                                       "i"(kPageShiftSize));
                        //                                 v_offsets[k0] =
                        //                                     ((page_idx[i_page] << kPageShiftSize)
                        //                                     + i_seq) * stride_v;
                        // #endif
                        //                             });
                        //                         }

                        v_dram_window.update_page_idx(v_offsets);
                    }
                    block_sync_lds();
                    gemm_1(o_acc,
                           get_slice_tile(
                               p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                           get_slice_tile(
                               v_lds_window,
                               sequence<(LdsSeq.at(number<k0_loops + i_k1>{})) * kN1, 0>{},
                               sequence<(LdsSeq.at(number<k0_loops + i_k1>{}) + 1) * kN1, kK1>{}));

                    if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
                    {
                        auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                            Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                        shuffle_tile(v_shuffle_tmp, v_buf);
                        auto v_lds_window_tmp = get_slice_tile(
                            v_lds_window,
                            sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{})) * kN1, 0>{},
                            sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{}) + 1) * kN1, kK1>{});
                        store_tile(v_lds_window_tmp,
                                   tile_elementwise_in(v_element_func,
                                                       v_shuffle_tmp)); // store the prefetch
                    }
                    else
                    {
                        auto v_lds_window_tmp = get_slice_tile(
                            v_lds_window,
                            sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{})) * kN1, 0>{},
                            sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{}) + 1) * kN1, kK1>{});
                        store_tile(v_lds_window_tmp,
                                   tile_elementwise_in(v_element_func, v_buf)); // store next v_buf
                    }
                    if constexpr(i_k1 < k1_loops - 1)
                        move_tile_window(v_dram_window, {0, kK1});
                });
            }
            i_total_loops++;
            if(i_total_loops < num_total_loop)
            {
                if constexpr(kIsSglangLayout)
                {
                    page_idx += kN0;
                }
                else
                {
                    page_idx += kN0 / kPageBlockSize;
                }
                // move K tile windows
                move_tile_window(k_dram_block_window, {kN0, 0});
                k_dram_window.set_window_origin(k_dram_block_window.get_window_origin());

                kv_offset_array_transform<statically_indexed_array<index_t, NRepeat>,
                                          decltype(k_coord),
                                          0,
                                          kPageBlockSize,
                                          kPageShiftSize,
                                          0,
                                          NRepeat,
                                          kN0 / NRepeat,
                                          kIsSglangLayout,
                                          true>(page_idx, stride_k, k_coord, k_offsets);

                //                 if constexpr(kIsSglangLayout)
                //                 {
                //                     static_for<0, NRepeat, 1>{}([&](auto n0) {
                //                         k_offsets[n0] = page_idx[k_coord[0] + kN0 / NRepeat *
                //                         n0.value] * stride_k;
                //                     });
                //                 }
                //                 else
                //                 {
                //                     static_for<0, NRepeat, 1>{}([&](auto n0) {
                // #if USED_VLLM_PAGE_TABLE_VERSION == 0
                //                         int32_t seqlen_k_idx_per_repeat = k_coord[0] + kN0 /
                //                         NRepeat * n0.value; int32_t i_page                  =
                //                         seqlen_k_idx_per_repeat / kPageBlockSize; int32_t i_seq
                //                         = seqlen_k_idx_per_repeat % kPageBlockSize; k_offsets[n0]
                //                         = (page_idx[i_page] * kPageBlockSize + i_seq) * stride_k;
                // #elif USED_VLLM_PAGE_TABLE_VERSION == 1
                //                         // constexpr index_t kPageMask     = (1 <<
                //                         kPageShiftSize) - 1;
                //                         // int32_t seqlen_k_idx_per_repeat = k_coord[0] +
                //                         n0.value * kN0 / NRepeat;
                //                         // int32_t i_page                  =
                //                         seqlen_k_idx_per_repeat >>
                //                         // kPageShiftSize; int32_t i_seq                   =
                //                         seqlen_k_idx_per_repeat
                //                         // & kPageMask; k_offsets[n0] = ((page_idx[i_page] <<
                //                         kPageShiftSize) +
                //                         // i_seq) * stride_k;

                //                         constexpr index_t kItemOffset = n0.value * kN0 / NRepeat
                //                         >> kPageShiftSize; int32_t i_page = (k_coord[0] >>
                //                         kPageShiftSize) + kItemOffset; k_offsets[n0] =
                //                             ((page_idx[i_page] << kPageShiftSize) + k_coord[0]) *
                //                             stride_k;
                // #elif USED_VLLM_PAGE_TABLE_VERSION == 2
                //                         constexpr index_t kPageMask   = (1 << kPageShiftSize) -
                //                         1; constexpr index_t kItemOffset = n0.value * kN0 /
                //                         NRepeat; index_t i_page; index_t i_seq;
                //                         // i_page = k_coord[0] + kItemOffset
                //                         // i_seq = i_page & (kPageBlockSize - 1)
                //                         // i_page = i_page >> log2(kPageBlockSize)
                //                         asm volatile(
                //                             "v_add_u32_e32 %[i_page], %[kItemOffset],
                //                             %[k_coord_0]\n\t" "v_and_b32_e32 %[i_seq],
                //                             %[kPageMask], %[i_page]\n\t" "v_lshrrev_b32_e32
                //                             %[i_page], %[kPageShiftSize], %[i_page]\n\t" :
                //                             [i_page] "+v"(i_page), [i_seq] "=v"(i_seq) :
                //                             [k_coord_0] "v"(k_coord[0]),
                //                               [kItemOffset] "i"(kItemOffset),
                //                               [kPageMask] "i"(kPageMask),
                //                               [kPageShiftSize] "i"(kPageShiftSize));
                //                         k_offsets[n0] = ((page_idx[i_page] << kPageShiftSize) +
                //                         i_seq) * stride_k;
                // #endif
                //                     });
                //                 }

                k_dram_window.update_page_idx(k_offsets);
                if constexpr(k1_loops >= 2 &&
                             LdsSeq.at(number<0>{}) == LdsSeq.at(number<k0_loops + k1_loops - 2>{}))
                    __builtin_amdgcn_s_barrier();
                async_load_tile_raw(k_lds_store(LdsSeq.at(number<0>{})),
                                    k_dram_window,
                                    number<-1>{},
                                    k_oob_ck,
                                    k_pre_np);
                move_tile_window(k_dram_window, {0, kK0});
            }
            // tail
            {
                block_sync_lds();
                gemm_1(
                    o_acc,
                    get_slice_tile(p, sequence<0, (k1_loops - 1) * kK1>{}, sequence<kM0, kN0>{}),
                    get_slice_tile(
                        v_lds_window,
                        sequence<(LdsSeq.at(number<k0_loops + k1_loops - 1>{})) * kN1, 0>{},
                        sequence<(LdsSeq.at(number<k0_loops + k1_loops - 1>{}) + 1) * kN1, kK1>{}));
            }
        } while(i_total_loops < num_total_loop);

        // store lse
        if constexpr(kStoreLSE)
        {
            auto lse = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

            constexpr auto lse_spans = decltype(lse)::get_distributed_spans();
            sweep_tile_span(lse_spans[number<0>{}], [&, m_ = m, l_ = l](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    lse(i_idx) = m_[i_idx] * R_LOG2E + log(l_[i_idx]);
                }
                else
                {
                    if constexpr(kHasLogitsSoftCap)
                    {
                        lse(i_idx) = m_[i_idx] * R_LOG2E + log(l_[i_idx]);
                    }
                    else
                    {
                        lse(i_idx) = m_[i_idx] * scale_s * R_LOG2E + log(l_[i_idx]);
                    }
                }
#else
                lse(i_idx) = m_[i_idx] + log(l_[i_idx]);
#endif
            });

            store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse));
        }

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = [&]() {
                if constexpr(FmhaMask::IsMasking)
                {
                    return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
                }
                else
                    return 1 / l[i_idx];
            }();
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        o_acc = tile_elementwise_in(o_acc_element_func, o_acc);

        return o_acc;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,       // M0*K0 tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp,       // N0*K0 tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp,       // N1*K1 tile
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp, // M0*N0 tile
               LSEDramBlockWindowTmp& lse_dram_block_window_tmp,         // M0*1 tile
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               const index_t* page_idx,
               const index_t stride_k,
               const index_t stride_v,
               DropoutType& dropout,
               const index_t page_block_size) const
    {
        return operator()(q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          identity{},
                          v_dram_block_window_tmp,
                          identity{},
                          bias_dram_block_window_tmp,
                          identity{},
                          randval_dram_block_window_tmp,
                          lse_dram_block_window_tmp,
                          identity{},
                          identity{},
                          identity{},
                          identity{},
                          mask,
                          position_encoding,
                          scale_s,
                          variant,
                          variant_params,
                          block_indices,
                          smem_ptr,
                          page_idx,
                          stride_k,
                          stride_v,
                          dropout,
                          page_block_size);
    }
};

} // namespace ck_tile
