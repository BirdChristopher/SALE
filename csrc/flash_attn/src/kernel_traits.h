/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

 #pragma once

 #include "cute/arch/copy.hpp"
 #include "cute/arch/copy_sm75.hpp"
 #include "cute/arch/copy_sm80.hpp"
 #include "cute/arch/mma_sm80.hpp"
 #include "cute/atom/copy_atom.hpp"
 #include "cute/container/tuple.hpp"
 #include "cute/layout_composed.hpp"
 #include "cute/numeric/int.hpp"
 #include "cute/numeric/numeric_types.hpp"
 #include "cute/tensor.hpp"
 
 #include "cutlass/cutlass.h"
 #include "cutlass/half.h"
 #include "cutlass/integer_subbyte.h"
 #include "cutlass/layout/layout.h"
 #include <cstdint>
 #include <cutlass/integer_subbyte.h>
 #include <cutlass/numeric_types.h>
 #include <type_traits>
 
 using namespace cute;
 
 template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t>
 struct Flash_kernel_traits {
 
 #if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
     using Element = elem_type;
     static constexpr bool Has_cp_async = true;
 #else
     using Element = cutlass::half_t;
     static constexpr bool Has_cp_async = false;
 #endif
 
     using ElementAccum = float;
     using index_t = int64_t;
     using vec_type = std::conditional_t<
         std::is_same_v<Element, cutlass::half_t>, 
         nv_half2,
         nv_bfloat162
     >;
 
 #if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
     using MMA_Atom_Arch = std::conditional_t<
         std::is_same_v<elem_type, cutlass::half_t>,
         MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
         MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
     >;
 #else
     using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
 #endif
 
 #if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
     using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
     using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
 #else
     using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
     using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
 #endif
 };
 
 
 template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, typename elem_type=cutlass::half_t,
          typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
 struct Flash_fwd_kernel_traits : public Base {
     using Element = typename Base::Element;
     using ElementAccum = typename Base::ElementAccum;
     using index_t = typename Base::index_t;
     static constexpr bool Has_cp_async = Base::Has_cp_async;
     using SmemCopyAtom = typename Base::SmemCopyAtom;
     using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;
 
     static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
     static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;
 
     // The number of threads.
     static constexpr int kNWarps = kNWarps_;
     static constexpr int kNThreads = kNWarps * 32;
 
     static constexpr int kBlockM = kBlockM_;
     static constexpr int kBlockN = kBlockN_;
     static constexpr int kHeadDim = kHeadDim_;
     static_assert(kHeadDim % 32 == 0);
     static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32; // 64 in most cases.
     static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32); // 128 in our cases.
     static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
 
     using TiledMma = TiledMMA<
         typename Base::MMA_Atom_Arch,       
         Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
         Tile<Int<16 * kNWarps>, _16, _16>>; // No permutation.
 
     using SmemLayoutAtomQ = decltype(
         composition(Swizzle<kSwizzle, 3, 3>{},
                     // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                     Layout<Shape<_8, Int<kBlockKSmem>>,
                            Stride<Int<kBlockKSmem>, _1>>{}));
     // ComposedLayout<Swizzle<B,M,S>, _0, Layout>
     // Do tiling(repeating the tiler)
     using SmemLayoutQ = decltype(tile_to_shape(
         SmemLayoutAtomQ{},
         Shape<Int<kBlockM>, Int<kHeadDim>>{}));
 
     // ComposedLayout<Swizzle<B,M,S>, _0, Layout> 
     using SmemLayoutKV = decltype(tile_to_shape(
         SmemLayoutAtomQ{},
         Shape<Int<kBlockN>, Int<kHeadDim>>{}));
 
     // https://github.com/ColfaxResearch/cutlass-kernels/blob/a222587e6d59b93ba704853d3946fb686d8b8892/src/fmha/fmha_forward.cu#L434
     using SmemLayoutVtransposed = decltype(
         composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
     using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));
 
     using SmemLayoutAtomO = decltype(
         composition(Swizzle<kSwizzle, 3, 3>{},
                     Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                            Stride<Int<kBlockKSmem>, _1>>{}));
     using SmemLayoutO = decltype(tile_to_shape(
         SmemLayoutAtomO{},
         Shape<Int<kBlockM>, Int<kHeadDim>>{}));
     using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
     using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;
 
     static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
     static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
     static constexpr int kSmemSize = Share_Q_K_smem ? std::max(kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize;
 
     static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
     static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
     // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
     // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
     // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
     // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
     // to the same banks.
 
     static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad; 
     static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
     
     using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                   Stride<Int<kGmemThreadsPerRow>, _1>>;
 
     // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
     // from the same address by the same threadblock. This is slightly faster.
     using Gmem_copy_struct = std::conditional_t<
         Has_cp_async,
         SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
         DefaultCopy
     >;
     using GmemTiledCopyQKV = decltype(
         make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                         GmemLayoutAtom{},
                         Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
     using GmemTiledCopyO = decltype(
         make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                         GmemLayoutAtom{},
                         Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
 
     using GmemLayoutAtomOaccum = std::conditional_t<
         kBlockKSmem == 32,
         Layout<Shape <_16, _8>,  // Thread layout, 8 threads per row
                Stride< _8, _1>>,
         Layout<Shape <_8, _16>,  // Thread layout, 16 threads per row
                Stride< _16, _1>>
     >;
     using GmemTiledCopyOaccum = decltype(
         make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                         GmemLayoutAtomOaccum{},
                         Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
     using GmemLayoutAtomRotcossin = GmemLayoutAtom;
     using GmemTiledCopyRotcossin = decltype(
         make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                         GmemLayoutAtomRotcossin{},
                         Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
     using GmemTiledCopyRotcossinCont = decltype(
         make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                         GmemLayoutAtomRotcossin{},
                         Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
 };
 
 template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, int kLocalWindow, int kBlockSeg_,
         bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, 
         bool Is_QQ_in_regs_=false, bool Share_QQ_QK_smem_=false,
         bool map_only=false,
         typename elem_type=cutlass::half_t,
         typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type>>
 struct Int4_Flash_fwd_kernel_traits_v1 : public Base {
     using Element = typename Base::Element;
     using ElementAccum = typename Base::ElementAccum;
     using Int4_Element = cute::int4b_t; 
     using UInt8_Type = cute::uint8_t;
 
     using index_t = typename Base::index_t;
     static constexpr bool Has_cp_async = Base::Has_cp_async;
     using SmemCopyAtom = typename Base::SmemCopyAtom;
     using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;
     
     using INT4_MMA_Atom_Arch = MMA_Atom<SM80_16x8x64_S32S4S4S32_TN>; // NOTE: satfinite modifier is unnecessary here.
     using INT4_SmemCopyAtom_x4 = Copy_Atom<SM75_U32x4_LDSM_N, cutlass::int4b_t>;
     using INT4_SmemCopyAtom_x2 = Copy_Atom<SM75_U32x2_LDSM_N, cutlass::int4b_t>; 
     
     // Used to do partition.
     using INT8_SmemCopyAtom_x4 = Copy_Atom<SM75_U32x4_LDSM_N, UInt8_Type>;
     using INT8_SmemCopyAtom_x2 = Copy_Atom<SM75_U32x2_LDSM_N, UInt8_Type>; 
     using SmemCopyAtom_SK_QMK = std::conditional_t<
         (kBlockN_ % 32 == 0),
         Copy_Atom<SM75_U32x4_LDSM_N, elem_type>,
         std::conditional_t<
             (kBlockN_ % 16 == 0), 
             Copy_Atom<SM75_U32x2_LDSM_N, elem_type>, 
             Copy_Atom<SM75_U32x1_LDSM_N, elem_type>
         >
     >;
 
     static_assert(sizeof(Element) == 2, "Only support fp16 prefill. Weird bugs will happend if using other types.");
     static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
     static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;
     static constexpr bool Share_QQ_QK_smem = Share_QQ_QK_smem_;
     static constexpr bool Is_QQ_in_regs = Is_QQ_in_regs_ || Share_QQ_QK_smem_;
 
     // The number of threads.
     static constexpr int kNWarps = kNWarps_; 
     static constexpr int kNThreads = kNWarps * 32; 
 
     static constexpr int kBlockM = kBlockM_;
     static constexpr int kBlockN = kBlockN_;
     static constexpr int kHeadDim = kHeadDim_;
     static_assert(kHeadDim % 32 == 0);
     static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32; // 64 in most cases. 
     static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32); // 128 in our cases.
     static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3; 
     static constexpr int kSwizzleInt4 = 3; 
 
     static constexpr int kBlockSeg = kBlockSeg_;
 
     static_assert(kBlockM_ % kBlockN_ == 0);
     static_assert(kBlockN_ >= 8, "Please set kBlockN_ more than 8. ");
     static_assert(kBlockM > kNWarps, "Block size is too small. Please set it bigger.");
     static_assert(kLocalWindow % kBlockN_ == 0);
 
     static constexpr int kNLocalBlocks = kLocalWindow / kBlockN_;
 
     using TiledMma = TiledMMA<
         typename Base::MMA_Atom_Arch,      
         Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
         Tile<Int<16 * kNWarps>, _16, _16>>; // No permutation. 
     static_assert(16 * kNWarps >= kBlockM, "16 * kNWarps should be larger than kBlockM");
     
     static constexpr int nWarpM = kBlockM_ / 16 > kNWarps_ ? kNWarps : kBlockM_ / 16;
     static constexpr int nWarpN = nWarpM > kNWarps_ ? 1 : kNWarps_ / nWarpM;
     static_assert(nWarpM * nWarpN >= kNWarps, "kNWarps is too large or M/N block size is too small");
 
     // Tile size should align with fp16 TiledMMA. 
     using Int4TiledMMA = TiledMMA<
         INT4_MMA_Atom_Arch, // 16 8 64
         Layout<Shape<Int<nWarpM>, Int<nWarpN>, _1>>,
         Tile<Int<nWarpM * 16>, Int<nWarpN  * 16>, _64>>; 
 
     using DummyUInt8TiledMMA = TiledMMA<
         MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>,
         Layout<Shape<Int<nWarpM>, Int<nWarpN>, _1>>,
         Tile<Int<nWarpM  * 16>, Int<nWarpN  * 16>, _32>>; 
 
     using SmemLayoutAtomQ = decltype(
         composition(Swizzle<kSwizzle, 3, 3>{},
                     // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                     Layout<Shape<_8, Int<kBlockKSmem>>, 
                             Stride<Int<kBlockKSmem>, _1>>{}));
 
     using SmemLayoutQ = decltype(tile_to_shape(
         SmemLayoutAtomQ{},
         Shape<Int<kBlockM>, Int<kHeadDim>>{}));
 
     // ComposedLayout<Swizzle<B,M,S>, _0, Layout> 
     using SmemLayoutKV = decltype(tile_to_shape(
         SmemLayoutAtomQ{},
         Shape<Int<kBlockN>, Int<kHeadDim>>{}));
 
     // https://github.com/ColfaxResearch/cutlass-kernels/blob/a222587e6d59b93ba704853d3946fb686d8b8892/src/fmha/fmha_forward.cu#L434
     using SmemLayoutVtransposed = decltype(
         composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
 
     // Useless in my case.
     using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));
 
     using SmemLayoutAtomO = decltype(
         composition(Swizzle<kSwizzle, 3, 3>{},
                     Layout<Shape<_8, Int<kBlockKSmem>>,
                            Stride<Int<kBlockKSmem>, _1>>{}));
     using SmemLayoutO = decltype(tile_to_shape(
         SmemLayoutAtomO{},
         Shape<Int<kBlockM>, Int<kHeadDim>>{}));
     using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopy, Element>;
     using SmemCopyAtomOaccum = Copy_Atom<AutoVectorizingCopy, ElementAccum>;
 
 
     static constexpr int smem_size = 99 * 1024; // for 4090
 
     static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
     static constexpr int kSmemKVSize = map_only ? size(SmemLayoutKV{}) *sizeof(Element) : size(SmemLayoutKV{}) * 2 * sizeof(Element);
     static constexpr int kSmemSize_2byte = Share_Q_K_smem ? std::max(kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize;
 
     static constexpr int int4QSmemSize = (kBlockM_ * kHeadDim_) / 2;
     static constexpr int int4StageSmemSize = (kBlockN_ * kHeadDim_) / 2 + 2 * kBlockN_ * sizeof(Element);
     
     static constexpr int residentSmemSize = Share_QQ_QK_smem_ ? kSmemSize_2byte : kSmemSize_2byte + int4QSmemSize;
     static constexpr int kPipeSize = 4; 
     static constexpr int allReduceBufSize = kNWarps * 4;
 
     static_assert(kPipeSize == 2 || kPipeSize == 4 || kPipeSize == 8);
     static constexpr uint32_t kPipemask = kPipeSize - 1;
 
     static constexpr int kSmemSize = map_only ? kSmemSize_2byte : int4StageSmemSize * kPipeSize + residentSmemSize + allReduceBufSize;
     static_assert(kSmemSize < smem_size, "Block size or kPipeSize is too large!");

     using SmemLayoutAtomUInt8Swizzled = decltype( 
         composition(Swizzle<kSwizzleInt4, 4, 3>{},
                     Layout<Shape<_16, Int<kBlockKGmem / 2>>, 
                             Stride<Int<kBlockKGmem / 2>, _1>>{})
     );
      using SmemLayoutAtomUInt8SwizzledPipe = decltype( 
         composition(Swizzle<kSwizzleInt4, 4, 3>{}, 
                     Layout<Shape<_16, Int<kBlockKGmem / 2>, _1>, 
                             Stride<Int<kBlockKGmem / 2>, _1, Int<kBlockKGmem * 8>>>{})
     );
 
     using SmemLayoutAtomInt4Swizzled = decltype( 
         composition(Swizzle<kSwizzleInt4, 5, 3>{}, // cover 2048 int4 elements. 
                     Layout<Shape<_16, Int<kBlockKGmem>>,
                             Stride<Int<kBlockKGmem>, _1>>{})
     );
      using SmemLayoutAtomInt4SwizzledPipe = decltype( 
         composition(Swizzle<kSwizzleInt4, 5, 3>{}, // cover 2048 int4 elements. 
                     Layout<Shape<_16, Int<kBlockKGmem>, _1>, // 2028 elem.
                             Stride<Int<kBlockKGmem>, _1, Int<kBlockKGmem * 16>>>{})
     );
 
 
     using SmemLayoutUInt8Q = decltype(tile_to_shape(
         SmemLayoutAtomUInt8Swizzled{},
         Shape<Int<kBlockM>, Int<kHeadDim / 2>>{}
     ));
 
     using SmemLayoutUInt8K = decltype(tile_to_shape(
         SmemLayoutAtomUInt8SwizzledPipe{}, 
         Shape<Int<kBlockN>, Int<kHeadDim / 2>, Int<kPipeSize>>{} // Pipelined.
     ));
 
     using SmemLayoutInt4Q = decltype(tile_to_shape(
         SmemLayoutAtomInt4Swizzled{},
         Shape<Int<kBlockM>, Int<kHeadDim>>{}
     ));
 
     using SmemLayoutInt4K = decltype(tile_to_shape(
         SmemLayoutAtomInt4SwizzledPipe{}, 
         Shape<Int<kBlockN>, Int<kHeadDim>, Int<kPipeSize>>{} // Pipelined.
     ));
 
     // using SmemLayoutSK = decltype(tile_to_shape(
     //     SmemLayoutAtomInt4Swizzled{},
     //     Shape<Int<kBlockN>, Int<kPipeSize>>{}
     // ));
     using SmemLayoutSK = Layout<Shape<Int<kBlockN>, Int<kPipeSize>>>;
     using SmemLayoutSK_repeat = Layout<Shape<_8, Int<kBlockN>, Int<kPipeSize>>,
                                         Stride<_0, _1, Int<kBlockN>>>;
 
     static constexpr int kCoreMatrixN  = kBlockN_ / 8; 
     using SmemKScaleTiledCopy = TiledCopy<
         SmemCopyAtom_SK_QMK, 
         Layout<Shape<Shape  <_4, _8>,  Shape <_2, Int<kCoreMatrixN>>>, 
                 Stride<Stride<_16, _1>, Stride<_8, _64>>>,
         Tile<_8, Int<kBlockN_>>
     >;
 
     static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element); // 8
     static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
     // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
     // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
     // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
     // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
     // to the same banks.
 
     static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad; // 8
     static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
     
     using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                   Stride<Int<kGmemThreadsPerRow>, _1>>;
     
     static constexpr int kGmemElemsPerLoad_SK_QMK = kGmemElemsPerLoad; 
     static constexpr int kGmemThreadsPerRow_SK_QMK = kBlockN / kGmemElemsPerLoad_SK_QMK;
     static_assert(kNThreads % kGmemThreadsPerRow_SK_QMK == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
 
     using GmemLayoutAtom_SQ_SK_QMK = Layout<Shape<Int<kGmemThreadsPerRow_SK_QMK>>>; 
 
     // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
     // from the same address by the same threadblock. This is slightly faster.
     using Gmem_copy_struct = std::conditional_t<
         Has_cp_async,
         SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
         DefaultCopy
     >; 

     using Gmem_copy_struct_SQ_SK_QMK = std::conditional_t<
         Has_cp_async,
         SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
         DefaultCopy
     >; 
 
     using GmemTiledCopyQKV = decltype(
         make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                         GmemLayoutAtom{},
                         Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
     using GmemTiledCopyO = decltype(
         make_tiled_copy(Copy_Atom<AutoVectorizingCopy, Element>{},
                         GmemLayoutAtom{},
                         Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
 
     static constexpr int kGmemInt4ElemsPerLoad = 32;
     static_assert(kHeadDim % kGmemInt4ElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
     static constexpr int kGmemThreadsPerRowInt4Load = kBlockKGmem / kGmemInt4ElemsPerLoad; 
     static_assert(kNThreads % kGmemThreadsPerRowInt4Load  == 0, "kNThreads must be a multiple of kGmemThreadsPerRowInt4Load");

     static constexpr int GmemThreadLayoutAtomInt4MaxRow = (kNThreads / kGmemThreadsPerRowInt4Load > kBlockN) ? kBlockN : kNThreads / kGmemThreadsPerRowInt4Load;
     static constexpr int GmemInt4QKCopyMaxThreadCnt = GmemThreadLayoutAtomInt4MaxRow * kGmemThreadsPerRowInt4Load;
     using GmemThreadLayoutAtomInt4 = Layout<
                                     Shape<Int<GmemThreadLayoutAtomInt4MaxRow>, Int<kGmemThreadsPerRowInt4Load>>,
                                     Stride<Int<kGmemThreadsPerRowInt4Load>, _1>
                                 >;
     using GmemInt4QKTiledCopy = decltype(
         make_tiled_copy(Copy_Atom<Gmem_copy_struct, cute::uint8_t>{}, 
                         GmemThreadLayoutAtomInt4{}, 
                         Layout<Shape<_1, _16>>{}
     ));

     static_assert(kBlockM_ < kNThreads, "kBlockM_ is too large! Please specify a kBlockM_ that is smaller than kNThreads.");
 

     using GmemKScaleTiledCopy = decltype(make_tiled_copy(
         Copy_Atom<Gmem_copy_struct_SQ_SK_QMK, Element>{}, 
         GmemLayoutAtom_SQ_SK_QMK{},
         Layout<Shape<_8>>{}
     ));
 
     using GmemQMKTiledCopy = GmemKScaleTiledCopy;
 
     using GmemLayoutAtomOaccum = std::conditional_t<
         kBlockKSmem == 32,
         Layout<Shape <_16, _8>,  // Thread layout, 8 threads per row
                Stride< _8, _1>>,
         Layout<Shape <_8, _16>,  // Thread layout, 16 threads per row
                Stride< _16, _1>>
     >;
     using GmemTiledCopyOaccum = decltype(
         make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                         GmemLayoutAtomOaccum{},
                         Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store,
     
     using GmemTiledCopyApprox = TiledCopy<
             Copy_Atom<AutoVectorizingCopy, Element>, 
             Layout< Shape< Shape<_4, _8, _4>, Shape<_2, _2, Int<kCoreMatrixN>>>, 
                     Stride<Stride<Int<2 * kBlockM>, _1, _16>, Stride<Int<kBlockM>, _8, Int<kBlockM * 8>>>>,
             Tile<Int<kBlockM>, Int<kBlockN>>
     >;
 
     using GmemLayoutAtomRotcossin = GmemLayoutAtom;
     using GmemTiledCopyRotcossin = decltype(
         make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                         GmemLayoutAtomRotcossin{},
                         Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
     using GmemTiledCopyRotcossinCont = decltype(
         make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                         GmemLayoutAtomRotcossin{},
                         Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
 
 
 
 };


template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, int kLocalWindow, int kBlockSeg_,
        bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, 
        typename elem_type=cutlass::half_t,
        typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type>>
struct Fp16_Flash_fwd_kernel_traits_v1 : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;

    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;


    static_assert(sizeof(Element) == 2, "Only support fp16 prefill. Weird bugs will happend if using other types.");
    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32; 

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32; // 64 in most cases. 
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32); // 128 in our cases.
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    static constexpr int kBlockSeg = kBlockSeg_;
    static constexpr int kPipeSize = 2;

    static_assert(kBlockM_ % kBlockN_ == 0);
    static_assert(kBlockN_ >= 8, "Please set kBlockN_ more than 8. ");
    static_assert(kBlockM > kNWarps, "Block size is too small. Please set it bigger.");
    static_assert(kLocalWindow % kBlockN_ == 0);

    static constexpr int kNLocalBlocks = kLocalWindow / kBlockN_;

    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,       
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * kNWarps>, _16, _16>>; // No permutation. 
    static_assert(16 * kNWarps >= kBlockM, "16 * kNWarps should be larger than kBlockM");
    
    static constexpr int nWarpM = kBlockM_ / 16 > kNWarps_ ? kNWarps : kBlockM_ / 16;
    static constexpr int nWarpN = nWarpM > kNWarps_ ? 1 : kNWarps_ / nWarpM;
    static_assert(nWarpM * nWarpN >= kNWarps, "kNWarps is too large or M/N block size is too small");

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3 , 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<kBlockKSmem>>, 
                            Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    // ComposedLayout<Swizzle<B,M,S>, _0, Layout> 
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>, Int<kPipeSize>>{}));
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<AutoVectorizingCopy, ElementAccum>;


    static constexpr int smem_size = 99 * 1024; // for 4090

    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutK{}) * sizeof(Element);
    static constexpr int kSmemSize_2byte = Share_Q_K_smem ? std::max(kSmemQSize, kSmemKSize) : kSmemQSize + kSmemKSize;
    static constexpr int allReduceBufSize = kNWarps * 4;
    static constexpr uint32_t kPipemask = 1;

    static constexpr int kSmemSize =  kSmemSize_2byte + allReduceBufSize;
    static_assert(kSmemSize < smem_size, "Block size or kPipeSize is too large!");

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element); // 8
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
    // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
    // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
    // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
    // to the same banks.

    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad; // 8
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >; 
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

    static_assert(kBlockM_ < kNThreads, "kBlockM_ is too large! Please specify a kBlockM_ that is smaller than kNThreads.");

};
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 