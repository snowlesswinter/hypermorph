//
// Hypermorph - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Hypermorph license (* see part 1 below)
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. Acknowledgement of the
//    original author is required if you publish this in a paper, or use it
//    in a product.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

#include <cassert>
#include <functional>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "cuda/aux_buffer_manager.h"
#include "cuda/block_arrangement.h"
#include "cuda/cuda_common_host.h"
#include "cuda/cuda_common_kern.h"
#include "cuda/cuda_debug.h"
#include "cuda/mem_piece.h"
#include "cuda/multi_precision.cuh"

surface<void, cudaSurfaceType3D> surf;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_0;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_1;
texture<float, cudaTextureType3D, cudaReadModeElementType> texf;
texture<float, cudaTextureType3D, cudaReadModeElementType> texf_0;
texture<float, cudaTextureType3D, cudaReadModeElementType> texf_1;
texture<long2, cudaTextureType3D, cudaReadModeElementType> texd;
texture<long2, cudaTextureType3D, cudaReadModeElementType> texd_0;
texture<long2, cudaTextureType3D, cudaReadModeElementType> texd_1;

struct UpperBoundaryHandlerNeumann
{
    __device__ void HandleUpperBoundary(float* north, float center, int y,
                                        int height)
    {
    }
};

struct UpperBoundaryHandlerOutflow
{
    __device__ void HandleUpperBoundary(float* north, float center, int y,
                                        int height)
    {
        if (y == height - 1) {
            if (center > 0.0f)
                *north = -center;
            else
                *north = 0.0f;
        }
    }
};

// =============================================================================

template <typename StorageType, typename UpperBoundaryHandler>
__global__ void ApplyStencilKernel(uint3 volume_size,
                                   UpperBoundaryHandler handler)
{
    using FPType = typename Tex3d<StorageType>::ValType;

    uint x = VolumeX();
    uint y = VolumeY();
    uint z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    Tex3d<StorageType> t3d;
    FPType near   = t3d(TexSel<StorageType>::Tex(tex, texf, texd), x,        y,        z - 1.0f);
    FPType south  = t3d(TexSel<StorageType>::Tex(tex, texf, texd), x,        y - 1.0f, z);
    FPType west   = t3d(TexSel<StorageType>::Tex(tex, texf, texd), x - 1.0f, y,        z);
    FPType center = t3d(TexSel<StorageType>::Tex(tex, texf, texd), x,        y,        z);
    FPType east   = t3d(TexSel<StorageType>::Tex(tex, texf, texd), x + 1.0f, y,        z);
    FPType north  = t3d(TexSel<StorageType>::Tex(tex, texf, texd), x,        y + 1.0f, z);
    FPType far    = t3d(TexSel<StorageType>::Tex(tex, texf, texd), x,        y,        z + 1.0f);

    //handler.HandleUpperBoundary(&north, center, y, volume_size.y);

    // NOTE: The coefficient 'h^2' is premultiplied in the divergence kernel.
    FPType v = (north + south + east + west + far + near - 6.0f * center);
    auto r = __float2half_rn(v);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

template <typename StorageType>
__global__ void ScaleVectorKernel(Tex3d<StorageType>::ValType* coef,
                                  uint3 volume_size)
{
    using FPType = typename Tex3d<StorageType>::ValType;

    uint x = VolumeX();
    uint y = VolumeY();
    uint z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    Tex3d<StorageType> t3d;
    FPType e1 = t3d(TexSel<StorageType>::Tex(tex_1, texf_1, texd_1), x, y, z);

    auto r = __float2half_rn(*coef * e1);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

template <typename StorageType>
__global__ void ScaledAddKernel(Tex3d<StorageType>::ValType* coef, float sign,
                                uint3 volume_size)
{
    using FPType = typename Tex3d<StorageType>::ValType;

    uint x = VolumeX();
    uint y = VolumeY();
    uint z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    Tex3d<StorageType> t3d;
    FPType e0 = t3d(TexSel<StorageType>::Tex(tex_0, texf_0, texd_0), x, y, z);
    FPType e1 = t3d(TexSel<StorageType>::Tex(tex_1, texf_1, texd_1), x, y, z);

    auto r = __float2half_rn(e0 + *coef * sign * e1);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

template <typename StorageType>
struct SchemeDefault
{
    using FPType = typename Tex3d<StorageType>::ValType;
    __device__ FPType Load(uint i, uint row_stride, uint slice_stride)
    {
        uint z = i / slice_stride;
        uint y = (i % slice_stride) / row_stride;
        uint x = i % row_stride;

        float xf = static_cast<float>(x);
        float yf = static_cast<float>(y);
        float zf = static_cast<float>(z);

        Tex3d<StorageType> t3d;
        FPType ¦Õ0 = t3d(TexSel<StorageType>::Tex(tex_0, texf_0, texd_0), xf, yf, zf);
        FPType ¦Õ1 = t3d(TexSel<StorageType>::Tex(tex_1, texf_1, texd_1), xf, yf, zf);
        return ¦Õ0 * ¦Õ1;
    }
    __device__ void Save(FPType* dest, FPType result)
    {
        *dest = result;
    }

    __host__ void Init() {}
};

template <typename StorageType>
struct SchemeAlpha : public SchemeDefault<StorageType>
{
    using FPType = typename Tex3d<StorageType>::ValType;
    __device__ void Save(FPType* dest, FPType result)
    {
        if (result > 0.00000001f || result < -0.00000001f)
            *dest = *rho_ / result;
        else
            *dest = 0.0f;
    }

    template <typename ScalarPackType>
    __host__ void Init(const ScalarPackType& rho)
    {
        AssignScalar<FPType>(&rho_, rho);
    }

    FPType* rho_;
};

template <typename StorageType>
struct SchemeBeta : public SchemeDefault<StorageType>
{
    using FPType = typename Tex3d<StorageType>::ValType;
    __device__ void Save(FPType* dest, FPType result)
    {
        *dest = result;

        FPType t = *rho_;
        if (t > 0.00000001f || t < -0.00000001f)
            *beta_ = result / t;
        else
            *beta_ = 0;
    }

    template <typename ScalarPackType>
    __host__ void Init(const ScalarPackType& beta, const ScalarPackType& rho)
    {
        AssignScalar<FPType>(&beta_, beta);
        AssignScalar<FPType>(&rho_, rho);
    }

    FPType* rho_;
    FPType* beta_;
};

#include "volume_reduction.cuh"

// =============================================================================

template <typename StorageType>
struct ApplyStencilMeta
{
    static void Invoke(const dim3& grid, const dim3& block,
                       const uint3& volume_size, bool outflow)
    {
        UpperBoundaryHandlerOutflow outflow_handler;
        UpperBoundaryHandlerNeumann neumann_handler;
        if (outflow)
            ApplyStencilKernel<StorageType><<<grid, block>>>(volume_size,
                                                             outflow_handler);
        else
            ApplyStencilKernel<StorageType><<<grid, block>>>(volume_size,
                                                             neumann_handler);
    }
};

DECLARE_KERNEL_META(
    ScaledAddKernel,
    MAKE_INVOKE_DECLARATION(const MemPiece& coef, float sign,
                            const uint3& volume_size),
    coef.AsType<FPType>(), sign, volume_size);

DECLARE_KERNEL_META(
    ScaleVectorKernel,
    MAKE_INVOKE_DECLARATION(const MemPiece& coef, const uint3& volume_size),
    coef.AsType<FPType>(), volume_size);

// =============================================================================

void LaunchApplyStencil(cudaArray* aux, cudaArray* search, bool outflow,
                        uint3 volume_size, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, aux) != cudaSuccess)
        return;

    auto bound = SelectiveBind(search, false, cudaFilterModePoint,
                               cudaAddressModeClamp, &tex, &texf, &texd);
    if (!bound.Succeeded())
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangeRowScan(&block, &grid, volume_size);

    InvokeKernel<ApplyStencilMeta>(bound, grid, block, volume_size, outflow);
    DCHECK_KERNEL();
}

void LaunchComputeAlpha(const MemPiece& alpha, const MemPiece& rho,
                        cudaArray* vec0, cudaArray* vec1, uint3 volume_size,
                        BlockArrangement* ba, AuxBufferManager* bm)
{
    auto bound_0 = SelectiveBind(vec0, false, cudaFilterModePoint,
                                 cudaAddressModeClamp, &tex_0, &texf_0,
                                 &texd_0);
    if (!bound_0.Succeeded())
        return;

    auto bound_1 = SelectiveBind(vec1, false, cudaFilterModePoint,
                                 cudaAddressModeClamp, &tex_1, &texf_1,
                                 &texd_1);
    if (!bound_1.Succeeded())
        return;

    ScalarPieces alpha_typed = CreateScalarPieces(alpha);
    ScalarPieces rho_typed   = CreateScalarPieces(rho);

    InvokeReduction<SchemeAlpha>(alpha_typed, bound_0, volume_size, ba, bm,
                                 rho_typed);
    DCHECK_KERNEL();
}

void LaunchComputeRho(const MemPiece& rho, cudaArray* search,
                      cudaArray* residual, uint3 volume_size,
                      BlockArrangement* ba, AuxBufferManager* bm)
{
    auto bound_0 = SelectiveBind(search, false, cudaFilterModePoint,
                                 cudaAddressModeClamp, &tex_0, &texf_0,
                                 &texd_0);
    if (!bound_0.Succeeded())
        return;

    auto bound_1 = SelectiveBind(residual, false, cudaFilterModePoint,
                                 cudaAddressModeClamp, &tex_1, &texf_1,
                                 &texd_1);
    if (!bound_1.Succeeded())
        return;

    ScalarPieces rho_typed = CreateScalarPieces(rho);

    InvokeReduction<SchemeDefault>(rho_typed, bound_0, volume_size, ba, bm);
    DCHECK_KERNEL();
}

void LaunchComputeRhoAndBeta(const MemPiece& beta, const MemPiece& rho_new,
                             const MemPiece& rho, cudaArray* vec0,
                             cudaArray* vec1, uint3 volume_size,
                             BlockArrangement* ba, AuxBufferManager* bm)
{
    auto bound_0 = SelectiveBind(vec0, false, cudaFilterModePoint,
                                 cudaAddressModeClamp, &tex_0, &texf_0,
                                 &texd_0);
    if (!bound_0.Succeeded())
        return;

    auto bound_1 = SelectiveBind(vec1, false, cudaFilterModePoint,
                                 cudaAddressModeClamp, &tex_1, &texf_1,
                                 &texd_1);
    if (!bound_1.Succeeded())
        return;

    ScalarPieces rho_new_typed = CreateScalarPieces(rho_new);
    ScalarPieces beta_typed    = CreateScalarPieces(beta);
    ScalarPieces rho_typed     = CreateScalarPieces(rho);

    InvokeReduction<SchemeBeta>(rho_new_typed, bound_0, volume_size, ba, bm,
                                beta_typed, rho_typed);
}

void LaunchScaledAdd(cudaArray* dest, cudaArray* v0, cudaArray* v1,
                     const MemPiece& coef, float sign, uint3 volume_size,
                     BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, dest) != cudaSuccess)
        return;

    auto bound_1 = SelectiveBind(v1, false, cudaFilterModePoint,
                                 cudaAddressModeClamp, &tex_1, &texf_1,
                                 &texd_1);
    if (!bound_1.Succeeded())
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangeRowScan(&block, &grid, volume_size);
    if (v0) {
        auto bound_0 = SelectiveBind(v0, false, cudaFilterModePoint,
                                     cudaAddressModeClamp, &tex_0, &texf_0,
                                     &texd_0);
        if (!bound_0.Succeeded())
            return;

        InvokeKernel<ScaledAddKernelMeta>(bound_1, grid, block, coef, sign,
                                          volume_size);
    } else {
        InvokeKernel<ScaleVectorKernelMeta>(bound_1, grid, block, coef,
                                            volume_size);
    }

    DCHECK_KERNEL();
}
