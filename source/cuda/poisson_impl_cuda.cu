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

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "cuda/block_arrangement.h"
#include "cuda/cuda_common_host.h"
#include "cuda/cuda_common_kern.h"
#include "cuda/cuda_debug.h"
#include "cuda/multi_precision.cuh"

surface<void, cudaSurfaceType3D> surf;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_b;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_fine;
texture<float, cudaTextureType3D, cudaReadModeElementType> texf;
texture<float, cudaTextureType3D, cudaReadModeElementType> texf_b;
texture<float, cudaTextureType3D, cudaReadModeElementType> texf_fine;
texture<long2, cudaTextureType3D, cudaReadModeElementType> texd;
texture<long2, cudaTextureType3D, cudaReadModeElementType> texd_b;
texture<long2, cudaTextureType3D, cudaReadModeElementType> texd_fine;

template <typename StorageType>
__global__ void ComputeResidualKernel(uint3 volume_size)
{
    using FPType = typename Tex3d<StorageType>::ValType;

    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    Tex3d<StorageType> t3d;
    FPType near   = t3d(TexSel<StorageType>::Tex(tex, texf, texd),       x,        y,        z - 1.0f);
    FPType south  = t3d(TexSel<StorageType>::Tex(tex, texf, texd),       x,        y - 1.0f, z);
    FPType west   = t3d(TexSel<StorageType>::Tex(tex, texf, texd),       x - 1.0f, y,        z);
    FPType center = t3d(TexSel<StorageType>::Tex(tex, texf, texd),       x,        y,        z);
    FPType east   = t3d(TexSel<StorageType>::Tex(tex, texf, texd),       x + 1.0f, y,        z);
    FPType north  = t3d(TexSel<StorageType>::Tex(tex, texf, texd),       x,        y + 1.0f, z);
    FPType far    = t3d(TexSel<StorageType>::Tex(tex, texf, texd),       x,        y,        z + 1.0f);
    FPType b      = t3d(TexSel<StorageType>::Tex(tex_b, texf_b, texd_b), x,        y,        z);

    // TODO: Should enforce the same boundary conditions as in relax kernel.

    FPType v = b - (north + south + east + west + far + near - 6.0f * center);
    t3d.Store(v, surf, x, y, z);
}

// TODO: Support fp64.
template <typename StorageType>
__global__ void ProlongateLerpKernel(uint3 volume_size)
{
    using FPType = typename Tex3d<StorageType>::ValType;

    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = (make_float3(x, y, z) + 0.5f) * 0.5f;

    Tex3d<StorageType> t3d;
    FPType v = t3d(TexSel<StorageType>::Tex(tex, texf, texd), coord.x, coord.y, coord.z);
    t3d.Store(v, surf, x, y, z);
}

template <typename StorageType>
__global__ void ProlongateErrorLerpKernel(uint3 volume_size)
{
    using FPType = typename Tex3d<StorageType>::ValType;

    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = (make_float3(x, y, z) + 0.5f) * 0.5f;

    Tex3d<StorageType> t3d;
    FPType coarse = t3d(TexSel<StorageType>::Tex(tex, texf, texd), coord.x, coord.y, coord.z);
    FPType fine   = t3d(TexSel<StorageType>::Tex(tex_fine, texf_fine, texd_fine), x, y, z);

    t3d.Store(fine + coarse, surf, x, y, z);
}

template <typename StorageType>
__global__ void RestrictLerpKernel(uint3 volume_size)
{
    using FPType = typename Tex3d<StorageType>::ValType;

    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = (make_float3(x, y, z) + 0.5f) * 2.0f;

    Tex3d<StorageType> t3d;
    FPType v = t3d(TexSel<StorageType>::Tex(tex, texf, texd), coord.x, coord.y, coord.z) * 4.0f;

    t3d.Store(v, surf, x, y, z);
}

// =============================================================================

DECLARE_KERNEL_META(
    ComputeResidualKernel,
    MAKE_INVOKE_DECLARATION(const uint3& volume_size),
    volume_size);

DECLARE_KERNEL_META(
    ProlongateLerpKernel,
    MAKE_INVOKE_DECLARATION(const uint3& volume_size),
    volume_size);

DECLARE_KERNEL_META(
    ProlongateErrorLerpKernel,
    MAKE_INVOKE_DECLARATION(const uint3& volume_size),
    volume_size);

DECLARE_KERNEL_META(
    RestrictLerpKernel,
    MAKE_INVOKE_DECLARATION(const uint3& volume_size),
    volume_size);

// =============================================================================
namespace kern_launcher
{
void ComputeResidual(cudaArray* r, cudaArray* u, cudaArray* b,
                     uint3 volume_size, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, r) != cudaSuccess)
        return;

    auto bound_u = SelectiveBind(u, false, cudaFilterModePoint,
                                 cudaAddressModeClamp, &tex, &texf, &texd);
    if (!bound_u.Succeeded())
        return;

    auto bound_b = SelectiveBind(b, false, cudaFilterModePoint,
                                 cudaAddressModeClamp, &tex_b, &texf_b,
                                 &texd_b);
    if (!bound_b.Succeeded())
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangeRowScan(&block, &grid, volume_size);

    InvokeKernel<ComputeResidualKernelMeta>(bound_u, grid, block, volume_size);
    DCHECK_KERNEL();
}

void Prolongate(cudaArray* fine, cudaArray* coarse, uint3 volume_size_fine,
                BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, fine) != cudaSuccess)
        return;

    auto bound_coarse = SelectiveBind(coarse, false, cudaFilterModeLinear,
                                      cudaAddressModeClamp, &tex, &texf, &texd);
    if (!bound_coarse.Succeeded())
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size_fine);
    InvokeKernel<ProlongateLerpKernelMeta>(bound_coarse, grid, block,
                                           volume_size_fine);
    DCHECK_KERNEL();
}

void ProlongateError(cudaArray* fine, cudaArray* coarse, uint3 volume_size_fine,
                     BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, fine) != cudaSuccess)
        return;

    auto bound_coarse = SelectiveBind(coarse, false, cudaFilterModeLinear,
                                      cudaAddressModeClamp, &tex, &texf, &texd);
    if (!bound_coarse.Succeeded())
        return;

    auto bound_fine = SelectiveBind(fine, false, cudaFilterModePoint,
                                    cudaAddressModeClamp, &tex_fine, &texf_fine,
                                    &texd_fine);
    if (!bound_fine.Succeeded())
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size_fine);
    InvokeKernel<ProlongateErrorLerpKernelMeta>(bound_coarse, grid, block,
                                                volume_size_fine);
    DCHECK_KERNEL();
}

void Restrict(cudaArray* coarse, cudaArray* fine, uint3 volume_size,
              BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, coarse) != cudaSuccess)
        return;

    auto bound = SelectiveBind(fine, false, cudaFilterModeLinear,
                               cudaAddressModeClamp, &tex, &texf, &texd);
    if (!bound.Succeeded())
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    InvokeKernel<RestrictLerpKernelMeta>(bound, grid, block, volume_size);
    DCHECK_KERNEL();
}
}
