//
// Fluid3d - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Fluid3d license (* see part 1 below)
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

#include "aux_buffer_manager.h"
#include "block_arrangement.h"
#include "cuda_common_host.h"
#include "cuda_common_kern.h"

surface<void, cudaSurfaceType3D> surf;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_0;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_1;

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

template <typename UpperBoundaryHandler>
__global__ void ApplyStencilKernel(uint3 volume_size,
                                   UpperBoundaryHandler handler)
{
    uint x = VolumeX();
    uint y = VolumeY();
    uint z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float near =   tex3D(tex, x,        y,        z - 1.0f);
    float south =  tex3D(tex, x,        y - 1.0f, z);
    float west =   tex3D(tex, x - 1.0f, y,        z);
    float center = tex3D(tex, x,        y,        z);
    float east =   tex3D(tex, x + 1.0f, y,        z);
    float north =  tex3D(tex, x,        y + 1.0f, z);
    float far =    tex3D(tex, x,        y,        z + 1.0f);

    handler.HandleUpperBoundary(&north, center, y, volume_size.y);

    // NOTE: The coefficient 'h^2' is premultiplied in the divergence kernel.
    float v = (north + south + east + west + far + near - 6.0f * center);
    auto r = __float2half_rn(v);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

__global__ void ScaleVectorKernel(float* coef, uint3 volume_size)
{
    uint x = VolumeX();
    uint y = VolumeY();
    uint z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float e1 = tex3D(tex_1, x, y, z);

    auto r = __float2half_rn(*coef * e1);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

__global__ void ScaledAddKernel(float* coef, float sign, uint3 volume_size)
{
    uint x = VolumeX();
    uint y = VolumeY();
    uint z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float e0 = tex3D(tex_0, x, y, z);
    float e1 = tex3D(tex_1, x, y, z);

    auto r = __float2half_rn(e0 + *coef * sign * e1);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

struct SchemeDefault
{
    __device__ float Load(uint i, uint row_stride, uint slice_stride)
    {
        uint z = i / slice_stride;
        uint y = (i % slice_stride) / row_stride;
        uint x = i % row_stride;

        float ��0 = tex3D(tex_0, static_cast<float>(x), static_cast<float>(y),
                         static_cast<float>(z));
        float ��1 = tex3D(tex_1, static_cast<float>(x), static_cast<float>(y),
                         static_cast<float>(z));
        return ��0 * ��1;
    }
    __device__ void Save(float* dest, float result)
    {
        *dest = result;
    }
};

struct SchemeAlpha : public SchemeDefault
{
    __device__ void Save(float* dest, float result)
    {
        if (result > 0.00000001f || result < -0.00000001f)
            *dest = *rho_ / result;
        else
            *dest = 0.0f;
    }

    float* rho_;
};

struct SchemeBeta : public SchemeDefault
{
    __device__ void Save(float* dest, float result)
    {
        *dest = result;

        float t = *rho_;
        if (t > 0.00000001f || t < -0.00000001f)
            *beta_ = result / t;
        else
            *beta_ = 0;
    }

    float* rho_;
    float* beta_;
};

#include "volume_reduction.cuh"

// =============================================================================

void LaunchApplyStencil(cudaArray* aux, cudaArray* search, bool outflow,
                        uint3 volume_size, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, aux) != cudaSuccess)
        return;

    auto bound = BindHelper::Bind(&tex, search, false, cudaFilterModePoint,
                                  cudaAddressModeClamp);
    if (bound.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangeRowScan(&block, &grid, volume_size);

    UpperBoundaryHandlerOutflow outflow_handler;
    UpperBoundaryHandlerNeumann neumann_handler;
    if (outflow)
        ApplyStencilKernel<<<grid, block>>>(volume_size, outflow_handler);
    else
        ApplyStencilKernel<<<grid, block>>>(volume_size, neumann_handler);
}

void LaunchComputeAlpha(float* alpha, float* rho, cudaArray* vec0,
                        cudaArray* vec1, uint3 volume_size,
                        BlockArrangement* ba, AuxBufferManager* bm)
{
    auto bound_0 = BindHelper::Bind(&tex_0, vec0, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_0.error() != cudaSuccess)
        return;

    auto bound_1 = BindHelper::Bind(&tex_1, vec1, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_1.error() != cudaSuccess)
        return;

    SchemeAlpha scheme;
    scheme.rho_ = rho;
    ReduceVolume(alpha, scheme, volume_size, ba, bm);
}

void LaunchComputeRho(float* rho, cudaArray* search, cudaArray* residual,
                      uint3 volume_size, BlockArrangement* ba,
                      AuxBufferManager* bm)
{
    auto bound_0 = BindHelper::Bind(&tex_0, search, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_0.error() != cudaSuccess)
        return;

    auto bound_1 = BindHelper::Bind(&tex_1, residual, false,
                                    cudaFilterModePoint, cudaAddressModeClamp);
    if (bound_1.error() != cudaSuccess)
        return;

    SchemeDefault scheme;
    ReduceVolume(rho, scheme, volume_size, ba, bm);
}

void LaunchComputeRhoAndBeta(float* beta, float* rho_new, float* rho,
                             cudaArray* vec0, cudaArray* vec1,
                             uint3 volume_size, BlockArrangement* ba,
                             AuxBufferManager* bm)
{
    
    auto bound_0 = BindHelper::Bind(&tex_0, vec0, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_0.error() != cudaSuccess)
        return;

    auto bound_1 = BindHelper::Bind(&tex_1, vec1, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_1.error() != cudaSuccess)
        return;

    SchemeBeta scheme;
    scheme.beta_ = beta;
    scheme.rho_ = rho;
    ReduceVolume(rho_new, scheme, volume_size, ba, bm);
}

void LaunchScaledAdd(cudaArray* dest, cudaArray* v0, cudaArray* v1, float* coef,
                     float sign, uint3 volume_size, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, dest) != cudaSuccess)
        return;

    auto bound_1 = BindHelper::Bind(&tex_1, v1, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_1.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangeRowScan(&block, &grid, volume_size);
    if (v0) {
        auto bound_0 = BindHelper::Bind(&tex_0, v0, false, cudaFilterModePoint,
                                        cudaAddressModeClamp);
        if (bound_0.error() != cudaSuccess)
            return;

        ScaledAddKernel<<<grid, block>>>(coef, sign, volume_size);
    } else {
        ScaleVectorKernel<<<grid, block>>>(coef, volume_size);
    }
}