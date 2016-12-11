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

#include "cuda_core.h"

#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "cuda_common_host.h"
#include "cuda_common_kern.h"
#include "cuda/cuda_debug.h"

surface<void, cudaSurfaceType3D> clear_volume;

template <typename T>
struct VolumeElementConstructor
{ 
    __device__ static inline T Construct(const float4& value)
    {
        return value.x;
    }
};

template <>
struct VolumeElementConstructor<float2>
{
    __device__ static inline float2 Construct(const float4& value)
    {
        return make_float2(value.x, value.y);
    }
};

template <>
struct VolumeElementConstructor<float4>
{
    __device__ static inline float4 Construct(const float4& value)
    {
        return value;
    }
};

template <>
struct VolumeElementConstructor<ushort>
{
    __device__ static inline ushort Construct(const float4& value)
    {
        return __float2half_rn(value.x);
    }
};

template <>
struct VolumeElementConstructor<ushort2>
{
    __device__ static inline ushort2 Construct(const float4& value)
    {
        return make_ushort2(__float2half_rn(value.x),
                            __float2half_rn(value.y));
    }
};

template <>
struct VolumeElementConstructor<ushort4>
{
    __device__ static inline ushort4 Construct(const float4& value)
    {
        return make_ushort4(__float2half_rn(value.x),
                            __float2half_rn(value.y),
                            __float2half_rn(value.z),
                            __float2half_rn(value.w));
    }
};

template <typename VolumeElementType>
__global__ void ClearVolumeKernel(float4 value, uint3 volume_size)
{
    uint x = VolumeX();
    uint y = VolumeY();
    uint z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    VolumeElementType raw =
        VolumeElementConstructor<VolumeElementType>::Construct(value);
    surf3Dwrite(raw, clear_volume, x * sizeof(raw), y, z, cudaBoundaryModeTrap);
}

__global__ void CopyToVboKernel(void* point_vbo, void* extra_vbo,
                                uint16_t* pos_x, uint16_t* pos_y,
                                uint16_t* pos_z, uint16_t* density,
                                uint16_t* temperature, float crit_density,
                                int* num_of_active_particles,
                                int num_of_particles)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num_of_particles)
        return;

    int stride = 3;
    bool skip = (num_of_active_particles && i >= *num_of_active_particles) ||
        __half2float(density[i]) < crit_density;

    uint16_t* p_buf = reinterpret_cast<uint16_t*>(point_vbo);
    p_buf[i * stride    ] = skip ? __float2half_rn(-100000.0f) : pos_x[i];
    p_buf[i * stride + 1] = pos_y[i];
    p_buf[i * stride + 2] = pos_z[i];


    //uint16_t* e_buf = reinterpret_cast<uint16_t*>(extra_vbo);
    //e_buf[i] = temperature[i];
}

// =============================================================================

bool IsHalf1Or2Or4(const cudaChannelFormatDesc& desc)
{
    if (desc.f != cudaChannelFormatKindFloat)
        return false;

    return desc.x == 16 &&
        ((desc.y == 0 && desc.z == 0 && desc.w == 0) ||
            (desc.y == 16 && desc.z == 0 && desc.w == 0)||
            (desc.y == 16 && desc.z == 16 && desc.w == 16));
}

bool IsFloat1Or2Or4(const cudaChannelFormatDesc& desc)
{
    if (desc.f != cudaChannelFormatKindFloat)
        return false;

    return desc.x == 32 &&
        ((desc.y == 0 && desc.z == 0 && desc.w == 0) ||
            (desc.y == 32 && desc.z == 0 && desc.w == 0)||
            (desc.y == 32 && desc.z == 32 && desc.w == 32));
}

bool IsCompliant(const cudaChannelFormatDesc& desc)
{
    return IsHalf1Or2Or4(desc) || IsFloat1Or2Or4(desc);
}

namespace kern_launcher
{
void ClearVolume(cudaArray* dest_array, const float4& value,
                 const uint3& volume_size, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&clear_volume, dest_array) != cudaSuccess)
        return;

    cudaChannelFormatDesc desc;
    cudaError_t result = cudaGetChannelDesc(&desc, dest_array);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 grid;
    dim3 block;
    ba->ArrangeRowScan(&grid, &block, volume_size);

    assert(IsCompliant(desc));
    if (desc.x == 16 && desc.y == 0 && desc.z == 0 && desc.w == 0 &&
            desc.f == cudaChannelFormatKindFloat)
        ClearVolumeKernel<ushort><<<grid, block>>>(value, volume_size);
    else if (desc.x == 16 && desc.y == 16 && desc.z == 0 && desc.w == 0 &&
             desc.f == cudaChannelFormatKindFloat)
        ClearVolumeKernel<ushort2><<<grid, block>>>(value, volume_size);
    else if (desc.x == 16 && desc.y == 16 && desc.z == 16 && desc.w == 16 &&
             desc.f == cudaChannelFormatKindFloat)
        ClearVolumeKernel<ushort4><<<grid, block>>>(value, volume_size);
    else if (desc.x == 32 && desc.y == 0 && desc.z == 0 && desc.w == 0 &&
            desc.f == cudaChannelFormatKindFloat)
        ClearVolumeKernel<float><<<grid, block>>>(value, volume_size);
    else if (desc.x == 32 && desc.y == 32 && desc.z == 0 && desc.w == 0 &&
             desc.f == cudaChannelFormatKindFloat)
        ClearVolumeKernel<float2><<<grid, block>>>(value, volume_size);
    else if (desc.x == 32 && desc.y == 32 && desc.z == 32 && desc.w == 32 &&
             desc.f == cudaChannelFormatKindFloat)
        ClearVolumeKernel<float4><<<grid, block>>>(value, volume_size);
}

void CopyToVbo(void* point_vbo, void* extra_vbo, uint16_t* pos_x,
               uint16_t* pos_y, uint16_t* pos_z, uint16_t* density,
               uint16_t* temperature, float crit_density,
               int* num_of_active_particles, int num_of_particles,
               BlockArrangement* ba)
{
    dim3 block;
    dim3 grid;
    ba->ArrangeLinear(&grid, &block, num_of_particles);
    CopyToVboKernel<<<grid, block>>>(point_vbo, extra_vbo, pos_x, pos_y, pos_z,
                                     density, temperature, crit_density,
                                     num_of_active_particles, num_of_particles);
    DCHECK_KERNEL();
}
}