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

#ifndef _CUDA_COMMON_KERN_H_
#define _CUDA_COMMON_KERN_H_

__device__ inline uint VolumeX()
{
    return __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
}

__device__ inline uint VolumeY()
{
    return __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
}

__device__ inline uint VolumeZ()
{
    return __umul24(blockIdx.z, blockDim.z) + threadIdx.z;
}

__device__ inline uint LinearIndex()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline uint LinearIndexBlock()
{
    return
        __umul24(__umul24(threadIdx.z, blockDim.y) + threadIdx.y, blockDim.x) +
            threadIdx.x;
}

__device__ inline uint LinearIndexVolume(uint x, uint y, uint z,
                                         const uint3& volume_size)
{
    return (__umul24(z, volume_size.y) + y) * volume_size.x + x;
}

__device__ inline float3 Half2Float(uint16_t x, uint16_t y, uint16_t z)
{
    return make_float3(__half2float(x), __half2float(y), __half2float(z));
}

__device__ inline int3 Float2Int(const float3& f)
{
    return make_int3(__float2int_rd(f.x), __float2int_rd(f.y),
                     __float2int_rd(f.z));
}

__device__ inline float3 Int2Float(const int3& i)
{
    return make_float3(__int2float_rn(i.x), __int2float_rn(i.y),
                       __int2float_rn(i.z));
}

#endif // _CUDA_COMMON_KERN_H_