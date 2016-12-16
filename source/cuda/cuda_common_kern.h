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

template <typename TexType>
__device__ inline float3 LoadVel(TexType tex_x, TexType tex_y, TexType tex_z,
                                 const float3& coord)
{
    return make_float3(tex3D(tex_x, coord.x + 0.5f, coord.y,        coord.z),
                       tex3D(tex_y, coord.x,        coord.y + 0.5f, coord.z),
                       tex3D(tex_z, coord.x,        coord.y,        coord.z + 0.5f));
}

#endif // _CUDA_COMMON_KERN_H_