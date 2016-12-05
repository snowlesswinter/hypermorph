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

#ifndef _RANDOM_H_
#define _RANDOM_H_

#include "third_party/opengl/glew.h"

#include <helper_math.h>

__device__ inline uint Tausworthe(uint z, int s1, int s2, int s3, uint M)
{
    uint b = (((z << s1) ^ z) >> s2);
    return (((z & M) << s3) ^ b);
}

__device__ inline float3 RandomCoord(uint* random_seed)
{
    uint seed = *random_seed;
    uint seed0 = Tausworthe(seed,  (blockIdx.x  + 1) & 0xF, (blockIdx.y  + 2) & 0xF, (blockIdx.z  + 3) & 0xF, 0xFFFFFFFE);
    uint seed1 = Tausworthe(seed0, (threadIdx.x + 1) & 0xF, (threadIdx.y + 2) & 0xF, (threadIdx.z + 3) & 0xF, 0xFFFFFFF8);
    uint seed2 = Tausworthe(seed1, (threadIdx.y + 1) & 0xF, (threadIdx.z + 2) & 0xF, (threadIdx.x + 3) & 0xF, 0xFFFFFFF0);
    uint seed3 = Tausworthe(seed2, (threadIdx.z + 1) & 0xF, (threadIdx.x + 2) & 0xF, (threadIdx.y + 3) & 0xF, 0xFFFFFFE0);

    float rand_x = (seed1 & 127) / 129.5918f - 0.49f;
    float rand_y = (seed2 & 127) / 129.5918f - 0.49f;
    float rand_z = (seed3 & 127) / 129.5918f - 0.49f;

    *random_seed = seed3;
    return make_float3(rand_x, rand_y, rand_z);
}

#endif  // _RANDOM_H_