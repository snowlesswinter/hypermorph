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
#include <math_constants.h>

// Reference:
// http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
//
// Comparison:
//
// * Performance
//
// curand_uniform(XORWOW) is 1.69x faster than the hash function, however,
// random generation is always a tiny part in kernel performance.
//
// Unlike cuRAND, hash function doesn't require initialization before running,
// which could be notable in the case of large amount of threads. 
//
// * Memory footprint
//
// No extra global memory is needed. This is a big advantage against cuRAND,
// since the per-thread curandState is really annoying, let alone the memory
// management code that brought in.
//
// * Portability
//
// Easy to port to HLSL, OpenCL, etc.

// In most scenario, what I need is something that acts randomly, which is not
// necessarily to be strictly statistically correct. The common PRNGs seem
// too expensive in this way.
//
// Besides, most PRNGs are making great efforts in delivering good 'sequences',
// but what I need is actually something that 'spread things out as early as
// possible(going wide)' inside the threads.

// Generates a random number of float between [0, 1).
__device__ inline float WangHash(uint* seed)
{
    uint local_seed = *seed;
    local_seed = (local_seed ^ 61) ^ (local_seed >> 16);
    local_seed *= 9;
    local_seed = local_seed ^ (local_seed >> 4);
    local_seed *= 0x27d4eb2d;
    local_seed = local_seed ^ (local_seed >> 15);
    *seed = local_seed;
    return local_seed * (1.0f / 4294967296.0f);
}

__device__ inline float3 RandomCoordCube(uint* seed)
{
    uint local_seed = *seed;
    float x = WangHash(&local_seed);
    float y = WangHash(&local_seed);
    float z = WangHash(&local_seed);

    *seed = local_seed;
    return make_float3(x, y, z) - 0.49999f;
}

// Generates uniform distributed coordinates within a sphere.
// 
// A possible solution could be found at:
// http://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
//
// But the above method requires Box-Muller transform, which is a bit
// computationally expensive. Moreover, it is not quite friendly to the 0-length
// vector, which may be generated by the hash function.
//
// So alternatively I turn to the spherical coordinates.
// https://cn.mathworks.com/help/matlab/math/numbers-placed-randomly-within-volume-of-sphere.html?requestedDomain=www.mathworks.com

__device__ inline float3 RandomCoordSphere(uint* seed)
{
    uint local_seed = *seed;
    float rvals = 2.0f * WangHash(&local_seed) - 0.9999f;
    float cos_elevation = __fsqrt_rn(1.0f - rvals * rvals);

    float azimuth = (2.0f * CUDART_PI_F) * WangHash(&local_seed);
    float cos_azi;
    float sin_azi;
    __sincosf(azimuth, &sin_azi, &cos_azi);

    float radii = cbrtf(WangHash(&local_seed));

    *seed = local_seed;

    float x = radii * cos_elevation * cos_azi;
    float y = radii * rvals;
    float z = radii * cos_elevation * sin_azi;

    return make_float3(x, y, z);
}

#endif  // _RANDOM_H_