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

#ifndef _FLIP_COMMON_H_
#define _FLIP_COMMON_H_

#include <stdint.h>

#include "flip.h"

const uint16_t kInvalidPos = 48128; // __float2half_rn(-1.0f);

__device__ inline int CellIndex(uint16_t pos_x, uint16_t pos_y,
                                uint16_t pos_z, const uint3& volume_size)
{
    float x = __half2float(pos_x);
    float y = __half2float(pos_y);
    float z = __half2float(pos_z);

    int xi = __float2int_rd(x);
    int yi = __float2int_rd(y);
    int zi = __float2int_rd(z);

    return __mul24(__mul24(z, volume_size.y) + y, volume_size.x) + x;
}

__device__ inline void SetUndefined(uint16_t* pos_x)
{
    *pos_x = kInvalidPos;
}

__device__ inline bool IsCellUndefined(uint16_t x)
{
    return x == kInvalidPos;
}

__device__ inline bool IsStopped(float v_x, float v_y, float v_z)
{
    // To determine the time to recycle particles.
    const float v_¦Å = 0.0001f;
    return !(v_x > v_¦Å || v_x < -v_¦Å || v_y > v_¦Å || v_y < -v_¦Å ||
             v_z > v_¦Å || v_z < -v_¦Å);
}

__device__ inline void FreeParticle(const FlipParticles& p, uint i)
{
    // Assign an invalid position value to indicate the binding kernel to
    // treat it as a free particle.
    p.position_x_[i] = kInvalidPos;
}

const uint32_t kMaxNumParticlesPerCell = 6;
const uint32_t kMinNumParticlesPerCell = 2;
const uint32_t kMaxNumSamplesForOneTime = 5;

#endif // _FLIP_COMMON_H_