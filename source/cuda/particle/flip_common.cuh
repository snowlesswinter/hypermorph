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

const uint32_t kCellUndefined = static_cast<uint32_t>(-1);

__device__ inline void SetUndefined(uint* cell_index)
{
    *cell_index = kCellUndefined;
}

__device__ inline bool IsCellUndefined(uint cell_index)
{
    return cell_index == kCellUndefined;
}

__device__ inline bool IsStopped(float v_x, float v_y, float v_z)
{
    // To determine the time to recycle particles.
    const float v_epsilon= 0.0001f;
    return !(v_x > v_epsilon || v_x < -v_epsilon || v_y > v_epsilon ||
        v_y < -v_epsilon || v_z > v_epsilon || v_z < -v_epsilon);
}

__device__ inline void FreeParticle(const FlipParticles& p, uint i)
{
    SetUndefined(&p.cell_index_[i]);

    // Assign an invalid position value to indicate the binding kernel to
    // treat it as a free particle.
    p.position_x_[i] = __float2half_rn(-1.0f);
}

const uint32_t kMaxNumParticlesPerCell = 6;
const uint32_t kMinNumParticlesPerCell = 2;
const uint32_t kMaxNumSamplesForOneTime = 5;

#endif // _FLIP_COMMON_H_