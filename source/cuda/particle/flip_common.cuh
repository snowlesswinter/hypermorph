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

#include "cuda/cuda_common_kern.h"
#include "cuda/particle/flip.h"

const uint16_t kInvalidPos = 0xFF00;
const uint32_t kMaxNumParticlesPerCell = 6;
const uint32_t kMaxNumSamplesForOneTime = 5;
const uint32_t kMinNumParticlesPerCell = 2;
const bool kHiResCoord = false;

namespace internal
{
__device__ inline float FromFP8(uint32_t x)
{
    uint32_t t = ((x & 0x00FF) << 15) + 0x3F800000;
    return __uint_as_float(t) - 1.0f;
}

__device__ inline float CoordFromFP16(uint16_t x)
{
    return __uint2float_rn(x >> 8) + FromFP8(x);
}

__device__ inline float3 CoordFromFP16(uint16_t x, uint16_t y, uint16_t z)
{
    return make_float3(CoordFromFP16(x), CoordFromFP16(y), CoordFromFP16(z));
}

// Round to nearest.
__device__ inline uint32_t ToFP8(float f)
{
    float t = f + 1.0f;
    return ((__float_as_uint(t) - 0x3F800000 + 0x4000) >> 15);
}

__device__ inline uint16_t CoordToFP16(float f)
{
    uint32_t u = __float2uint_rd(f);
    return static_cast<uint16_t>((u << 8) + ToFP8(fracf(f)));
}
} // namespace internal

__device__ inline void SetUndefined(uint16_t* pos_x)
{
    *pos_x = kInvalidPos;
}

__device__ inline bool IsParticleUndefined(uint16_t x)
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

__device__ inline uint ParticleIndex(uint cell_index)
{
    return cell_index * kMaxNumParticlesPerCell;
}

namespace flip
{
__device__ inline float3 Position(const FlipParticles& p, uint i)
{
    if (kHiResCoord)
        return internal::CoordFromFP16(p.position_x_[i], p.position_y_[i], p.position_z_[i]);

    return Half2Float(p.position_x_[i], p.position_y_[i], p.position_z_[i]);
}

__device__ inline float3 Position32(uint16_t x, uint16_t y, uint16_t z)
{
    if (kHiResCoord)
        return internal::CoordFromFP16(x, y, z);

    return Half2Float(x, y, z);
}

__device__ inline uint16_t Position16(float x)
{
    if (kHiResCoord)
        return internal::CoordToFP16(x);

    return __float2half_rn(x);
}

__device__ inline uint CellIndex(uint16_t pos_x, uint16_t pos_y,
                                 uint16_t pos_z, const uint3& volume_size)
{
    uint xi;
    uint yi;
    uint zi;
    if (kHiResCoord) {
        xi = pos_x >> 8;
        yi = pos_y >> 8;
        zi = pos_z >> 8;
    } else {
        float x = __half2float(pos_x);
        float y = __half2float(pos_y);
        float z = __half2float(pos_z);

        xi = __float2uint_rd(x);
        yi = __float2uint_rd(y);
        zi = __float2uint_rd(z);
    }

    return (__umul24(zi, volume_size.y) + yi) * volume_size.x + xi;
}
} // namespace flip

#endif // _FLIP_COMMON_H_