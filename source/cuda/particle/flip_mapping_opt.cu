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
#include <functional>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "cuda/aux_buffer_manager.h"
#include "cuda/block_arrangement.h"
#include "cuda/cuda_common_host.h"
#include "cuda/cuda_common_kern.h"
#include "cuda/cuda_debug.h"
#include "cuda/particle/flip_common.cuh"
#include "flip.h"

surface<void, cudaSurfaceType3D> surf;
surface<void, cudaSurfaceType3D> surf_x;
surface<void, cudaSurfaceType3D> surf_y;
surface<void, cudaSurfaceType3D> surf_z;
surface<void, cudaSurfaceType3D> surf_d;
surface<void, cudaSurfaceType3D> surf_t;

const uint32_t kMaxParticlesInCell = 4;
//const uint16_t kInvalidPos = 48128; // __float2half_rn(-1.0f);

struct ParticleFields
{
    uint16_t* position_x_;
    uint16_t* position_y_;
    uint16_t* position_z_;
    uint16_t* velocity_x_;
    uint16_t* velocity_y_;
    uint16_t* velocity_z_;
    uint16_t* density_;
    uint16_t* temperature_;
};

struct FieldWeightAndWeightedSum
{
    float weight_vel_x_;
    float weight_vel_y_;
    float weight_vel_z_;
    float weight_density_;
    float weight_temperature_;
    float sum_vel_x_;
    float sum_vel_y_;
    float sum_vel_z_;
    float sum_density_;
    float sum_temperature_;
};

struct TemporaryStorage
{
    __device__ inline void Store(float* field, float value)
    {
        *field = value;
    }
};

struct AccumulativeStorage
{
    __device__ inline void Store(float* field, float value)
    {
        *field += value;
    }
};

union FloatSignExtraction
{
    float f_;
    struct
    {
        unsigned int mantisa_ : 23;
        unsigned int exponent_ : 8;
        unsigned int sign_ : 1;
    } parts_;
};

// =============================================================================

__global__ void TransferToGridLopKernel(FlipParticles particles,
                                        uint3 volume_size)
{
    // save the lookup table into smem

    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= *particles.num_of_actives_)
        return;

    int x = __float2int_rd(particles.position_x_[i]);
    int y = __float2int_rd(particles.position_y_[i]);
    int z = __float2int_rd(particles.position_z_[i]);
    for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
        // Calculate weighted average to the nearest cell.

        // Use atomic add to the cell fields.
    }
}

namespace kern_launcher
{
void TransferToGridOpt(cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z,
                       cudaArray* density, cudaArray* temperature,
                       const FlipParticles& particles, const FlipParticles& aux,
                       uint3 volume_size, BlockArrangement* ba)
{
    
}
}
