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
    __shared__ int8_t adj_route[8];
    __shared__ float smem_vx_wsum  [8 * 8 * 8];
    __shared__ float smem_vy_wsum  [8 * 8 * 8];
    __shared__ float smem_vz_wsum  [8 * 8 * 8];
    __shared__ float smem_d_wsum   [8 * 8 * 8];
    __shared__ float smem_t_wsum   [8 * 8 * 8];
    __shared__ float smem_vx_weight[8 * 8 * 8];
    __shared__ float smem_vy_weight[8 * 8 * 8];
    __shared__ float smem_vz_weight[8 * 8 * 8];
    __shared__ float smem_d_weight [8 * 8 * 8];
    __shared__ float smem_t_weight [8 * 8 * 8];

    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (i == 0) {
        adj_route[0] = -64 - 8 - 1;
        adj_route[1] = -64 - 8 - 0;
        adj_route[2] = -64 - 0 - 1;
        adj_route[3] = -64 - 0 - 0;
        adj_route[4] = -0  - 8 - 1;
        adj_route[5] = -0  - 8 - 0;
        adj_route[6] = -0  - 0 - 1;
        adj_route[7] = -0  - 0 - 0;
    }

    __syncthreads();

    if (IsCellUndefined(particles.position_x_[i]))
        return;

    float3 coord = Half2Float(particles.position_x_[i],
                              particles.position_y_[i],
                              particles.position_z_[i]);

    int3 coord_i = Float2Int(coord);

    float3 center = Int2Float(coord_i);
    int3 table_index = Float2Int((coord - center) * 2.0f);
    int table_i = (table_index.z << 1 + table_index.y) << 1 + table_index.x;
    int offset = adj_route[table_i];

    // Classify the particle by its situation of adjacency.
    float3 diff = coord - center;

    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) for (int k = 0; k < 2; k++) {
        int n = (i << 1 + j) << 1 + k;
        int v = adj_table[n];
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
