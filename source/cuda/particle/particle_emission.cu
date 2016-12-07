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
#include "cuda/fluid_impulse.h"
#include "cuda/particle/flip_common.cuh"
#include "flip.h"
#include "random.cuh"

__global__ void EmitParticlesFromSphereKernel(uint16_t* pos_x, uint16_t* pos_y,
                                              uint16_t* pos_z,
                                              uint16_t* density, uint16_t* life,
                                              int* tail, int num_of_particles,
                                              int num_to_emit, float3 location,
                                              float radius, float density_value,
                                              uint random_seed)
{
    uint l = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (l >= num_to_emit)
        return;

    uint i = *tail + l;
    i = i >= num_of_particles ? i - num_of_particles : i;

    uint seed = random_seed + l;
    float3 coord = location + RandomCoordSphere(&seed) * radius;
    pos_x[i]   = __float2half_rn(coord.x);
    pos_y[i]   = __float2half_rn(coord.y);
    pos_z[i]   = __float2half_rn(coord.z);
    density[i] = __float2half_rn(density_value);
    life[i]    = __float2half_rn(1.0f);

    __syncthreads();
    if (l == num_to_emit - 1)
        *tail = i;
}

// =============================================================================

namespace kern_launcher
{
void EmitParticles(uint16_t* pos_x, uint16_t* pos_y, uint16_t* pos_z,
                   uint16_t* density, uint16_t* life, int* tail,
                   int num_of_particles, int num_to_emit, float3 location,
                   float radius, float density_value, uint random_seed,
                   BlockArrangement* ba)
{
    dim3 grid;
    dim3 block;
    ba->ArrangeLinear(&grid, &block, num_to_emit);
    EmitParticlesFromSphereKernel<<<grid, block>>>(
        pos_x, pos_y, pos_z, density, life, tail, num_of_particles, num_to_emit,
        location, radius, density_value, random_seed);
    DCHECK_KERNEL();
}
}
