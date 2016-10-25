//
// Fluid3d - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Fluid3d license (* see part 1 below)
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

#include "cuda/block_arrangement.h"
#include "cuda/cuda_common_host.h"
#include "cuda/cuda_common_kern.h"
#include "cuda/cuda_debug.h"
#include "cuda/particle/flip_common.cuh"
#include "flip.h"

texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_x;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_y;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_z;

// =============================================================================

struct AdvectionEuler
{
    __device__ float3 Advect(float3 pos_0, float3 vel_0,
                             float time_step_over_cell_size)
    {
        float pos_x = pos_0.x + vel_0.x * time_step_over_cell_size;
        float pos_y = pos_0.y + vel_0.y * time_step_over_cell_size;
        float pos_z = pos_0.z + vel_0.z * time_step_over_cell_size;

        return make_float3(pos_x, pos_y, pos_z);
    }
};

struct AdvectionMidPoint
{
    __device__ float3 Advect(float3 pos_0, float3 vel_0,
                             float time_step_over_cell_size)
    {
        float mid_x = pos_0.x + 0.5f * time_step_over_cell_size * vel_0.x;
        float mid_y = pos_0.y + 0.5f * time_step_over_cell_size * vel_0.y;
        float mid_z = pos_0.z + 0.5f * time_step_over_cell_size * vel_0.z;
        
        float v_x2 = tex3D(tex_x, mid_x + 0.5f, mid_y,        mid_z);
        float v_y2 = tex3D(tex_y, mid_x,        mid_y + 0.5f, mid_z);
        float v_z2 = tex3D(tex_z, mid_x,        mid_y,        mid_z + 0.5f);
        
        float pos_x = pos_0.x + v_x2 * time_step_over_cell_size;
        float pos_y = pos_0.y + v_y2 * time_step_over_cell_size;
        float pos_z = pos_0.z + v_z2 * time_step_over_cell_size;

        return make_float3(pos_x, pos_y, pos_z);
    }
};

struct AdvectionBogackiShampine
{
    __device__ float3 Advect(float3 pos_0, float3 vel_0,
                             float time_step_over_cell_size)
    {
        float mid_x = pos_0.x + 0.5f * time_step_over_cell_size * vel_0.x;
        float mid_y = pos_0.y + 0.5f * time_step_over_cell_size * vel_0.y;
        float mid_z = pos_0.z + 0.5f * time_step_over_cell_size * vel_0.z;

        float v_x2 = tex3D(tex_x, mid_x + 0.5f, mid_y,        mid_z);
        float v_y2 = tex3D(tex_y, mid_x,        mid_y + 0.5f, mid_z);
        float v_z2 = tex3D(tex_z, mid_x,        mid_y,        mid_z + 0.5f);

        float mid_x2 = pos_0.x + 0.75f * time_step_over_cell_size * v_x2;
        float mid_y2 = pos_0.y + 0.75f * time_step_over_cell_size * v_y2;
        float mid_z2 = pos_0.z + 0.75f * time_step_over_cell_size * v_z2;

        float v_x3 = tex3D(tex_x, mid_x2 + 0.5f, mid_y2,        mid_z2);
        float v_y3 = tex3D(tex_y, mid_x2,        mid_y2 + 0.5f, mid_z2);
        float v_z3 = tex3D(tex_z, mid_x2,        mid_y2,        mid_z2 + 0.5f);

        float c1 = 2.0f / 9.0f * time_step_over_cell_size;
        float c2 = 3.0f / 9.0f * time_step_over_cell_size;
        float c3 = 4.0f / 9.0f * time_step_over_cell_size;

        float pos_x = pos_0.x + c1 * vel_0.x + c2 * v_x2 + c3 * v_x3;
        float pos_y = pos_0.y + c1 * vel_0.y + c2 * v_y2 + c3 * v_y3;
        float pos_z = pos_0.z + c1 * vel_0.z + c2 * v_z2 + c3 * v_z3;

        return make_float3(pos_x, pos_y, pos_z);
    }
};

struct AdvectionRK4
{
    __device__ float3 Advect(float3 pos_0, float3 vel_0,
                             float time_step_over_cell_size)
    {
        float mid_x = pos_0.x + 0.5f * time_step_over_cell_size * vel_0.x;
        float mid_y = pos_0.y + 0.5f * time_step_over_cell_size * vel_0.y;
        float mid_z = pos_0.z + 0.5f * time_step_over_cell_size * vel_0.z;

        float v_x2 = tex3D(tex_x, mid_x + 0.5f, mid_y,        mid_z);
        float v_y2 = tex3D(tex_y, mid_x,        mid_y + 0.5f, mid_z);
        float v_z2 = tex3D(tex_z, mid_x,        mid_y,        mid_z + 0.5f);

        float mid_x2 = pos_0.x + 0.5f * time_step_over_cell_size * v_x2;
        float mid_y2 = pos_0.y + 0.5f * time_step_over_cell_size * v_y2;
        float mid_z2 = pos_0.z + 0.5f * time_step_over_cell_size * v_z2;

        float v_x3 = tex3D(tex_x, mid_x2 + 0.5f, mid_y2,        mid_z2);
        float v_y3 = tex3D(tex_y, mid_x2,        mid_y2 + 0.5f, mid_z2);
        float v_z3 = tex3D(tex_z, mid_x2,        mid_y2,        mid_z2 + 0.5f);

        float mid_x3 = pos_0.x + time_step_over_cell_size * v_x3;
        float mid_y3 = pos_0.y + time_step_over_cell_size * v_y3;
        float mid_z3 = pos_0.z + time_step_over_cell_size * v_z3;

        float v_x4 = tex3D(tex_x, mid_x3 + 0.5f, mid_y3,        mid_z3);
        float v_y4 = tex3D(tex_y, mid_x3,        mid_y3 + 0.5f, mid_z3);
        float v_z4 = tex3D(tex_z, mid_x3,        mid_y3,        mid_z3 + 0.5f);

        float c1 = 1.0f / 6.0f * time_step_over_cell_size;
        float c2 = 2.0f / 6.0f * time_step_over_cell_size;
        float c3 = 2.0f / 6.0f * time_step_over_cell_size;
        float c4 = 1.0f / 6.0f * time_step_over_cell_size;

        float pos_x = pos_0.x + c1 * vel_0.x + c2 * v_x2 + c3 * v_x3 + c4 * v_x4;
        float pos_y = pos_0.y + c1 * vel_0.y + c2 * v_y2 + c3 * v_y3 + c4 * v_y4;
        float pos_z = pos_0.z + c1 * vel_0.z + c2 * v_z2 + c3 * v_z3 + c4 * v_z4;

        return make_float3(pos_x, pos_y, pos_z);
    }
};

// Active particles should be consecutive, but could be freed during the
// routine.
template <typename AdvectionImpl>
__global__ void AdvectParticlesKernel(FlipParticles particles,
                                      uint3 volume_size,
                                      float time_step_over_cell_size,
                                      AdvectionImpl advect)
{
    FlipParticles& p = particles;

    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= *p.num_of_actives_) // Maybe dynamic parallelism is a better
                                 // choice.
        return;

    float x = __half2float(p.position_x_[i]);
    float y = __half2float(p.position_y_[i]);
    float z = __half2float(p.position_z_[i]);

    // The fluid looks less bumpy with the re-sampled velocity. Don't know
    // the exact reason yet.
    float v_x = tex3D(tex_x, x + 0.5f, y,        z);
    float v_y = tex3D(tex_y, x,        y + 0.5f, z);
    float v_z = tex3D(tex_z, x,        y,        z + 0.5f);

    if (IsStopped(v_x, v_y, v_z)) {
        // Don't eliminate the particle. It may contains density/temperature
        // information.
        //
        // We don't need the number of active particles until the sorting is
        // done.
        return;
    }

    // TODO: Velocity dissipation.
    float3 result = advect.Advect(make_float3(x, y, z),
                                  make_float3(v_x, v_y, v_z),
                                  time_step_over_cell_size);
    float pos_x = result.x;
    float pos_y = result.y;
    float pos_z = result.z;

    if (pos_x < 0.0f || pos_x >= volume_size.x)
        p.velocity_x_[i] = 0;

    if (pos_y < 0.0f || pos_y >= volume_size.y)
        p.velocity_y_[i] = 0;

    if (pos_z < 0.0f || pos_z >= volume_size.z)
        p.velocity_z_[i] = 0;

    pos_x = min(max(pos_x, 0.0f), volume_size.x - 1.0f);
    pos_y = min(max(pos_y, 0.0f), volume_size.y - 1.0f);
    pos_z = min(max(pos_z, 0.0f), volume_size.z - 1.0f);

    p.position_x_[i] = __float2half_rn(pos_x);
    p.position_y_[i] = __float2half_rn(pos_y);
    p.position_z_[i] = __float2half_rn(pos_z);

    int xi = static_cast<int>(pos_x);
    int yi = static_cast<int>(pos_y);
    int zi = static_cast<int>(pos_z);

    uint cell_index = (zi * volume_size.y + yi) * volume_size.x + xi;
    p.cell_index_[i] = cell_index;
}

// =============================================================================

namespace kern_launcher
{
void AdvectParticles(const FlipParticles& particles, cudaArray* vel_x,
                     cudaArray* vel_y, cudaArray* vel_z, float time_step,
                     float cell_size, uint3 volume_size, BlockArrangement* ba)
{
    dim3 block;
    dim3 grid;
    ba->ArrangeLinear(&grid, &block, particles.num_of_particles_);

    auto bound_x = BindHelper::Bind(&tex_x, vel_x, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, vel_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, vel_z, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    AdvectionEuler adv_fe;
    AdvectionMidPoint adv_mp;
    AdvectionBogackiShampine adv_bs;
    AdvectionRK4 adv_rk4;

    int order = 3;
    switch (order) {
        case 1:
            AdvectParticlesKernel<<<grid, block>>>(particles, volume_size,
                                                   time_step / cell_size,
                                                   adv_fe);
            break;
        case 2:
            AdvectParticlesKernel<<<grid, block>>>(particles, volume_size,
                                                   time_step / cell_size,
                                                   adv_mp);
            break;
        case 3:
            AdvectParticlesKernel<<<grid, block>>>(particles, volume_size,
                                                   time_step / cell_size,
                                                   adv_bs);
            break;
        case 4:
            AdvectParticlesKernel<<<grid, block>>>(particles, volume_size,
                                                   time_step / cell_size,
                                                   adv_rk4);
            break;
    }

    DCHECK_KERNEL();
}
}
