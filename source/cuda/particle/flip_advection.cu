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

#include "cuda/block_arrangement.h"
#include "cuda/cuda_common_host.h"
#include "cuda/cuda_common_kern.h"
#include "cuda/cuda_debug.h"
#include "cuda/particle/flip_common.cuh"
#include "flip.h"

texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_x;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_y;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_z;

#include "particle_advection.cuh"

namespace
{
// Active particles are always *NOT* consecutive.
template <typename AdvectionImpl>
__global__ void AdvectParticlesKernel(FlipParticles particles, float3 bounds,
                                      float time_step_over_cell_size,
                                      bool outflow, AdvectionImpl advect)
{
    FlipParticles& p = particles;

    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= p.num_of_particles_)
        return;

    uint16_t xh = p.position_x_[i];
    if (IsCellUndefined(xh))
        return;

    float x = __half2float(xh);
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

    float3 result = advect.Advect(make_float3(x, y, z),
                                  make_float3(v_x, v_y, v_z),
                                  time_step_over_cell_size);

    bool out_of_bounds = false;
    if (result.x < 0.0f || result.x > bounds.x)
        p.velocity_x_[i] = 0;

    if (result.y < 0.0f || result.y > bounds.y) {
        if (outflow)
            out_of_bounds = true;
        else
            p.velocity_y_[i] = 0;
    }

    if (result.z < 0.0f || result.z > bounds.z)
        p.velocity_z_[i] = 0;

    if (out_of_bounds) {
        FreeParticle(particles, i);
    } else {
        float3 pos = clamp(result, make_float3(0.0f), bounds);

        p.position_x_[i] = __float2half_rn(pos.x);
        p.position_y_[i] = __float2half_rn(pos.y);
        p.position_z_[i] = __float2half_rn(pos.z);
    }
}

template <typename TexType>
__device__ void LoadVelToSmem_8x8x8(float3* smem, TexType tex_x_obj,
                                    TexType tex_y_obj, TexType tex_z_obj,
                                    const uint3& volume_size)
{
    // Linear index in the kernel block.
    uint i = LinearIndexBlock();

    uint smem_block_width = 16;
    uint smem_slice_size = 16 * 16;
    uint smem_stride = 8 * 8 * 8;

    // Shared memory block(including halo) coordinates.
    uint sz = i > smem_slice_size ? 1 : 0;
    uint t =  i - __umul24(sz, smem_slice_size);
    uint sy = t / smem_block_width;
    uint sx = t - __umul24(sy, smem_block_width);

    // Grid coordinates.
    //
    // Note that we still need to use linear filtering, as some particles may
    // go out of the block.
    float gx = __uint2float_rn(blockIdx.x * blockDim.x + sx - 4) + 0.5f;
    float gy = __uint2float_rn(blockIdx.y * blockDim.y + sy - 4) + 0.5f;
    float gz = __uint2float_rn(blockIdx.z * blockDim.z + sz - 4) + 0.5f;

    for (int j = 0; j < 8; j++)
        smem[i + j * smem_stride].x = tex3D(tex_x_obj, gx + 0.5f, gy, gz + 2.0f * j);

    for (int j = 0; j < 8; j++)
        smem[i + j * smem_stride].y = tex3D(tex_y_obj, gx, gy + 0.5f, gz + 2.0f * j);

    for (int j = 0; j < 8; j++)
        smem[i + j * smem_stride].z = tex3D(tex_z_obj, gx, gy, gz + 0.5f + 2.0f * j);

}

__device__ float3 LoadVelFromSmem_8x8x8(const float3* smem, const float3& coord,
                                        uint si)
{
    uint3 p0 = make_uint3(__float2uint_rd(coord.x), __float2uint_rd(coord.y),
                          __float2uint_rd(coord.z));
    float3 t = coord - make_float3(__uint2float_rn(p0.x), __uint2float_rn(p0.y),
                                   __uint2float_rn(p0.z));

    uint row_stride = 16;
    uint slice_stride = 16 * 16;

    uint i = si + p0.z * slice_stride + p0.y * row_stride + p0.x;

    float3 x0y0z0 = smem[i];
    float3 x1y0z0 = smem[i + 1];
    float3 x0y1z0 = smem[i + row_stride];
    float3 x0y0z1 = smem[i + slice_stride];
    float3 x1y1z0 = smem[i + row_stride + 1];
    float3 x0y1z1 = smem[i + slice_stride + row_stride];
    float3 x1y0z1 = smem[i + slice_stride + 1];
    float3 x1y1z1 = smem[i + slice_stride + row_stride + 1];

    float3 a = lerp(x0y0z0, x1y0z0, t.x);
    float3 b = lerp(x0y1z0, x1y1z0, t.x);
    float3 c = lerp(x0y0z1, x1y0z1, t.x);
    float3 d = lerp(x0y1z1, x1y1z1, t.x);

    float3 e = lerp(a, b, t.y);
    float3 f = lerp(c, d, t.y);

    return lerp(e, f, t.z);
}

template <typename AdvectionImpl>
__global__ void AdvectParticlesKernel_smem(FlipParticles particles,
                                           float3 bounds,
                                           float time_step_over_cell_size,
                                           bool outflow, AdvectionImpl advect,
                                           uint3 volume_size)
{
    __shared__ float3 smem[16 * 16 * 16]; // 4096

    LoadVelToSmem_8x8x8(smem, tex_x, tex_y, tex_z, volume_size);
    __syncthreads();

    uint x = VolumeX();
    uint y = VolumeY();
    uint z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    FlipParticles& p = particles;
    float3 my_coord = make_float3(x, y, z) + 0.5f;

    // Coordinates in the shared memory block(including halo).
    uint3 ti = threadIdx + 4;

    // Linear index in the shared memory block(including halo).
    uint si = __umul24(__umul24(ti.z, 16) + ti.y, 16) + ti.x;

    uint cell_index = LinearIndexVolume(x, y, z, volume_size);
    int count = p.particle_count_[cell_index];
    for (int n = 0; n < count; n++) {
        int i = cell_index * kMaxNumParticlesPerCell + n;

        float3 coord = make_float3(__half2float(p.position_x_[i]),
                                   __half2float(p.position_y_[i]),
                                   __half2float(p.position_z_[i]));

        // The fluid looks less bumpy with the grid velocity.
        float3 vel = LoadVelFromSmem_8x8x8(smem, coord - my_coord, si);

        if (IsStopped(vel.x, vel.y, vel.z)) {
            // Don't eliminate the particle. It may contains density/temperature
            // information.
            //
            // We don't need the number of active particles until the sorting is
            // done.
            return;
        }

        float3 result = advect.Advect(make_float3(x, y, z), vel,
                                      time_step_over_cell_size);

        if (result.x < 0.0f || result.x > bounds.x)
            p.velocity_x_[i] = 0;

        if (result.y < 0.0f || result.y > bounds.y) {
            if (outflow)
                FreeParticle(particles, i);
            else
                p.velocity_y_[i] = 0;
        }

        if (result.z < 0.0f || result.z > bounds.z)
            p.velocity_z_[i] = 0;

        float3 pos = clamp(result, make_float3(0.0f), bounds);

        p.position_x_[i] = __float2half_rn(pos.x);
        p.position_y_[i] = __float2half_rn(pos.y);
        p.position_z_[i] = __float2half_rn(pos.z);
    }
}

void AdvectParticles_smem(const FlipParticles& particles, cudaArray* vel_x,
                          cudaArray* vel_y, cudaArray* vel_z, float time_step,
                          float cell_size, bool outflow, uint3 volume_size,
                          BlockArrangement* ba)
{
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

    dim3 block;
    dim3 grid;
    ba->Arrange8x8x8(&grid, &block, volume_size);

    AdvectionBogackiShampine adv_rk3;
    float3 bounds = make_float3(volume_size) - 1.0f;
    AdvectParticlesKernel_smem<<<grid, block>>>(particles, bounds,
                                                time_step / cell_size,
                                                outflow, adv_rk3, volume_size);
    DCHECK_KERNEL();
}
}

// =============================================================================

namespace kern_launcher
{
void AdvectFlipParticles(const FlipParticles& particles, cudaArray* vel_x,
                         cudaArray* vel_y, cudaArray* vel_z, float time_step,
                         float cell_size, bool outflow, uint3 volume_size,
                         BlockArrangement* ba)
{
    bool smem = false;
    if (smem) {
        AdvectParticles_smem(particles, vel_x, vel_y, vel_z, time_step,
                             cell_size, outflow, volume_size, ba);
        return;
    }

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

    dim3 block;
    dim3 grid;
    ba->ArrangeLinear(&grid, &block, particles.num_of_particles_);

    AdvectionEuler adv_fe;
    AdvectionMidPoint adv_mp;
    AdvectionBogackiShampine adv_bs;
    AdvectionRK4 adv_rk4;

    float3 bounds = make_float3(volume_size) - 1.0f;
    int order = 3;
    switch (order) {
        case 1:
            AdvectParticlesKernel<<<grid, block>>>(particles, bounds,
                                                   time_step / cell_size,
                                                   outflow, adv_fe);
            break;
        case 2:
            AdvectParticlesKernel<<<grid, block>>>(particles, bounds,
                                                   time_step / cell_size,
                                                   outflow, adv_mp);
            break;
        case 3:
            AdvectParticlesKernel<<<grid, block>>>(particles, bounds,
                                                   time_step / cell_size,
                                                   outflow, adv_bs);
            break;
        case 4:
            AdvectParticlesKernel<<<grid, block>>>(particles, bounds,
                                                   time_step / cell_size,
                                                   outflow, adv_rk4);
            break;
    }

    DCHECK_KERNEL();
}
}
