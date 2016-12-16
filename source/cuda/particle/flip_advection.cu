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
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_xn;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_yn;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_zn;

#include "particle_advection.cuh"

namespace flip
{
__device__ void InterpolateVelocity(const FlipParticles& p, uint i,
                                    const float3& coord, const float3& vnp1)
{
    float3 vn = LoadVel(tex_xn, tex_yn, tex_zn, coord);
    float3 delta = vnp1 - vn;

    // v_np1 = (1 - ¦Á) * v_n_pic + ¦Á * v_n_flip.
    // We are using ¦Á = 1.
    p.velocity_x_[i] = __float2half_rn(__half2float(p.velocity_x_[i]) + delta.x);
    p.velocity_y_[i] = __float2half_rn(__half2float(p.velocity_y_[i]) + delta.y);
    p.velocity_z_[i] = __float2half_rn(__half2float(p.velocity_z_[i]) + delta.z);
}

// To save memory access, velocity interpolation is merged into the advection
// kernel. Even though we need to do a bit more interpolation(some new
// particles were added in resampling kernel), we have saved one velocity
// texture read during the FLIP step.
//
// Not gonna use shared memory yet. Same reason as in former interpolation
// kernel.
//
// Active particles are always *NOT* consecutive.
template <typename AdvectionImpl>
__global__ void AdvectParticlesKernel(FlipParticles particles, float3 bounds,
                                      float time_step_over_cell_size,
                                      bool outflow, AdvectionImpl advect)
{
    FlipParticles& p = particles;

    uint i = LinearIndex();
    if (i >= p.num_of_particles_)
        return;

    uint16_t xh = p.position_x_[i];
    if (IsCellUndefined(xh))
        return;

    float3 coord = Coordinates(p, i);

    // The fluid looks less bumpy with the re-sampled velocity. Don't know
    // the exact reason yet.
    float3 v = LoadVel(tex_x, tex_y, tex_z, coord);
    InterpolateVelocity(particles, i, coord, v);

    if (IsStopped(v.x, v.y, v.z)) {
        // Don't eliminate the particle. It may contains density/temperature
        // information.

        return;
    }

    float3 result = advect.Advect(coord, v, time_step_over_cell_size);

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
} // namespace flip

// =============================================================================

namespace kern_launcher
{
void AdvectFlipParticles(const FlipParticles& particles, cudaArray* vnp1_x,
                         cudaArray* vnp1_y, cudaArray* vnp1_z, cudaArray* vn_x,
                         cudaArray* vn_y, cudaArray* vn_z,  float time_step,
                         float cell_size, bool outflow, uint3 volume_size,
                         BlockArrangement* ba)
{
    auto bound_x = BindHelper::Bind(&tex_x, vnp1_x, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, vnp1_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, vnp1_z, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    auto bound_xn = BindHelper::Bind(&tex_xn, vn_x, false, cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_xn.error() != cudaSuccess)
        return;

    auto bound_yn = BindHelper::Bind(&tex_yn, vn_y, false, cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_yn.error() != cudaSuccess)
        return;

    auto bound_zn = BindHelper::Bind(&tex_zn, vn_z, false, cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_zn.error() != cudaSuccess)
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
            flip::AdvectParticlesKernel<<<grid, block>>>(particles, bounds,
                                                         time_step / cell_size,
                                                         outflow, adv_fe);
            break;
        case 2:
            flip::AdvectParticlesKernel<<<grid, block>>>(particles, bounds,
                                                         time_step / cell_size,
                                                         outflow, adv_mp);
            break;
        case 3:
            flip::AdvectParticlesKernel<<<grid, block>>>(particles, bounds,
                                                         time_step / cell_size,
                                                         outflow, adv_bs);
            break;
        case 4:
            flip::AdvectParticlesKernel<<<grid, block>>>(particles, bounds,
                                                         time_step / cell_size,
                                                         outflow, adv_rk4);
            break;
    }

    DCHECK_KERNEL();
}
}
