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

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "cuda/advection_method.h"
#include "cuda/block_arrangement.h"
#include "cuda/field_offset.h"
#include "cuda/cuda_common_host.h"
#include "cuda/cuda_common_kern.h"
#include "cuda/cuda_debug.h"

surface<void, cudaSurfaceType3D> surf;
surface<void, cudaSurfaceType3D> surf_x;
surface<void, cudaSurfaceType3D> surf_y;
surface<void, cudaSurfaceType3D> surf_z;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_x;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_y;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_z;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_aux;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_vx;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_vy;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_vz;

__device__ float3 GetVelocityStaggeredOffset(float3 pos)
{
    float v_x = tex3D(tex_vx, pos.x + 0.5f, pos.y,        pos.z       );
    float v_y = tex3D(tex_vy, pos.x,        pos.y + 0.5f, pos.z       );
    float v_z = tex3D(tex_vz, pos.x,        pos.y,        pos.z + 0.5f);
    return make_float3(v_x, v_y, v_z);
}

// =============================================================================

template <bool MidPoint>
__device__ inline float3 AdvectImpl(float3 vel, float3 pos, float time_step_over_cell_size)
{
    return pos - vel * time_step_over_cell_size;
}

template <>
__device__ inline float3 AdvectImpl<true>(float3 vel, float3 pos, float time_step_over_cell_size)
{
    float3 mid_point = pos - vel * 0.5f * time_step_over_cell_size;
    vel = GetVelocityStaggeredOffset(mid_point);
    return pos - vel * time_step_over_cell_size;
}

template <bool MidPoint>
__global__ void AdvectFieldBfeccStaggeredOffsetKernel(float3 offset, float time_step_over_cell_size, float dissipation, uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 vel = GetVelocityStaggeredOffset(coord + offset);
    float3 back_traced = AdvectImpl<MidPoint>(vel, coord, time_step_over_cell_size);

    float ¦Õ0 = tex3D(tex, back_traced.x - 0.5f, back_traced.y - 0.5f, back_traced.z - 0.5f);
    float ¦Õ1 = tex3D(tex, back_traced.x - 0.5f, back_traced.y - 0.5f, back_traced.z + 0.5f);
    float ¦Õ2 = tex3D(tex, back_traced.x - 0.5f, back_traced.y + 0.5f, back_traced.z - 0.5f);
    float ¦Õ3 = tex3D(tex, back_traced.x - 0.5f, back_traced.y + 0.5f, back_traced.z + 0.5f);
    float ¦Õ4 = tex3D(tex, back_traced.x + 0.5f, back_traced.y - 0.5f, back_traced.z - 0.5f);
    float ¦Õ5 = tex3D(tex, back_traced.x + 0.5f, back_traced.y - 0.5f, back_traced.z + 0.5f);
    float ¦Õ6 = tex3D(tex, back_traced.x + 0.5f, back_traced.y + 0.5f, back_traced.z - 0.5f);
    float ¦Õ7 = tex3D(tex, back_traced.x + 0.5f, back_traced.y + 0.5f, back_traced.z + 0.5f);

    float ¦Õ_min = fminf(fminf(fminf(fminf(fminf(fminf(fminf(¦Õ0, ¦Õ1), ¦Õ2), ¦Õ3), ¦Õ4), ¦Õ5), ¦Õ6), ¦Õ7);
    float ¦Õ_max = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(¦Õ0, ¦Õ1), ¦Õ2), ¦Õ3), ¦Õ4), ¦Õ5), ¦Õ6), ¦Õ7);

    float ¦Õ_new = tex3D(tex_aux, back_traced.x, back_traced.y, back_traced.z);
    float clamped = fmaxf(fminf(¦Õ_new, ¦Õ_max), ¦Õ_min);
    if (clamped != ¦Õ_new) // New extrema found, revert to the first order
                          // accurate semi-Lagrangian method.
        ¦Õ_new = tex3D(tex, back_traced.x, back_traced.y, back_traced.z);

    auto r = __float2half_rn(dissipation * ¦Õ_new);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

template <bool MidPoint>
__global__ void AdvectFieldMacCormackStaggeredOffsetKernel(float3 offset, float time_step_over_cell_size, float dissipation, uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float3 vel = GetVelocityStaggeredOffset(coord + offset);
    float3 back_traced = AdvectImpl<MidPoint>(vel, coord, time_step_over_cell_size);

    float ¦Õ_n = tex3D(tex, coord.x, coord.y, coord.z);

    float ¦Õ0 = tex3D(tex, back_traced.x - 0.5f, back_traced.y - 0.5f, back_traced.z - 0.5f);
    float ¦Õ1 = tex3D(tex, back_traced.x - 0.5f, back_traced.y - 0.5f, back_traced.z + 0.5f);
    float ¦Õ2 = tex3D(tex, back_traced.x - 0.5f, back_traced.y + 0.5f, back_traced.z - 0.5f);
    float ¦Õ3 = tex3D(tex, back_traced.x - 0.5f, back_traced.y + 0.5f, back_traced.z + 0.5f);
    float ¦Õ4 = tex3D(tex, back_traced.x + 0.5f, back_traced.y - 0.5f, back_traced.z - 0.5f);
    float ¦Õ5 = tex3D(tex, back_traced.x + 0.5f, back_traced.y - 0.5f, back_traced.z + 0.5f);
    float ¦Õ6 = tex3D(tex, back_traced.x + 0.5f, back_traced.y + 0.5f, back_traced.z - 0.5f);
    float ¦Õ7 = tex3D(tex, back_traced.x + 0.5f, back_traced.y + 0.5f, back_traced.z + 0.5f);

    float ¦Õ_min = fminf(fminf(fminf(fminf(fminf(fminf(fminf(¦Õ0, ¦Õ1), ¦Õ2), ¦Õ3), ¦Õ4), ¦Õ5), ¦Õ6), ¦Õ7);
    float ¦Õ_max = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(¦Õ0, ¦Õ1), ¦Õ2), ¦Õ3), ¦Õ4), ¦Õ5), ¦Õ6), ¦Õ7);

    float ¦Õ_np1_hat = tex3D(tex_aux, coord.x, coord.y, coord.z);

    float3 forward_trace = AdvectImpl<MidPoint>(vel, coord, -time_step_over_cell_size);
    float ¦Õ_n_hat = tex3D(tex_aux, forward_trace.x, forward_trace.y, forward_trace.z);

    float ¦Õ_new = ¦Õ_np1_hat + 0.5f * (¦Õ_n - ¦Õ_n_hat);
    float clamped = fmaxf(fminf(¦Õ_new, ¦Õ_max), ¦Õ_min);
    if (clamped != ¦Õ_new)
        ¦Õ_new = ¦Õ_np1_hat;

    auto r = __float2half_rn(¦Õ_new * dissipation);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

template <bool MidPoint>
__global__ void AdvectFieldSemiLagrangianStaggeredOffsetKernel(
    float3 offset, float time_step_over_cell_size, float dissipation,
    uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float3 vel = GetVelocityStaggeredOffset(coord + offset);
    float3 back_traced = AdvectImpl<MidPoint>(vel, coord,
                                              time_step_over_cell_size);
    float ¦Õ = tex3D(tex, back_traced.x, back_traced.y, back_traced.z);
    auto r = __float2half_rn(¦Õ * dissipation);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

template <bool MidPoint>
__global__ void BfeccRemoveErrorStaggeredOffsetKernel(
    float3 offset, float time_step_over_cell_size, uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 vel = GetVelocityStaggeredOffset(coord + offset);
    float3 forward_trace = AdvectImpl<MidPoint>(vel, coord,
                                                -time_step_over_cell_size);

    float ¦Õ = tex3D(tex, coord.x, coord.y, coord.z);
    float r = tex3D(tex_aux, forward_trace.x, forward_trace.y, forward_trace.z);
    r = 0.5f * (3.0f * ¦Õ - r);
    surf3Dwrite(__float2half_rn(r), surf, x * sizeof(ushort), y, z,
                cudaBoundaryModeTrap);
}

void AdvectFieldsBfeccStaggeredOffset(cudaArray** fnp1, cudaArray** fn,
                                      float3* offset, int num_of_fields,
                                      cudaArray* vel_x, cudaArray* vel_y,
                                      cudaArray* vel_z, cudaArray* aux,
                                      float cell_size, float time_step,
                                      float dissipation, uint3 volume_size,
                                      bool mid_point, BlockArrangement* ba)
{
    auto bound_vx = BindHelper::Bind(&tex_vx, vel_x, false,
                                     cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_vx.error() != cudaSuccess)
        return;

    auto bound_vy = BindHelper::Bind(&tex_vy, vel_y, false,
                                     cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_vy.error() != cudaSuccess)
        return;

    auto bound_vz = BindHelper::Bind(&tex_vz, vel_z, false,
                                     cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_vz.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    for (int i = 0; i < num_of_fields; i++) {
        // Pass 1: Calculate ¦Õ_n_plus_1_hat, and store in |fnp1[i]|.
        if (BindCudaSurfaceToArray(&surf, fnp1[i]) != cudaSuccess)
            return;

        auto bound = BindHelper::Bind(&tex, fn[i], false, cudaFilterModeLinear,
                                      cudaAddressModeClamp);
        if (bound.error() != cudaSuccess)
            return;

        if (mid_point)
            AdvectFieldSemiLagrangianStaggeredOffsetKernel<true><<<grid, block>>>(
                offset[i], time_step / cell_size, 1.0f, volume_size);
        else
            AdvectFieldSemiLagrangianStaggeredOffsetKernel<false><<<grid, block>>>(
                offset[i], time_step / cell_size, 1.0f, volume_size);

        // Pass 2: Calculate ¦Õ_n_hat, and store in |aux|.
        if (BindCudaSurfaceToArray(&surf, aux) != cudaSuccess)
            return;

        {
            auto bound_a = BindHelper::Bind(&tex_aux, fnp1[i], false,
                                            cudaFilterModeLinear,
                                            cudaAddressModeClamp);
            if (bound_a.error() != cudaSuccess)
                return;

            if (mid_point)
                BfeccRemoveErrorStaggeredOffsetKernel<true><<<grid, block>>>(
                    offset[i], time_step / cell_size, volume_size);
            else
                BfeccRemoveErrorStaggeredOffsetKernel<false><<<grid, block>>>(
                    offset[i], time_step / cell_size, volume_size);
        }

        // Pass 3: Calculate the final result.
        if (BindCudaSurfaceToArray(&surf, fnp1[i]) != cudaSuccess)
            return;

        auto bound_a = BindHelper::Bind(&tex_aux, aux, false,
                                        cudaFilterModeLinear,
                                        cudaAddressModeClamp);
        if (bound_a.error() != cudaSuccess)
            return;

        if (mid_point)
            AdvectFieldBfeccStaggeredOffsetKernel<true><<<grid, block>>>(
                offset[i], time_step / cell_size,
                1.0f - dissipation * time_step, volume_size);
        else
            AdvectFieldBfeccStaggeredOffsetKernel<false><<<grid, block>>>(
                offset[i], time_step / cell_size,
                1.0f - dissipation * time_step, volume_size);
    }
}

void AdvectFieldsMacCormackStaggeredOffset(cudaArray** fnp1, cudaArray** fn,
                                           float3* offset, int num_of_fields,
                                           cudaArray* vel_x, cudaArray* vel_y,
                                           cudaArray* vel_z, cudaArray* aux,
                                           float cell_size, float time_step,
                                           float dissipation, uint3 volume_size,
                                           bool mid_point, BlockArrangement* ba)
{
    auto bound_vx = BindHelper::Bind(&tex_vx, vel_x, false,
                                     cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_vx.error() != cudaSuccess)
        return;

    auto bound_vy = BindHelper::Bind(&tex_vy, vel_y, false,
                                     cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_vy.error() != cudaSuccess)
        return;

    auto bound_vz = BindHelper::Bind(&tex_vz, vel_z, false,
                                     cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_vz.error() != cudaSuccess)
        return;

    auto bound_a = BindHelper::Bind(&tex_aux, aux, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_a.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    for (int i = 0; i < num_of_fields; i++) {
        if (BindCudaSurfaceToArray(&surf, aux) != cudaSuccess)
            return;

        auto bound = BindHelper::Bind(&tex, fn[i], false, cudaFilterModeLinear,
                                      cudaAddressModeClamp);
        if (bound.error() != cudaSuccess)
            return;

        if (mid_point)
            AdvectFieldSemiLagrangianStaggeredOffsetKernel<true><<<grid, block>>>(
                offset[i], time_step / cell_size, 1.0f, volume_size);
        else
            AdvectFieldSemiLagrangianStaggeredOffsetKernel<false><<<grid, block>>>(
                offset[i], time_step / cell_size, 1.0f, volume_size);

        if (BindCudaSurfaceToArray(&surf, fnp1[i]) != cudaSuccess)
            return;

        if (mid_point)
            AdvectFieldMacCormackStaggeredOffsetKernel<true><<<grid, block>>>(
                offset[i], time_step / cell_size,
                1.0f - dissipation * time_step, volume_size);
        else
            AdvectFieldMacCormackStaggeredOffsetKernel<false><<<grid, block>>>(
                offset[i], time_step / cell_size,
                1.0f - dissipation * time_step, volume_size);
    }
}

void AdvectFieldsSemiLagrangianStaggeredOffset(cudaArray** fnp1, cudaArray** fn,
                                               float3* offset,
                                               int num_of_fields,
                                               cudaArray* vel_x,
                                               cudaArray* vel_y,
                                               cudaArray* vel_z,
                                               float cell_size, float time_step,
                                               float dissipation,
                                               uint3 volume_size,
                                               bool mid_point,
                                               BlockArrangement* ba)
{
    auto bound_vx = BindHelper::Bind(&tex_vx, vel_x, false,
                                     cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_vx.error() != cudaSuccess)
        return;

    auto bound_vy = BindHelper::Bind(&tex_vy, vel_y, false,
                                     cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_vy.error() != cudaSuccess)
        return;

    auto bound_vz = BindHelper::Bind(&tex_vz, vel_z, false,
                                     cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_vz.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    for (int i = 0; i < num_of_fields; i++) {
        if (BindCudaSurfaceToArray(&surf, fnp1[i]) != cudaSuccess)
            return;

        auto bound = BindHelper::Bind(&tex, fn[i], false, cudaFilterModeLinear,
                                      cudaAddressModeClamp);
        if (bound.error() != cudaSuccess)
            return;

        if (mid_point)
            AdvectFieldSemiLagrangianStaggeredOffsetKernel<true><<<grid, block>>>(
                offset[i], time_step / cell_size,
                1.0f - dissipation * time_step, volume_size);
        else
            AdvectFieldSemiLagrangianStaggeredOffsetKernel<false><<<grid, block>>>(
                offset[i], time_step / cell_size,
                1.0f - dissipation * time_step, volume_size);
    }
}

namespace kern_launcher
{
void AdvectScalarFieldStaggered(cudaArray* fnp1, cudaArray* fn,
                                cudaArray* vel_x, cudaArray* vel_y,
                                cudaArray* vel_z, cudaArray* aux,
                                float cell_size, float time_step,
                                float dissipation, AdvectionMethod method,
                                uint3 volume_size, bool mid_point,
                                BlockArrangement* ba)
{
    cudaArray* fnp1s[] = {fnp1};
    cudaArray* fns[] = {fn};
    float3 offsets[] = {make_float3(0.0f)};
    int num_of_fields = sizeof(fnp1s) / sizeof(fnp1s[0]);
    if (method == MACCORMACK_SEMI_LAGRANGIAN) {
        AdvectFieldsMacCormackStaggeredOffset(fnp1s, fns, offsets,
                                              num_of_fields, vel_x, vel_y,
                                              vel_z, aux, cell_size, time_step,
                                              dissipation, volume_size,
                                              mid_point, ba);
    } else if (method == BFECC_SEMI_LAGRANGIAN) {
        AdvectFieldsBfeccStaggeredOffset(fnp1s, fns, offsets, num_of_fields,
                                         vel_x, vel_y, vel_z, aux, cell_size,
                                         time_step, dissipation, volume_size,
                                         mid_point, ba);
    } else {
        AdvectFieldsSemiLagrangianStaggeredOffset(fnp1s, fns, offsets,
                                                  num_of_fields, vel_x, vel_y,
                                                  vel_z, cell_size, time_step,
                                                  dissipation, volume_size,
                                                  mid_point, ba);
    }
    DCHECK_KERNEL();
}

void AdvectVelocityStaggered(cudaArray* fnp1_x, cudaArray* fnp1_y,
                             cudaArray* fnp1_z, cudaArray* fn_x,
                             cudaArray* fn_y, cudaArray* fn_z, cudaArray* vel_x,
                             cudaArray* vel_y, cudaArray* vel_z, cudaArray* aux,
                             float cell_size, float time_step,
                             float dissipation, AdvectionMethod method,
                             uint3 volume_size, bool mid_point,
                             BlockArrangement* ba)
{
    cudaArray* fnp1s[] = {fnp1_x, fnp1_y, fnp1_z};
    cudaArray* fns[] = {fn_x, fn_y, fn_z};
    float3 offsets[] = {
        -GetOffsetVelocityField(0),
        -GetOffsetVelocityField(1),
        -GetOffsetVelocityField(2)
    };
    int num_of_fields = sizeof(fnp1s) / sizeof(fnp1s[0]);
    if (method == MACCORMACK_SEMI_LAGRANGIAN) {
        AdvectFieldsMacCormackStaggeredOffset(fnp1s, fns, offsets,
                                              num_of_fields, vel_x, vel_y,
                                              vel_z, aux, cell_size, time_step,
                                              dissipation, volume_size,
                                              mid_point, ba);
    } else if (method == BFECC_SEMI_LAGRANGIAN) {
        AdvectFieldsBfeccStaggeredOffset(fnp1s, fns, offsets, num_of_fields,
                                         vel_x, vel_y, vel_z, aux, cell_size,
                                         time_step, dissipation, volume_size,
                                         mid_point, ba);
    } else {
        AdvectFieldsSemiLagrangianStaggeredOffset(fnp1s, fns, offsets,
                                                  num_of_fields, vel_x, vel_y,
                                                  vel_z, cell_size, time_step,
                                                  dissipation, volume_size,
                                                  mid_point, ba);
    }
    DCHECK_KERNEL();
}

void AdvectVorticityStaggered(cudaArray* fnp1_x, cudaArray* fnp1_y,
                              cudaArray* fnp1_z, cudaArray* fn_x,
                              cudaArray* fn_y, cudaArray* fn_z,
                              cudaArray* vel_x, cudaArray* vel_y,
                              cudaArray* vel_z, cudaArray* aux, float cell_size,
                              float time_step,  float dissipation,
                              AdvectionMethod method, uint3 volume_size,
                              bool mid_point, BlockArrangement* ba)
{
    cudaArray* fnp1s[] = {fnp1_x, fnp1_y, fnp1_z};
    cudaArray* fns[] = {fn_x, fn_y, fn_z};
    float3 offsets[] = {
        -GetOffsetVorticityField(0),
        -GetOffsetVorticityField(1),
        -GetOffsetVorticityField(2)
    };
    int num_of_fields = sizeof(fnp1s) / sizeof(fnp1s[0]);
    if (method == MACCORMACK_SEMI_LAGRANGIAN) {
        AdvectFieldsMacCormackStaggeredOffset(fnp1s, fns, offsets,
                                              num_of_fields, vel_x, vel_y,
                                              vel_z, aux, cell_size, time_step,
                                              dissipation, volume_size,
                                              mid_point, ba);
    } else if (method == BFECC_SEMI_LAGRANGIAN) {
        AdvectFieldsBfeccStaggeredOffset(fnp1s, fns, offsets, num_of_fields,
                                         vel_x, vel_y, vel_z, aux, cell_size,
                                         time_step, dissipation, volume_size,
                                         mid_point, ba);
    } else {
        AdvectFieldsSemiLagrangianStaggeredOffset(fnp1s, fns, offsets,
                                                  num_of_fields, vel_x, vel_y,
                                                  vel_z, cell_size, time_step,
                                                  dissipation, volume_size,
                                                  mid_point, ba);
    }

    DCHECK_KERNEL();
}
}