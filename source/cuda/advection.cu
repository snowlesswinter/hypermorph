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

#include "advection_method.h"
#include "block_arrangement.h"
#include "cuda_common_host.h"
#include "cuda_common_kern.h"

surface<void, cudaSurfaceType3D> surf;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_vx;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_vy;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_vz;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_aux;

__device__ float TrilinearInterpolationSingle(float x0y0z0, float x1y0z0,
                                              float x0y1z0, float x0y0z1,
                                              float x1y1z0, float x0y1z1,
                                              float x1y0z1, float x1y1z1,
                                              float alpha, float beta,
                                              float gamma)
{
    float xy0z0 = (1 - alpha) * x0y0z0 + alpha * x1y0z0;
    float xy1z0 = (1 - alpha) * x0y1z0 + alpha * x1y1z0;
    float xy0z1 = (1 - alpha) * x0y0z1 + alpha * x1y0z1;
    float xy1z1 = (1 - alpha) * x0y1z1 + alpha * x1y1z1;

    float yz0 = (1 - beta) * xy0z0 + beta * xy1z0;
    float yz1 = (1 - beta) * xy0z1 + beta * xy1z1;

    return (1 - gamma) * yz0 + gamma * yz1;
}

__device__ float3 TrilinearInterpolation(float3* cache, float3 coord,
                                         int slice_stride, int row_stride)
{
    float int_x = floorf(coord.x);
    float int_y = floorf(coord.y);
    float int_z = floorf(coord.z);

    float alpha = fracf(coord.x);
    float beta = fracf(coord.y);
    float gamma = fracf(coord.z);

    int index = int_z * slice_stride + int_y * row_stride + int_x;
    float3 x0y0z0 = cache[index];
    float3 x1y0z0 = cache[index + 1];
    float3 x0y1z0 = cache[index + row_stride];
    float3 x0y0z1 = cache[index + slice_stride];
    float3 x1y1z0 = cache[index + row_stride + 1];
    float3 x0y1z1 = cache[index + slice_stride + row_stride];
    float3 x1y0z1 = cache[index + slice_stride + 1];
    float3 x1y1z1 = cache[index + slice_stride + row_stride + 1];

    float x = TrilinearInterpolationSingle(x0y0z0.x, x1y0z0.x, x0y1z0.x, x0y0z1.x, x1y1z0.x, x0y1z1.x, x1y0z1.x, x1y1z1.x, alpha, beta, gamma);
    float y = TrilinearInterpolationSingle(x0y0z0.y, x1y0z0.y, x0y1z0.y, x0y0z1.y, x1y1z0.y, x0y1z1.y, x1y0z1.y, x1y1z1.y, alpha, beta, gamma);
    float z = TrilinearInterpolationSingle(x0y0z0.z, x1y0z0.z, x0y1z0.z, x0y0z1.z, x1y1z0.z, x0y1z1.z, x1y0z1.z, x1y1z1.z, alpha, beta, gamma);
    return make_float3(x, y, z);
}

// Only ~45% hit rate, serious block effect, deprecated.
__global__ void AdvectVelocityKernel_smem(float time_step, float dissipation)
{
    __shared__ float3 cached_block[600];

    int base_x = blockIdx.x * blockDim.x;
    int base_y = blockIdx.y * blockDim.y;
    int base_z = blockIdx.z * blockDim.z;

    int x = base_x + threadIdx.x;
    int y = base_y + threadIdx.y;
    int z = base_z + threadIdx.z;

    int bw = blockDim.x;
    int bh = blockDim.y;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;

    int index = threadIdx.z * bw * bh + threadIdx.y * bw + threadIdx.x;
    cached_block[index] =
        make_float3(tex3D(tex, coord.x, coord.y, coord.z));
    float3 velocity = cached_block[index];
    __syncthreads();

    float3 back_traced = coord - time_step * velocity;

    float3 new_velocity;
    if (back_traced.x >= base_x + 0.5f && back_traced.x < base_x + blockDim.x + 0.5f &&
            back_traced.y >= base_y + 0.5f && back_traced.y < base_y + blockDim.y + 0.5f &&
            back_traced.z >= base_z + 0.5f && back_traced.z < base_z + blockDim.z + 0.5f) {

        new_velocity = TrilinearInterpolation(
            cached_block, back_traced - make_float3(base_x + 0.5f, base_y + 0.5f, base_z + 0.5f),
            bw * bh, bw);
    } else {
        new_velocity =
            make_float3( 
                tex3D(tex, back_traced.x, back_traced.y,
                      back_traced.z));
    }
    new_velocity *= 1.0f - dissipation * time_step;
    ushort4 result = make_ushort4(__float2half_rn(new_velocity.x),
                                  __float2half_rn(new_velocity.y),
                                  __float2half_rn(new_velocity.z),
                                  0);
    surf3Dwrite(result, surf, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

// =============================================================================

__device__ float3 GetVelocity(float3 pos)
{
    float v_x = tex3D(tex_vx, pos.x, pos.y, pos.z);
    float v_y = tex3D(tex_vy, pos.x, pos.y, pos.z);
    float v_z = tex3D(tex_vz, pos.x, pos.y, pos.z);
    return make_float3(v_x, v_y, v_z);
}

template <bool MidPoint>
__device__ inline float3 AdvectImpl(float3 vel, float3 pos, float time_step_over_cell_size)
{
    return pos - vel * time_step_over_cell_size;
}

template <>
__device__ inline float3 AdvectImpl<true>(float3 vel, float3 pos, float time_step_over_cell_size)
{
    float3 mid_point = pos - vel * 0.5f * time_step_over_cell_size;
    vel = GetVelocity(mid_point);
    return pos - vel * time_step_over_cell_size;
}

template <bool MidPoint>
__global__ void AdvectFieldBfeccKernel(float time_step_over_cell_size, float dissipation, uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 vel = GetVelocity(coord);
    float3 back_traced = AdvectImpl<MidPoint>(vel, coord, time_step_over_cell_size);

    float phi0 = tex3D(tex, back_traced.x - 0.5f, back_traced.y - 0.5f, back_traced.z - 0.5f);
    float phi1 = tex3D(tex, back_traced.x - 0.5f, back_traced.y - 0.5f, back_traced.z + 0.5f);
    float phi2 = tex3D(tex, back_traced.x - 0.5f, back_traced.y + 0.5f, back_traced.z - 0.5f);
    float phi3 = tex3D(tex, back_traced.x - 0.5f, back_traced.y + 0.5f, back_traced.z + 0.5f);
    float phi4 = tex3D(tex, back_traced.x + 0.5f, back_traced.y - 0.5f, back_traced.z - 0.5f);
    float phi5 = tex3D(tex, back_traced.x + 0.5f, back_traced.y - 0.5f, back_traced.z + 0.5f);
    float phi6 = tex3D(tex, back_traced.x + 0.5f, back_traced.y + 0.5f, back_traced.z - 0.5f);
    float phi7 = tex3D(tex, back_traced.x + 0.5f, back_traced.y + 0.5f, back_traced.z + 0.5f);

    float phi_min = fminf(fminf(fminf(fminf(fminf(fminf(fminf(phi0, phi1), phi2), phi3), phi4), phi5), phi6), phi7);
    float phi_max = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(phi0, phi1), phi2), phi3), phi4), phi5), phi6), phi7);

    float phi_new = tex3D(tex_aux, back_traced.x, back_traced.y, back_traced.z);
    float clamped = fmaxf(fminf(phi_new, phi_max), phi_min);
    if (clamped != phi_new) // New extrema found, revert to the first order
                            // accurate semi-Lagrangian method.
        phi_new = tex3D(tex, back_traced.x, back_traced.y, back_traced.z);

    auto r = __float2half_rn(dissipation * phi_new);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

template <bool MidPoint>
__global__ void AdvectFieldMacCormackKernel(float time_step_over_cell_size, float dissipation, uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    float3 coord = make_float3(x, y, z) + 0.5f;

    float3 vel = GetVelocity(coord);
    float3 back_traced = AdvectImpl<MidPoint>(vel, coord, time_step_over_cell_size);

    float phi_n = tex3D(tex, coord.x, coord.y, coord.z);

    float phi0 = tex3D(tex, back_traced.x - 0.5f, back_traced.y - 0.5f, back_traced.z - 0.5f);
    float phi1 = tex3D(tex, back_traced.x - 0.5f, back_traced.y - 0.5f, back_traced.z + 0.5f);
    float phi2 = tex3D(tex, back_traced.x - 0.5f, back_traced.y + 0.5f, back_traced.z - 0.5f);
    float phi3 = tex3D(tex, back_traced.x - 0.5f, back_traced.y + 0.5f, back_traced.z + 0.5f);
    float phi4 = tex3D(tex, back_traced.x + 0.5f, back_traced.y - 0.5f, back_traced.z - 0.5f);
    float phi5 = tex3D(tex, back_traced.x + 0.5f, back_traced.y - 0.5f, back_traced.z + 0.5f);
    float phi6 = tex3D(tex, back_traced.x + 0.5f, back_traced.y + 0.5f, back_traced.z - 0.5f);
    float phi7 = tex3D(tex, back_traced.x + 0.5f, back_traced.y + 0.5f, back_traced.z + 0.5f);

    float phi_min = fminf(fminf(fminf(fminf(fminf(fminf(fminf(phi0, phi1), phi2), phi3), phi4), phi5), phi6), phi7);
    float phi_max = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(phi0, phi1), phi2), phi3), phi4), phi5), phi6), phi7);

    float phi_np1_hat = tex3D(tex_aux, coord.x, coord.y, coord.z);

    float3 forward_trace = AdvectImpl<MidPoint>(vel, coord, -time_step_over_cell_size);
    float phi_n_hat = tex3D(tex_aux, forward_trace.x, forward_trace.y, forward_trace.z);

    float phi_new = phi_np1_hat + 0.5f * (phi_n - phi_n_hat);
    float clamped = fmaxf(fminf(phi_new, phi_max), phi_min);
    if (clamped != phi_new)
        phi_new = phi_np1_hat;

    auto r = __float2half_rn(phi_new * dissipation);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

template <bool MidPoint>
__global__ void AdvectFieldSemiLagrangianKernel(float time_step_over_cell_size,
                                                float dissipation,
                                                uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    float3 coord = make_float3(x, y, z) + 0.5f;

    float3 vel = GetVelocity(coord);
    float3 back_traced = AdvectImpl<MidPoint>(vel, coord,
                                              time_step_over_cell_size);

    float phi = tex3D(tex, back_traced.x, back_traced.y, back_traced.z);
    auto r = __float2half_rn(phi * dissipation);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

template <bool MidPoint>
__global__ void BfeccRemoveErrorKernel(float time_step_over_cell_size,
                                       uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 vel = GetVelocity(coord);
    float3 forward_trace = AdvectImpl<MidPoint>(vel, coord,
                                                -time_step_over_cell_size);

    float phi = tex3D(tex, coord.x, coord.y, coord.z);
    float r = tex3D(tex_aux, forward_trace.x, forward_trace.y, forward_trace.z);
    r = 0.5f * (3.0f * phi - r);
    surf3Dwrite(__float2half_rn(r), surf, x * sizeof(ushort), y, z,
                cudaBoundaryModeTrap);
}

// =============================================================================

void AdvectFieldsBfecc(cudaArray** fnp1, cudaArray** fn, int num_of_fields,
                       cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z,
                       cudaArray* aux, float cell_size, float time_step,
                       float dissipation, uint3 volume_size, bool mid_point,
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

    dim3 grid;
    dim3 block;
    ba->ArrangePrefer3dLocality(&grid, &block, volume_size);
    for (int i = 0; i < num_of_fields; i++) {
        // Pass 1: Calculate phi_n_plus_1_hat, and store in |fnp1[i]|.
        if (BindCudaSurfaceToArray(&surf, fnp1[i]) != cudaSuccess)
            return;

        auto bound = BindHelper::Bind(&tex, fn[i], false, cudaFilterModeLinear,
                                      cudaAddressModeClamp);
        if (bound.error() != cudaSuccess)
            return;

        if (mid_point)
            AdvectFieldSemiLagrangianKernel<true><<<grid, block>>>(
                time_step / cell_size, 1.0f, volume_size);
        else
            AdvectFieldSemiLagrangianKernel<false><<<grid, block>>>(
                time_step / cell_size, 1.0f, volume_size);

        // Pass 2: Calculate phi_n_hat, and store in |aux|.
        if (BindCudaSurfaceToArray(&surf, aux) != cudaSuccess)
            return;

        {
            auto bound_a = BindHelper::Bind(&tex_aux, fnp1[i], false,
                                            cudaFilterModeLinear,
                                            cudaAddressModeClamp);
            if (bound_a.error() != cudaSuccess)
                return;

            if (mid_point)
                BfeccRemoveErrorKernel<true><<<grid, block>>>(
                    time_step / cell_size, volume_size);
            else
                BfeccRemoveErrorKernel<false><<<grid, block>>>(
                    time_step / cell_size, volume_size);
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
            AdvectFieldBfeccKernel<true><<<grid, block>>>(
                time_step / cell_size, 1.0f - dissipation * time_step,
                volume_size);
        else
            AdvectFieldBfeccKernel<false><<<grid, block>>>(
                time_step / cell_size, 1.0f - dissipation * time_step,
                volume_size);
    }
}

void AdvectFieldsMacCormack(cudaArray** fnp1, cudaArray** fn,
                            int num_of_fields, cudaArray* vel_x,
                            cudaArray* vel_y, cudaArray* vel_z, cudaArray* aux,
                            float cell_size, float time_step, float dissipation,
                            uint3 volume_size, bool mid_point,
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

    auto bound_a = BindHelper::Bind(&tex_aux, aux, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_a.error() != cudaSuccess)
        return;

    dim3 grid;
    dim3 block;
    ba->ArrangePrefer3dLocality(&grid, &block, volume_size);
    for (int i = 0; i < num_of_fields; i++) {
        if (BindCudaSurfaceToArray(&surf, aux) != cudaSuccess)
            return;

        auto bound = BindHelper::Bind(&tex, fn[i], false, cudaFilterModeLinear,
                                      cudaAddressModeClamp);
        if (bound.error() != cudaSuccess)
            return;

        if (mid_point)
            AdvectFieldSemiLagrangianKernel<true><<<grid, block>>>(
                time_step / cell_size, 1.0f, volume_size);
        else
            AdvectFieldSemiLagrangianKernel<false><<<grid, block>>>(
                time_step / cell_size, 1.0f, volume_size);

        if (BindCudaSurfaceToArray(&surf, fnp1[i]) != cudaSuccess)
            return;

        if (mid_point)
            AdvectFieldMacCormackKernel<true><<<grid, block>>>(
                time_step / cell_size, 1.0f - dissipation * time_step,
                volume_size);
        else
            AdvectFieldMacCormackKernel<false><<<grid, block>>>(
                time_step / cell_size, 1.0f - dissipation * time_step,
                volume_size);
    }
}

void AdvectFieldsSemiLagrangian(cudaArray** fnp1, cudaArray** fn,
                                int num_of_fields, cudaArray* vel_x,
                                cudaArray* vel_y, cudaArray* vel_z,
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

    dim3 grid;
    dim3 block;
    ba->ArrangePrefer3dLocality(&grid, &block, volume_size);
    for (int i = 0; i < num_of_fields; i++) {
        if (BindCudaSurfaceToArray(&surf, fnp1[i]) != cudaSuccess)
            return;

        auto bound = BindHelper::Bind(&tex, fn[i], false, cudaFilterModeLinear,
                                      cudaAddressModeClamp);
        if (bound.error() != cudaSuccess)
            return;

        if (mid_point)
            AdvectFieldSemiLagrangianKernel<true><<<grid, block>>>(
                time_step / cell_size, 1.0f - dissipation * time_step,
                volume_size);
        else
            AdvectFieldSemiLagrangianKernel<false><<<grid, block>>>(
                time_step / cell_size, 1.0f - dissipation * time_step,
                volume_size);
    }
}

namespace kern_launcher
{
void AdvectScalarField(cudaArray* fnp1, cudaArray* fn, cudaArray* vel_x,
                       cudaArray* vel_y, cudaArray* vel_z, cudaArray* aux,
                       float cell_size, float time_step, float dissipation,
                       AdvectionMethod method, uint3 volume_size,
                       bool mid_point, BlockArrangement* ba)
{
    cudaArray* fnp1s[] = {fnp1};
    cudaArray* fns[] = {fn};
    int num_of_fields = sizeof(fnp1s) / sizeof(fnp1s[0]);
    if (method == MACCORMACK_SEMI_LAGRANGIAN) {
        AdvectFieldsMacCormack(fnp1s, fns, num_of_fields, vel_x, vel_y, vel_z,
                               aux, cell_size, time_step, dissipation,
                               volume_size, mid_point, ba);
    } else if (method == BFECC_SEMI_LAGRANGIAN) {
        AdvectFieldsBfecc(fnp1s, fns, num_of_fields, vel_x, vel_y, vel_z, aux,
                          cell_size, time_step, dissipation, volume_size,
                          mid_point, ba);
    } else {
        AdvectFieldsSemiLagrangian(fnp1s, fns, num_of_fields, vel_x, vel_y,
                                   vel_z, cell_size, time_step, dissipation,
                                   volume_size, mid_point, ba);
    }
}

void AdvectVectorField(cudaArray* fnp1_x, cudaArray* fnp1_y, cudaArray* fnp1_z,
                       cudaArray* fn_x, cudaArray* fn_y, cudaArray* fn_z,
                       cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z,
                       cudaArray* aux, float cell_size, float time_step,
                       float dissipation, AdvectionMethod method,
                       uint3 volume_size, bool mid_point, BlockArrangement* ba)
{
    cudaArray* fnp1s[] = {fnp1_x, fnp1_y, fnp1_z};
    cudaArray* fns[] = {fn_x, fn_y, fn_z};
    int num_of_fields = sizeof(fnp1s) / sizeof(fnp1s[0]);
    if (method == MACCORMACK_SEMI_LAGRANGIAN) {
        AdvectFieldsMacCormack(fnp1s, fns, num_of_fields, vel_x, vel_y, vel_z,
                               aux, cell_size, time_step, dissipation,
                               volume_size, mid_point, ba);
    } else if (method == BFECC_SEMI_LAGRANGIAN) {
        AdvectFieldsBfecc(fnp1s, fns, num_of_fields, vel_x, vel_y, vel_z, aux,
                          cell_size, time_step, dissipation, volume_size,
                          mid_point, ba);
    } else {
        AdvectFieldsSemiLagrangian(fnp1s, fns, num_of_fields, vel_x, vel_y,
                                   vel_z, cell_size, time_step, dissipation,
                                   volume_size, mid_point, ba);
    }
}
}
