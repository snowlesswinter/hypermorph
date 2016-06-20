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

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "block_arrangement.h"
#include "cuda_common.h"

surface<void, cudaSurfaceType3D> surf;
surface<void, cudaSurfaceType3D> surf_x;
surface<void, cudaSurfaceType3D> surf_y;
surface<void, cudaSurfaceType3D> surf_z;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_x;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_y;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_z;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_b;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_t;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_d;

struct UpperBoundaryHandlerNeumann
{
    __device__ void HandleUpperBoundary(float* diff_ns, float base_y)
    {
        *diff_ns = -base_y;
    }
};

struct UpperBoundaryHandlerOutflow
{
    __device__ void HandleUpperBoundary(float* diff_ns, float base_y)
    {
        if (base_y < 0.0f)
            *diff_ns = -base_y;
        else
            *diff_ns = 0.0f;
    }
};

// =============================================================================

__global__ void ApplyBuoyancyKernel(float time_step, float ambient_temperature,
                                    float accel_factor, float gravity,
                                    uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float t = tex3D(tex_t, coord.x, coord.y, coord.z);
    float d = tex3D(tex_d, coord.x, coord.y, coord.z);
    float accel =
        time_step * ((t - ambient_temperature) * accel_factor - d * gravity);

    float velocity = tex3D(tex, coord.x, coord.y, coord.z);
    auto result = __float2half_rn(velocity + accel);
    surf3Dwrite(result, surf, x * sizeof(result), y, z, cudaBoundaryModeTrap);
}

__global__ void ApplyBuoyancyStaggeredKernel(float time_step,
                                             float ambient_temperature,
                                             float accel_factor, float gravity,
                                             uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, 0, z) + 0.5f;
    float t_prev = tex3D(tex_t, coord.x, coord.y, coord.z);
    float d_prev = tex3D(tex_d, coord.x, coord.y, coord.z);
    float accel_prev = time_step *
        ((t_prev - ambient_temperature) * accel_factor - d_prev * gravity);

    float y = 1.5f;
    for (int i = 1; i < volume_size.y; i++, y += 1.0f) {
        float t = tex3D(tex_t, coord.x, y, coord.z);
        float d = tex3D(tex_d, coord.x, y, coord.z);
        float accel = time_step *
            ((t - ambient_temperature) * accel_factor - d * gravity);

        float velocity = tex3D(tex, coord.x, y, coord.z);

        auto r = __float2half_rn(velocity + (accel_prev + accel) * 0.5f);
        surf3Dwrite(r, surf, x * sizeof(r), i, z, cudaBoundaryModeTrap);

        t_prev = t;
        d_prev = d;
        accel_prev = accel;
    }
}

template <typename UpperBoundaryHandler>
__global__ void ComputeDivergenceKernel(float half_inverse_cell_size,
                                        uint3 volume_size,
                                        UpperBoundaryHandler handler)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float west =     tex3D(tex_x, coord.x - 1.0f, coord.y,        coord.z);
    float center_x = tex3D(tex_x, coord.x,        coord.y,        coord.z);
    float east =     tex3D(tex_x, coord.x + 1.0f, coord.y,        coord.z);
    float south =    tex3D(tex_y, coord.x,        coord.y - 1.0f, coord.z);
    float center_y = tex3D(tex_y, coord.x,        coord.y,        coord.z);
    float north =    tex3D(tex_y, coord.x,        coord.y + 1.0f, coord.z);
    float near =     tex3D(tex_z, coord.x,        coord.y,        coord.z - 1.0f);
    float center_z = tex3D(tex_z, coord.x,        coord.y,        coord.z);
    float far =      tex3D(tex_z, coord.x,        coord.y,        coord.z + 1.0f);

    float diff_ew = east - west;
    float diff_ns = north - south;
    float diff_fn = far - near;

    // Handle boundary problem.
    if (x >= volume_size.x - 1)
        diff_ew = (center_x + west) * -0.5f;

    if (x <= 0)
        diff_ew = (east + center_x) * 0.5f;

    if (y >= volume_size.y - 1)
        handler.HandleUpperBoundary(&diff_ns, (center_y + south) * 0.5f);

    if (y <= 0)
        diff_ns = (north + center_y) * 0.5f;

    if (z >= volume_size.z - 1)
        diff_fn = (center_z + near) * -0.5f;

    if (z <= 0)
        diff_fn = (far + center_z) * 0.5f;

    float div = half_inverse_cell_size * (diff_ew + diff_ns + diff_fn);
    auto r = __float2half_rn(div);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

template <typename UpperBoundaryHandler>
__global__ void ComputeDivergenceStaggeredKernel(float inverse_cell_size,
                                                 uint3 volume_size,
                                                 UpperBoundaryHandler handler)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float base_x = tex3D(tex_x, coord.x,        coord.y,        coord.z);
    float base_y = tex3D(tex_y, coord.x,        coord.y,        coord.z);
    float base_z = tex3D(tex_z, coord.x,        coord.y,        coord.z);
    float east =   tex3D(tex_x, coord.x + 1.0f, coord.y,        coord.z);
    float north =  tex3D(tex_y, coord.x,        coord.y + 1.0f, coord.z);
    float far =    tex3D(tex_z, coord.x,        coord.y,        coord.z + 1.0f);

    float diff_ew = east  - base_x;
    float diff_ns = north - base_y;
    float diff_fn = far   - base_z;

    // Handle boundary problem
    if (x >= volume_size.x - 1)
        diff_ew = -base_x;

    if (y >= volume_size.y - 1)
        handler.HandleUpperBoundary(&diff_ns, base_y);

    if (z >= volume_size.z - 1)
        diff_fn = -base_z;

    float div = inverse_cell_size * (diff_ew + diff_ns + diff_fn);
    auto r = __float2half_rn(div);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

__global__ void ComputeResidualDiagnosisKernel(float inverse_h_square,
                                               uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z);

    float near =   tex3D(tex, coord.x, coord.y, coord.z - 1.0f);
    float south =  tex3D(tex, coord.x, coord.y - 1.0f, coord.z);
    float west =   tex3D(tex, coord.x - 1.0f, coord.y, coord.z);
    float center = tex3D(tex, coord.x, coord.y, coord.z);
    float east =   tex3D(tex, coord.x + 1.0f, coord.y, coord.z);
    float north =  tex3D(tex, coord.x, coord.y + 1.0f, coord.z);
    float far =    tex3D(tex, coord.x, coord.y, coord.z + 1.0f);
    float b =      tex3D(tex_b, coord.x, coord.y, coord.z);

    if (coord.y == volume_size.y - 1)
        north = center;

    if (coord.y == 0)
        south = center;

    if (coord.x == volume_size.x - 1)
        east = center;

    if (coord.x == 0)
        west = center;

    if (coord.z == volume_size.z - 1)
        far = center;

    if (coord.z == 0)
        near = center;

    float v = b -
        (north + south + east + west + far + near - 6.0 * center) *
        inverse_h_square;
    surf3Dwrite(fabsf(v), surf, x * sizeof(float), y, z, cudaBoundaryModeTrap);
}

__global__ void RoundPassedKernel(int* dest_array, int round, int x)
{
    dest_array[0] = x * x - round * round;
}

__global__ void SubtractGradientKernel(float half_inverse_cell_size,
                                       uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float near =   tex3D(tex, coord.x, coord.y, coord.z - 1.0f);
    float south =  tex3D(tex, coord.x, coord.y - 1.0f, coord.z);
    float west =   tex3D(tex, coord.x - 1.0f, coord.y, coord.z);
    float center = tex3D(tex, coord.x, coord.y, coord.z);
    float east =   tex3D(tex, coord.x + 1.0f, coord.y, coord.z);
    float north =  tex3D(tex, coord.x, coord.y + 1.0f, coord.z);
    float far =    tex3D(tex, coord.x, coord.y, coord.z + 1.0f);

    float diff_ew = east - west;
    float diff_ns = north - south;
    float diff_fn = far - near;

    // Handle boundary problem
    float3 mask = make_float3(1.0f);
    if (x >= volume_size.x - 1)
        mask.x = 0.0f;

    if (x <= 0)
        mask.x = 0.0f;

    if (y >= volume_size.y - 1)
        mask.y = 0.0f;

    if (y <= 0)
        mask.y = 0.0f;

    if (z >= volume_size.z - 1)
        mask.z = 0.0f;

    if (z <= 0)
        mask.z = 0.0f;

    float old_x = tex3D(tex_x, coord.x, coord.y, coord.z);
    float grad_x = diff_ew * half_inverse_cell_size;
    float new_x = old_x - grad_x;
    auto r_x = __float2half_rn(new_x * mask.x);
    surf3Dwrite(r_x, surf_x, x * sizeof(r_x), y, z, cudaBoundaryModeTrap);

    float old_y = tex3D(tex_y, coord.x, coord.y, coord.z);
    float grad_y = diff_ns * half_inverse_cell_size;
    float new_y = old_y - grad_y;
    auto r_y = __float2half_rn(new_y * mask.y);
    surf3Dwrite(r_y, surf_y, x * sizeof(r_y), y, z, cudaBoundaryModeTrap);

    float old_z = tex3D(tex_z, coord.x, coord.y, coord.z);
    float grad_z = diff_fn * half_inverse_cell_size;
    float new_z = old_z - grad_z;
    auto r_z = __float2half_rn(new_z * mask.z);
    surf3Dwrite(r_z, surf_z, x * sizeof(r_z), y, z, cudaBoundaryModeTrap);
}

__global__ void SubtractGradientStaggeredKernel(float inverse_cell_size,
                                                uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;
    
    float near =  tex3D(tex, coord.x,        coord.y,          coord.z - 1.0f);
    float south = tex3D(tex, coord.x,        coord.y - 1.0f,   coord.z);
    float west =  tex3D(tex, coord.x - 1.0f, coord.y,          coord.z);
    float base =  tex3D(tex, coord.x,        coord.y,          coord.z);

    // Handle boundary problem.
    float mask = 1.0f;
    if (x <= 0)
        mask = 0;

    if (y <= 0)
        mask = 0;

    if (z <= 0)
        mask = 0;

    float old_x = tex3D(tex_x, coord.x, coord.y, coord.z);
    float grad_x = (base - west) * inverse_cell_size;
    float new_x = old_x - grad_x;
    auto r_x = __float2half_rn(new_x * mask);
    surf3Dwrite(r_x, surf_x, x * sizeof(r_x), y, z, cudaBoundaryModeTrap);

    float old_y = tex3D(tex_y, coord.x, coord.y, coord.z);
    float grad_y = (base - south) * inverse_cell_size;
    float new_y = old_y - grad_y;
    auto r_y = __float2half_rn(new_y * mask);
    surf3Dwrite(r_y, surf_y, x * sizeof(r_y), y, z, cudaBoundaryModeTrap);

    float old_z = tex3D(tex_z, coord.x, coord.y, coord.z);
    float grad_z = (base - near) * inverse_cell_size;
    float new_z = old_z - grad_z;
    auto r_z = __float2half_rn(new_z * mask);
    surf3Dwrite(r_z, surf_z, x * sizeof(r_z), y, z, cudaBoundaryModeTrap);
}

// =============================================================================

void LaunchApplyBuoyancy(cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z,
                         cudaArray* temperature, cudaArray* density,
                         float time_step, float ambient_temperature,
                         float accel_factor, float gravity, bool staggered,
                         uint3 volume_size, BlockArrangement* ba)
{

    if (BindCudaSurfaceToArray(&surf, vel_y) != cudaSuccess)
        return;

    auto bound_v = BindHelper::Bind(&tex, vel_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_v.error() != cudaSuccess)
        return;

    auto bound_t = BindHelper::Bind(&tex_t, temperature, false,
                                    cudaFilterModeLinear, cudaAddressModeClamp);
    if (bound_t.error() != cudaSuccess)
        return;

    auto bound_d = BindHelper::Bind(&tex_d, density, false,
                                    cudaFilterModeLinear, cudaAddressModeClamp);
    if (bound_d.error() != cudaSuccess)
        return;

    if (staggered) {
        dim3 block(16, 1, 16);
        dim3 grid(volume_size.x / block.x, 1, volume_size.z / block.z);
        ba->ArrangeGrid(&grid, block, volume_size);
        grid.y = 1;
        ApplyBuoyancyStaggeredKernel<<<grid, block>>>(time_step,
                                                      ambient_temperature,
                                                      accel_factor, gravity,
                                                      volume_size);
    } else {
        dim3 block;
        dim3 grid;
        ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
        ApplyBuoyancyKernel<<<grid, block>>>(time_step, ambient_temperature,
                                             accel_factor, gravity,
                                             volume_size);
    }
}

void LaunchComputeDivergence(cudaArray* div, cudaArray* vel_x, cudaArray* vel_y,
                             cudaArray* vel_z, float cell_size, bool outflow,
                             bool staggered, uint3 volume_size,
                             BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, div) != cudaSuccess)
        return;

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
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);

    UpperBoundaryHandlerOutflow outflow_handler;
    UpperBoundaryHandlerNeumann neumann_handler;
    if (staggered) {
        if (outflow) {
            ComputeDivergenceStaggeredKernel<<<grid, block>>>(1.0f / cell_size,
                                                              volume_size,
                                                              outflow_handler);
        } else {
            ComputeDivergenceStaggeredKernel<<<grid, block>>>(1.0f / cell_size,
                                                              volume_size,
                                                              neumann_handler);
        }
    } else {
        if (outflow) {
            ComputeDivergenceKernel<<<grid, block>>>(0.5f / cell_size,
                                                     volume_size,
                                                     outflow_handler);
        } else {
            ComputeDivergenceKernel<<<grid, block>>>(0.5f / cell_size,
                                                     volume_size,
                                                     neumann_handler);
        }
    }
}

void LaunchComputeResidualDiagnosis(cudaArray* residual, cudaArray* u,
                                    cudaArray* b, float cell_size,
                                    uint3 volume_size, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, residual) != cudaSuccess)
        return;

    auto bound_u = BindHelper::Bind(&tex, u, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_u.error() != cudaSuccess)
        return;

    auto bound_b = BindHelper::Bind(&tex_b, b, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_b.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    ComputeResidualDiagnosisKernel<<<grid, block>>>(
        1.0f / (cell_size * cell_size), volume_size);
}

void LaunchRoundPassed(int* dest_array, int round, int x)
{
    RoundPassedKernel<<<1, 1>>>(dest_array, round, x);
}

void LaunchSubtractGradient(cudaArray* vel_x, cudaArray* vel_y,
                            cudaArray* vel_z, cudaArray* pressure,
                            float cell_size, bool staggered, uint3 volume_size,
                            BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf_x, vel_x) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_y, vel_y) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_z, vel_z) != cudaSuccess)
        return;

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

    auto bound = BindHelper::Bind(&tex, pressure, false, cudaFilterModeLinear,
                                  cudaAddressModeClamp);
    if (bound.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);

    if (staggered)
        SubtractGradientStaggeredKernel<<<grid, block>>>(1.0f / cell_size,
                                                         volume_size);
    else
        SubtractGradientKernel<<<grid, block>>>(0.5f / cell_size, volume_size);
}
