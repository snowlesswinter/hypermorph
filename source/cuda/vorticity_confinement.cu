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
#include "cuda_common_host.h"
#include "cuda_common_kern.h"

surface<void, cudaSurfaceType3D> surf;
surface<void, cudaSurfaceType3D> surf_x;
surface<void, cudaSurfaceType3D> surf_y;
surface<void, cudaSurfaceType3D> surf_z;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_x;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_y;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_z;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_vx;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_vy;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_vz;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_xp;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_yp;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_zp;

__global__ void AddCurlPsiKernel(float inverse_cell_size, uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    // Keep away from the shearing boundaries.
    int d = 8;
    if (x <= d || y <= d || z <= d || x >= volume_size.x - d ||
            y >= volume_size.y - d || z >= volume_size.z - d)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float ¦·_y0 = tex3D(tex_y, coord.x, coord.y, coord.z);
    float ¦·_y1 = tex3D(tex_y, coord.x, coord.y, coord.z + 1.0f);
    float ¦·_z0 = tex3D(tex_z, coord.x, coord.y, coord.z);
    float ¦·_z1 = tex3D(tex_z, coord.x, coord.y + 1.0f, coord.z);

    float u = inverse_cell_size * (¦·_z1 - ¦·_z0 - ¦·_y1 + ¦·_y0);

    float ¦·_x0 = tex3D(tex_x, coord.x, coord.y, coord.z);
    float ¦·_x1 = tex3D(tex_x, coord.x, coord.y, coord.z + 1.0f);
    float ¦·_z2 = tex3D(tex_z, coord.x + 1.0f, coord.y, coord.z);

    float v = inverse_cell_size * (¦·_x1 - ¦·_x0 - ¦·_z2 + ¦·_z0);

    float ¦·_y2 = tex3D(tex_y, coord.x + 1.0f, coord.y, coord.z);
    float ¦·_x2 = tex3D(tex_x, coord.x, coord.y + 1.0f, coord.z);
    float w = inverse_cell_size * (¦·_y2 - ¦·_y0 - ¦·_x2 + ¦·_x0);

    float v_x = tex3D(tex_vx, coord.x, coord.y, coord.z);
    auto r_x = __float2half_rn(v_x + u);
    surf3Dwrite(r_x, surf_x, x * sizeof(r_x), y, z, cudaBoundaryModeTrap);

    float v_y = tex3D(tex_vy, coord.x, coord.y, coord.z);
    auto r_y = __float2half_rn(v_y + v);
    surf3Dwrite(r_y, surf_y, x * sizeof(r_y), y, z, cudaBoundaryModeTrap);

    float v_z = tex3D(tex_vz, coord.x, coord.y, coord.z);
    auto r_z = __float2half_rn(v_z + w);
    surf3Dwrite(r_z, surf_z, x * sizeof(r_z), y, z, cudaBoundaryModeTrap);
}

__global__ void ApplyVorticityConfinementStaggeredKernel(uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x == 0 || y == 0 || z == 0)
        return;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float conf_x = tex3D(tex_x, coord.x - 0.5f, coord.y + 0.5f, coord.z + 0.5f);
    float vel_x = tex3D(tex_vx, coord.x, coord.y, coord.z);
    auto r_x = __float2half_rn(vel_x + conf_x);
    surf3Dwrite(r_x, surf_x, x * sizeof(r_x), y, z, cudaBoundaryModeTrap);

    float conf_y = tex3D(tex_y, coord.x + 0.5f, coord.y - 0.5f, coord.z + 0.5f);
    float vel_y = tex3D(tex_vy, coord.x, coord.y, coord.z);
    auto r_y = __float2half_rn(vel_y + conf_y);
    surf3Dwrite(r_y, surf_y, x * sizeof(r_y), y, z, cudaBoundaryModeTrap);

    float conf_z = tex3D(tex_z, coord.x + 0.5f, coord.y + 0.5f, coord.z - 0.5f);
    float vel_z = tex3D(tex_vz, coord.x, coord.y, coord.z);
    auto r_z = __float2half_rn(vel_z + conf_z);
    surf3Dwrite(r_z, surf_z, x * sizeof(r_z), y, z, cudaBoundaryModeTrap);
}

__global__ void BuildVorticityConfinementStaggeredKernel(
    float coeff, float cell_size, float half_inverse_cell_size,
    uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;

    // Calculate the gradient of vorticity.
    float near_vort_x =   tex3D(tex_x, coord.x,        coord.y + 0.5f, coord.z - 0.5f);
    float near_vort_y =   tex3D(tex_y, coord.x + 0.5f, coord.y,        coord.z - 0.5f);
    float near_vort_z =   tex3D(tex_z, coord.x + 0.5f, coord.y + 0.5f, coord.z - 1.0f);
    float near_vort = __fsqrt_rn(near_vort_x * near_vort_x + near_vort_y * near_vort_y + near_vort_z * near_vort_z);

    float south_vort_x =  tex3D(tex_x, coord.x,        coord.y - 0.5f, coord.z + 0.5f);
    float south_vort_y =  tex3D(tex_y, coord.x + 0.5f, coord.y - 1.0f, coord.z + 0.5f);
    float south_vort_z =  tex3D(tex_z, coord.x + 0.5f, coord.y - 0.5f, coord.z);
    float south_vort = __fsqrt_rn(south_vort_x * south_vort_x + south_vort_y * south_vort_y + south_vort_z * south_vort_z);

    float west_vort_x =   tex3D(tex_x, coord.x - 1.0f, coord.y + 0.5f, coord.z + 0.5f);
    float west_vort_y =   tex3D(tex_y, coord.x - 0.5f, coord.y,        coord.z + 0.5f);
    float west_vort_z =   tex3D(tex_z, coord.x - 0.5f, coord.y + 0.5f, coord.z);
    float west_vort = __fsqrt_rn(west_vort_x * west_vort_x + west_vort_y * west_vort_y + west_vort_z * west_vort_z);

    float center_vort_x = tex3D(tex_x, coord.x,        coord.y,        coord.z);
    float center_vort_y = tex3D(tex_y, coord.x,        coord.y,        coord.z);
    float center_vort_z = tex3D(tex_z, coord.x,        coord.y,        coord.z);

    float east_vort_x =   tex3D(tex_x, coord.x + 1.0f, coord.y + 0.5f, coord.z + 0.5f);
    float east_vort_y =   tex3D(tex_y, coord.x + 1.5f, coord.y,        coord.z + 0.5f);
    float east_vort_z =   tex3D(tex_z, coord.x + 1.5f, coord.y + 0.5f, coord.z);
    float east_vort = __fsqrt_rn(east_vort_x * east_vort_x + east_vort_y * east_vort_y + east_vort_z * east_vort_z);

    float north_vort_x =  tex3D(tex_x, coord.x,        coord.y + 1.5f, coord.z + 0.5f);
    float north_vort_y =  tex3D(tex_y, coord.x + 0.5f, coord.y + 1.0f, coord.z + 0.5f);
    float north_vort_z =  tex3D(tex_z, coord.x + 0.5f, coord.y + 1.5f, coord.z);
    float north_vort = __fsqrt_rn(north_vort_x * north_vort_x + north_vort_y * north_vort_y + north_vort_z * north_vort_z);

    float far_vort_x =    tex3D(tex_x, coord.x,        coord.y + 0.5f, coord.z + 1.5f);
    float far_vort_y =    tex3D(tex_y, coord.x + 0.5f, coord.y,        coord.z + 1.5f);
    float far_vort_z =    tex3D(tex_z, coord.x + 0.5f, coord.y + 0.5f, coord.z + 1.0f);
    float far_vort = __fsqrt_rn(far_vort_x * far_vort_x + far_vort_y * far_vort_y + far_vort_z * far_vort_z);

    // Calculate normalized ¦Ç.
    float ¦Ç_x = half_inverse_cell_size * (east_vort - west_vort);
    float ¦Ç_y = half_inverse_cell_size * (north_vort - south_vort);
    float ¦Ç_z = half_inverse_cell_size * (far_vort - near_vort);

    float r_¦Ç_mag = __frsqrt_rn(¦Ç_x * ¦Ç_x + ¦Ç_y * ¦Ç_y + ¦Ç_z * ¦Ç_z + 0.00001f);
    ¦Ç_x *= r_¦Ç_mag;
    ¦Ç_y *= r_¦Ç_mag;
    ¦Ç_z *= r_¦Ç_mag;

    // Vorticity confinement at the center of the grid.
    float conf_x = coeff * cell_size * (¦Ç_y * center_vort_z - ¦Ç_z * center_vort_y);
    float conf_y = coeff * cell_size * (¦Ç_z * center_vort_x - ¦Ç_x * center_vort_z);
    float conf_z = coeff * cell_size * (¦Ç_x * center_vort_y - ¦Ç_y * center_vort_x);

    ushort result_x = __float2half_rn(conf_x);
    surf3Dwrite(result_x, surf_x, x * sizeof(result_x), y, z, cudaBoundaryModeTrap);

    ushort result_y = __float2half_rn(conf_y);
    surf3Dwrite(result_y, surf_y, x * sizeof(result_y), y, z, cudaBoundaryModeTrap);

    ushort result_z = __float2half_rn(conf_z);
    surf3Dwrite(result_z, surf_z, x * sizeof(result_z), y, z, cudaBoundaryModeTrap);
}

__global__ void ComputeCurlStaggeredKernel(uint3 volume_size,
                                           float inverse_cell_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 result;
    if (x < volume_size.x - 1 && y < volume_size.y - 1 && z < volume_size.z - 1) {
        if (x > 0 && y > 0 && z > 0) {
            float v_z =       tex3D(tex_vz, coord.x,        coord.y,        coord.z);
            float v_z_west =  tex3D(tex_vz, coord.x - 1.0f, coord.y,        coord.z);
            float v_z_south = tex3D(tex_vz, coord.x,        coord.y - 1.0f, coord.z);
            float v_y =       tex3D(tex_vy, coord.x,        coord.y,        coord.z);
            float v_y_near =  tex3D(tex_vy, coord.x,        coord.y,        coord.z - 1.0f);
            float v_y_west =  tex3D(tex_vy, coord.x - 1.0f, coord.y,        coord.z);
            float v_x =       tex3D(tex_vx, coord.x,        coord.y,        coord.z);
            float v_x_near =  tex3D(tex_vx, coord.x,        coord.y,        coord.z - 1.0f);
            float v_x_south = tex3D(tex_vx, coord.x,        coord.y - 1.0f, coord.z);

            result.x = inverse_cell_size * (v_z - v_z_south - v_y + v_y_near);
            result.y = inverse_cell_size * (v_x - v_x_near -  v_z + v_z_west);
            result.z = inverse_cell_size * (v_y - v_y_west -  v_x + v_x_south);
        } else if (x == 0) {
            result.x = tex3D(tex_x, coord.x + 1.0f, coord.y, coord.z);
            result.y = tex3D(tex_y, coord.x + 1.0f, coord.y, coord.z);
            result.z = tex3D(tex_z, coord.x + 1.0f, coord.y, coord.z);
        } else if (y == 0) {
            result.x = tex3D(tex_x, coord.x, coord.y + 1.0f, coord.z);
            result.y = tex3D(tex_y, coord.x, coord.y + 1.0f, coord.z);
            result.z = tex3D(tex_z, coord.x, coord.y + 1.0f, coord.z);
        } else {
            result.x = tex3D(tex_x, coord.x, coord.y, coord.z + 1.0f);
            result.y = tex3D(tex_y, coord.x, coord.y, coord.z + 1.0f);
            result.z = tex3D(tex_z, coord.x, coord.y, coord.z + 1.0f);
        }
    } else if (x == volume_size.x - 1) {
        result.x = tex3D(tex_x, coord.x - 1.0f, coord.y, coord.z);
        result.y = tex3D(tex_y, coord.x - 1.0f, coord.y, coord.z);
        result.z = tex3D(tex_z, coord.x - 1.0f, coord.y, coord.z);
    } else if (y == volume_size.y - 1) {
        result.x = tex3D(tex_x, coord.x, coord.y - 1.0f, coord.z);
        result.y = tex3D(tex_y, coord.x, coord.y - 1.0f, coord.z);
        result.z = tex3D(tex_z, coord.x, coord.y - 1.0f, coord.z);
    } else {
        result.x = tex3D(tex_x, coord.x, coord.y, coord.z - 1.0f);
        result.y = tex3D(tex_y, coord.x, coord.y, coord.z - 1.0f);
        result.z = tex3D(tex_z, coord.x, coord.y, coord.z - 1.0f);
    }

    auto raw_x = __float2half_rn(result.x);
    surf3Dwrite(raw_x, surf_x, x * sizeof(raw_x), y, z, cudaBoundaryModeTrap);

    auto raw_y = __float2half_rn(result.y);
    surf3Dwrite(raw_y, surf_y, x * sizeof(raw_y), y, z, cudaBoundaryModeTrap);

    auto raw_z = __float2half_rn(result.z);
    surf3Dwrite(raw_z, surf_z, x * sizeof(raw_z), y, z, cudaBoundaryModeTrap);
}

__global__ void ComputeDeltaVorticityKernel(uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z);

    float v_x =  tex3D(tex_x,  coord.x, coord.y, coord.z);
    float v_xp = tex3D(tex_xp, coord.x, coord.y, coord.z);
    auto raw_x = __float2half_rn(v_xp - v_x);
    surf3Dwrite(raw_x, surf_x, x * sizeof(raw_x), y, z, cudaBoundaryModeTrap);

    float v_y =  tex3D(tex_y,  coord.x, coord.y, coord.z);
    float v_yp = tex3D(tex_yp, coord.x, coord.y, coord.z);
    auto raw_y = __float2half_rn(v_yp - v_y);
    surf3Dwrite(raw_y, surf_y, x * sizeof(raw_y), y, z, cudaBoundaryModeTrap);
    
    float v_z =  tex3D(tex_z,  coord.x, coord.y, coord.z);
    float v_zp = tex3D(tex_zp, coord.x, coord.y, coord.z);
    auto raw_z = __float2half_rn(v_zp - v_z);
    surf3Dwrite(raw_z, surf_z, x * sizeof(raw_z), y, z, cudaBoundaryModeTrap);
}

__global__ void DecayVorticesStaggeredKernel(float time_step, uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float div_x = tex3D(tex, coord.x, coord.y - 0.5f, coord.z - 0.5f);
    float coef_x = fminf(0.0f, -div_x * time_step);

    float vort_x = tex3D(tex_x, coord.x, coord.y, coord.z);
    auto r_x = __float2half_rn(vort_x * __expf(coef_x));
    surf3Dwrite(r_x, surf_x, x * sizeof(r_x), y, z, cudaBoundaryModeTrap);

    float div_y = tex3D(tex, coord.x - 0.5f, coord.y, coord.z - 0.5f);
    float coef_y = fminf(0.0f, -div_y * time_step);

    float vort_y = tex3D(tex_y, coord.x, coord.y, coord.z);
    auto r_y = __float2half_rn(vort_y * __expf(coef_y));
    surf3Dwrite(r_y, surf_y, x * sizeof(r_y), y, z, cudaBoundaryModeTrap);

    float div_z = tex3D(tex, coord.x - 0.5f, coord.y - 0.5f, coord.z);
    float coef_z = fminf(0.0f, -div_z * time_step);

    float vort_z = tex3D(tex_z, coord.x, coord.y, coord.z);
    auto r_z = __float2half_rn(vort_z * __expf(coef_z));
    surf3Dwrite(r_z, surf_z, x * sizeof(r_z), y, z, cudaBoundaryModeTrap);
}

__global__ void StretchVorticesStaggeredKernel(float scale, uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float ¦Ø_xx = tex3D(tex_x, coord.x, coord.y, coord.z);
    float ¦Ø_xy = tex3D(tex_y, coord.x + 0.5f, coord.y - 0.5f, coord.z);
    float ¦Ø_xz = tex3D(tex_z, coord.x + 0.5f, coord.y, coord.z - 0.5f);

    float mag_x = sqrtf(¦Ø_xx * ¦Ø_xx + ¦Ø_xy * ¦Ø_xy + ¦Ø_xz * ¦Ø_xz + 0.00001f);
    float dx_x = ¦Ø_xx / mag_x;
    float dy_x = ¦Ø_xy / mag_x;
    float dz_x = ¦Ø_xz / mag_x;

    float v_x0 = tex3D(tex_vx, coord.x + dx_x + 0.5f, coord.y + dy_x - 0.5f, coord.z + dz_x - 0.5f);
    float v_x1 = tex3D(tex_vx, coord.x - dx_x + 0.5f, coord.y - dy_x - 0.5f, coord.z - dz_x - 0.5f);

    auto result_x = __float2half_rn(scale * (v_x0 - v_x1) * mag_x + ¦Ø_xx);
    surf3Dwrite(result_x, surf_x, x * sizeof(result_x), y, z, cudaBoundaryModeTrap);

    float ¦Ø_yx = tex3D(tex_x, coord.x - 0.5f, coord.y + 0.5f, coord.z);
    float ¦Ø_yy = tex3D(tex_y, coord.x, coord.y, coord.z);
    float ¦Ø_yz = tex3D(tex_z, coord.x, coord.y + 0.5f, coord.z - 0.5f);

    float mag_y = sqrtf(¦Ø_yx * ¦Ø_yx + ¦Ø_yy * ¦Ø_yy + ¦Ø_yz * ¦Ø_yz + 0.00001f);
    float dx_y = ¦Ø_yx / mag_y;
    float dy_y = ¦Ø_yy / mag_y;
    float dz_y = ¦Ø_yz / mag_y;

    float v_y0 = tex3D(tex_vy, coord.x + dx_y - 0.5f, coord.y + dy_y + 0.5f, coord.z + dz_y - 0.5f);
    float v_y1 = tex3D(tex_vy, coord.x - dx_y - 0.5f, coord.y - dy_y + 0.5f, coord.z - dz_y - 0.5f);

    auto result_y = __float2half_rn(scale * (v_y0 - v_y1) * mag_y + ¦Ø_yy);
    surf3Dwrite(result_y, surf_y, x * sizeof(result_y), y, z, cudaBoundaryModeTrap);

    float ¦Ø_zx = tex3D(tex_x, coord.x - 0.5f, coord.y, coord.z + 0.5f);
    float ¦Ø_zy = tex3D(tex_y, coord.x, coord.y - 0.5f, coord.z + 0.5f);
    float ¦Ø_zz = tex3D(tex_z, coord.x, coord.y, coord.z);

    float mag_z = sqrtf(¦Ø_zx * ¦Ø_zx + ¦Ø_zy * ¦Ø_zy + ¦Ø_zz * ¦Ø_zz + 0.00001f);
    float dx_z = ¦Ø_zx / mag_z;
    float dy_z = ¦Ø_zy / mag_z;
    float dz_z = ¦Ø_zz / mag_z;

    float v_z0 = tex3D(tex_vz, coord.x + dx_z - 0.5f, coord.y + dy_z - 0.5f, coord.z + dz_z + 0.5f);
    float v_z1 = tex3D(tex_vz, coord.x - dx_z - 0.5f, coord.y - dy_z - 0.5f, coord.z - dz_z + 0.5f);

    auto result_z = __float2half_rn(scale * (v_z0 - v_z1) * mag_z + ¦Ø_zz);
    surf3Dwrite(result_z, surf_z, x * sizeof(result_z), y, z, cudaBoundaryModeTrap);
}

// =============================================================================

void LaunchAddCurlPsi(cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z,
                      cudaArray* psi_x, cudaArray* psi_y, cudaArray* psi_z,
                      float cell_size, uint3 volume_size, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf_x, vel_x) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_y, vel_y) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_z, vel_z) != cudaSuccess)
        return;

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

    auto bound_x = BindHelper::Bind(&tex_x, psi_x, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, psi_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, psi_z, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    AddCurlPsiKernel<<<grid, block>>>(1.0f / cell_size, volume_size);
}

void LaunchApplyVorticityConfinementStaggered(cudaArray* vel_x,
                                              cudaArray* vel_y,
                                              cudaArray* vel_z,
                                              cudaArray* conf_x,
                                              cudaArray* conf_y,
                                              cudaArray* conf_z,
                                              uint3 volume_size,
                                              BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf_x, vel_x) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_y, vel_y) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_z, vel_z) != cudaSuccess)
        return;

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

    auto bound_x = BindHelper::Bind(&tex_x, conf_x, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, conf_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, conf_z, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    ApplyVorticityConfinementStaggeredKernel<<<grid, block>>>(volume_size);
}

void LaunchBuildVorticityConfinementStaggered(cudaArray* conf_x,
                                              cudaArray* conf_y,
                                              cudaArray* conf_z,
                                              cudaArray* vort_x,
                                              cudaArray* vort_y,
                                              cudaArray* vort_z,
                                              float coeff, float cell_size,
                                              uint3 volume_size,
                                              BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf_x, conf_x) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_y, conf_y) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_z, conf_z) != cudaSuccess)
        return;

    auto bound_x = BindHelper::Bind(&tex_x, vort_x, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, vort_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, vort_z, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    BuildVorticityConfinementStaggeredKernel<<<grid, block>>>(coeff, cell_size,
                                                              0.5f / cell_size,
                                                              volume_size);
}

void LaunchComputeCurlStaggered(cudaArray* vort_x, cudaArray* vort_y,
                                cudaArray* vort_z, cudaArray* vel_x,
                                cudaArray* vel_y, cudaArray* vel_z,
                                float cell_size, uint3 volume_size,
                                BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf_x, vort_x) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_y, vort_y) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_z, vort_z) != cudaSuccess)
        return;

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

    auto bound_x = BindHelper::Bind(&tex_x, vort_x, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, vort_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, vort_z, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    ComputeCurlStaggeredKernel<<<grid, block>>>(volume_size, 1.0f / cell_size);
}

void LaunchComputeDeltaVorticity(cudaArray* delta_x, cudaArray* delta_y,
                                 cudaArray* delta_z,  cudaArray* vn_x,
                                 cudaArray* vn_y, cudaArray* vn_z,
                                 uint3 volume_size, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf_x, delta_x) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_y, delta_y) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_z, delta_z) != cudaSuccess)
        return;

    auto bound_x = BindHelper::Bind(&tex_x, vn_x, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, vn_y, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, vn_z, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    auto bound_xp = BindHelper::Bind(&tex_xp, delta_x, false,
                                     cudaFilterModePoint, cudaAddressModeClamp);
    if (bound_xp.error() != cudaSuccess)
        return;

    auto bound_yp = BindHelper::Bind(&tex_yp, delta_y, false,
                                     cudaFilterModePoint, cudaAddressModeClamp);
    if (bound_yp.error() != cudaSuccess)
        return;

    auto bound_zp = BindHelper::Bind(&tex_zp, delta_z, false,
                                     cudaFilterModePoint, cudaAddressModeClamp);
    if (bound_zp.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    ComputeDeltaVorticityKernel<<<grid, block>>>(volume_size);
}

void LaunchDecayVorticesStaggered(cudaArray* vort_x, cudaArray* vort_y,
                                  cudaArray* vort_z, cudaArray* div,
                                  float time_step, uint3 volume_size,
                                  BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf_x, vort_x) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_y, vort_y) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_z, vort_z) != cudaSuccess)
        return;

    auto bound = BindHelper::Bind(&tex, div, false, cudaFilterModeLinear,
                                  cudaAddressModeClamp);
    if (bound.error() != cudaSuccess)
        return;

    auto bound_x = BindHelper::Bind(&tex_x, vort_x, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, vort_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, vort_z, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    DecayVorticesStaggeredKernel<<<grid, block>>>(time_step, volume_size);
}

void LaunchStretchVorticesStaggered(cudaArray* vnp1_x, cudaArray* vnp1_y,
                                    cudaArray* vnp1_z, cudaArray* vel_x,
                                    cudaArray* vel_y, cudaArray* vel_z,
                                    cudaArray* vort_x, cudaArray* vort_y,
                                    cudaArray* vort_z, float cell_size,
                                    float time_step, uint3 volume_size,
                                    BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf_x, vnp1_x) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_y, vnp1_y) != cudaSuccess)
        return;

    if (BindCudaSurfaceToArray(&surf_z, vnp1_z) != cudaSuccess)
        return;

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

    auto bound_x = BindHelper::Bind(&tex_x, vort_x, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, vort_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, vort_z, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    float scale = time_step / (cell_size * 2.0f);

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    StretchVorticesStaggeredKernel<<<grid, block>>>(scale, volume_size);
}
