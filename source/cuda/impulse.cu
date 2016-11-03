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
#include "fluid_impulse.h"

surface<void, cudaSurfaceType3D> surf;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex;

__global__ void ApplyImpulse1Kernel(float3 center_point, float3 hotspot,
                                    float radius, float value,
                                    uint3 volume_size)
{
    int x = VolumeX();
    int y = 1 + threadIdx.y;
    int z = VolumeZ();

    if (x >= volume_size.x || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float2 diff = make_float2(coord.x, coord.z) -
        make_float2(center_point.x, center_point.z);
    float d = hypotf(diff.x, diff.y);
    if (d < radius) {
        diff = make_float2(coord.x, coord.z) -
            make_float2(hotspot.x, hotspot.z);
        float scale = (radius - hypotf(diff.x, diff.y)) / radius;
        scale = fmaxf(scale, 0.1f);
        surf3Dwrite(__float2half_rn(scale * value), surf,
                    x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
        return;
    }
}

__global__ void HotFloorKernel(float3 center_point, float3 hotspot,
                               float radius, float value, uint3 volume_size)
{
    int x = VolumeX();
    int y = 1 + threadIdx.y;
    int z = VolumeZ();

    if (x >= volume_size.x || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float2 diff =
        make_float2(coord.x, coord.z) - make_float2(hotspot.x, hotspot.z);
    float d = hypotf(diff.x, diff.y);
    if (d < radius) {
        surf3Dwrite(__float2half_rn(value), surf, x * sizeof(ushort), y, z,
                    cudaBoundaryModeTrap);
        return;
    }
}

__global__ void ImpulseDensityKernel(float3 center_point, float radius,
                                     float value, uint3 volume_size)
{
    int x = VolumeX();
    int y = 1 + threadIdx.y;
    int z = VolumeZ();

    if (x >= volume_size.x || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float2 diff = make_float2(coord.x, coord.z) -
        make_float2(center_point.x, center_point.z);
    float d = hypotf(diff.x, diff.y);
    if (d < radius) {
        surf3Dwrite(__float2half_rn(value), surf,
                    x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
    }
}

__global__ void GenerateHeatSphereKernel(float3 center_point, float radius,
                                         float value, uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 diff = make_float3(coord.x, coord.y, coord.z) -
        make_float3(center_point.x, center_point.y, center_point.z);
    float d = norm3df(diff.x, diff.y, diff.z);
    if (d < radius && d > radius * 0.5f) {
        surf3Dwrite(__float2half_rn(value), surf,
                    x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
        return;
    }
}

__global__ void ImpulseDensitySphereKernel(float3 center_point, float radius,
                                           float value, uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 diff = make_float3(coord.x, coord.y, coord.z) -
        make_float3(center_point.x, center_point.y, center_point.z);
    float d = norm3df(diff.x, diff.y, diff.z);
    if (d < radius && d > radius * 0.5f) {
        surf3Dwrite(__float2half_rn(value), surf,
                    x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
        return;
    }
}

__global__ void BuoyantJetKernel(float3 hotspot, float radius, float value,
                                 uint3 volume_size)
{
    int x = 1 + threadIdx.x;
    int y = VolumeY();
    int z = VolumeZ();

    if (y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float2 diff =
        make_float2(coord.y, coord.z) - make_float2(hotspot.y, hotspot.z);
    float d = hypotf(diff.x, diff.y);
    if (d < radius) {
        surf3Dwrite(__float2half_rn(value), surf, x * sizeof(ushort), y, z,
                    cudaBoundaryModeTrap);
        return;
    }
}

// =============================================================================

void LaunchImpulseScalar(cudaArray* dest, cudaArray* original,
                         float3 center_point, float3 hotspot, float radius,
                         float value, FluidImpulse impulse, uint3 volume_size,
                         BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, dest) != cudaSuccess)
        return;

    const int kHeatLayerThickness = 8;
    switch (impulse) {
        case IMPULSE_HOT_FLOOR: {
            dim3 block(volume_size.x, kHeatLayerThickness, 1);
            dim3 grid;
            ba->ArrangeGrid(&grid, block, volume_size);
            grid.y = 1;
            HotFloorKernel<<<grid, block>>>(center_point, hotspot, radius,
                                            value, volume_size);
            break;
        }
        case IMPULSE_SPHERE: {
            uint3 actual_size = volume_size;
            actual_size.y = static_cast<uint>(radius + center_point.y) + 1;

            dim3 block;
            dim3 grid;
            ba->ArrangeRowScan(&block, &grid, actual_size);
            GenerateHeatSphereKernel<<<grid, block>>>(center_point, radius,
                                                      value, volume_size);
            break;
        }
        case IMPULSE_BUOYANT_JET: {
            dim3 block(kHeatLayerThickness, volume_size.y, 1);
            dim3 grid;
            ba->ArrangeGrid(&grid, block, volume_size);
            grid.x = 1;
            BuoyantJetKernel<<<grid, block>>>(center_point, radius, value,
                                              volume_size);
            break;
        }
    }
}

void LaunchImpulseDensity(cudaArray* dest, cudaArray* original,
                          float3 center_point, float radius, float value,
                          FluidImpulse impulse, uint3 volume_size,
                          BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, dest) != cudaSuccess)
        return;

    switch (impulse) {
        case IMPULSE_HOT_FLOOR: {
            dim3 block(volume_size.x, 8, 1);
            dim3 grid;
            ba->ArrangeGrid(&grid, block, volume_size);
            grid.y = 1;
            ImpulseDensityKernel<<<grid, block>>>(center_point, radius, value,
                                                  volume_size);
            break;
        }
        case IMPULSE_SPHERE: {
            uint3 actual_size = volume_size;
            actual_size.y = static_cast<uint>(radius + center_point.y) + 1;

            dim3 block;
            dim3 grid;
            ba->ArrangeRowScan(&block, &grid, actual_size);
            grid.y = 1;
            ImpulseDensitySphereKernel<<<grid, block>>>(center_point, radius,
                                                        value, volume_size);
            break;
        }
        case IMPULSE_BUOYANT_JET: {
            dim3 block(2, volume_size.y, 1);
            dim3 grid;
            ba->ArrangeGrid(&grid, block, volume_size);
            grid.x = 1;
            BuoyantJetKernel<<<grid, block>>>(center_point, radius, value,
                                              volume_size);
            break;
        }
    }
}