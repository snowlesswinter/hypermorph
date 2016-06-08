#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "block_arrangement.h"
#include "cuda_common.h"

surface<void, cudaSurfaceType3D> surf;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex;

__global__ void ApplyImpulse1Kernel(float3 center_point, float3 hotspot,
                                    float radius, float value,
                                    uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = 1 + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

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

__global__ void ApplyImpulse1Kernel2(float3 center_point, float3 hotspot,
                                     float radius, float value,
                                     uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = 1 + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float2 diff =
        make_float2(coord.x, coord.z) - make_float2(hotspot.x, hotspot.z);
    float d = hypotf(diff.x, diff.y);
    if (d < 2.0f) {
        surf3Dwrite(__float2half_rn(value), surf, x * sizeof(ushort), y, z,
                    cudaBoundaryModeTrap);
        return;
    }
}

__global__ void ApplyImpulse3Kernel(float3 center_point, float3 hotspot,
                                    float radius, float3 value,
                                    uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = 1 + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

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
        ushort4 result = make_ushort4(__float2half_rn(scale * value.x),
                                      __float2half_rn(scale * value.y),
                                      __float2half_rn(scale * value.z),
                                      0);
        surf3Dwrite(result, surf, x * sizeof(ushort4), y, z,
                    cudaBoundaryModeTrap);
        return;
    }
}

__global__ void ImpulseDensityKernel(float3 center_point, float radius,
                                     float value, uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = 1 + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

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
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 diff = make_float3(coord.x, coord.y, coord.z) -
        make_float3(center_point.x, center_point.y, center_point.z);
    float d = norm3df(diff.x, diff.y, diff.z);
    if (d < radius && d > radius * 0.9f) {
        surf3Dwrite(__float2half_rn(value), surf,
                    x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
        return;
    }
}

__global__ void ImpulseDensitySphereKernel(float3 center_point, float radius,
                                           float value, uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 diff = make_float3(coord.x, coord.y, coord.z) -
        make_float3(center_point.x, center_point.y, center_point.z);
    float d = norm3df(diff.x, diff.y, diff.z);
    if (d < radius && d > radius * 0.9f) {
        surf3Dwrite(__float2half_rn(value), surf,
                    x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
        return;
    }
}

// =============================================================================

void LaunchApplyImpulse(cudaArray* dest_array, cudaArray* original_array,
                        float3 center_point, float3 hotspot, float radius,
                        float3 value, uint32_t mask, uint3 volume_size,
                        BlockArrangement* ba)
{
    assert(mask == 1 || mask == 7);
    if (mask != 1 && mask != 7)
        return;

    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    dim3 block(volume_size.x, 2, 1);
    dim3 grid;
    ba->ArrangeGrid(&grid, block, volume_size);
    grid.y = 1;
    if (mask == 1) {
        if (BindCudaSurfaceToArray(&surf, dest_array) != cudaSuccess)
            return;

        ApplyImpulse1Kernel<<<grid, block>>>(center_point, hotspot, radius,
                                             value.x, volume_size);
    } else if (mask == 7) {
        if (BindCudaSurfaceToArray(&surf, dest_array) != cudaSuccess)
            return;

        ApplyImpulse3Kernel<<<grid, block>>>(center_point, hotspot, radius,
                                             value, volume_size);
    }
}

void LaunchGenerateHeatSphere(cudaArray* dest, cudaArray* original,
                              float3 center_point, float radius, float3 value,
                              uint3 volume_size, BlockArrangement* ba)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest);

    if (BindCudaSurfaceToArray(&surf, dest) != cudaSuccess)
        return;

    uint3 actual_size = volume_size;
    actual_size.y = static_cast<uint>(radius + center_point.y) + 1;

    dim3 block;
    dim3 grid;
    ba->ArrangeRowScan(&block, &grid, actual_size);
    GenerateHeatSphereKernel<<<grid, block>>>(center_point, radius, value.x,
                                              volume_size);
}

void LaunchImpulseDensity(cudaArray* dest_array, cudaArray* original_array,
                          float3 center_point, float radius, float3 value,
                          uint3 volume_size, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, dest_array) != cudaSuccess)
        return;

    dim3 block(volume_size.x, 2, 1);
    dim3 grid;
    ba->ArrangeGrid(&grid, block, volume_size);
    grid.y = 1;
    ImpulseDensityKernel<<<grid, block>>>(center_point, radius, value.x,
                                          volume_size);
}

void LaunchImpulseDensitySphere(cudaArray* dest, cudaArray* original,
                                float3 center_point, float radius, float3 value,
                                uint3 volume_size, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, dest) != cudaSuccess)
        return;

    uint3 actual_size = volume_size;
    actual_size.y = static_cast<uint>(radius + center_point.y) + 1;

    dim3 block;
    dim3 grid;
    ba->ArrangeRowScan(&block, &grid, actual_size);
    ImpulseDensitySphereKernel<<<grid, block>>>(center_point, radius, value.x,
                                                volume_size);
}
