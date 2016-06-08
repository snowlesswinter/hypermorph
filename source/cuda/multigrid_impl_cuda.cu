#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "block_arrangement.h"
#include "cuda_common.h"

surface<void, cudaSurfaceType3D> surf;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_b;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_fine;

__global__ void ComputeResidualKernel(float inverse_h_square, uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float near =   tex3D(tex, x, y, z - 1.0f);
    float south =  tex3D(tex, x, y - 1.0f, z);
    float west =   tex3D(tex, x - 1.0f, y, z);
    float center = tex3D(tex, x, y, z);
    float east =   tex3D(tex, x + 1.0f, y, z);
    float north =  tex3D(tex, x, y + 1.0f, z);
    float far =    tex3D(tex, x, y, z + 1.0f);
    float b =      tex3D(tex_b, x, y, z);

    float v = b -
        (north + south + east + west + far + near - 6.0f * center) *
        inverse_h_square;
    auto r = __float2half_rn(v);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

__global__ void ProlongateLerpKernel(uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = (make_float3(x, y, z) + 0.5f) * 0.5f;
    auto raw = __float2half_rn(tex3D(tex, coord.x, coord.y, coord.z));
    surf3Dwrite(raw, surf, x * sizeof(raw), y, z, cudaBoundaryModeTrap);
}

__global__ void ProlongateErrorLerpKernel(uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = (make_float3(x, y, z) + 0.5f) * 0.5f;

    float coarse = tex3D(tex, coord.x, coord.y, coord.z);
    float fine = tex3D(tex_fine, x, y, z);
    auto raw = __float2half_rn(fine + coarse);
    surf3Dwrite(raw, surf, x * sizeof(raw), y, z, cudaBoundaryModeTrap);
}

__global__ void RelaxWithZeroGuessKernel(float alpha_omega_over_beta,
                                         float one_minus_omega,
                                         float minus_h_square,
                                         float omega_times_inverse_beta,
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

    float u = one_minus_omega * (alpha_omega_over_beta * center) +
        (alpha_omega_over_beta * (north + south + east + west + far + near) +
        minus_h_square * center) * omega_times_inverse_beta;

    auto r = __float2half_rn(u);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

__global__ void RestrictLerpKernel(uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = (make_float3(x, y, z) + 0.5f) * 2.0f;

    auto r = __float2half_rn(tex3D(tex, coord.x, coord.y, coord.z));
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

// =============================================================================

void LaunchComputeResidual(cudaArray* r, cudaArray* u, cudaArray* b,
                           float cell_size, uint3 volume_size,
                           BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, r) != cudaSuccess)
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
    ba->ArrangeRowScan(&block, &grid, volume_size);
    ComputeResidualKernel<<<grid, block>>>(1.0f / (cell_size * cell_size),
                                           volume_size);
}

void LaunchProlongate(cudaArray* fine, cudaArray* coarse,
                      uint3 volume_size_fine, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, fine) != cudaSuccess)
        return;

    auto bound_coarse = BindHelper::Bind(&tex, coarse, false,
                                         cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_coarse.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size_fine);
    ProlongateLerpKernel<<<grid, block>>>(volume_size_fine);
}

void LaunchProlongateError(cudaArray* fine, cudaArray* coarse,
                           uint3 volume_size_fine, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, fine) != cudaSuccess)
        return;

    auto bound_coarse = BindHelper::Bind(&tex, coarse, false,
                                         cudaFilterModeLinear,
                                         cudaAddressModeClamp);
    if (bound_coarse.error() != cudaSuccess)
        return;

    auto bound_fine = BindHelper::Bind(&tex_fine, fine, false,
                                       cudaFilterModePoint,
                                       cudaAddressModeClamp);
    if (bound_fine.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size_fine);
    ProlongateErrorLerpKernel<<<grid, block>>>(volume_size_fine);
}

void LaunchRelaxWithZeroGuess(cudaArray* u, cudaArray* b, float cell_size,
                              uint3 volume_size, BlockArrangement* ba)
{
    float alpha_omega_over_beta = -(cell_size * cell_size) * 0.11111111f;
    float one_minus_omega = 0.33333333f;
    float minus_h_square = -(cell_size * cell_size);
    float omega_times_inverse_beta = 0.11111111f;

    if (BindCudaSurfaceToArray(&surf, u) != cudaSuccess)
        return;

    auto bound = BindHelper::Bind(&tex, b, false, cudaFilterModePoint,
                                  cudaAddressModeClamp);
    if (bound.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    RelaxWithZeroGuessKernel<<<grid, block>>>(alpha_omega_over_beta,
                                              one_minus_omega,
                                              minus_h_square,
                                              omega_times_inverse_beta,
                                              volume_size);
}

void LaunchRestrict(cudaArray* coarse, cudaArray* fine, uint3 volume_size,
                    BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, coarse) != cudaSuccess)
        return;

    auto bound = BindHelper::Bind(&tex, fine, false, cudaFilterModeLinear,
                                  cudaAddressModeClamp);
    if (bound.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    RestrictLerpKernel<<<grid, block>>>(volume_size);
}
