#include <cassert>
#include <functional>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "aux_buffer_manager.h"
#include "block_arrangement.h"
#include "cuda_common.h"

surface<void, cudaSurfaceType3D> surf;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_0;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_1;

__global__ void ApplyStencilKernel(float inverse_square_cell_size,
                                   uint3 volume_size)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float near =   tex3D(tex, x,        y,        z - 1.0f);
    float south =  tex3D(tex, x,        y - 1.0f, z);
    float west =   tex3D(tex, x - 1.0f, y,        z);
    float center = tex3D(tex, x,        y,        z);
    float east =   tex3D(tex, x + 1.0f, y,        z);
    float north =  tex3D(tex, x,        y + 1.0f, z);
    float far =    tex3D(tex, x,        y,        z + 1.0f);

    float v = (north + south + east + west + far + near - 6.0f * center) *
        inverse_square_cell_size;
    auto r = __float2half_rn(v);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

__global__ void UpdateVectorKernel(float* coef, float sign, uint3 volume_size)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float e0 = tex3D(tex_0, x, y, z);
    float e1 = tex3D(tex_1, x, y, z);

    auto r = __float2half_rn(e0 + *coef * sign * e1);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

struct SchemeDefault
{
    __device__ float Load(uint i, uint row_stride, uint slice_stride)
    {
        uint z = i / slice_stride;
        uint y = (i % slice_stride) / row_stride;
        uint x = i % row_stride;

        float ¦Õ0 = tex3D(tex_0, static_cast<float>(x), static_cast<float>(y),
                         static_cast<float>(z));
        float ¦Õ1 = tex3D(tex_1, static_cast<float>(x), static_cast<float>(y),
                         static_cast<float>(z));
        return ¦Õ0 * ¦Õ1;
    }
    __device__ void Save(float* dest, float result)
    {
        *dest = result;
    }
};

struct SchemeAlpha : public SchemeDefault
{
    __device__ void Save(float* dest, float result)
    {
        if (result > 0.00000001f || result < -0.00000001f)
            *dest = *rho_ / result;
        else
            *dest = 0.0f;
    }

    float* rho_;
};

struct SchemeBeta : public SchemeDefault
{
    __device__ void Save(float* dest, float result)
    {
        *dest = result;

        float t = *rho_;
        if (t > 0.00000001f || t < -0.00000001f)
            *beta_ = result / t;
        else
            *beta_ = 0;
    }

    float* rho_;
    float* beta_;
};

#include "volume_reduction.cuh"

// =============================================================================

void LaunchApplyStencil(cudaArray* aux, cudaArray* search, float cell_size,
                        uint3 volume_size, BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, aux) != cudaSuccess)
        return;

    auto bound = BindHelper::Bind(&tex, search, false, cudaFilterModePoint,
                                  cudaAddressModeClamp);
    if (bound.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangeRowScan(&block, &grid, volume_size);
    ApplyStencilKernel<<<grid, block>>>(1.0f / (cell_size * cell_size),
                                        volume_size);
}

void LaunchComputeAlpha(float* alpha, float* rho, cudaArray* vec0,
                        cudaArray* vec1, uint3 volume_size,
                        BlockArrangement* ba, AuxBufferManager* bm)
{
    auto bound_0 = BindHelper::Bind(&tex_0, vec0, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_0.error() != cudaSuccess)
        return;

    auto bound_1 = BindHelper::Bind(&tex_1, vec1, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_1.error() != cudaSuccess)
        return;

    SchemeAlpha scheme;
    scheme.rho_ = rho;
    ReduceVolume(alpha, scheme, volume_size, ba, bm);
}

void LaunchComputeRho(float* rho, cudaArray* search, cudaArray* residual,
                      uint3 volume_size, BlockArrangement* ba,
                      AuxBufferManager* bm)
{
    auto bound_0 = BindHelper::Bind(&tex_0, search, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_0.error() != cudaSuccess)
        return;

    auto bound_1 = BindHelper::Bind(&tex_1, residual, false,
                                    cudaFilterModePoint, cudaAddressModeClamp);
    if (bound_1.error() != cudaSuccess)
        return;

    SchemeDefault scheme;
    ReduceVolume(rho, scheme, volume_size, ba, bm);
}

void LaunchComputeRhoAndBeta(float* beta, float* rho_new, float* rho,
                             cudaArray* vec0, cudaArray* vec1,
                             uint3 volume_size, BlockArrangement* ba,
                             AuxBufferManager* bm)
{
    
    auto bound_0 = BindHelper::Bind(&tex_0, vec0, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_0.error() != cudaSuccess)
        return;

    auto bound_1 = BindHelper::Bind(&tex_1, vec1, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_1.error() != cudaSuccess)
        return;

    SchemeBeta scheme;
    scheme.beta_ = beta;
    scheme.rho_ = rho;
    ReduceVolume(rho_new, scheme, volume_size, ba, bm);
}

void LaunchUpdateVector(cudaArray* dest, cudaArray* v0, cudaArray* v1,
                        float* coef, float sign, uint3 volume_size,
                        BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&surf, dest) != cudaSuccess)
        return;

    auto bound_0 = BindHelper::Bind(&tex_0, v0, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_0.error() != cudaSuccess)
        return;

    auto bound_1 = BindHelper::Bind(&tex_1, v1, false, cudaFilterModePoint,
                                    cudaAddressModeClamp);
    if (bound_1.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangeRowScan(&block, &grid, volume_size);
    UpdateVectorKernel<<<grid, block>>>(coef, sign, volume_size);
}
