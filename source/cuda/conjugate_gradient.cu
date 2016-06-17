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

__device__ float ReadFromTexture(uint i, uint row_stride, uint slice_stride)
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

struct SchemeDefault {};
struct SchemeAlpha {};

template <typename SaveScheme>
__device__ void SaveResult(float* dest, float result)
{
    *dest = result;
}

template <>
__device__ void SaveResult<SchemeAlpha>(float* dest, float result)
{
    *dest = *dest / result;
}

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

void LaunchComputeDotProductOfVectors(float* rho, cudaArray* vec0,
                                      cudaArray* vec1, uint3 volume_size,
                                      BlockArrangement* ba,
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

    std::unique_ptr<float, std::function<void (void*)>> aux_buffer(
        reinterpret_cast<float*>(bm->Allocate(64 * sizeof(float))),
        [&bm](void* p) { bm->Free(p); });

    ReduceVolume<SchemeAlpha>(rho, aux_buffer.get(), volume_size, ba);
    ReduceVolume<SchemeDefault>(rho, aux_buffer.get(), volume_size, ba);

}