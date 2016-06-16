#include <cassert>
#include <functional>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "aux_buffer_manager.h"
#include "block_arrangement.h"
#include "cuda_common.h"

surface<void, cudaSurfaceType3D> surf;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_0;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_1;

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