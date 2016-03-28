#include "cuda_core.h"

#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

texture<float1, cudaTextureType3D, cudaReadModeElementType> in_tex;
surface<void, cudaTextureType3D> clear_volume;

__global__ void AbsoluteKernel(float* out_data, int w, int h, int d)
{
    int block_offset = gridDim.x * gridDim.y * blockIdx.z +
        gridDim.x * blockIdx.y + blockIdx.x;
    int index = block_offset * blockDim.x*blockDim.y*blockDim.z +
        blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
    float3 coord;
    coord.x = (float(blockIdx.x) * blockDim.x + threadIdx.x + 0.5f) / w;
    coord.y = (float(blockIdx.y) * blockDim.y + threadIdx.y + 0.5f) / h;
    coord.z = (float(blockIdx.z) * blockDim.z + threadIdx.x + 0.5f) / d;

    float1 cc = tex3D(in_tex, coord.x, coord.y, coord.z);
    out_data[index] = cc.x;
}

__global__ void ClearVolume4Kernel(float4 value)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    ushort4 raw = make_ushort4(__float2half_rn(value.x),
                               __float2half_rn(value.y),
                               __float2half_rn(value.z),
                               __float2half_rn(value.w));
    surf3Dwrite(raw, clear_volume, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

__global__ void ClearVolume1Kernel(float4 value)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    ushort1 raw = make_ushort1(__float2half_rn(value.x));
    surf3Dwrite(raw, clear_volume, x * sizeof(ushort1), y, z,
                cudaBoundaryModeTrap);
}

// =============================================================================

void LaunchClearVolumeKernel(cudaArray* dest_array, float4 value,
                             int3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaError_t result = cudaGetChannelDesc(&desc, dest_array);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    result = cudaBindSurfaceToArray(&clear_volume, dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);

    assert(
        desc.x ==16 &&
           ((desc.y == 0 && desc.z == 0) ||
            (desc.y == 16 && desc.z == 16)));
    if (desc.x == 16 && desc.y == 0 && desc.z == 0 && desc.w == 0)
        ClearVolume1Kernel<<<grid, block>>>(value);
    else if (desc.x == 16 && desc.y == 16 && desc.z == 16 && desc.w == 0)
        ClearVolume4Kernel<<<grid, block>>>(value);
}