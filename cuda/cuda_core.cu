#include "cuda_core.h"

#include <cassert>

#include "opengl/glew.h"

#include <helper_math.h>

// cudaReadModeNormalizedFloat
// cudaReadModeElementType
texture<float1, cudaTextureType3D, cudaReadModeElementType> in_tex;
texture<float4, cudaTextureType3D, cudaReadModeElementType> prolongate_coarse;
texture<float4, cudaTextureType3D, cudaReadModeElementType> prolongate_fine;

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

__global__ void ProlongatePackedKernel(float4* out_data,
                                       int num_of_block_per_slice,
                                       int tile_height, int tile_width,
                                       int3 coarse_dim, float slice_stride,
                                       int fine_width)
{
    int block_offset = gridDim.x * gridDim.y * blockIdx.z +
        gridDim.x * blockIdx.y + blockIdx.x;

    int x = threadIdx.z * tile_width + threadIdx.x;
    int z = block_offset / num_of_block_per_slice;
    int y = (block_offset - z * num_of_block_per_slice) * tile_height + threadIdx.y;

    int index = slice_stride * z + fine_width * y + x;

    float3 c = make_float3(x, y, z);
    c *= 0.5f;

    int odd_x = x - ((x >> 1) << 1);
    int odd_y = y - ((y >> 1) << 1);
    int odd_z = z - ((z >> 1) << 1);

    float t_x = -1.0f * (1 - odd_x) * 0.08333333f;
    float t_y = -1.0f * (1 - odd_y) * 0.08333333f;
    float t_z = -1.0f * (1 - odd_z) * 0.08333333f;

    float3 t_c = make_float3(c.x + t_x, c.y + t_y, c.z + t_z);
    float4 result_float = tex3D(prolongate_coarse, t_c.x, t_c.y, t_c.z);

    float3 f_coord = make_float3(float(x) + 0.5f, float(y) + 0.5f,
                                 float(z) + 0.5f);

    float4 original = tex3D(prolongate_fine, f_coord.x, f_coord.y, f_coord.z);
    float4 result = make_float4(original.x + result_float.x, original.y, 0, 0);

    out_data[index] = result;
}

void LaunchProlongatePacked(float4* dest_array, cudaArray* coarse_array,
                            cudaArray* fine_array, int coarse_width)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    prolongate_coarse.normalized = false;
    prolongate_coarse.filterMode = cudaFilterModeLinear;
    prolongate_coarse.addressMode[0] = cudaAddressModeClamp;
    prolongate_coarse.addressMode[1] = cudaAddressModeClamp;
    prolongate_coarse.addressMode[2] = cudaAddressModeClamp;
    prolongate_coarse.channelDesc = desc;

    cudaError_t result = cudaBindTextureToArray(&prolongate_coarse,
                                                coarse_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    prolongate_fine.normalized = false;
    prolongate_fine.filterMode = cudaFilterModeLinear;
    prolongate_fine.addressMode[0] = cudaAddressModeClamp;
    prolongate_fine.addressMode[1] = cudaAddressModeClamp;
    prolongate_fine.addressMode[2] = cudaAddressModeClamp;
    prolongate_fine.channelDesc = desc;

    result = cudaBindTextureToArray(&prolongate_fine, fine_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    int fine_width = coarse_width * 2;
    dim3 block(8, 8, 16);
    dim3 grid(fine_width / block.x, fine_width / block.y, fine_width / block.z);
    int3 coarse_dim = make_int3(coarse_width, coarse_width, coarse_width);
    int num_of_block_per_slice = fine_width / 8;
    int tile_height = 8;
    int tile_width = 8;
    int slice_stride = fine_width * fine_width;
    ProlongatePackedKernel<<<grid, block>>>(dest_array, num_of_block_per_slice,
                                            tile_height, tile_width, coarse_dim,
                                            slice_stride, fine_width);

    cudaUnbindTexture(&prolongate_fine);
    cudaUnbindTexture(&prolongate_coarse);
}