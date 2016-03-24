#include "cuda_core.h"

#include <cassert>

#include "opengl/glew.h"

#include <helper_math.h>

// cudaReadModeNormalizedFloat
// cudaReadModeElementType
texture<float1, cudaTextureType3D, cudaReadModeElementType> in_tex;
texture<float4, cudaTextureType3D, cudaReadModeElementType> prolongate_coarse;
texture<float4, cudaTextureType3D, cudaReadModeElementType> prolongate_fine;
texture<float4, cudaTextureType3D, cudaReadModeElementType> advect_velocity;
texture<float, cudaTextureType3D, cudaReadModeElementType> advect_source;

__global__ void RoundPassedKernel(int* dest_array, int round, int x)
{
    dest_array[0] = x * x - round * round;
}

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
                                       int num_of_blocks_per_slice,
                                       int slice_stride, int fine_width)
{
    int block_offset = gridDim.x * gridDim.y * blockIdx.z +
        gridDim.x * blockIdx.y + blockIdx.x;

    int x = threadIdx.z * blockDim.x + threadIdx.x;
    int z = block_offset / num_of_blocks_per_slice;
    int y = (block_offset - z * num_of_blocks_per_slice) * blockDim.y +
        threadIdx.y;

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

__global__ void AdvectVelocityKernel(float4* out_data, float time_step,
                                     float dissipation,
                                     int num_of_blocks_per_slice,
                                     int slice_stride, int width)
{
    int block_offset = gridDim.x * gridDim.y * blockIdx.z +
        gridDim.x * blockIdx.y + blockIdx.x;

    int x = threadIdx.z * blockDim.x + threadIdx.x;
    int z = block_offset / num_of_blocks_per_slice;
    int y = (block_offset - z * num_of_blocks_per_slice) * blockDim.y +
        threadIdx.y;

    int index = slice_stride * z + width * y + x;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;
    float4 velocity = tex3D(advect_velocity, coord.x, coord.y, coord.z);
    float3 back_traced =
        coord - time_step * make_float3(velocity.x, velocity.y, velocity.z);

    out_data[index] = dissipation * tex3D(advect_velocity, back_traced.x,
                                          back_traced.y, back_traced.z);
}

__global__ void AdvectKernel(float* out_data, float time_step,
                             float dissipation, int num_of_blocks_per_slice,
                             int slice_stride, int width)
{
    int block_offset = gridDim.x * gridDim.y * blockIdx.z +
        gridDim.x * blockIdx.y + blockIdx.x;

    int x = threadIdx.z * blockDim.x + threadIdx.x;
    int z = block_offset / num_of_blocks_per_slice;
    int y = (block_offset - z * num_of_blocks_per_slice) * blockDim.y +
        threadIdx.y;

    int index = slice_stride * z + width * y + x;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;
    float4 velocity = tex3D(advect_velocity, coord.x, coord.y, coord.z);
    float3 back_traced =
        coord - time_step * make_float3(velocity.x, velocity.y, velocity.z);

    out_data[index] = dissipation * tex3D(advect_source, back_traced.x,
                                          back_traced.y, back_traced.z);
}

// =============================================================================

void LaunchRoundPassed(int* dest_array, int round, int x)
{
    RoundPassedKernel<<<1, 1>>>(dest_array, round, x);
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

    // TODO: Disabling the linear filter mode may slightly speed up the kernel.
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
    int num_of_blocks_per_slice = fine_width / 8;
    int slice_stride = fine_width * fine_width;
    ProlongatePackedKernel<<<grid, block>>>(dest_array, num_of_blocks_per_slice,
                                            slice_stride, fine_width);

    cudaUnbindTexture(&prolongate_fine);
    cudaUnbindTexture(&prolongate_coarse);
}

void LaunchAdvectVelocity(float4* dest_array, cudaArray* velocity_array,
                          float time_step, float dissipation, int width)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    advect_velocity.normalized = false;
    advect_velocity.filterMode = cudaFilterModeLinear;
    advect_velocity.addressMode[0] = cudaAddressModeClamp;
    advect_velocity.addressMode[1] = cudaAddressModeClamp;
    advect_velocity.addressMode[2] = cudaAddressModeClamp;
    advect_velocity.channelDesc = desc;

    cudaError_t result = cudaBindTextureToArray(&advect_velocity,
                                                velocity_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, 16);
    dim3 grid(width / block.x, width / block.y, width / block.z);
    int num_of_blocks_per_slice = width / 8;
    int slice_stride = width * width;

    AdvectVelocityKernel<<<grid, block>>>(dest_array, time_step, dissipation,
                                          num_of_blocks_per_slice, slice_stride,
                                          width);

    cudaUnbindTexture(&advect_velocity);
}

void LaunchAdvect(float* dest_array, cudaArray* velocity_array,
                  cudaArray* source_array, float time_step,
                  float dissipation, int width)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    advect_velocity.normalized = false;
    advect_velocity.filterMode = cudaFilterModeLinear;
    advect_velocity.addressMode[0] = cudaAddressModeClamp;
    advect_velocity.addressMode[1] = cudaAddressModeClamp;
    advect_velocity.addressMode[2] = cudaAddressModeClamp;
    advect_velocity.channelDesc = desc;

    cudaError_t result = cudaBindTextureToArray(&advect_velocity,
                                                velocity_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    desc = cudaCreateChannelDesc<float>();
    advect_source.normalized = false;
    advect_source.filterMode = cudaFilterModeLinear;
    advect_source.addressMode[0] = cudaAddressModeClamp;
    advect_source.addressMode[1] = cudaAddressModeClamp;
    advect_source.addressMode[2] = cudaAddressModeClamp;
    advect_source.channelDesc = desc;

    result = cudaBindTextureToArray(&advect_source, source_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, 16);
    dim3 grid(width / block.x, width / block.y, width / block.z);
    int num_of_blocks_per_slice = width / 8;
    int slice_stride = width * width;

    AdvectKernel<<<grid, block>>>(dest_array, time_step, dissipation,
                                  num_of_blocks_per_slice, slice_stride,
                                  width);

    cudaUnbindTexture(&advect_source);
    cudaUnbindTexture(&advect_velocity);
}