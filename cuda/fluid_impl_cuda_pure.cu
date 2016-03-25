#include "fluid_impl_cuda.h"

#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_velocity;

__global__ void AdvectVelocityPureKernel(ushort4* out_data, float time_step,
                                         float dissipation,
                                         int num_of_blocks_per_slice,
                                         int slice_stride, int3 volume_size)
{
    int block_offset = gridDim.x * gridDim.y * blockIdx.z +
        gridDim.x * blockIdx.y + blockIdx.x;

    int x = threadIdx.z * blockDim.x + threadIdx.x;
    int z = block_offset / num_of_blocks_per_slice;
    int y = (block_offset - z * num_of_blocks_per_slice) * blockDim.y +
        threadIdx.y;

    int index = slice_stride * z + volume_size.x * y + x;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;
    float4 velocity = tex3D(advect_velocity, coord.x, coord.y, coord.z);
    float3 back_traced =
        coord - time_step * make_float3(velocity.x, velocity.y, velocity.z);

    float4 result = dissipation * tex3D(advect_velocity, back_traced.x,
                                        back_traced.y, back_traced.z);
    out_data[index] = make_ushort4(__float2half_rn(result.x),
                                   __float2half_rn(result.y),
                                   __float2half_rn(result.z),
                                   0);
}

// =============================================================================

void LaunchAdvectVelocityPure(void* dest_array, void* velocity_array,
                              float time_step, float dissipation,
                              int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
    advect_velocity.normalized = false;
    advect_velocity.filterMode = cudaFilterModeLinear;
    advect_velocity.addressMode[0] = cudaAddressModeClamp;
    advect_velocity.addressMode[1] = cudaAddressModeClamp;
    advect_velocity.addressMode[2] = cudaAddressModeClamp;
    advect_velocity.channelDesc = desc;

    size_t offset = 0;
    cudaError_t result = cudaBindTexture(&offset, &advect_velocity,
                                         velocity_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    int num_of_blocks_per_slice = volume_size.y / 8;
    int slice_stride = volume_size.x * volume_size.y;

    AdvectVelocityPureKernel<<<grid, block>>>(
        reinterpret_cast<ushort4*>(dest_array), time_step, dissipation,
        num_of_blocks_per_slice, slice_stride, volume_size);

    cudaUnbindTexture(&advect_velocity);
}