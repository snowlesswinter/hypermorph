#include "fluid_impl_cuda.h"

#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

surface<void, cudaTextureType3D> advect_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_velocity;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_source;

__global__ void AdvectVelocityPureKernel(float time_step,
                                         float dissipation,
                                         int slice_stride, int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;
    float4 velocity = tex3D(advect_velocity, coord.x, coord.y, coord.z);
    float3 back_traced =
        coord - time_step * make_float3(velocity.x, velocity.y, velocity.z);

    float4 new_velocity = dissipation * tex3D(advect_velocity, back_traced.x,
                                              back_traced.y, back_traced.z);
    ushort4 result = make_ushort4(__float2half_rn(new_velocity.x),
                                  __float2half_rn(new_velocity.y),
                                  __float2half_rn(new_velocity.z),
                                  0);
    surf3Dwrite(result, advect_dest, x, y, z, cudaBoundaryModeTrap);
}

__global__ void AdvectPureKernel(float time_step, float dissipation,
                                 int slice_stride, int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;
    float4 velocity = tex3D(advect_velocity, coord.x, coord.y, coord.z);
    float3 back_traced =
        coord - time_step * make_float3(velocity.x, velocity.y, velocity.z);

    float result = dissipation * tex3D(advect_source, back_traced.x,
                                       back_traced.y, back_traced.z);
    surf3Dwrite(__float2half_rn(result), advect_dest, x, y, z,
                cudaBoundaryModeTrap);
}

// =============================================================================

void LaunchAdvectVelocityPure(cudaArray_t dest_array,
                              cudaArray_t velocity_array,
                              float time_step, float dissipation,
                              int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
    advect_dest.channelDesc = desc;

    cudaError_t result = cudaBindSurfaceToArray(&advect_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    desc = cudaCreateChannelDescHalf4();
    advect_velocity.normalized = false;
    advect_velocity.filterMode = cudaFilterModeLinear;
    advect_velocity.addressMode[0] = cudaAddressModeClamp;
    advect_velocity.addressMode[1] = cudaAddressModeClamp;
    advect_velocity.addressMode[2] = cudaAddressModeClamp;
    advect_velocity.channelDesc = desc;

    result = cudaBindTextureToArray(&advect_velocity, velocity_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    int slice_stride = volume_size.x * volume_size.y;

    AdvectVelocityPureKernel<<<grid, block>>>(time_step, dissipation,
                                              slice_stride, volume_size);

    cudaUnbindTexture(&advect_velocity);
}

void LaunchAdvectPure(cudaArray_t dest_array, cudaArray_t velocity_array,
                      cudaArray_t source_array, float time_step,
                      float dissipation, int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
    advect_dest.channelDesc = desc;

    cudaError_t result = cudaBindSurfaceToArray(&advect_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    desc = cudaCreateChannelDescHalf4();
    advect_velocity.normalized = false;
    advect_velocity.filterMode = cudaFilterModeLinear;
    advect_velocity.addressMode[0] = cudaAddressModeClamp;
    advect_velocity.addressMode[1] = cudaAddressModeClamp;
    advect_velocity.addressMode[2] = cudaAddressModeClamp;
    advect_velocity.channelDesc = desc;

    result = cudaBindTextureToArray(&advect_velocity, velocity_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    desc = cudaCreateChannelDescHalf();
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

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    int slice_stride = volume_size.x * volume_size.y;

    AdvectPureKernel<<<grid, block>>>(time_step, dissipation, slice_stride,
                                      volume_size);

    cudaUnbindTexture(&advect_source);
    cudaUnbindTexture(&advect_velocity);
}
