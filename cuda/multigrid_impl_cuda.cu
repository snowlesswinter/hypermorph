#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

surface<void, cudaTextureType3D> advect_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_velocity;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_source;
texture<float4, cudaTextureType3D, cudaReadModeElementType> prolongate_coarse;
texture<float4, cudaTextureType3D, cudaReadModeElementType> prolongate_fine;
surface<void, cudaTextureType3D> prolongate_pure_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> prolongate_pure_coarse;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> prolongate_pure_fine;
surface<void, cudaTextureType3D> guess_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> guess_source;

__global__ void AdvectKernel(float time_step, float dissipation,
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
    surf3Dwrite(__float2half_rn(result), advect_dest, x * sizeof(ushort), y, z,
                cudaBoundaryModeTrap);
}

__global__ void ProlongatePackedKernel(float4* out_data,
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

__global__ void ProlongatePackedPureKernel(int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 c = make_float3(x, y, z);
    c *= 0.5f;

    int odd_x = x - ((x >> 1) << 1);
    int odd_y = y - ((y >> 1) << 1);
    int odd_z = z - ((z >> 1) << 1);

    float t_x = -1.0f * (1 - odd_x) * 0.08333333f;
    float t_y = -1.0f * (1 - odd_y) * 0.08333333f;
    float t_z = -1.0f * (1 - odd_z) * 0.08333333f;

    float3 t_c = make_float3(c.x + t_x, c.y + t_y, c.z + t_z);
    float4 result_float = tex3D(prolongate_pure_coarse, t_c.x, t_c.y, t_c.z);

    float3 f_coord = make_float3(float(x) + 0.5f, float(y) + 0.5f,
                                 float(z) + 0.5f);

    float4 original = tex3D(prolongate_pure_fine, f_coord.x, f_coord.y, f_coord.z);
    float4 result = make_float4(original.x + result_float.x, original.y, 0.0f,
                                0.0f);

    ushort4 raw = make_ushort4(__float2half_rn(result.x),
                               __float2half_rn(result.y),
                               __float2half_rn(result.z),
                               0);
    surf3Dwrite(raw, prolongate_pure_dest, x * sizeof(ushort), y,
                z, cudaBoundaryModeTrap);
}

__global__ void RelaxWithZeroGuessPackedPureKernel(
    float alpha_omega_over_beta, float one_minus_omega, float minus_h_square,
    float omega_times_inverse_beta, int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;

    float near =    tex3D(guess_source, coord.x, coord.y, coord.z - 1.0f).x;
    float south =   tex3D(guess_source, coord.x, coord.y - 1.0f, coord.z).x;
    float west =    tex3D(guess_source, coord.x - 1.0f, coord.y, coord.z).x;
    float4 center = tex3D(guess_source, coord.x, coord.y, coord.z);
    float east =    tex3D(guess_source, coord.x + 1.0f, coord.y, coord.z).x;
    float north =   tex3D(guess_source, coord.x, coord.y + 1.0f, coord.z).x;
    float far =     tex3D(guess_source, coord.x, coord.y, coord.z + 1.0f).x;
    float b_center = center.y;

    if (coord.y == volume_size.y - 1)
        north = b_center;

    if (coord.y == 0)
        south = b_center;

    if (coord.x == volume_size.x - 1)
        east = b_center;

    if (coord.x == 0)
        west = b_center;

    if (coord.z == volume_size.z - 1)
        far = b_center;

    if (coord.z == 0)
        near = b_center;

    float v = one_minus_omega * (alpha_omega_over_beta * b_center) +
        (alpha_omega_over_beta * (north + south + east + west + far + near) +
        minus_h_square * b_center) * omega_times_inverse_beta;

    ushort4 raw = make_ushort4(__float2half_rn(v),
                               __float2half_rn(b_center),
                               0, 0);
    surf3Dwrite(raw, guess_dest, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap); 
}

// =============================================================================

void LaunchAdvect(cudaArray_t dest_array, cudaArray_t velocity_array,
                      cudaArray_t source_array, float time_step,
                      float dissipation, int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf();
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

    AdvectKernel<<<grid, block>>>(time_step, dissipation, slice_stride,
                                      volume_size);

    cudaUnbindTexture(&advect_source);
    cudaUnbindTexture(&advect_velocity);
}

void LaunchProlongatePacked(float4* dest_array, cudaArray* coarse_array,
                            cudaArray* fine_array, int3 volume_size_fine)
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

    int3 volume_size = volume_size_fine;
    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    int num_of_blocks_per_slice = volume_size.y / 8;
    int slice_stride = volume_size.x * volume_size.y;

    ProlongatePackedKernel << <grid, block >> >(dest_array, num_of_blocks_per_slice,
                                                slice_stride, volume_size);

    cudaUnbindTexture(&prolongate_fine);
    cudaUnbindTexture(&prolongate_coarse);
}

void LaunchProlongatePackedPure(cudaArray* dest_array, cudaArray* coarse_array,
                                cudaArray* fine_array, int3 volume_size_fine)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
    prolongate_pure_dest.channelDesc = desc;

    cudaError_t result = cudaBindSurfaceToArray(&prolongate_pure_dest,
                                                dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    desc = cudaCreateChannelDescHalf4();
    prolongate_pure_coarse.normalized = false;
    prolongate_pure_coarse.filterMode = cudaFilterModeLinear;
    prolongate_pure_coarse.addressMode[0] = cudaAddressModeClamp;
    prolongate_pure_coarse.addressMode[1] = cudaAddressModeClamp;
    prolongate_pure_coarse.addressMode[2] = cudaAddressModeClamp;
    prolongate_pure_coarse.channelDesc = desc;

    result = cudaBindTextureToArray(&prolongate_pure_coarse, coarse_array,
                                    &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    desc = cudaCreateChannelDescHalf4();
    prolongate_pure_fine.normalized = false;
    prolongate_pure_fine.filterMode = cudaFilterModeLinear;
    prolongate_pure_fine.addressMode[0] = cudaAddressModeClamp;
    prolongate_pure_fine.addressMode[1] = cudaAddressModeClamp;
    prolongate_pure_fine.addressMode[2] = cudaAddressModeClamp;
    prolongate_pure_fine.channelDesc = desc;

    result = cudaBindTextureToArray(&prolongate_pure_fine, fine_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size_fine.x / 8);
    dim3 grid(volume_size_fine.x / block.x, volume_size_fine.y / block.y,
              volume_size_fine.z / block.z);

    ProlongatePackedPureKernel<<<grid, block>>>(volume_size_fine);

    cudaUnbindTexture(&prolongate_pure_fine);
    cudaUnbindTexture(&prolongate_pure_coarse);
}

void LaunchRelaxWithZeroGuessPackedPure(cudaArray* dest_array,
                                        cudaArray* source_array,
                                        float alpha_omega_over_beta,
                                        float one_minus_omega,
                                        float minus_h_square,
                                        float omega_times_inverse_beta,
                                        int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
    guess_dest.channelDesc = desc;

    cudaError_t result = cudaBindSurfaceToArray(&guess_dest, dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    desc = cudaCreateChannelDescHalf4();
    guess_source.normalized = false;
    guess_source.filterMode = cudaFilterModeLinear;
    guess_source.addressMode[0] = cudaAddressModeClamp;
    guess_source.addressMode[1] = cudaAddressModeClamp;
    guess_source.addressMode[2] = cudaAddressModeClamp;
    guess_source.channelDesc = desc;

    result = cudaBindTextureToArray(&guess_source, source_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    RelaxWithZeroGuessPackedPureKernel<<<grid, block>>>(
        alpha_omega_over_beta, one_minus_omega, minus_h_square,
        omega_times_inverse_beta, volume_size);

    cudaUnbindTexture(&guess_source);
}