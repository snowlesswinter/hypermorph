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
texture<float4, cudaTextureType3D, cudaReadModeElementType> buoyancy_velocity;
texture<float, cudaTextureType3D, cudaReadModeElementType> buoyancy_temperature;
texture<float, cudaTextureType3D, cudaReadModeElementType> impulse_original;
texture<float4, cudaTextureType3D, cudaReadModeElementType> divergence_velocity;

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

__global__ void AdvectVelocityKernel(float4* out_data, float time_step,
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

    out_data[index] = dissipation * tex3D(advect_velocity, back_traced.x,
                                          back_traced.y, back_traced.z);
}

__global__ void AdvectKernel(float* out_data, float time_step,
                             float dissipation, int num_of_blocks_per_slice,
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

    out_data[index] = dissipation * tex3D(advect_source, back_traced.x,
                                          back_traced.y, back_traced.z);
}

__global__ void ApplyBuoyancyKernel(float4* out_data, float time_step,
                                    float ambient_temperature,
                                    float accel_factor, float gravity,
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
    float4 velocity = tex3D(buoyancy_velocity, coord.x, coord.y, coord.z);
    float t = tex3D(buoyancy_temperature, coord.x, coord.y, coord.z);

    out_data[index] = velocity;
    if (t > ambient_temperature)
        out_data[index] += time_step * ((t - ambient_temperature) *
            accel_factor - gravity) * make_float4(0.0f, 1.0f, 0.0f, 0.0f);
}

__global__ void ApplyImpulseKernel(float* out_data, float3 center_point,
                                   float3 hotspot, float radius, float value,
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
    float original = tex3D(impulse_original, coord.x, coord.y, coord.z);

    if (coord.x > 1.0f && coord.y < 3.0f)
    {
        float2 diff = make_float2(coord.x, coord.z) -
            make_float2(center_point.x, center_point.z);
        float d = hypotf(diff.x, diff.y);
        if (d < radius)
        {
            diff = make_float2(coord.x, coord.z) -
                make_float2(hotspot.x, hotspot.z);
            float scale = (radius - hypotf(diff.x, diff.y)) / radius;
            scale = max(scale, 0.5f);
            out_data[index] = scale * value;
            return;
        }
    }

    out_data[index] = original;
}

__global__ void ComputeDivergenceKernel(float4* out_data,
                                        float half_inverse_cell_size,
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

    float4 near = tex3D(divergence_velocity, coord.x, coord.y, coord.z - 1.0f);
    float4 south = tex3D(divergence_velocity, coord.x, coord.y - 1.0f, coord.z);
    float4 west = tex3D(divergence_velocity, coord.x - 1.0f, coord.y, coord.z);
    float4 center = tex3D(divergence_velocity, coord.x, coord.y, coord.z);
    float4 east = tex3D(divergence_velocity, coord.x + 1.0f, coord.y, coord.z);
    float4 north = tex3D(divergence_velocity, coord.x, coord.y + 1.0f, coord.z);
    float4 far = tex3D(divergence_velocity, coord.x, coord.y, coord.z + 1.0f);

    float diff_ew = east.x - west.x;
    float diff_ns = north.y - south.y;
    float diff_fn = far.z - near.z;

    // Handle boundary problem
    if (x >= volume_size.x - 1)
        diff_ew = -center.x - west.x;

    if (x <= 0)
        diff_ew = east.x + center.x;

    if (y >= volume_size.y - 1)
        diff_ns = -center.y - south.y;

    if (y <= 0)
        diff_ns = north.y + center.y;

    if (z >= volume_size.z - 1)
        diff_fn = -center.z - far.z;

    if (z <= 0)
        diff_fn = near.z + center.z;

    float alpha = 0;
    if (diff_ew != 0 || diff_ns != 0 || diff_fn != 0)
        alpha = 1;

    out_data[index] = make_float4(
        0.0f, half_inverse_cell_size * (diff_ew + diff_ns + diff_fn), 0.0f,
        alpha);// 0.0f);
}

// =============================================================================

void LaunchRoundPassed(int* dest_array, int round, int x)
{
    RoundPassedKernel<<<1, 1>>>(dest_array, round, x);
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

    ProlongatePackedKernel<<<grid, block>>>(dest_array, num_of_blocks_per_slice,
                                            slice_stride, volume_size);

    cudaUnbindTexture(&prolongate_fine);
    cudaUnbindTexture(&prolongate_coarse);
}

void LaunchAdvectVelocity(float4* dest_array, cudaArray* velocity_array,
                          float time_step, float dissipation, int3 volume_size)
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

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    int num_of_blocks_per_slice = volume_size.y / 8;
    int slice_stride = volume_size.x * volume_size.y;

    AdvectVelocityKernel<<<grid, block>>>(dest_array, time_step, dissipation,
                                          num_of_blocks_per_slice, slice_stride,
                                          volume_size);

    cudaUnbindTexture(&advect_velocity);
}

void LaunchAdvect(float* dest_array, cudaArray* velocity_array,
                  cudaArray* source_array, float time_step,
                  float dissipation, int3 volume_size)
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

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    int num_of_blocks_per_slice = volume_size.y / 8;
    int slice_stride = volume_size.x * volume_size.y;

    AdvectKernel<<<grid, block>>>(dest_array, time_step, dissipation,
                                  num_of_blocks_per_slice, slice_stride,
                                  volume_size);

    cudaUnbindTexture(&advect_source);
    cudaUnbindTexture(&advect_velocity);
}

void LaunchApplyBuoyancy(float4* dest_array, cudaArray* velocity_array,
                         cudaArray* temperature_array, float time_step,
                         float ambient_temperature, float accel_factor,
                         float gravity, int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    buoyancy_velocity.normalized = false;
    buoyancy_velocity.filterMode = cudaFilterModeLinear;
    buoyancy_velocity.addressMode[0] = cudaAddressModeClamp;
    buoyancy_velocity.addressMode[1] = cudaAddressModeClamp;
    buoyancy_velocity.addressMode[2] = cudaAddressModeClamp;
    buoyancy_velocity.channelDesc = desc;

    cudaError_t result = cudaBindTextureToArray(&buoyancy_velocity,
                                                velocity_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    desc = cudaCreateChannelDesc<float>();
    buoyancy_temperature.normalized = false;
    buoyancy_temperature.filterMode = cudaFilterModeLinear;
    buoyancy_temperature.addressMode[0] = cudaAddressModeClamp;
    buoyancy_temperature.addressMode[1] = cudaAddressModeClamp;
    buoyancy_temperature.addressMode[2] = cudaAddressModeClamp;
    buoyancy_temperature.channelDesc = desc;

    result = cudaBindTextureToArray(&buoyancy_temperature,
                                    temperature_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    int num_of_blocks_per_slice = volume_size.y / 8;
    int slice_stride = volume_size.x * volume_size.y;

    ApplyBuoyancyKernel<<<grid, block>>>(dest_array, time_step,
                                         ambient_temperature, accel_factor,
                                         gravity, num_of_blocks_per_slice,
                                         slice_stride, volume_size);

    cudaUnbindTexture(&buoyancy_temperature);
    cudaUnbindTexture(&buoyancy_velocity);
}

void LaunchApplyImpulse(float* dest_array, cudaArray* original_array,
                        float3 center_point, float3 hotspot, float radius,
                        float value, int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    impulse_original.normalized = false;
    impulse_original.filterMode = cudaFilterModeLinear;
    impulse_original.addressMode[0] = cudaAddressModeClamp;
    impulse_original.addressMode[1] = cudaAddressModeClamp;
    impulse_original.addressMode[2] = cudaAddressModeClamp;
    impulse_original.channelDesc = desc;

    cudaError_t result = cudaBindTextureToArray(&impulse_original,
                                                original_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    int num_of_blocks_per_slice = volume_size.y / 8;
    int slice_stride = volume_size.x * volume_size.y;

    ApplyImpulseKernel<<<grid, block>>>(dest_array, center_point, hotspot,
                                        radius, value, num_of_blocks_per_slice,
                                        slice_stride, volume_size);

    cudaUnbindTexture(&impulse_original);
}

void LaunchComputeDivergence(float4* dest_array, cudaArray* velocity_array,
                             float half_inverse_cell_size, int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    divergence_velocity.normalized = false;
    divergence_velocity.filterMode = cudaFilterModeLinear;
    divergence_velocity.addressMode[0] = cudaAddressModeClamp;
    divergence_velocity.addressMode[1] = cudaAddressModeClamp;
    divergence_velocity.addressMode[2] = cudaAddressModeClamp;
    divergence_velocity.channelDesc = desc;

    cudaError_t result = cudaBindTextureToArray(&divergence_velocity,
                                                velocity_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    int num_of_blocks_per_slice = volume_size.y / 8;
    int slice_stride = volume_size.x * volume_size.y;

    ComputeDivergenceKernel<<<grid, block>>>(dest_array, half_inverse_cell_size,
                                             num_of_blocks_per_slice,
                                             slice_stride, volume_size);

    cudaUnbindTexture(&divergence_velocity);
}