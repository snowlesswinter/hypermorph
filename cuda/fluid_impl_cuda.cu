#include "fluid_impl_cuda.h"

#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_velocity;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_source;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> buoyancy_velocity;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> buoyancy_temperature;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> impulse_original;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> divergence_velocity;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> gradient_packed;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> gradient_velocity;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> jacobi;

__global__ void RoundPassedKernel(int* dest_array, int round, int x)
{
    dest_array[0] = x * x - round * round;
}

__global__ void AdvectVelocityKernel(ushort4* out_data, float time_step,
                                     float dissipation, int slice_stride,
                                     int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

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

__global__ void AdvectKernel(ushort* out_data, float time_step,
                             float dissipation, int slice_stride,
                             int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int index = slice_stride * z + volume_size.x * y + x;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;
    float4 velocity = tex3D(advect_velocity, coord.x, coord.y, coord.z);
    float3 back_traced =
        coord - time_step * make_float3(velocity.x, velocity.y, velocity.z);

    float result = dissipation * tex3D(advect_source, back_traced.x,
                                       back_traced.y, back_traced.z);
    out_data[index] = __float2half_rn(result);
}

__global__ void ApplyBuoyancyKernel(ushort4* out_data, float time_step,
                                    float ambient_temperature,
                                    float accel_factor, float gravity,
                                    int slice_stride, int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int index = slice_stride * z + volume_size.x * y + x;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;
    float4 velocity = tex3D(buoyancy_velocity, coord.x, coord.y, coord.z);
    float t = tex3D(buoyancy_temperature, coord.x, coord.y, coord.z);

    out_data[index] = make_ushort4(__float2half_rn(velocity.x),
                                   __float2half_rn(velocity.y),
                                   __float2half_rn(velocity.z),
                                   0);
    if (t > ambient_temperature) {
        float accel = time_step * ((t - ambient_temperature) * accel_factor -
            gravity);
        out_data[index].y = __float2half_rn(velocity.y + accel);
    }
}

__global__ void ApplyImpulseKernel(ushort* out_data, float3 center_point,
                                   float3 hotspot, float radius, float value,
                                   int slice_stride, int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

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
            out_data[index] = __float2half_rn(scale * value);
            return;
        }
    }

    out_data[index] = __float2half_rn(original);
}

__global__ void ComputeDivergenceKernel(ushort4* out_data,
                                        float half_inverse_cell_size,
                                        int slice_stride, int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

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

    float result = half_inverse_cell_size * (diff_ew + diff_ns + diff_fn);
    out_data[index] = make_ushort4(0, __float2half_rn(result), 0, 0);
}

__global__ void SubstractGradientKernel(ushort4* out_data,
                                        float gradient_scale,
                                        int slice_stride, int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int index = slice_stride * z + volume_size.x * y + x;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;

    float4 near = tex3D(gradient_packed, coord.x, coord.y, coord.z - 1.0f);
    float4 south = tex3D(gradient_packed, coord.x, coord.y - 1.0f, coord.z);
    float4 west = tex3D(gradient_packed, coord.x - 1.0f, coord.y, coord.z);
    float4 center = tex3D(gradient_packed, coord.x, coord.y, coord.z);
    float4 east = tex3D(gradient_packed, coord.x + 1.0f, coord.y, coord.z);
    float4 north = tex3D(gradient_packed, coord.x, coord.y + 1.0f, coord.z);
    float4 far = tex3D(gradient_packed, coord.x, coord.y, coord.z + 1.0f);

    float diff_ew = east.x - west.x;
    float diff_ns = north.x - south.x;
    float diff_fn = far.x - near.x;

    // Handle boundary problem
    float4 mask = make_float4(1.0f, 1.0f, 1.0f, 0.0f);
    if (x >= volume_size.x - 1)
        mask.x = 0;

    if (x <= 0)
        mask.x = 0;

    if (y >= volume_size.y - 1)
        mask.y = 0;

    if (y <= 0)
        mask.y = 0;

    if (z >= volume_size.z - 1)
        mask.z = 0;

    if (z <= 0)
        mask.z = 0;

    float4 old_v = tex3D(gradient_velocity, coord.x, coord.y, coord.z);
    float4 grad = make_float4(diff_ew, diff_ns, diff_fn, 0.0f) * gradient_scale;
    float4 new_v = old_v - grad;
    float4 result = mask * new_v; // Velocity goes to 0 when hit ???
    out_data[index] = make_ushort4(__float2half_rn(result.x),
                                   __float2half_rn(result.y),
                                   __float2half_rn(result.z),
                                   0);
}

__global__ void DampedJacobiKernel(ushort4* out_data, float one_minus_omega,
                                   float minus_square_cell_size,
                                   float omega_over_beta,
                                   int slice_stride, int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int index = slice_stride * z + volume_size.x * y + x;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;

    float near =            tex3D(jacobi, coord.x, coord.y, coord.z - 1.0f).x;
    float south =           tex3D(jacobi, coord.x, coord.y - 1.0f, coord.z).x;
    float west =            tex3D(jacobi, coord.x - 1.0f, coord.y, coord.z).x;
    float4 packed_center =  tex3D(jacobi, coord.x, coord.y, coord.z);
    float east =            tex3D(jacobi, coord.x + 1.0f, coord.y, coord.z).x;
    float north =           tex3D(jacobi, coord.x, coord.y + 1.0f, coord.z).x;
    float far =             tex3D(jacobi, coord.x, coord.y, coord.z + 1.0f).x;

    float center = packed_center.x;

    // Handle boundary problem
    if (x >= volume_size.x - 1)
        east = center;

    if (x <= 0)
        west = center;

    if (y >= volume_size.y - 1)
        north = center;

    if (y <= 0)
        south = center;

    if (z >= volume_size.z - 1)
        far = center;

    if (z <= 0)
        near = center;

    float b_center = packed_center.y;
    float u = one_minus_omega * center +
        (west + east + south + north + far + near + minus_square_cell_size *
        b_center) * omega_over_beta;
    out_data[index] = make_ushort4(__float2half_rn(u),
                                   __float2half_rn(b_center), 0, 0);
}

// =============================================================================

void LaunchRoundPassed(int* dest_array, int round, int x)
{
    RoundPassedKernel<<<1, 1>>>(dest_array, round, x);
}

void LaunchAdvectVelocity(ushort4* dest_array, cudaArray* velocity_array,
                          float time_step, float dissipation, int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
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
    int slice_stride = volume_size.x * volume_size.y;

    AdvectVelocityKernel<<<grid, block>>>(dest_array, time_step, dissipation,
                                          slice_stride, volume_size);

    cudaUnbindTexture(&advect_velocity);
}

void LaunchAdvect(ushort* dest_array, cudaArray* velocity_array,
                  cudaArray* source_array, float time_step,
                  float dissipation, int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
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

    AdvectKernel<<<grid, block>>>(dest_array, time_step, dissipation,
                                  slice_stride, volume_size);

    cudaUnbindTexture(&advect_source);
    cudaUnbindTexture(&advect_velocity);
}

void LaunchApplyBuoyancy(ushort4* dest_array, cudaArray* velocity_array,
                         cudaArray* temperature_array, float time_step,
                         float ambient_temperature, float accel_factor,
                         float gravity, int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
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

    desc = cudaCreateChannelDescHalf();
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
    int slice_stride = volume_size.x * volume_size.y;

    ApplyBuoyancyKernel<<<grid, block>>>(dest_array, time_step,
                                         ambient_temperature, accel_factor,
                                         gravity, slice_stride, volume_size);

    cudaUnbindTexture(&buoyancy_temperature);
    cudaUnbindTexture(&buoyancy_velocity);
}

void LaunchApplyImpulse(ushort* dest_array, cudaArray* original_array,
                        float3 center_point, float3 hotspot, float radius,
                        float value, int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf();
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
    int slice_stride = volume_size.x * volume_size.y;

    ApplyImpulseKernel<<<grid, block>>>(dest_array, center_point, hotspot,
                                        radius, value, slice_stride,
                                        volume_size);

    cudaUnbindTexture(&impulse_original);
}

void LaunchComputeDivergence(ushort4* dest_array, cudaArray* velocity_array,
                             float half_inverse_cell_size, int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
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
    int slice_stride = volume_size.x * volume_size.y;

    ComputeDivergenceKernel<<<grid, block>>>(dest_array, half_inverse_cell_size,
                                             slice_stride, volume_size);

    cudaUnbindTexture(&divergence_velocity);
}

void LaunchSubstractGradient(ushort4* dest_array, cudaArray* velocity_array,
                             cudaArray* packed_array, float gradient_scale,
                             int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
    gradient_velocity.normalized = false;
    gradient_velocity.filterMode = cudaFilterModeLinear;
    gradient_velocity.addressMode[0] = cudaAddressModeClamp;
    gradient_velocity.addressMode[1] = cudaAddressModeClamp;
    gradient_velocity.addressMode[2] = cudaAddressModeClamp;
    gradient_velocity.channelDesc = desc;

    cudaError_t result = cudaBindTextureToArray(&gradient_velocity,
                                                velocity_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    desc = cudaCreateChannelDescHalf4();
    gradient_packed.normalized = false;
    gradient_packed.filterMode = cudaFilterModeLinear;
    gradient_packed.addressMode[0] = cudaAddressModeClamp;
    gradient_packed.addressMode[1] = cudaAddressModeClamp;
    gradient_packed.addressMode[2] = cudaAddressModeClamp;
    gradient_packed.channelDesc = desc;

    result = cudaBindTextureToArray(&gradient_packed, packed_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    int slice_stride = volume_size.x * volume_size.y;

    SubstractGradientKernel<<<grid, block>>>(dest_array, gradient_scale,
                                             slice_stride, volume_size);

    cudaUnbindTexture(&gradient_packed);
    cudaUnbindTexture(&gradient_velocity);
}

void LaunchDampedJacobi(ushort4* dest_array, cudaArray* packed_array,
                        float one_minus_omega, float minus_square_cell_size,
                        float omega_over_beta, int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
    jacobi.normalized = false;
    jacobi.filterMode = cudaFilterModeLinear;
    jacobi.addressMode[0] = cudaAddressModeClamp;
    jacobi.addressMode[1] = cudaAddressModeClamp;
    jacobi.addressMode[2] = cudaAddressModeClamp;
    jacobi.channelDesc = desc;

    cudaError_t result = cudaBindTextureToArray(&jacobi, packed_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    int slice_stride = volume_size.x * volume_size.y;

    DampedJacobiKernel<<<grid, block>>>(dest_array, one_minus_omega,
                                        minus_square_cell_size, omega_over_beta,
                                        slice_stride, volume_size);

    cudaUnbindTexture(&jacobi);
}
