#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

surface<void, cudaTextureType3D> advect_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_velocity;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_source;
surface<void, cudaTextureType3D> buoyancy_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> buoyancy_velocity;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> buoyancy_temperature;
surface<void, cudaTextureType3D> impulse_dest;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> impulse_original;
surface<void, cudaTextureType3D> divergence_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> divergence_velocity;
surface<void, cudaTextureType3D> gradient_velocity;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> gradient_packed;
surface<void, cudaTextureType3D> jacobi;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> jacobi_packed;
surface<void, cudaTextureType3D> diagnosis;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> diagnosis_source;

__global__ void AdvectPureKernel(float time_step, float dissipation,
                                 int3 volume_size)
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

__global__ void AdvectVelocityPureKernel(float time_step, float dissipation,
                                         int3 volume_size)
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
    surf3Dwrite(result, advect_dest, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

__global__ void ApplyBuoyancyPureKernel(float time_step,
                                        float ambient_temperature,
                                        float accel_factor, float gravity,
                                        int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;
    float4 velocity = tex3D(buoyancy_velocity, coord.x, coord.y, coord.z);
    float t = tex3D(buoyancy_temperature, coord.x, coord.y, coord.z);

    ushort4 result = make_ushort4(__float2half_rn(velocity.x),
                                  __float2half_rn(velocity.y),
                                  __float2half_rn(velocity.z),
                                  0);
    if (t > ambient_temperature) {
        float accel = time_step * ((t - ambient_temperature) * accel_factor -
                                   gravity);
        result.y = __float2half_rn(velocity.y + accel);
    }
    surf3Dwrite(result, buoyancy_dest, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

__global__ void ApplyImpulsePureKernel(float3 center_point, float3 hotspot,
                                       float radius, float value,
                                       int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;
    float original = tex3D(impulse_original, coord.x, coord.y, coord.z);

    if (coord.x > 1.0f && coord.y < 3.0f) {
        float2 diff = make_float2(coord.x, coord.z) -
            make_float2(center_point.x, center_point.z);
        float d = hypotf(diff.x, diff.y);
        if (d < radius) {
            diff = make_float2(coord.x, coord.z) -
                make_float2(hotspot.x, hotspot.z);
            float scale = (radius - hypotf(diff.x, diff.y)) / radius;
            scale = max(scale, 0.5f);
            surf3Dwrite(__float2half_rn(scale * value), impulse_dest,
                        x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
            return;
        }
    }

    surf3Dwrite(__float2half_rn(original), impulse_dest, x * sizeof(ushort), y,
                z, cudaBoundaryModeTrap);
}

__global__ void ComputeDivergencePureKernel(float half_inverse_cell_size,
                                            int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;

    float4 near =   tex3D(divergence_velocity, coord.x, coord.y, coord.z - 1.0f);
    float4 south =  tex3D(divergence_velocity, coord.x, coord.y - 1.0f, coord.z);
    float4 west =   tex3D(divergence_velocity, coord.x - 1.0f, coord.y, coord.z);
    float4 center = tex3D(divergence_velocity, coord.x, coord.y, coord.z);
    float4 east =   tex3D(divergence_velocity, coord.x + 1.0f, coord.y, coord.z);
    float4 north =  tex3D(divergence_velocity, coord.x, coord.y + 1.0f, coord.z);
    float4 far =    tex3D(divergence_velocity, coord.x, coord.y, coord.z + 1.0f);

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

    float div = half_inverse_cell_size * (diff_ew + diff_ns + diff_fn);
    ushort4 result = make_ushort4(0, __float2half_rn(div), 0, 0);
    surf3Dwrite(result, divergence_dest, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

__global__ void ComputeResidualPackedDiagnosisKernel(float inverse_h_square,
                                                     int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;

    float near =    tex3D(diagnosis_source, coord.x, coord.y, coord.z - 1.0f).x;
    float south =   tex3D(diagnosis_source, coord.x, coord.y - 1.0f, coord.z).x;
    float west =    tex3D(diagnosis_source, coord.x - 1.0f, coord.y, coord.z).x;
    float4 center = tex3D(diagnosis_source, coord.x, coord.y, coord.z);
    float east =    tex3D(diagnosis_source, coord.x + 1.0f, coord.y, coord.z).x;
    float north =   tex3D(diagnosis_source, coord.x, coord.y + 1.0f, coord.z).x;
    float far =     tex3D(diagnosis_source, coord.x, coord.y, coord.z + 1.0f).x;
    float b_center = center.y;

    if (coord.y == volume_size.y - 1)
        north = center.x;

    if (coord.y == 0)
        south = center.x;

    if (coord.x == volume_size.x - 1)
        east = center.x;

    if (coord.x == 0)
        west = center.x;

    if (coord.z == volume_size.z - 1)
        far = center.x;

    if (coord.z == 0)
        near = center.x;

    float v = b_center -
        (north + south + east + west + far + near - 6.0 * center.x) *
        inverse_h_square;
    surf3Dwrite(fabsf(v), diagnosis, x * sizeof(float), y, z,
                cudaBoundaryModeTrap);
}

__global__ void DampedJacobiPureKernel(float one_minus_omega,
                                       float minus_square_cell_size,
                                       float omega_over_beta, int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;

    float near =           tex3D(jacobi_packed, coord.x, coord.y, coord.z - 1.0f).x;
    float south =          tex3D(jacobi_packed, coord.x, coord.y - 1.0f, coord.z).x;
    float west =           tex3D(jacobi_packed, coord.x - 1.0f, coord.y, coord.z).x;
    float4 packed_center = tex3D(jacobi_packed, coord.x, coord.y, coord.z);
    float east =           tex3D(jacobi_packed, coord.x + 1.0f, coord.y, coord.z).x;
    float north =          tex3D(jacobi_packed, coord.x, coord.y + 1.0f, coord.z).x;
    float far =            tex3D(jacobi_packed, coord.x, coord.y, coord.z + 1.0f).x;

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
    ushort4 raw = make_ushort4(__float2half_rn(u),
                               __float2half_rn(b_center), 0, 0);
    surf3Dwrite(raw, jacobi, x * sizeof(ushort4), y, z, cudaBoundaryModeTrap);
}

__global__ void SubstractGradientPureKernel(float gradient_scale,
                                            int3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

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
    float3 mask = make_float3(1.0f, 1.0f, 1.0f);
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

    ushort4 raw;
    surf3Dread(&raw, gradient_velocity, x * sizeof(ushort4), y, z);
    float3 old_v = make_float3(__half2float(raw.x), __half2float(raw.y),
                               __half2float(raw.z));
    float3 grad = make_float3(diff_ew, diff_ns, diff_fn) * gradient_scale;
    float3 new_v = old_v - grad;
    float3 result = mask * new_v; // Velocity goes to 0 when hit ???
    raw = make_ushort4(__float2half_rn(result.x), __float2half_rn(result.y),
                       __float2half_rn(result.z), 0);
    surf3Dwrite(raw, gradient_velocity, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

// =============================================================================

void LaunchAdvectPure(cudaArray_t dest_array, cudaArray_t velocity_array,
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
    AdvectPureKernel<<<grid, block>>>(time_step, dissipation, volume_size);

    cudaUnbindTexture(&advect_source);
    cudaUnbindTexture(&advect_velocity);
}

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
    AdvectVelocityPureKernel<<<grid, block>>>(time_step, dissipation,
                                              volume_size);

    cudaUnbindTexture(&advect_velocity);
}

void LaunchApplyBuoyancyPure(cudaArray* dest_array, cudaArray* velocity_array,
                             cudaArray* temperature_array, float time_step,
                             float ambient_temperature, float accel_factor,
                             float gravity, int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
    buoyancy_dest.channelDesc = desc;

    cudaError_t result = cudaBindSurfaceToArray(&buoyancy_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    desc = cudaCreateChannelDescHalf4();
    buoyancy_velocity.normalized = false;
    buoyancy_velocity.filterMode = cudaFilterModeLinear;
    buoyancy_velocity.addressMode[0] = cudaAddressModeClamp;
    buoyancy_velocity.addressMode[1] = cudaAddressModeClamp;
    buoyancy_velocity.addressMode[2] = cudaAddressModeClamp;
    buoyancy_velocity.channelDesc = desc;

    result = cudaBindTextureToArray(&buoyancy_velocity, velocity_array, &desc);
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
    ApplyBuoyancyPureKernel<<<grid, block>>>(time_step, ambient_temperature,
                                             accel_factor, gravity,
                                             volume_size);

    cudaUnbindTexture(&buoyancy_temperature);
    cudaUnbindTexture(&buoyancy_velocity);
}

void LaunchApplyImpulsePure(cudaArray* dest_array, cudaArray* original_array,
                            float3 center_point, float3 hotspot, float radius,
                            float value, int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf();
    impulse_dest.channelDesc = desc;

    cudaError_t result = cudaBindSurfaceToArray(&impulse_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    desc = cudaCreateChannelDescHalf();
    impulse_original.normalized = false;
    impulse_original.filterMode = cudaFilterModeLinear;
    impulse_original.addressMode[0] = cudaAddressModeClamp;
    impulse_original.addressMode[1] = cudaAddressModeClamp;
    impulse_original.addressMode[2] = cudaAddressModeClamp;
    impulse_original.channelDesc = desc;

    result = cudaBindTextureToArray(&impulse_original, original_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    ApplyImpulsePureKernel<<<grid, block>>>(center_point, hotspot, radius,
                                            value, volume_size);

    cudaUnbindTexture(&impulse_original);
}

void LaunchComputeDivergencePure(cudaArray* dest_array,
                                 cudaArray* velocity_array,
                                 float half_inverse_cell_size, int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
    divergence_dest.channelDesc = desc;

    cudaError_t result = cudaBindSurfaceToArray(&divergence_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    desc = cudaCreateChannelDescHalf4();
    divergence_velocity.normalized = false;
    divergence_velocity.filterMode = cudaFilterModeLinear;
    divergence_velocity.addressMode[0] = cudaAddressModeClamp;
    divergence_velocity.addressMode[1] = cudaAddressModeClamp;
    divergence_velocity.addressMode[2] = cudaAddressModeClamp;
    divergence_velocity.channelDesc = desc;

    result = cudaBindTextureToArray(&divergence_velocity, velocity_array,
                                    &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    ComputeDivergencePureKernel<<<grid, block>>>(half_inverse_cell_size,
                                                 volume_size);

    cudaUnbindTexture(&divergence_velocity);
}

void LaunchComputeResidualPackedDiagnosis(cudaArray* dest_array,
                                          cudaArray* source_array,
                                          float inverse_h_square,
                                          int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    diagnosis.channelDesc = desc;

    cudaError_t result = cudaBindSurfaceToArray(&diagnosis, dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    desc = cudaCreateChannelDescHalf4();
    diagnosis_source.normalized = false;
    diagnosis_source.filterMode = cudaFilterModeLinear;
    diagnosis_source.addressMode[0] = cudaAddressModeClamp;
    diagnosis_source.addressMode[1] = cudaAddressModeClamp;
    diagnosis_source.addressMode[2] = cudaAddressModeClamp;
    diagnosis_source.channelDesc = desc;

    result = cudaBindTextureToArray(&diagnosis_source, source_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    ComputeResidualPackedDiagnosisKernel<<<grid, block>>>(inverse_h_square,
                                                          volume_size);

    cudaUnbindTexture(&diagnosis_source);
}

void LaunchDampedJacobiPure(cudaArray* dest_array, cudaArray* packed_array,
                            float one_minus_omega, float minus_square_cell_size,
                            float omega_over_beta, int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
    jacobi.channelDesc = desc;

    cudaError_t result = cudaBindSurfaceToArray(&jacobi, dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    desc = cudaCreateChannelDescHalf4();
    jacobi_packed.normalized = false;
    jacobi_packed.filterMode = cudaFilterModeLinear;
    jacobi_packed.addressMode[0] = cudaAddressModeClamp;
    jacobi_packed.addressMode[1] = cudaAddressModeClamp;
    jacobi_packed.addressMode[2] = cudaAddressModeClamp;
    jacobi_packed.channelDesc = desc;

    result = cudaBindTextureToArray(&jacobi_packed, packed_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, volume_size.x / 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    DampedJacobiPureKernel<<<grid, block>>>(one_minus_omega,
                                            minus_square_cell_size,
                                            omega_over_beta, volume_size);

    cudaUnbindTexture(&jacobi_packed);
}

void LaunchSubstractGradientPure(cudaArray* dest_array, cudaArray* packed_array,
                                 float gradient_scale, int3 volume_size)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDescHalf4();
    gradient_velocity.channelDesc = desc;

    cudaError_t result = cudaBindSurfaceToArray(&gradient_velocity, dest_array,
                                                &desc);
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
    SubstractGradientPureKernel<<<grid, block>>>(gradient_scale, volume_size);

    cudaUnbindTexture(&gradient_packed);
}
