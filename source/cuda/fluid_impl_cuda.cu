#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "block_arrangement.h"

surface<void, cudaSurfaceType3D> advect_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_velocity;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_source;
surface<void, cudaSurfaceType3D> buoyancy_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> buoyancy_velocity;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> buoyancy_temperature;
surface<void, cudaSurfaceType3D> impulse_dest1;
surface<void, cudaSurfaceType3D> impulse_dest4;
surface<void, cudaSurfaceType3D> divergence_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> divergence_velocity;
surface<void, cudaSurfaceType3D> gradient_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> gradient_velocity;
texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> gradient_packed;
surface<void, cudaSurfaceType3D> diagnosis;
texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> diagnosis_source;

__global__ void AdvectKernel(float time_step, float dissipation)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;
    float4 velocity = tex3D(advect_velocity, coord.x, coord.y, coord.z);
    float3 back_traced =
        coord - time_step * make_float3(velocity.x, velocity.y, velocity.z);

    float result = (1.0f - dissipation * time_step) *
        tex3D(advect_source, back_traced.x, back_traced.y, back_traced.z);
    surf3Dwrite(__float2half_rn(result), advect_dest, x * sizeof(ushort), y, z,
                cudaBoundaryModeTrap);
}

__global__ void AdvectDensityKernel(float time_step, float dissipation)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;
    float4 velocity = tex3D(advect_velocity, coord.x, coord.y, coord.z);
    float3 back_traced =
        coord - time_step * make_float3(velocity.x, velocity.y, velocity.z);

    float density = tex3D(advect_source, back_traced.x, back_traced.y,
                          back_traced.z);

    float ¦Ø = dissipation * time_step * (1.0f - density);
    float result = (1.0f - ¦Ø) * density;
    surf3Dwrite(__float2half_rn(result), advect_dest, x * sizeof(ushort), y, z,
                cudaBoundaryModeTrap);
}

__global__ void AdvectVelocityKernel(float time_step, float dissipation)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;
    float4 velocity = tex3D(advect_velocity, coord.x, coord.y, coord.z);
    float3 back_traced =
        coord - time_step * make_float3(velocity.x, velocity.y, velocity.z);

    float4 new_velocity = (1.0f - dissipation * time_step) *
        tex3D(advect_velocity, back_traced.x,
                                              back_traced.y, back_traced.z);
    ushort4 result = make_ushort4(__float2half_rn(new_velocity.x),
                                  __float2half_rn(new_velocity.y),
                                  __float2half_rn(new_velocity.z),
                                  0);
    surf3Dwrite(result, advect_dest, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

__device__ float3 Float4ToFloat3(float4 v)
{
    return make_float3(v.x, v.y, v.z);
}

__device__ float TrilinearInterpolationSingle(float x0y0z0, float x1y0z0,
                                              float x0y1z0, float x0y0z1,
                                              float x1y1z0, float x0y1z1,
                                              float x1y0z1, float x1y1z1,
                                              float ¦Á, float ¦Â, float ¦Ã)
{
    float xy0z0 = (1 - ¦Á) * x0y0z0 + ¦Á * x1y0z0;
    float xy1z0 = (1 - ¦Á) * x0y1z0 + ¦Á * x1y1z0;
    float xy0z1 = (1 - ¦Á) * x0y0z1 + ¦Á * x1y0z1;
    float xy1z1 = (1 - ¦Á) * x0y1z1 + ¦Á * x1y1z1;

    float yz0 = (1 - ¦Â) * xy0z0 + ¦Â * xy1z0;
    float yz1 = (1 - ¦Â) * xy0z1 + ¦Â * xy1z1;

    return (1 - ¦Ã) * yz0 + ¦Ã * yz1;
}

__device__ float3 TrilinearInterpolation(float3* cache, float3 coord,
                                         int slice_stride, int row_stride)
{
    float int_x = floorf(coord.x);
    float int_y = floorf(coord.y);
    float int_z = floorf(coord.z);

    float ¦Á = fracf(coord.x);
    float ¦Â = fracf(coord.y);
    float ¦Ã = fracf(coord.z);

    int index = int_z * slice_stride + int_y * row_stride + int_x;
    float3 x0y0z0 = cache[index];
    float3 x1y0z0 = cache[index + 1];
    float3 x0y1z0 = cache[index + row_stride];
    float3 x0y0z1 = cache[index + slice_stride];
    float3 x1y1z0 = cache[index + row_stride + 1];
    float3 x0y1z1 = cache[index + slice_stride + row_stride];
    float3 x1y0z1 = cache[index + slice_stride + 1];
    float3 x1y1z1 = cache[index + slice_stride + row_stride + 1];

    float x = TrilinearInterpolationSingle(x0y0z0.x, x1y0z0.x, x0y1z0.x, x0y0z1.x, x1y1z0.x, x0y1z1.x, x1y0z1.x, x1y1z1.x, ¦Á, ¦Â, ¦Ã);
    float y = TrilinearInterpolationSingle(x0y0z0.y, x1y0z0.y, x0y1z0.y, x0y0z1.y, x1y1z0.y, x0y1z1.y, x1y0z1.y, x1y1z1.y, ¦Á, ¦Â, ¦Ã);
    float z = TrilinearInterpolationSingle(x0y0z0.z, x1y0z0.z, x0y1z0.z, x0y0z1.z, x1y1z0.z, x0y1z1.z, x1y0z1.z, x1y1z1.z, ¦Á, ¦Â, ¦Ã);
    return make_float3(x, y, z);
}

// Only ~45% hit rate, serious block effect, deprecated.
__global__ void AdvectVelocityKernel_smem(float time_step, float dissipation)
{
    __shared__ float3 cached_block[600];

    int base_x = blockIdx.x * blockDim.x;
    int base_y = blockIdx.y * blockDim.y;
    int base_z = blockIdx.z * blockDim.z;

    int x = base_x + threadIdx.x;
    int y = base_y + threadIdx.y;
    int z = base_z + threadIdx.z;

    int bw = blockDim.x;
    int bh = blockDim.y;

    float3 coord = make_float3(x, y, z);
    coord += 0.5f;

    int index = threadIdx.z * bw * bh + threadIdx.y * bw + threadIdx.x;
    cached_block[index] = Float4ToFloat3(
        tex3D(advect_velocity, coord.x, coord.y, coord.z));
    float3 velocity = cached_block[index];
    __syncthreads();

    float3 back_traced = coord - time_step * velocity;

    float3 new_velocity;
    if (back_traced.x >= base_x + 0.5f && back_traced.x < base_x + blockDim.x + 0.5f &&
            back_traced.y >= base_y + 0.5f && back_traced.y < base_y + blockDim.y + 0.5f &&
            back_traced.z >= base_z + 0.5f && back_traced.z < base_z + blockDim.z + 0.5f) {

        new_velocity = TrilinearInterpolation(
            cached_block, back_traced - make_float3(base_x + 0.5f, base_y + 0.5f, base_z + 0.5f),
            bw * bh, bw);
    } else {
        new_velocity = Float4ToFloat3(
            tex3D(advect_velocity, back_traced.x, back_traced.y,
                  back_traced.z));
    }
    new_velocity *= 1.0f - dissipation * time_step;
    ushort4 result = make_ushort4(__float2half_rn(new_velocity.x),
                                  __float2half_rn(new_velocity.y),
                                  __float2half_rn(new_velocity.z),
                                  0);
    surf3Dwrite(result, advect_dest, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

__global__ void ApplyBuoyancyKernel(float time_step, float ambient_temperature,
                                    float accel_factor, float gravity)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);

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

__global__ void ApplyImpulse1Kernel(float3 center_point, float3 hotspot,
                                    float radius, float value)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = 1 + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float2 diff = make_float2(coord.x, coord.z) -
        make_float2(center_point.x, center_point.z);
    float d = hypotf(diff.x, diff.y);
    if (d < radius) {
        diff = make_float2(coord.x, coord.z) -
            make_float2(hotspot.x, hotspot.z);
        float scale = (radius - hypotf(diff.x, diff.y)) / radius;
        scale = fmaxf(scale, 0.1f);
        surf3Dwrite(__float2half_rn(scale * value), impulse_dest1,
                    x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
        return;
    }
}

__global__ void ImpulseDensityKernel(float3 center_point, float radius,
                                     float value)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = 1 + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float2 diff = make_float2(coord.x, coord.z) -
        make_float2(center_point.x, center_point.z);
    float d = hypotf(diff.x, diff.y);
    if (d < radius) {
        surf3Dwrite(__float2half_rn(value), impulse_dest1,
                    x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
    }
}

__global__ void ApplyImpulse3Kernel(float3 center_point, float3 hotspot,
                                    float radius, float3 value)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = 1 + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float2 diff = make_float2(coord.x, coord.z) -
        make_float2(center_point.x, center_point.z);
    float d = hypotf(diff.x, diff.y);
    if (d < radius) {
        diff = make_float2(coord.x, coord.z) -
            make_float2(hotspot.x, hotspot.z);
        float scale = (radius - hypotf(diff.x, diff.y)) / radius;
        scale = fmaxf(scale, 0.1f);
        ushort4 result = make_ushort4(__float2half_rn(scale * value.x),
                                      __float2half_rn(scale * value.y),
                                      __float2half_rn(scale * value.z),
                                      0);
        surf3Dwrite(result, impulse_dest4, x * sizeof(ushort4), y, z,
                    cudaBoundaryModeTrap);
        return;
    }
}

__global__ void ComputeDivergenceKernel(float half_inverse_cell_size,
                                        uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);

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
        diff_fn = -center.z - near.z;

    if (z <= 0)
        diff_fn = far.z + center.z;

    float div = half_inverse_cell_size * (diff_ew + diff_ns + diff_fn);
    ushort2 result = make_ushort2(0, __float2half_rn(div));
    surf3Dwrite(result, divergence_dest, x * sizeof(ushort2), y, z,
                cudaBoundaryModeTrap);
}

__global__ void ComputeResidualPackedDiagnosisKernel(float inverse_h_square,
                                                     uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);

    float  near =   tex3D(diagnosis_source, coord.x, coord.y, coord.z - 1.0f).x;
    float  south =  tex3D(diagnosis_source, coord.x, coord.y - 1.0f, coord.z).x;
    float  west =   tex3D(diagnosis_source, coord.x - 1.0f, coord.y, coord.z).x;
    float2 center = tex3D(diagnosis_source, coord.x, coord.y, coord.z);
    float  east =   tex3D(diagnosis_source, coord.x + 1.0f, coord.y, coord.z).x;
    float  north =  tex3D(diagnosis_source, coord.x, coord.y + 1.0f, coord.z).x;
    float  far =    tex3D(diagnosis_source, coord.x, coord.y, coord.z + 1.0f).x;
    float  b_center = center.y;

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

__global__ void RoundPassedKernel(int* dest_array, int round, int x)
{
    dest_array[0] = x * x - round * round;
}

__global__ void SubtractGradientKernel(float gradient_scale, uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z);

    float near =   tex3D(gradient_packed, coord.x, coord.y, coord.z - 1.0f).x;
    float south =  tex3D(gradient_packed, coord.x, coord.y - 1.0f, coord.z).x;
    float west =   tex3D(gradient_packed, coord.x - 1.0f, coord.y, coord.z).x;
    float center = tex3D(gradient_packed, coord.x, coord.y, coord.z).x;
    float east =   tex3D(gradient_packed, coord.x + 1.0f, coord.y, coord.z).x;
    float north =  tex3D(gradient_packed, coord.x, coord.y + 1.0f, coord.z).x;
    float far =    tex3D(gradient_packed, coord.x, coord.y, coord.z + 1.0f).x;

    float diff_ew = east - west;
    float diff_ns = north - south;
    float diff_fn = far - near;

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

    float3 old_v = Float4ToFloat3(
        tex3D(gradient_velocity, coord.x, coord.y, coord.z));
    float3 grad = make_float3(diff_ew, diff_ns, diff_fn) * gradient_scale;
    float3 new_v = old_v - grad;
    float3 result = mask * new_v; // Velocity goes to 0 when hit ???
    ushort4 raw = make_ushort4(__float2half_rn(result.x),
                               __float2half_rn(result.y),
                               __float2half_rn(result.z),
                               0);
    surf3Dwrite(raw, gradient_dest, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

// =============================================================================

void LaunchAdvect(cudaArray_t dest_array, cudaArray_t velocity_array,
                  cudaArray_t source_array, float time_step, float dissipation,
                  uint3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&advect_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, velocity_array);
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

    cudaGetChannelDesc(&desc, source_array);
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

    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    AdvectKernel<<<grid, block>>>(time_step, dissipation);

    cudaUnbindTexture(&advect_source);
    cudaUnbindTexture(&advect_velocity);
}

void LaunchAdvectDensity(cudaArray_t dest_array, cudaArray_t velocity_array,
                         cudaArray_t source_array, float time_step,
                         float dissipation, uint3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&advect_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, velocity_array);
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

    cudaGetChannelDesc(&desc, source_array);
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

    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    AdvectDensityKernel<<<grid, block>>>(time_step, dissipation);

    cudaUnbindTexture(&advect_source);
    cudaUnbindTexture(&advect_velocity);
}

void LaunchAdvectVelocity(cudaArray_t dest_array, cudaArray_t velocity_array,
                          float time_step, float dissipation, uint3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&advect_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, velocity_array);
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

    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    AdvectVelocityKernel<<<grid, block>>>(time_step, dissipation);

    cudaUnbindTexture(&advect_velocity);
}

void LaunchApplyBuoyancy(cudaArray* dest_array, cudaArray* velocity_array,
                         cudaArray* temperature_array, float time_step,
                         float ambient_temperature, float accel_factor,
                         float gravity, uint3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&buoyancy_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, velocity_array);
    buoyancy_velocity.normalized = false;
    buoyancy_velocity.filterMode = cudaFilterModePoint;
    buoyancy_velocity.addressMode[0] = cudaAddressModeClamp;
    buoyancy_velocity.addressMode[1] = cudaAddressModeClamp;
    buoyancy_velocity.addressMode[2] = cudaAddressModeClamp;
    buoyancy_velocity.channelDesc = desc;

    result = cudaBindTextureToArray(&buoyancy_velocity, velocity_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, temperature_array);
    buoyancy_temperature.normalized = false;
    buoyancy_temperature.filterMode = cudaFilterModePoint;
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
    ApplyBuoyancyKernel<<<grid, block>>>(time_step, ambient_temperature,
                                         accel_factor, gravity);

    cudaUnbindTexture(&buoyancy_temperature);
    cudaUnbindTexture(&buoyancy_velocity);
}

void LaunchApplyImpulse(cudaArray* dest_array, cudaArray* original_array,
                        float3 center_point, float3 hotspot, float radius,
                        float3 value, uint32_t mask, uint3 volume_size)
{
    assert(mask == 1 || mask == 7);
    if (mask != 1 && mask != 7)
        return;

    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    dim3 block(128, 2, 1);
    dim3 grid(volume_size.x / block.x, 1, volume_size.z / block.z);
    if (mask == 1) {
        cudaError_t result = cudaBindSurfaceToArray(&impulse_dest1, dest_array,
                                                    &desc);
        assert(result == cudaSuccess);
        if (result != cudaSuccess)
            return;

        ApplyImpulse1Kernel<<<grid, block>>>(center_point, hotspot, radius,
                                             value.x);
    } else if (mask == 7) {
        cudaError_t result = cudaBindSurfaceToArray(&impulse_dest4, dest_array,
                                                    &desc);
        assert(result == cudaSuccess);
        if (result != cudaSuccess)
            return;

        ApplyImpulse3Kernel<<<grid, block>>>(center_point, hotspot, radius,
                                             value);
    }
}

void LaunchComputeDivergence(cudaArray* dest_array, cudaArray* velocity_array,
                             float half_inverse_cell_size, uint3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&divergence_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, velocity_array);
    divergence_velocity.normalized = false;
    divergence_velocity.filterMode = cudaFilterModePoint;
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
    ComputeDivergenceKernel<<<grid, block>>>(half_inverse_cell_size,
                                             volume_size);

    cudaUnbindTexture(&divergence_velocity);
}

void LaunchComputeResidualPackedDiagnosis(cudaArray* dest_array,
                                          cudaArray* source_array,
                                          float inverse_h_square,
                                          uint3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&diagnosis, dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, source_array);
    diagnosis_source.normalized = false;
    diagnosis_source.filterMode = cudaFilterModePoint;
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

void LaunchImpulseDensity(cudaArray* dest_array, cudaArray* original_array,
                          float3 center_point, float radius, float3 value,
                          uint3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    dim3 block(128, 2, 1);
    dim3 grid(volume_size.x / block.x, 1, volume_size.z / block.z);

    cudaError_t result = cudaBindSurfaceToArray(&impulse_dest1, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    ImpulseDensityKernel<<<grid, block>>>(center_point, radius, value.x);
}

void LaunchRoundPassed(int* dest_array, int round, int x)
{
    RoundPassedKernel<<<1, 1>>>(dest_array, round, x);
}

void LaunchSubtractGradient(cudaArray* dest_array, cudaArray* packed_array,
                            float gradient_scale, uint3 volume_size,
                            BlockArrangement* ba)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&gradient_dest, dest_array,
                                                &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, dest_array);
    gradient_velocity.normalized = false;
    gradient_velocity.filterMode = cudaFilterModePoint;
    gradient_velocity.addressMode[0] = cudaAddressModeClamp;
    gradient_velocity.addressMode[1] = cudaAddressModeClamp;
    gradient_velocity.addressMode[2] = cudaAddressModeClamp;
    gradient_velocity.channelDesc = desc;

    // Reading as texture would be more efficient. Hardware half-float
    // conversion?
    result = cudaBindTextureToArray(&gradient_velocity, dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, packed_array);
    gradient_packed.normalized = false;
    gradient_packed.filterMode = cudaFilterModePoint;
    gradient_packed.addressMode[0] = cudaAddressModeClamp;
    gradient_packed.addressMode[1] = cudaAddressModeClamp;
    gradient_packed.addressMode[2] = cudaAddressModeClamp;
    gradient_packed.channelDesc = desc;

    result = cudaBindTextureToArray(&gradient_packed, packed_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    SubtractGradientKernel<<<grid, block>>>(gradient_scale, volume_size);

    cudaUnbindTexture(&gradient_packed);
    cudaUnbindTexture(&gradient_velocity);
}
