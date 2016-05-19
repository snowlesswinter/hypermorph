#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "advection_method.h"
#include "block_arrangement.h"

surface<void, cudaSurfaceType3D> advect_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_velocity;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_intermediate;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_intermediate1;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_source;

__global__ void AdvectMacCormackKernel(float time_step, float dissipation,
                                       bool quadratic_dissipation)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 velocity = make_float3(tex3D(advect_velocity, coord.x, coord.y, coord.z));
    float3 back_traced = coord - time_step * velocity;
    float ¦Õ = tex3D(advect_source, coord.x, coord.y, coord.z);

    float ¦Õ0 = tex3D(advect_source, back_traced.x - 0.5f, back_traced.y - 0.5f, back_traced.z - 0.5f);
    float ¦Õ1 = tex3D(advect_source, back_traced.x - 0.5f, back_traced.y - 0.5f, back_traced.z + 0.5f);
    float ¦Õ2 = tex3D(advect_source, back_traced.x - 0.5f, back_traced.y + 0.5f, back_traced.z - 0.5f);
    float ¦Õ3 = tex3D(advect_source, back_traced.x - 0.5f, back_traced.y + 0.5f, back_traced.z + 0.5f);
    float ¦Õ4 = tex3D(advect_source, back_traced.x + 0.5f, back_traced.y - 0.5f, back_traced.z - 0.5f);
    float ¦Õ5 = tex3D(advect_source, back_traced.x + 0.5f, back_traced.y - 0.5f, back_traced.z + 0.5f);
    float ¦Õ6 = tex3D(advect_source, back_traced.x + 0.5f, back_traced.y + 0.5f, back_traced.z - 0.5f);
    float ¦Õ7 = tex3D(advect_source, back_traced.x + 0.5f, back_traced.y + 0.5f, back_traced.z + 0.5f);

    float ¦Õ_min = fminf(fminf(fminf(fminf(fminf(fminf(fminf(¦Õ0, ¦Õ1), ¦Õ2), ¦Õ3), ¦Õ4), ¦Õ5), ¦Õ6), ¦Õ7);
    float ¦Õ_max = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(¦Õ0, ¦Õ1), ¦Õ2), ¦Õ3), ¦Õ4), ¦Õ5), ¦Õ6), ¦Õ7);

    float ¦Õ_n_plus_1_hat = tex3D(advect_intermediate1, coord.x, coord.y, coord.z);
    float3 forward_traced = coord + time_step * velocity;
    float ¦Õ_n_hat = tex3D(advect_intermediate1, forward_traced.x, forward_traced.y, forward_traced.z);

    float ¦Õ_new = (¦Õ_n_plus_1_hat + 0.5f * (¦Õ - ¦Õ_n_hat));
    float clamped = fmaxf(fminf(¦Õ_new, ¦Õ_max), ¦Õ_min);
    if (clamped != ¦Õ_new) // New extrema found, revert to the first order
                          // accurate semi-Lagrangian method.
        ¦Õ_new = ¦Õ_n_plus_1_hat;

    float result = quadratic_dissipation ?
        (1.0f - dissipation * time_step * (1.0f - ¦Õ_new)) * ¦Õ_new :
        (1.0f - dissipation * time_step) * ¦Õ_new;
    surf3Dwrite(__float2half_rn(result), advect_dest, x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
}

__global__ void AdvectSemiLagrangianKernel(float time_step, float dissipation,
                                           bool quadratic_dissipation)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float4 velocity = tex3D(advect_velocity, coord.x, coord.y, coord.z);
    float3 back_traced = coord - time_step * make_float3(velocity);

    float ¦Õ = tex3D(advect_source, back_traced.x, back_traced.y, back_traced.z);
    float result = quadratic_dissipation ?
        (1.0f - dissipation * time_step * (1.0f - ¦Õ)) * ¦Õ :
        (1.0f - dissipation * time_step) * ¦Õ;
    surf3Dwrite(__float2half_rn(result), advect_dest, x * sizeof(ushort), y, z,
                cudaBoundaryModeTrap);
}

__global__ void AdvectVelocityMacCormackKernel(float time_step,
                                               float dissipation)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 v_n = make_float3(tex3D(advect_velocity, coord.x, coord.y, coord.z));
    float3 back_traced = coord - time_step * v_n;

    float3 v0 = make_float3(tex3D(advect_velocity, back_traced.x - 0.5f, back_traced.y - 0.5f, back_traced.z - 0.5f));
    float3 v1 = make_float3(tex3D(advect_velocity, back_traced.x - 0.5f, back_traced.y - 0.5f, back_traced.z + 0.5f));
    float3 v2 = make_float3(tex3D(advect_velocity, back_traced.x - 0.5f, back_traced.y + 0.5f, back_traced.z - 0.5f));
    float3 v3 = make_float3(tex3D(advect_velocity, back_traced.x - 0.5f, back_traced.y + 0.5f, back_traced.z + 0.5f));
    float3 v4 = make_float3(tex3D(advect_velocity, back_traced.x + 0.5f, back_traced.y - 0.5f, back_traced.z - 0.5f));
    float3 v5 = make_float3(tex3D(advect_velocity, back_traced.x + 0.5f, back_traced.y - 0.5f, back_traced.z + 0.5f));
    float3 v6 = make_float3(tex3D(advect_velocity, back_traced.x + 0.5f, back_traced.y + 0.5f, back_traced.z - 0.5f));
    float3 v7 = make_float3(tex3D(advect_velocity, back_traced.x + 0.5f, back_traced.y + 0.5f, back_traced.z + 0.5f));

    float3 v_min = fminf(fminf(fminf(fminf(fminf(fminf(fminf(v0, v1), v2), v3), v4), v5), v6), v7);
    float3 v_max = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0, v1), v2), v3), v4), v5), v6), v7);

    float3 v_n_plus_1_hat = make_float3(tex3D(advect_intermediate, coord.x, coord.y, coord.z));
    float3 forward_trace = coord + time_step * v_n;
    float3 v_n_hat = make_float3(tex3D(advect_intermediate, forward_trace.x, forward_trace.y, forward_trace.z));

    float3 v_new = (v_n_plus_1_hat + 0.5f * (v_n - v_n_hat));
    float3 clamped = fmaxf(fminf(v_new, v_max), v_min);
    if (clamped.x != v_new.x || clamped.y != v_new.y || clamped.z != v_new.z)
        v_new = v_n_plus_1_hat;

    v_new = (1.0f - dissipation * time_step) * v_new;
    ushort4 result = make_ushort4(__float2half_rn(v_new.x), __float2half_rn(v_new.y), __float2half_rn(v_new.z), 0);
    surf3Dwrite(result, advect_dest, x * sizeof(ushort4), y, z, cudaBoundaryModeTrap);
}

__global__ void AdvectVelocitySemiLagrangianKernel(float time_step,
                                                   float dissipation)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 velocity =
        make_float3(tex3D(advect_velocity, coord.x, coord.y, coord.z));
    float3 back_traced = coord - time_step * velocity;

    float3 new_velocity =
        (1.0f - dissipation * time_step) * 
            make_float3(
                tex3D(advect_velocity, back_traced.x, back_traced.y,
                      back_traced.z));
    ushort4 result = make_ushort4(__float2half_rn(new_velocity.x),
                                  __float2half_rn(new_velocity.y),
                                  __float2half_rn(new_velocity.z),
                                  0);
    surf3Dwrite(result, advect_dest, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
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
    cached_block[index] =
        make_float3(tex3D(advect_velocity, coord.x, coord.y, coord.z));
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
        new_velocity =
            make_float3( 
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

// =============================================================================

void LaunchAdvectMacCormack(cudaArray_t dest_array, cudaArray_t velocity_array,
                            cudaArray_t source_array,
                            cudaArray_t intermediate_array, float time_step,
                            float dissipation, bool quadratic_dissipation,
                            uint3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, intermediate_array);
    cudaError_t result = cudaBindSurfaceToArray(&advect_dest,
                                                intermediate_array, &desc);
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
    AdvectSemiLagrangianKernel<<<grid, block>>>(time_step, 0.0f,
                                                quadratic_dissipation);

    cudaGetChannelDesc(&desc, dest_array);
    result = cudaBindSurfaceToArray(&advect_dest, dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, intermediate_array);
    advect_intermediate1.normalized = false;
    advect_intermediate1.filterMode = cudaFilterModeLinear;
    advect_intermediate1.addressMode[0] = cudaAddressModeClamp;
    advect_intermediate1.addressMode[1] = cudaAddressModeClamp;
    advect_intermediate1.addressMode[2] = cudaAddressModeClamp;
    advect_intermediate1.channelDesc = desc;

    result = cudaBindTextureToArray(&advect_intermediate1, intermediate_array,
                                    &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    AdvectMacCormackKernel<<<grid, block>>>(time_step, dissipation,
                                            quadratic_dissipation);

    cudaUnbindTexture(&advect_intermediate1);
    cudaUnbindTexture(&advect_source);
    cudaUnbindTexture(&advect_velocity);
}

void LaunchAdvect(cudaArray_t dest_array, cudaArray_t velocity_array,
                  cudaArray_t source_array, cudaArray_t intermediate_array,
                  float time_step, float dissipation,
                  bool quadratic_dissipation, uint3 volume_size,
                  AdvectionMethod method)
{
    if (method == MACCORMACK_SEMI_LAGRANGIAN) {
        LaunchAdvectMacCormack(dest_array, velocity_array, source_array,
                               intermediate_array, time_step, dissipation,
                               false, volume_size);
        return;
    }

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
    AdvectSemiLagrangianKernel<<<grid, block>>>(time_step, dissipation,
                                                quadratic_dissipation);

    cudaUnbindTexture(&advect_source);
    cudaUnbindTexture(&advect_velocity);
}

void LaunchAdvectVelocityMacCormack(cudaArray_t dest_array,
                                    cudaArray_t velocity_array,
                                    cudaArray_t intermediate_array,
                                    float time_step, float time_step_prev,
                                    float dissipation, uint3 volume_size)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, intermediate_array);
    cudaError_t result = cudaBindSurfaceToArray(&advect_dest,
                                                intermediate_array, &desc);
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
    AdvectVelocitySemiLagrangianKernel<<<grid, block>>>(time_step, 0.0f);

    cudaGetChannelDesc(&desc, dest_array);
    result = cudaBindSurfaceToArray(&advect_dest, dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, intermediate_array);
    advect_intermediate.normalized = false;
    advect_intermediate.filterMode = cudaFilterModeLinear;
    advect_intermediate.addressMode[0] = cudaAddressModeClamp;
    advect_intermediate.addressMode[1] = cudaAddressModeClamp;
    advect_intermediate.addressMode[2] = cudaAddressModeClamp;
    advect_intermediate.channelDesc = desc;

    result = cudaBindTextureToArray(&advect_intermediate, intermediate_array,
                                    &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    AdvectVelocityMacCormackKernel<<<grid, block>>>(time_step, dissipation);

    cudaUnbindTexture(&advect_intermediate);
    cudaUnbindTexture(&advect_velocity);
}

void LaunchAdvectVelocity(cudaArray_t dest_array, cudaArray_t velocity_array,
                          cudaArray_t intermediate_array, float time_step,
                          float time_step_prev, float dissipation,
                          uint3 volume_size, AdvectionMethod method)
{
    if (method == MACCORMACK_SEMI_LAGRANGIAN) {
        LaunchAdvectVelocityMacCormack(dest_array, velocity_array,
                                       intermediate_array, time_step,
                                       time_step_prev, dissipation,
                                       volume_size);
        return;
    }

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
    AdvectVelocitySemiLagrangianKernel<<<grid, block>>>(time_step,
                                                        dissipation);
    cudaUnbindTexture(&advect_velocity);
}
