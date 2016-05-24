#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "advection_method.h"
#include "block_arrangement.h"
#include "cuda_common.h"

surface<void, cudaSurfaceType3D> advect_dest;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_velocity;
texture<ushort4, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_intermediate;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_intermediate1;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> advect_source;

template <typename TextureType>
__device__ __inline float3 GetVelocityXStaggered(TextureType tex, float3 pos)
{
    float v_x = tex3D(tex, pos.x, pos.y, pos.z).x;
    float v_y = tex3D(tex, pos.x - 0.5f, pos.y + 0.5f, pos.z).y;
    float v_z = tex3D(tex, pos.x - 0.5f, pos.y, pos.z + 0.5f).z;
    return make_float3(v_x, v_y, v_z);
}

template <typename TextureType>
__device__ __inline float3 GetVelocityYStaggered(TextureType tex, float3 pos)
{
    float v_x = tex3D(tex, pos.x + 0.5f, pos.y - 0.5f, pos.z).x;
    float v_y = tex3D(tex, pos.x, pos.y, pos.z).y;
    float v_z = tex3D(tex, pos.x, pos.y - 0.5f, pos.z + 0.5f).z;
    return make_float3(v_x, v_y, v_z);
}

template <typename TextureType>
__device__ __inline float3 GetVelocityZStaggered(TextureType tex, float3 pos)
{
    float v_x = tex3D(tex, pos.x + 0.5f, pos.y, pos.z - 0.5f).x;
    float v_y = tex3D(tex, pos.x, pos.y + 0.5f, pos.z - 0.5f).y;
    float v_z = tex3D(tex, pos.x, pos.y, pos.z).z;
    return make_float3(v_x, v_y, v_z);
}

template <typename TextureType>
__device__ float3 BackTraceVelocityStaggered(TextureType vel_field,
                                             TextureType tex, float3 pos,
                                             float time_step, float3* vel)
{
    float3 v0 = GetVelocityXStaggered(vel_field, pos);
    vel->x = v0.x;
    float3 back_traced0 = pos - v0 * time_step;
    float v_x = tex3D(tex, back_traced0.x, back_traced0.y, back_traced0.z).x;

    float3 v1 = GetVelocityYStaggered(vel_field, pos);
    vel->y = v1.y;
    float3 back_traced1 = pos - v1 * time_step;
    float v_y = tex3D(tex, back_traced1.x, back_traced1.y, back_traced1.z).y;

    float3 v2 = GetVelocityZStaggered(vel_field, pos);
    vel->z = v2.z;
    float3 back_traced2 = pos - v2 * time_step;
    float v_z = tex3D(tex, back_traced2.x, back_traced2.y, back_traced2.z).z;

    return make_float3(v_x, v_y, v_z);
}

template <int n> __device__ __inline float GetComponent(float4 v);
template <> __device__ __inline float GetComponent<0>(float4 v) { return v.x; }
template <> __device__ __inline float GetComponent<1>(float4 v) { return v.y; }
template <> __device__ __inline float GetComponent<2>(float4 v) { return v.z; }

template <typename TextureType, int component>
__device__ void GetVelocityLimitStaggered(TextureType tex, float3 back_traced,
                                          float* min, float* max)
{
    float v0 = GetComponent<component>(tex3D(tex, back_traced.x - 0.5f, back_traced.y - 0.5f, back_traced.z - 0.5f));
    float v1 = GetComponent<component>(tex3D(tex, back_traced.x - 0.5f, back_traced.y - 0.5f, back_traced.z + 0.5f));
    float v2 = GetComponent<component>(tex3D(tex, back_traced.x - 0.5f, back_traced.y + 0.5f, back_traced.z - 0.5f));
    float v3 = GetComponent<component>(tex3D(tex, back_traced.x - 0.5f, back_traced.y + 0.5f, back_traced.z + 0.5f));
    float v4 = GetComponent<component>(tex3D(tex, back_traced.x + 0.5f, back_traced.y - 0.5f, back_traced.z - 0.5f));
    float v5 = GetComponent<component>(tex3D(tex, back_traced.x + 0.5f, back_traced.y - 0.5f, back_traced.z + 0.5f));
    float v6 = GetComponent<component>(tex3D(tex, back_traced.x + 0.5f, back_traced.y + 0.5f, back_traced.z - 0.5f));
    float v7 = GetComponent<component>(tex3D(tex, back_traced.x + 0.5f, back_traced.y + 0.5f, back_traced.z + 0.5f));

    *min = fminf(fminf(fminf(fminf(fminf(fminf(fminf(v0, v1), v2), v3), v4), v5), v6), v7);
    *max = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0, v1), v2), v3), v4), v5), v6), v7);
}

template <typename TextureType>
__device__ void GetVelocityLimitsStaggered(TextureType tex,
                                           float3 back_traced_x,
                                           float3 back_traced_y,
                                           float3 back_traced_z, float3* min,
                                           float3* max)
{
    GetVelocityLimitStaggered<TextureType, 0>(tex, back_traced_x, &min->x, &max->x);
    GetVelocityLimitStaggered<TextureType, 1>(tex, back_traced_y, &min->y, &max->y);
    GetVelocityLimitStaggered<TextureType, 2>(tex, back_traced_z, &min->z, &max->z);
}

template <typename TextureType>
__device__ float3 GetCenterVelocity(TextureType tex, float3 pos,
                                    float3 volume_size)
{
    float3 v0 = make_float3(tex3D(tex, pos.x, pos.y, pos.z));
    float3 v1;
    v1.x = pos.x > volume_size.x - 0.5001f ? 0.0f : tex3D(tex, pos.x + 1.0f, pos.y, pos.z).x;
    v1.y = pos.y > volume_size.y - 0.5001f ? 0.0f : tex3D(tex, pos.x, pos.y + 1.0f, pos.z).y;
    v1.z = pos.z > volume_size.z - 0.5001f ? 0.0f : tex3D(tex, pos.x, pos.y, pos.z + 1.0f).z;

    return 0.5f * (v0 + v1);
}

// =============================================================================

__global__ void AdvectScalarBfeccStaggeredKernel(float time_step,
                                                 float dissipation,
                                                 bool quadratic_dissipation,
                                                 float3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 velocity = GetCenterVelocity(advect_velocity, coord, volume_size);
    float3 back_traced = coord - time_step * velocity;

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

    float ¦Õ_new = tex3D(advect_intermediate1, back_traced.x, back_traced.y, back_traced.z);
    float clamped = fmaxf(fminf(¦Õ_new, ¦Õ_max), ¦Õ_min);
    if (clamped != ¦Õ_new) // New extrema found, revert to the first order
                          // accurate semi-Lagrangian method.
        ¦Õ_new = tex3D(advect_source, back_traced.x, back_traced.y, back_traced.z);

    float result = quadratic_dissipation ?
        (1.0f - dissipation * time_step * (1.0f - ¦Õ_new)) * ¦Õ_new :
        (1.0f - dissipation * time_step) * ¦Õ_new;
    surf3Dwrite(__float2half_rn(result), advect_dest, x * sizeof(ushort), y, z, cudaBoundaryModeTrap);
}

__global__ void AdvectScalarBfeccRemoveErrorStaggeredKernel(float time_step,
                                                            float3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 velocity = GetCenterVelocity(advect_velocity, coord, volume_size);
    float3 back_traced = coord - time_step * velocity;

    float ¦Õ = tex3D(advect_source, coord.x, coord.y, coord.z);
    float result = tex3D(advect_intermediate1, back_traced.x, back_traced.y,
                         back_traced.z);
    result = 0.5f * (3.0f * ¦Õ - result);
    surf3Dwrite(__float2half_rn(result), advect_dest, x * sizeof(ushort), y, z,
                cudaBoundaryModeTrap);
}

__global__ void AdvectScalarMacCormackStaggeredKernel(
    float time_step, float dissipation, bool quadratic_dissipation,
    float3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 velocity = GetCenterVelocity(advect_velocity, coord, volume_size);
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

__global__ void AdvectScalarSemiLagrangianStaggeredKernel(
    float time_step, float dissipation, bool quadratic_dissipation,
    float3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 velocity = GetCenterVelocity(advect_velocity, coord, volume_size);
    float3 back_traced = coord - time_step * velocity;

    float ¦Õ = tex3D(advect_source, back_traced.x, back_traced.y, back_traced.z);
    float result = quadratic_dissipation ?
        (1.0f - dissipation * time_step * (1.0f - ¦Õ)) * ¦Õ :
        (1.0f - dissipation * time_step) * ¦Õ;
    surf3Dwrite(__float2half_rn(result), advect_dest, x * sizeof(ushort), y, z,
                cudaBoundaryModeTrap);
}

__global__ void AdvectVelocityBfeccStaggeredKernel(float time_step,
                                                   float dissipation)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 v0 = GetVelocityXStaggered(advect_velocity, coord);
    float3 back_traced_x = coord - v0 * time_step;

    float3 v1 = GetVelocityYStaggered(advect_velocity, coord);
    float3 back_traced_y = coord - v1 * time_step;

    float3 v2 = GetVelocityZStaggered(advect_velocity, coord);
    float3 back_traced_z = coord - v2 * time_step;

    float3 v_min;
    float3 v_max;
    GetVelocityLimitsStaggered(advect_velocity, back_traced_x, back_traced_y, back_traced_z, &v_min, &v_max);

    float3 v_new;
    v_new.x = tex3D(advect_intermediate, back_traced_x.x, back_traced_x.y, back_traced_x.z).x;
    v_new.y = tex3D(advect_intermediate, back_traced_y.x, back_traced_y.y, back_traced_y.z).y;
    v_new.z = tex3D(advect_intermediate, back_traced_z.x, back_traced_z.y, back_traced_z.z).z;

    float3 clamped = fmaxf(fminf(v_new, v_max), v_min);
    if (clamped.x != v_new.x || clamped.y != v_new.y || clamped.z != v_new.z) {
        v_new.x = tex3D(advect_velocity, back_traced_x.x, back_traced_x.y, back_traced_x.z).x;
        v_new.y = tex3D(advect_velocity, back_traced_y.x, back_traced_y.y, back_traced_y.z).y;
        v_new.z = tex3D(advect_velocity, back_traced_z.x, back_traced_z.y, back_traced_z.z).z;
    }

    v_new *= (1.0f - dissipation * time_step);
    ushort4 result = make_ushort4(__float2half_rn(v_new.x), __float2half_rn(v_new.y), __float2half_rn(v_new.z), 0);
    surf3Dwrite(result, advect_dest, x * sizeof(ushort4), y, z, cudaBoundaryModeTrap);
}

__global__ void AdvectVelocityBfeccRemoveErrorStaggeredKernel(float time_step)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 velocity;
    float3 new_velocity = BackTraceVelocityStaggered(advect_velocity,
                                                     advect_intermediate, coord,
                                                     time_step, &velocity);

    new_velocity = 0.5f * (3.0f * velocity - new_velocity);
    ushort4 result = make_ushort4(__float2half_rn(new_velocity.x),
                                  __float2half_rn(new_velocity.y),
                                  __float2half_rn(new_velocity.z),
                                  0);
    surf3Dwrite(result, advect_dest, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

__global__ void AdvectVelocityMacCormackStaggeredKernel(float time_step,
                                                        float dissipation)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 v0 = GetVelocityXStaggered(advect_velocity, coord);
    float v_n_x = v0.x;
    float3 back_traced_x = coord - v0 * time_step;
    float3 forward_trace_x = coord + v0 * time_step;

    float3 v1 = GetVelocityYStaggered(advect_velocity, coord);
    float v_n_y = v1.y;
    float3 back_traced_y = coord - v1 * time_step;
    float3 forward_trace_y = coord + v1 * time_step;

    float3 v2 = GetVelocityZStaggered(advect_velocity, coord);
    float v_n_z = v2.z;
    float3 back_traced_z = coord - v2 * time_step;
    float3 forward_trace_z = coord + v2 * time_step;

    float3 v_min;
    float3 v_max;
    GetVelocityLimitsStaggered(advect_velocity, back_traced_x, back_traced_y, back_traced_z, &v_min, &v_max);

    float3 v_n_plus_1_hat = make_float3(tex3D(advect_intermediate, coord.x, coord.y, coord.z));

    float3 v_n_hat;
    v_n_hat.x = tex3D(advect_intermediate, forward_trace_x.x, forward_trace_x.y, forward_trace_x.z).x;
    v_n_hat.y = tex3D(advect_intermediate, forward_trace_y.x, forward_trace_y.y, forward_trace_y.z).y;
    v_n_hat.z = tex3D(advect_intermediate, forward_trace_z.x, forward_trace_z.y, forward_trace_z.z).z;

    float3 v_new = (v_n_plus_1_hat + 0.5f * (make_float3(v_n_x, v_n_y, v_n_z) - v_n_hat));
    float3 clamped = fmaxf(fminf(v_new, v_max), v_min);
    if (clamped.x != v_new.x || clamped.y != v_new.y || clamped.z != v_new.z)
        v_new = v_n_plus_1_hat;

    v_new *= (1.0f - dissipation * time_step);
    ushort4 result = make_ushort4(__float2half_rn(v_new.x), __float2half_rn(v_new.y), __float2half_rn(v_new.z), 0);
    surf3Dwrite(result, advect_dest, x * sizeof(ushort4), y, z, cudaBoundaryModeTrap);
}

__global__ void AdvectVelocitySemiLagrangianStaggeredKernel(float time_step,
                                                            float dissipation)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 velocity;
    float3 new_velocity = BackTraceVelocityStaggered(advect_velocity,
                                                     advect_velocity, coord,
                                                     time_step, &velocity);
    new_velocity *= (1.0f - dissipation * time_step);
    ushort4 result = make_ushort4(__float2half_rn(new_velocity.x),
                                  __float2half_rn(new_velocity.y),
                                  __float2half_rn(new_velocity.z),
                                  0);
    surf3Dwrite(result, advect_dest, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

// =============================================================================

void LaunchAdvectScalarBfeccStaggered(cudaArray_t dest_array,
                                      cudaArray_t velocity_array,
                                      cudaArray_t source_array,
                                      cudaArray_t intermediate_array,
                                      float time_step, float dissipation,
                                      bool quadratic_dissipation,
                                      uint3 volume_size)
{
    // Pass 1: Calculate ¦Õ_n_plus_1_hat, and store in |dest_array|.
    if (BindCudaSurfaceToArray(&advect_dest, dest_array) != cudaSuccess)
        return;

    auto bound_vel = BindHelper::Bind(&advect_velocity, velocity_array,
                                      false, cudaFilterModeLinear);
    if (bound_vel.error() != cudaSuccess)
        return;

    auto bound_source = BindHelper::Bind(&advect_source, source_array,
                                         false, cudaFilterModeLinear);
    if (bound_source.error() != cudaSuccess)
        return;

    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    AdvectScalarSemiLagrangianStaggeredKernel<<<grid, block>>>(
        time_step, 0.0f, quadratic_dissipation, make_float3(volume_size));

    // Pass 2: Calculate ¦Õ_n_hat, and store in |intermediate_array|.
    if (BindCudaSurfaceToArray(&advect_dest, intermediate_array) != cudaSuccess)
        return;

    auto bound_intermediate1 = BindHelper::Bind(&advect_intermediate1,
                                                dest_array, false,
                                                cudaFilterModeLinear);
    if (bound_intermediate1.error() != cudaSuccess)
        return;

    AdvectScalarBfeccRemoveErrorStaggeredKernel<<<grid, block>>>(
        -time_step, make_float3(volume_size));

    // Pass 3: Calculate the final result.
    if (BindCudaSurfaceToArray(&advect_dest, dest_array) != cudaSuccess)
        return;

    bound_intermediate1.Take(
        BindHelper::Bind(&advect_intermediate1, intermediate_array, false,
                         cudaFilterModeLinear));
    if (bound_intermediate1.error() != cudaSuccess)
        return;

    AdvectScalarBfeccStaggeredKernel<<<grid, block>>>(
        time_step, dissipation, quadratic_dissipation,
        make_float3(volume_size));
}

void LaunchAdvectScalarMacCormackStaggered(cudaArray_t dest_array,
                                           cudaArray_t velocity_array,
                                           cudaArray_t source_array,
                                           cudaArray_t intermediate_array,
                                           float time_step, float dissipation,
                                           bool quadratic_dissipation,
                                           uint3 volume_size)
{
    if (BindCudaSurfaceToArray(&advect_dest, intermediate_array) != cudaSuccess)
        return;

    auto bound_vel = BindHelper::Bind(&advect_velocity, velocity_array, false,
                                      cudaFilterModeLinear);
    if (bound_vel.error() != cudaSuccess)
        return;

    auto bound_source = BindHelper::Bind(&advect_source, source_array, false,
                                      cudaFilterModeLinear);
    if (bound_source.error() != cudaSuccess)
        return;

    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    AdvectScalarSemiLagrangianStaggeredKernel<<<grid, block>>>(
        time_step, 0.0f, quadratic_dissipation, make_float3(volume_size));

    if (BindCudaSurfaceToArray(&advect_dest, dest_array) != cudaSuccess)
        return;

    auto bound_intermediate1 = BindHelper::Bind(&advect_intermediate1,
                                                intermediate_array, false,
                                                cudaFilterModeLinear);
    if (bound_intermediate1.error() != cudaSuccess)
        return;

    AdvectScalarMacCormackStaggeredKernel<<<grid, block>>>(
        time_step, dissipation, quadratic_dissipation,
        make_float3(volume_size));
}

void LaunchAdvectScalarStaggered(cudaArray_t dest_array,
                                 cudaArray_t velocity_array,
                                 cudaArray_t source_array,
                                 cudaArray_t intermediate_array,
                                 float time_step, float dissipation,
                                 bool quadratic_dissipation, uint3 volume_size,
                                 AdvectionMethod method)
{
    if (method == MACCORMACK_SEMI_LAGRANGIAN) {
        LaunchAdvectScalarMacCormackStaggered(dest_array, velocity_array,
                                              source_array, intermediate_array,
                                              time_step, dissipation, false,
                                              volume_size);
        return;
    } else if (method == BFECC_SEMI_LAGRANGIAN) {
        LaunchAdvectScalarBfeccStaggered(dest_array, velocity_array,
                                         source_array, intermediate_array,
                                         time_step, dissipation, false,
                                         volume_size);
        return;
    }

    if (BindCudaSurfaceToArray(&advect_dest, dest_array) != cudaSuccess)
        return;

    auto bound_vel = BindHelper::Bind(&advect_velocity, velocity_array, false,
                                      cudaFilterModeLinear);
    if (bound_vel.error() != cudaSuccess)
        return;

    auto bound_source = BindHelper::Bind(&advect_source, source_array, false,
                                         cudaFilterModeLinear);
    if (bound_source.error() != cudaSuccess)
        return;

    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    AdvectScalarSemiLagrangianStaggeredKernel<<<grid, block>>>(
        time_step, dissipation, quadratic_dissipation,
        make_float3(volume_size));
}

void LaunchAdvectVelocityBfeccStaggered(cudaArray_t dest_array,
                                        cudaArray_t velocity_array,
                                        cudaArray_t intermediate_array,
                                        float time_step, float time_step_prev,
                                        float dissipation, uint3 volume_size)
{
    // Pass 1: Calculate ¦Õ_n_plus_1_hat, and store in |dest_array|.
    if (BindCudaSurfaceToArray(&advect_dest, dest_array) != cudaSuccess)
        return;

    auto bound_vel = BindHelper::Bind(&advect_velocity, velocity_array, false,
                                      cudaFilterModeLinear);
    if (bound_vel.error() != cudaSuccess)
        return;

    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    AdvectVelocitySemiLagrangianStaggeredKernel<<<grid, block>>>(time_step,
                                                                 0.0f);

    // Pass 2: Calculate ¦Õ_n_hat, and store in |intermediate_array|.
    if (BindCudaSurfaceToArray(&advect_dest, intermediate_array) != cudaSuccess)
        return;

    auto bound_intermediate = BindHelper::Bind(&advect_intermediate, dest_array,
                                               false, cudaFilterModeLinear);
    if (bound_intermediate.error() != cudaSuccess)
        return;

    AdvectVelocityBfeccRemoveErrorStaggeredKernel<<<grid, block>>>(-time_step);

    // Pass 3: Calculate the final result.
    if (BindCudaSurfaceToArray(&advect_dest, dest_array) != cudaSuccess)
        return;

    bound_intermediate.Take(
        BindHelper::Bind(&advect_intermediate, intermediate_array, false,
                         cudaFilterModeLinear));
    if (bound_intermediate.error() != cudaSuccess)
        return;

    AdvectVelocityBfeccStaggeredKernel<<<grid, block>>>(time_step, dissipation);
}

void LaunchAdvectVelocityMacCormackStaggered(cudaArray_t dest_array,
                                             cudaArray_t velocity_array,
                                             cudaArray_t intermediate_array,
                                             float time_step,
                                             float time_step_prev,
                                             float dissipation,
                                             uint3 volume_size)
{
    if (BindCudaSurfaceToArray(&advect_dest, intermediate_array) != cudaSuccess)
        return;

    auto bound_vel = BindHelper::Bind(&advect_velocity, velocity_array, false,
                                      cudaFilterModeLinear);
    if (bound_vel.error() != cudaSuccess)
        return;

    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    AdvectVelocitySemiLagrangianStaggeredKernel<<<grid, block>>>(time_step,
                                                                 0.0f);

    if (BindCudaSurfaceToArray(&advect_dest, dest_array) != cudaSuccess)
        return;

    auto bound_intermediate = BindHelper::Bind(&advect_intermediate,
                                               intermediate_array, false,
                                               cudaFilterModeLinear);
    if (bound_intermediate.error() != cudaSuccess)
        return;

    AdvectVelocityMacCormackStaggeredKernel<<<grid, block>>>(time_step,
                                                             dissipation);
}

void LaunchAdvectVelocityStaggered(cudaArray_t dest_array,
                                   cudaArray_t velocity_array,
                                   cudaArray_t intermediate_array,
                                   float time_step, float time_step_prev,
                                   float dissipation, uint3 volume_size,
                                   AdvectionMethod method)
{
    if (method == MACCORMACK_SEMI_LAGRANGIAN) {
        LaunchAdvectVelocityMacCormackStaggered(dest_array, velocity_array,
                                                intermediate_array, time_step,
                                                time_step_prev, dissipation,
                                                volume_size);
        return;
    } else if (method == BFECC_SEMI_LAGRANGIAN) {
        LaunchAdvectVelocityBfeccStaggered(dest_array, velocity_array,
                                           intermediate_array, time_step,
                                           time_step_prev, dissipation,
                                           volume_size);
        return;
    }

    if (BindCudaSurfaceToArray(&advect_dest, dest_array) != cudaSuccess)
        return;

    auto bound_vel = BindHelper::Bind(&advect_velocity, velocity_array, false,
                                      cudaFilterModeLinear);
    if (bound_vel.error() != cudaSuccess)
        return;

    dim3 block(8, 8, 8);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);
    AdvectVelocitySemiLagrangianStaggeredKernel<<<grid, block>>>(time_step,
                                                                 dissipation);
}
