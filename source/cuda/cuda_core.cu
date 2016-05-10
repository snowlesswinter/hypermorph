#include "cuda_core.h"

#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "third_party/glm/common.hpp"
#include "third_party/glm/glm.hpp"
#include "third_party/glm/mat3x3.hpp"
#include "third_party/glm/mat4x4.hpp"
#include "third_party/glm/vec2.hpp"
#include "third_party/glm/vec3.hpp"

texture<float1, cudaTextureType3D, cudaReadModeElementType> in_tex;
surface<void, cudaSurfaceType3D> clear_volume;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> raycast_density;
surface<void, cudaSurfaceType2D> raycast_dest;

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

__global__ void ClearVolume4Kernel(glm::vec4 value)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    surf3Dwrite(make_float4(value.x, value.y, value.z, value.w), clear_volume,
                x * sizeof(float4), y, z, cudaBoundaryModeTrap);
}

__global__ void ClearVolume2Kernel(glm::vec4 value)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    surf3Dwrite(make_float2(value.x, value.y), clear_volume, x * sizeof(float2),
                y, z, cudaBoundaryModeTrap);
}

__global__ void ClearVolume1Kernel(glm::vec4 value)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    surf3Dwrite(value.x, clear_volume, x * sizeof(float), y, z,
                cudaBoundaryModeTrap);
}

__global__ void ClearVolumeHalf4Kernel(glm::vec4 value)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    ushort4 raw = make_ushort4(__float2half_rn(value.x),
                               __float2half_rn(value.y),
                               __float2half_rn(value.z),
                               __float2half_rn(value.w));
    surf3Dwrite(raw, clear_volume, x * sizeof(ushort4), y, z,
                cudaBoundaryModeTrap);
}

__global__ void ClearVolumeHalf2Kernel(glm::vec4 value)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    ushort2 raw = make_ushort2(__float2half_rn(value.x),
                               __float2half_rn(value.y));
    surf3Dwrite(raw, clear_volume, x * sizeof(ushort2), y, z,
                cudaBoundaryModeTrap);
}

__global__ void ClearVolumeHalf1Kernel(glm::vec4 value)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    ushort1 raw = make_ushort1(__float2half_rn(value.x));
    surf3Dwrite(raw, clear_volume, x * sizeof(ushort1), y, z,
                cudaBoundaryModeTrap);
}

__device__ bool IntersectAABB(glm::vec3 ray_dir, glm::vec3 eye_pos,
                              glm::vec3 min_pos, glm::vec3 max_pos, float* near,
                              float* far)
{
    glm::vec3 inverse_ray_dir = 1.0f / ray_dir;
    glm::vec3 bottom = inverse_ray_dir * (min_pos - eye_pos);
    glm::vec3 top = inverse_ray_dir * (max_pos - eye_pos);
    glm::vec3 near_corner_dist = glm::min(top, bottom);
    glm::vec3 far_corner_dist = glm::max(top, bottom);
    glm::vec2 t = glm::max(glm::vec2(near_corner_dist.x),
                           glm::vec2(near_corner_dist.y, near_corner_dist.z));
    *near = glm::max(t.x, t.y);
    t = glm::min(glm::vec2(far_corner_dist.x),
                 glm::vec2(far_corner_dist.y, far_corner_dist.z));
    *far = glm::min(t.x, t.y);
    return near <= far;
}

__global__ void RaycastKernel(glm::mat3 model_view, glm::vec2 viewport_size,
                              glm::vec3 eye_pos, float focal_length)
{
    const float kMaxDistance = sqrt(2.0f);
    const int kNumSamples = 128;
    const float kStepSize = kMaxDistance / static_cast<float>(kNumSamples);
    const int kNumLightSamples = 32;
    const float kLightScale =
        kMaxDistance / static_cast<float>(kNumLightSamples);
    const float kAbsorption = 10.0f;
    const float kDensityFactor = 10.0f;
    const glm::vec3 light_pos(1.0f, 1.0f, 2.0f);
    const glm::vec3 light_intensity(10.0f);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= static_cast<int>(viewport_size.x) ||
            y >= static_cast<int>(viewport_size.y))
        return;

    glm::vec3 ray_dir;

    // Normalize ray direction vector and transfrom to world space.
    ray_dir.x = 2.0f * x / viewport_size.x - 1.0f;
    ray_dir.y = 2.0f * y / viewport_size.y - 1.0f;
    ray_dir.z = -focal_length;

    // Transform the ray direction vector to model space.
    ray_dir = glm::normalize(ray_dir * model_view);

    // Ray origin is already in model space.
    float near;
    float far;
    IntersectAABB(ray_dir, eye_pos, glm::vec3(-1.0f), glm::vec3(1.0f),
                  &near, &far);
    if (near < 0.0f)
        near = 0.0f;

    glm::vec3 ray_start = eye_pos + ray_dir * near;
    glm::vec3 ray_stop = eye_pos + ray_dir * far;

    ray_start = 0.5f * (ray_start + 1.0f);
    ray_stop = 0.5f * (ray_stop + 1.0f);

    glm::vec3 pos = ray_start;
    glm::vec3 step = glm::normalize(ray_stop - ray_start) * kStepSize;
    float travel = glm::distance(ray_stop, ray_start);
    float transparency = 1.0f;
    glm::vec3 accumulated(0.0f);

    for (int i = 0; i < kNumSamples && travel > 0.0f;
             i++, pos += step, travel -= kStepSize) {
        float density =
            tex3D(raycast_density, pos.x, pos.y, pos.z) * kDensityFactor;
        if (density < 0.000001f)
            continue;

        transparency *= 1.0f - density * kStepSize * kAbsorption;
        if (transparency <= 0.01f)
            break;

        glm::vec3 light_dir = glm::normalize(light_pos - pos) * kLightScale;
        float light_weight = 1.0f;
        glm::vec3 light_pos = pos + light_dir;

        for (int j = 0; j < kNumLightSamples; j++) {
            float alpha = tex3D(raycast_density, light_pos.x, light_pos.y,
                                light_pos.z);
            light_weight *= 1.0f - kAbsorption * kStepSize * alpha;
            if (light_weight <= 0.01f)
                light_pos += light_dir;
        }

        accumulated +=
            light_intensity * light_weight * transparency * density * kStepSize;
    }

    ushort4 raw = make_ushort4(__float2half_rn(accumulated.x),
                               __float2half_rn(accumulated.y),
                               __float2half_rn(accumulated.z),
                               __float2half_rn(1.0f - transparency));
    surf2Dwrite(raw, raycast_dest, x * sizeof(ushort4), y,
                cudaBoundaryModeTrap);
}

// =============================================================================

bool IsHalf1Or2Or4(const cudaChannelFormatDesc& desc)
{
    if (desc.f != cudaChannelFormatKindFloat)
        return false;

    return desc.x == 16 &&
        ((desc.y == 0 && desc.z == 0 && desc.w == 0) ||
            (desc.y == 16 && desc.z == 0 && desc.w == 0)||
            (desc.y == 16 && desc.z == 16 && desc.w == 16));
}

bool IsFloat1Or2Or4(const cudaChannelFormatDesc& desc)
{
    if (desc.f != cudaChannelFormatKindFloat)
        return false;

    return desc.x == 32 &&
        ((desc.y == 0 && desc.z == 0 && desc.w == 0) ||
            (desc.y == 32 && desc.z == 0 && desc.w == 0)||
            (desc.y == 32 && desc.z == 32 && desc.w == 32));
}

bool IsCompliant(const cudaChannelFormatDesc& desc)
{
    return IsHalf1Or2Or4(desc) || IsFloat1Or2Or4(desc);
}

void LaunchClearVolumeKernel(cudaArray* dest_array, const glm::vec4& value,
                             const glm::ivec3& volume_size)
{
    cudaChannelFormatDesc desc;
    cudaError_t result = cudaGetChannelDesc(&desc, dest_array);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    result = cudaBindSurfaceToArray(&clear_volume, dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(8, 8, 16);
    dim3 grid(volume_size.x / block.x, volume_size.y / block.y,
              volume_size.z / block.z);

    assert(IsCompliant(desc));
    if (desc.x == 16 && desc.y == 0 && desc.z == 0 && desc.w == 0 &&
            desc.f == cudaChannelFormatKindFloat)
        ClearVolumeHalf1Kernel<<<grid, block>>>(value);
    else if (desc.x == 16 && desc.y == 16 && desc.z == 0 && desc.w == 0 &&
             desc.f == cudaChannelFormatKindFloat)
        ClearVolumeHalf2Kernel<<<grid, block>>>(value);
    else if (desc.x == 16 && desc.y == 16 && desc.z == 16 && desc.w == 16 &&
             desc.f == cudaChannelFormatKindFloat)
        ClearVolumeHalf4Kernel<<<grid, block>>>(value);
    else if (desc.x == 32 && desc.y == 0 && desc.z == 0 && desc.w == 0 &&
            desc.f == cudaChannelFormatKindFloat)
        ClearVolume1Kernel<<<grid, block>>>(value);
    else if (desc.x == 32 && desc.y == 32 && desc.z == 0 && desc.w == 0 &&
             desc.f == cudaChannelFormatKindFloat)
        ClearVolume2Kernel<<<grid, block>>>(value);
    else if (desc.x == 32 && desc.y == 32 && desc.z == 32 && desc.w == 32 &&
             desc.f == cudaChannelFormatKindFloat)
        ClearVolume4Kernel<<<grid, block>>>(value);
}

void LaunchRaycastKernel(cudaArray* dest_array, cudaArray* density_array,
                         const glm::mat4& model_view,
                         const glm::ivec2& surface_size,
                         const glm::vec3& eye_pos, float focal_length)
{
    cudaChannelFormatDesc desc;
    cudaError_t result = cudaGetChannelDesc(&desc, dest_array);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    result = cudaBindSurfaceToArray(&raycast_dest, dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, density_array);
    raycast_density.normalized = true;
    raycast_density.filterMode = cudaFilterModeLinear;
    raycast_density.addressMode[0] = cudaAddressModeClamp;
    raycast_density.addressMode[1] = cudaAddressModeClamp;
    raycast_density.addressMode[2] = cudaAddressModeClamp;
    raycast_density.channelDesc = desc;

    result = cudaBindTextureToArray(&raycast_density, density_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block(16, 8, 1);
    dim3 grid((surface_size.x + block.x - 1) / block.x,
              (surface_size.y + block.y - 1) / block.y,
              1);

    glm::vec2 viewport_size(surface_size);
    glm::mat3 m(model_view);
    RaycastKernel<<<grid, block>>>(m, viewport_size, eye_pos, focal_length);
}