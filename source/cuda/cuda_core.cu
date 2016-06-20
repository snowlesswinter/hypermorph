//
// Fluid3d - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Fluid3d license (* see part 1 below)
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. Acknowledgement of the
//    original author is required if you publish this in a paper, or use it
//    in a product.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

#include "cuda_core.h"

#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "cuda_common.h"
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

__global__ void ClearVolume4Kernel(glm::vec4 value, uint3 volume_size)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    auto raw = make_float4(value.x, value.y, value.z, value.w);
    surf3Dwrite(raw, clear_volume, x * sizeof(raw), y, z, cudaBoundaryModeTrap);
}

__global__ void ClearVolume2Kernel(glm::vec4 value, uint3 volume_size)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    auto raw = make_float2(value.x, value.y);
    surf3Dwrite(raw, clear_volume, x * sizeof(raw), y, z, cudaBoundaryModeTrap);
}

__global__ void ClearVolume1Kernel(glm::vec4 value, uint3 volume_size)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    surf3Dwrite(value.x, clear_volume, x * sizeof(value.x), y, z,
                cudaBoundaryModeTrap);
}

__global__ void ClearVolumeHalf4Kernel(glm::vec4 value, uint3 volume_size)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    auto raw = make_ushort4(__float2half_rn(value.x),
                            __float2half_rn(value.y),
                            __float2half_rn(value.z),
                            __float2half_rn(value.w));
    surf3Dwrite(raw, clear_volume, x * sizeof(raw), y, z,
                cudaBoundaryModeTrap);
}

__global__ void ClearVolumeHalf2Kernel(glm::vec4 value, uint3 volume_size)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    auto raw = make_ushort2(__float2half_rn(value.x),
                            __float2half_rn(value.y));
    surf3Dwrite(raw, clear_volume, x * sizeof(raw), y, z,
                cudaBoundaryModeTrap);
}

__global__ void ClearVolumeHalf1Kernel(glm::vec4 value, uint3 volume_size)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    auto raw = make_ushort1(__float2half_rn(value.x));
    surf3Dwrite(raw, clear_volume, x * sizeof(raw), y, z,
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
                              glm::vec3 eye_pos, float focal_length,
                              glm::vec2 offset, glm::vec3 light_pos,
                              glm::vec3 light_intensity, int num_samples,
                              float step_size, int num_light_samples,
                              float light_scale, float step_absorption,
                              float density_factor, float occlusion_factor)
{
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

    // Transform the ray direction vector to model-view space.
    ray_dir = glm::normalize(ray_dir * model_view);

    // Ray origin is already in model space.
    float near;
    float far;
    IntersectAABB(ray_dir, eye_pos, glm::vec3(-1.0f), glm::vec3(1.0f),
                  &near, &far);
    if (far - near < 0.0001f) {
        ushort4 raw = make_ushort4(0, __float2half_rn(1.0f), 0,
                                   __float2half_rn(0.0f));
        surf2Dwrite(raw, raycast_dest, (x + offset.x) * sizeof(ushort4),
                    (y + offset.y), cudaBoundaryModeTrap);
        return;
    }
    if (near < 0.0f)
        near = 0.0f;

    glm::vec3 ray_start = eye_pos + ray_dir * near;
    glm::vec3 ray_stop = eye_pos + ray_dir * far;

    // Transfrom to [0, 1) model space.
    ray_start = 0.5f * (ray_start + 1.0f);
    ray_stop = 0.5f * (ray_stop + 1.0f);

    glm::vec3 pos = ray_start;
    glm::vec3 step = glm::normalize(ray_stop - ray_start) * step_size;
    float travel = glm::distance(ray_stop, ray_start);
    float transmittance = 1.0f;
    float luminance = 0.0f;

    for (int i = 0; i < num_samples && travel > 0.0f;
            i++, pos += step, travel -= step_size) {
        float density =
            tex3D(raycast_density, pos.x, pos.y, pos.z) * density_factor;
        if (density < 0.01f)
            continue;

        glm::vec3 light_dir = glm::normalize(light_pos - pos) * light_scale;
        float light_weight = 1.0f;
        glm::vec3 l_pos = pos + light_dir;

        for (int j = 0; j < num_light_samples; j++) {
            float d = tex3D(raycast_density, l_pos.x, l_pos.y, l_pos.z);
            light_weight *= __expf(-step_absorption * d * occlusion_factor);
            if (light_weight <= 0.01f)
                break;

            // Early termination. Great performance gain.
            if (l_pos.x < 0.0f || l_pos.y < 0.0f || l_pos.z < 0.0f ||
                    l_pos.x > 1.0f || l_pos.y > 1.0f || l_pos.z > 1.0f)
                break;

            l_pos += light_dir;
        }

        transmittance *= __expf(-density * step_absorption);
        luminance += light_weight * transmittance * density;

        if (transmittance <= 0.01f)
            break;
    }

    auto raw = make_ushort4(
        __float2half_rn(light_intensity.x * luminance * step_size),
        __float2half_rn(light_intensity.y * luminance * step_size),
        __float2half_rn(light_intensity.z * luminance * step_size),
        __float2half_rn(1.0f - transmittance));
    surf2Dwrite(raw, raycast_dest, (x + offset.x) * sizeof(raw),
                (y + offset.y), cudaBoundaryModeTrap);
}

__global__ void RaycastFastKernel(glm::mat3 model_view, glm::vec2 viewport_size,
                                  glm::vec3 eye_pos, float focal_length,
                                  glm::vec2 offset, glm::vec3 light_intensity,
                                  int num_samples, float step_size,
                                  int num_light_samples, float light_scale,
                                  float step_absorption, float density_factor,
                                  float occlusion_factor)
{
    const glm::vec3 light_pos(1.5f, 0.7f, 0.0f);

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

    // Transform the ray direction vector to model-view space.
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

    // Transfrom to [0, 1) model space.
    ray_start = 0.5f * (ray_start + 1.0f);
    ray_stop = 0.5f * (ray_stop + 1.0f);

    glm::vec3 pos = ray_start;
    glm::vec3 step = glm::normalize(ray_stop - ray_start) * step_size;
    float travel = glm::distance(ray_stop, ray_start);
    float transmittance = 1.0f;

    for (int i = 0; i < num_samples && travel > 0.0f;
            i++, pos += step, travel -= step_size) {
        float density =
            tex3D(raycast_density, pos.x, pos.y, pos.z) * density_factor;
        if (density < 0.01f)
            continue;

        transmittance *= 1.0f - density * step_absorption;
        if (transmittance <= 0.01f)
            break;
    }

    auto raw = make_ushort4(__float2half_rn(light_intensity.x),
                            __float2half_rn(light_intensity.y),
                            __float2half_rn(light_intensity.z),
                            __float2half_rn(1.0f - transmittance));
    surf2Dwrite(raw, raycast_dest, (x + offset.x) * sizeof(raw),
                (y + offset.y), cudaBoundaryModeTrap);
}

__global__ void RaycastKernel_color(glm::mat3 model_view,
                                    glm::vec2 viewport_size, glm::vec3 eye_pos,
                                    float focal_length, glm::vec2 offset,
                                    glm::vec3 light_intensity, int num_samples,
                                    float step_size, int num_light_samples,
                                    float light_scale, float step_absorption,
                                    float density_factor,
                                    float occlusion_factor)
{
    const glm::vec3 light_pos(1.5f, 0.7f, 0.0f);
    glm::vec3 smoke_color(57.0f, 88.0f, 78.0f);
    glm::vec3 dark = glm::normalize(smoke_color) * 0.2f;
    smoke_color = glm::normalize(glm::vec3(140.0f, 190.0f, 154.0f)) * 15.0f;
    smoke_color = glm::normalize(glm::vec3(128.0f, 190.0f, 234.0f)) * 15.0f;

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

    // Transform the ray direction vector to model-view space.
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

    // Transfrom to [0, 1) model space.
    ray_start = 0.5f * (ray_start + 1.0f);
    ray_stop = 0.5f * (ray_stop + 1.0f);

    glm::vec3 pos = ray_start;
    glm::vec3 step = glm::normalize(ray_stop - ray_start) * step_size;
    float travel = glm::distance(ray_stop, ray_start);
    float transmittance = 1.0f;
    float luminance = 0.0f;

    for (int i = 0; i < num_samples && travel > 0.0f;
         i++, pos += step, travel -= step_size)
    {
        float density =
            tex3D(raycast_density, pos.x, pos.y, pos.z) * density_factor;
        if (density < 0.01f)
            continue;

        glm::vec3 light_dir = glm::normalize(light_pos - pos) * light_scale;
        float light_weight = 1.0f;
        glm::vec3 l_pos = pos + light_dir;

        for (int j = 0; j < num_light_samples; j++)
        {
            float d = tex3D(raycast_density, l_pos.x, l_pos.y, l_pos.z);
            light_weight *= 1.0f - step_absorption * d * occlusion_factor;
            if (light_weight <= 0.01f)
                break;

            // Early termination. Great performance gain.
            if (l_pos.x < 0.0f || l_pos.y < 0.0f || l_pos.z < 0.0f ||
                l_pos.x > 1.0f || l_pos.y > 1.0f || l_pos.z > 1.0f)
                break;

            l_pos += light_dir;
        }

        transmittance *= 1.0f - density * step_absorption;
        luminance += light_weight * transmittance * density;

        if (transmittance <= 0.01f)
            break;
    }

    float alpha = glm::max(0.0f, glm::min(luminance / 20.0f, 1.0f));
    auto raw = make_ushort4(
        __float2half_rn(dark.x + (light_intensity.x * alpha + (1.0f - alpha) * smoke_color.x) * luminance * step_size),
        __float2half_rn(dark.y + (light_intensity.y * alpha + (1.0f - alpha) * smoke_color.y) * luminance * step_size),
        __float2half_rn(dark.z + (light_intensity.z * alpha + (1.0f - alpha) * smoke_color.z) * luminance * step_size),
        __float2half_rn(1.0f - transmittance));
    surf2Dwrite(raw, raycast_dest, (x + offset.x) * sizeof(raw),
                (y + offset.y), cudaBoundaryModeTrap);
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
                             const uint3& volume_size,
                             BlockArrangement* ba)
{
    if (BindCudaSurfaceToArray(&clear_volume, dest_array) != cudaSuccess)
        return;

    cudaChannelFormatDesc desc;
    cudaError_t result = cudaGetChannelDesc(&desc, dest_array);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangeRowScan(&block, &grid, volume_size);

    assert(IsCompliant(desc));
    if (desc.x == 16 && desc.y == 0 && desc.z == 0 && desc.w == 0 &&
            desc.f == cudaChannelFormatKindFloat)
        ClearVolumeHalf1Kernel<<<grid, block>>>(value, volume_size);
    else if (desc.x == 16 && desc.y == 16 && desc.z == 0 && desc.w == 0 &&
             desc.f == cudaChannelFormatKindFloat)
        ClearVolumeHalf2Kernel<<<grid, block>>>(value, volume_size);
    else if (desc.x == 16 && desc.y == 16 && desc.z == 16 && desc.w == 16 &&
             desc.f == cudaChannelFormatKindFloat)
        ClearVolumeHalf4Kernel<<<grid, block>>>(value, volume_size);
    else if (desc.x == 32 && desc.y == 0 && desc.z == 0 && desc.w == 0 &&
            desc.f == cudaChannelFormatKindFloat)
        ClearVolume1Kernel<<<grid, block>>>(value, volume_size);
    else if (desc.x == 32 && desc.y == 32 && desc.z == 0 && desc.w == 0 &&
             desc.f == cudaChannelFormatKindFloat)
        ClearVolume2Kernel<<<grid, block>>>(value, volume_size);
    else if (desc.x == 32 && desc.y == 32 && desc.z == 32 && desc.w == 32 &&
             desc.f == cudaChannelFormatKindFloat)
        ClearVolume4Kernel<<<grid, block>>>(value, volume_size);
}

void LaunchRaycastKernel(cudaArray* dest_array, cudaArray* density_array,
                         const glm::mat4& model_view,
                         const glm::ivec2& surface_size,
                         const glm::vec3& eye_pos, const glm::vec3& light_color,
                         const glm::vec3& light_pos, float light_intensity,
                         float focal_length, int num_samples,
                         int num_light_samples, float absorption,
                         float density_factor, float occlusion_factor)
{
    if (BindCudaSurfaceToArray(&raycast_dest, dest_array) != cudaSuccess)
        return;

    auto bound_density = BindHelper::Bind(&raycast_density, density_array, true,
                                          cudaFilterModeLinear,
                                          cudaAddressModeBorder);
    if (bound_density.error() != cudaSuccess)
        return;

    glm::vec2 viewport_size = surface_size;
    glm::ivec2 offset(0);
    glm::mat3 m(model_view);

    dim3 block(32, 8, 1);
    dim3 grid((static_cast<int>(viewport_size.x) + block.x - 1) / block.x,
              (static_cast<int>(viewport_size.y) + block.y - 1) / block.y, 1);

    glm::vec3 intensity = glm::normalize(light_color);
    intensity *= light_intensity;
    const float kMaxDistance = sqrt(2.0f);
    const float kStepSize = kMaxDistance / static_cast<float>(num_samples);
    const float kLightScale =
        kMaxDistance / static_cast<float>(num_light_samples);
    const float kAbsorptionTimesStepSize = absorption * kStepSize;
    
    const bool fast = false;
    if (fast)
        RaycastFastKernel<<<grid, block>>>(m, viewport_size, eye_pos,
                                           focal_length, offset, intensity,
                                           num_samples, kStepSize,
                                           num_light_samples, kLightScale,
                                           kAbsorptionTimesStepSize,
                                           density_factor, occlusion_factor);
    else
        RaycastKernel<<<grid, block>>>(m, viewport_size, eye_pos, focal_length,
                                       offset, light_pos, intensity,
                                       num_samples, kStepSize,
                                       num_light_samples, kLightScale,
                                       kAbsorptionTimesStepSize, density_factor,
                                       occlusion_factor);
}