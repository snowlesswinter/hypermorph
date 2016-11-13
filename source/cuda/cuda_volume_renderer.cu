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

#include "cuda_common_host.h"
#include "cuda_common_kern.h"
#include "cuda/cuda_debug.h"
#include "third_party/glm/common.hpp"
#include "third_party/glm/glm.hpp"
#include "third_party/glm/mat3x3.hpp"
#include "third_party/glm/mat4x4.hpp"
#include "third_party/glm/vec2.hpp"
#include "third_party/glm/vec3.hpp"

texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> raycast_density;
surface<void, cudaSurfaceType2D> raycast_dest;

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

__global__ void RaycastKernel(glm::mat4 inv_rotation, glm::vec2 viewport_size,
                              glm::vec3 eye_pos, float focal_length,
                              glm::vec2 offset, glm::vec3 light_pos,
                              glm::vec3 light_intensity, int num_samples,
                              float step_size, int num_light_samples,
                              float light_scale, float step_absorption,
                              float density_factor, float occlusion_factor,
                              glm::vec2 screen_size, glm::vec3 normalized_size)
{
    int x = VolumeX();
    int y = VolumeY();

    if (x >= static_cast<int>(viewport_size.x) ||
            y >= static_cast<int>(viewport_size.y))
        return;

    glm::vec4 ray_dir4;

    // Normalize ray direction vector and transform to model space.
    ray_dir4.x = (2.0f * x / viewport_size.x - 1.0f) * screen_size.x;
    ray_dir4.y = (2.0f * y / viewport_size.y - 1.0f) * screen_size.y;
    ray_dir4.z = -focal_length; // Right handed, towards far end.
    ray_dir4.w = 0.0f;

    // Transform the ray direction vector to model space.
    glm::vec3 ray_dir = glm::vec3(glm::normalize(inv_rotation * ray_dir4));

    // Ray origin is already in model space.
    float near;
    float far;
    IntersectAABB(ray_dir, eye_pos, -normalized_size, normalized_size,
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

    // Transform to [0, 1) model space.
    ray_start = 0.5f * (ray_start / normalized_size + 1.0f);
    ray_stop = 0.5f * (ray_stop / normalized_size + 1.0f);

    glm::vec3 pos = ray_start;
    glm::vec3 step = glm::normalize(ray_stop - ray_start) * step_size;
    float travel = glm::distance(ray_stop, ray_start);
    float visibility = 1.0f;
    float luminance = 0.0f;

    for (int i = 0; i < num_samples && travel > 0.0f;
            i++, pos += step, travel -= step_size) {
        float density =
            tex3D(raycast_density, pos.x, pos.y, pos.z) * density_factor;
        if (density < 0.02f)
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

        visibility *= __expf(-density * step_absorption);
        luminance += light_weight * visibility * density;

        if (visibility <= 0.01f)
            break;
    }

    auto raw = make_ushort4(
        __float2half_rn(light_intensity.x * luminance * step_size),
        __float2half_rn(light_intensity.y * luminance * step_size),
        __float2half_rn(light_intensity.z * luminance * step_size),
        __float2half_rn(1.0f - visibility));
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

    int x = VolumeX();
    int y = VolumeY();

    if (x >= static_cast<int>(viewport_size.x) ||
            y >= static_cast<int>(viewport_size.y))
        return;

    glm::vec3 ray_dir;

    // Normalize ray direction vector and transform to world space.
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

    // Transform to [0, 1) model space.
    ray_start = 0.5f * (ray_start + 1.0f);
    ray_stop = 0.5f * (ray_stop + 1.0f);

    glm::vec3 pos = ray_start;
    glm::vec3 step = glm::normalize(ray_stop - ray_start) * step_size;
    float travel = glm::distance(ray_stop, ray_start);
    float visibility = 1.0f;

    for (int i = 0; i < num_samples && travel > 0.0f;
            i++, pos += step, travel -= step_size) {
        float density =
            tex3D(raycast_density, pos.x, pos.y, pos.z) * density_factor;
        if (density < 0.01f)
            continue;

        visibility *= 1.0f - density * step_absorption;
        if (visibility <= 0.01f)
            break;
    }

    auto raw = make_ushort4(__float2half_rn(light_intensity.x),
                            __float2half_rn(light_intensity.y),
                            __float2half_rn(light_intensity.z),
                            __float2half_rn(1.0f - visibility));
    surf2Dwrite(raw, raycast_dest, (x + offset.x) * sizeof(raw),
                (y + offset.y), cudaBoundaryModeTrap);
}

__device__ float3 hsv2rgb(const float3& c)
{
    float4 K = make_float4(1.0f, 2.0f / 3.0f, 1.0f / 3.0f, 3.0f);
    float3 fr = fracf(make_float3(c.x) + make_float3(K.x, K.y, K.z)) * 6.0f - make_float3(K.w);
    float3 p = make_float3(fabsf(fr.x), fabsf(fr.y), fabsf(fr.z));
    return c.z * lerp(make_float3(K.x), clamp(p - make_float3(K.x), 0.0f, 1.0f), c.y);
}

__global__ void RaycastKernel_dir_light(glm::mat4 inv_rotation, glm::vec2 viewport_size,
                                        glm::vec3 eye_pos, float focal_length,
                                        glm::vec2 offset, glm::vec3 light_dir,
                                        glm::vec3 light_intensity, int num_samples,
                                        float step_size, int num_light_samples,
                                        float step_absorption,
                                        float density_factor, float occlusion_factor,
                                        glm::vec2 screen_size, glm::vec3 normalized_size)
{
    int x = VolumeX();
    int y = VolumeY();

    if (x >= static_cast<int>(viewport_size.x) ||
            y >= static_cast<int>(viewport_size.y))
        return;

    glm::vec4 ray_dir4;

    // Normalize ray direction vector and transform to model space.
    ray_dir4.x = (2.0f * x / viewport_size.x - 1.0f) * screen_size.x;
    ray_dir4.y = (2.0f * y / viewport_size.y - 1.0f) * screen_size.y;
    ray_dir4.z = -focal_length; // Right handed, towards far end.
    ray_dir4.w = 0.0f;

    // Transform the ray direction vector to model space.
    glm::vec3 ray_dir = glm::vec3(glm::normalize(inv_rotation * ray_dir4));

    // Ray origin is already in model space.
    float near;
    float far;
    IntersectAABB(ray_dir, eye_pos, -normalized_size, normalized_size,
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

    // Transform to [0, 1) model space.
    ray_start = 0.5f * (ray_start / normalized_size + 1.0f);
    ray_stop = 0.5f * (ray_stop / normalized_size + 1.0f);

    glm::vec3 pos = ray_start;
    glm::vec3 step = glm::normalize(ray_stop - ray_start) * step_size;
    float travel = glm::distance(ray_stop, ray_start);
    float visibility = 1.0f;
    float luminance = 0.0f;

    for (int i = 0; i < num_samples && travel > 0.0f;
            i++, pos += step, travel -= step_size) {
        float density =
            tex3D(raycast_density, pos.x, pos.y, pos.z) * density_factor;
        if (density < 0.02f)
            continue;

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

        visibility *= __expf(-density * step_absorption);
        luminance += light_weight * visibility * density;

        if (visibility <= 0.01f)
            break;
    }

    float3 hsv_color = make_float3(205.0f / 360.0f, 0.75f, 0.45f);
    hsv_color.x += luminance * step_size * 100.0f / 360.0f;
    hsv_color.y -= luminance * step_size * 6.5f;
    hsv_color.z += luminance * step_size * 4.5f;

    float3 rgb_color = hsv2rgb(hsv_color);
    auto raw = make_ushort4(__float2half_rn(rgb_color.x),
                            __float2half_rn(rgb_color.y),
                            __float2half_rn(rgb_color.z),
                            __float2half_rn(1.0f - visibility));
    surf2Dwrite(raw, raycast_dest, (x + offset.x) * sizeof(raw),
                (y + offset.y), cudaBoundaryModeTrap);
}

// =============================================================================

namespace kern_launcher
{
void Raycast(cudaArray* dest_array, cudaArray* density_array,
             const glm::mat4& inv_rotation, const glm::ivec2& surface_size,
             const glm::vec3& eye_pos, const glm::vec3& light_color,
             const glm::vec3& light_pos, float light_intensity,
             float focal_length, const glm::vec2& screen_size, int num_samples,
             int num_light_samples, float absorption, float density_factor,
             float occlusion_factor, const glm::vec3& volume_size)
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

    dim3 block(32, 8, 1);
    dim3 grid((static_cast<int>(viewport_size.x) + block.x - 1) / block.x,
              (static_cast<int>(viewport_size.y) + block.y - 1) / block.y, 1);

    glm::vec3 intensity = glm::normalize(light_color);
    intensity *= light_intensity;

    float max_length =
        glm::max(glm::max(volume_size.x, volume_size.y), volume_size.z);
    glm::vec3 normalized_size = volume_size / max_length;
    const float kMaxDistance = sqrt(3.0f);
    const float kStepSize = kMaxDistance / static_cast<float>(num_samples);
    const float kLightScale =
        kMaxDistance / static_cast<float>(num_light_samples);
    const float kAbsorptionTimesStepSize = absorption * kStepSize;

    bool rotate_light = false;
    glm::vec3 light = light_pos;
    if (!rotate_light)
        light = glm::vec3(inv_rotation * glm::vec4(light_pos, 1));

    bool directional_light = true;
    if (directional_light) {
        light = glm::normalize(light) * kLightScale;
        RaycastKernel_dir_light<<<grid, block>>>(
            inv_rotation, viewport_size, eye_pos, focal_length, offset, light,
            intensity, num_samples, kStepSize, num_light_samples,
            kAbsorptionTimesStepSize, density_factor, occlusion_factor,
            screen_size, normalized_size);
    } else {
        RaycastKernel<<<grid, block>>>(
            inv_rotation, viewport_size, eye_pos, focal_length, offset, light,
            intensity, num_samples, kStepSize, num_light_samples, kLightScale,
            kAbsorptionTimesStepSize, density_factor, occlusion_factor,
            screen_size, normalized_size);
    }

    DCHECK_KERNEL();
}
}