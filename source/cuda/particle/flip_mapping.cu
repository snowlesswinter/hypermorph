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

#include <cassert>
#include <functional>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "cuda/aux_buffer_manager.h"
#include "cuda/block_arrangement.h"
#include "cuda/cuda_common_host.h"
#include "cuda/cuda_common_kern.h"
#include "cuda/cuda_debug.h"
#include "cuda/particle/flip_common.cuh"
#include "flip.h"

surface<void, cudaSurfaceType3D> surf;
surface<void, cudaSurfaceType3D> surf_x;
surface<void, cudaSurfaceType3D> surf_y;
surface<void, cudaSurfaceType3D> surf_z;
surface<void, cudaSurfaceType3D> surf_d;
surface<void, cudaSurfaceType3D> surf_t;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_x;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_y;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_z;

const uint32_t kMaxParticlesInCell = 4;
const uint16_t kInvalidPos = 48128; // __float2half_rn(-1.0f);

struct ParticleFields
{
    uint16_t* position_x_;
    uint16_t* position_y_;
    uint16_t* position_z_;
    uint16_t* velocity_x_;
    uint16_t* velocity_y_;
    uint16_t* velocity_z_;
    uint16_t* density_;
    uint16_t* temperature_;
};

struct FieldWeightAndWeightedSum
{
    float weight_vel_x_;
    float weight_vel_y_;
    float weight_vel_z_;
    float weight_density_;
    float weight_temperature_;
    float sum_vel_x_;
    float sum_vel_y_;
    float sum_vel_z_;
    float sum_density_;
    float sum_temperature_;
};

__device__ inline void ResetWeight(FieldWeightAndWeightedSum* w)
{
    const float kInitialWeight = 0.0001f;

    w->weight_vel_x_       = kInitialWeight;
    w->weight_vel_y_       = kInitialWeight;
    w->weight_vel_z_       = kInitialWeight;
    w->weight_density_     = kInitialWeight;
    w->weight_temperature_ = kInitialWeight;
    w->sum_vel_x_          = 0.0f;
    w->sum_vel_y_          = 0.0f;
    w->sum_vel_z_          = 0.0f;
    w->sum_density_        = 0.0f;
    w->sum_temperature_    = 0.0f;
}

__device__ float WeightKernel(float r)
{
    if (r >= -1.0f && r <= 0.0f)
        return 1.0f + r;

    if (r < 1.0f && r > 0.0f)
        return 1.0f - r;

    return 0.0f;
}

__device__ float DistanceWeight(float x, float y, float z, float x0,
                                float y0, float z0)
{
    return WeightKernel(x - x0) * WeightKernel(y - y0) * WeightKernel(z - z0);
}

__device__ void ComputeWeightedAverage(float* total_value, float* total_weight,
                                       const uint16_t* pos_x,
                                       const uint16_t* pos_y,
                                       const uint16_t* pos_z, float3 pos,
                                       const uint16_t* field, int count)
{
    for (int i = 0; i < count; i++) {
        float x = __half2float(*(pos_x + i));
        float y = __half2float(*(pos_y + i));
        float z = __half2float(*(pos_z + i));

        float weight = DistanceWeight(x, y, z, pos.x, pos.y, pos.z);

        *total_weight += weight;
        *total_value += weight * __half2float(*(field + i));
    }
}

__device__ void ReadPosAndField_all(ParticleFields* smem_fields,
                                    const FlipParticles& particles, uint grid_x,
                                    uint grid_y, uint grid_z, uint smem_i,
                                    uint3 volume_size)
{
    uint32_t* p_index = particles.particle_index_;
    uint32_t* p_count = particles.particle_count_;

    int cell = (grid_z * volume_size.y + grid_y) * volume_size.x + grid_x;
    int count = p_count[cell];
    int index = p_index[cell];

    const uint16_t* pos_x       = particles.position_x_  + index;
    const uint16_t* pos_y       = particles.position_y_  + index;
    const uint16_t* pos_z       = particles.position_z_  + index;
    const uint16_t* vel_x       = particles.velocity_x_  + index;
    const uint16_t* vel_y       = particles.velocity_y_  + index;
    const uint16_t* vel_z       = particles.velocity_z_  + index;
    const uint16_t* density     = particles.density_     + index;
    const uint16_t* temperature = particles.temperature_ + index;

    int i = 0;
    int base_index = smem_i * kMaxParticlesInCell;
    for (;i < count; i++) {
        smem_fields->position_x_ [base_index + i] = *pos_x++;
        smem_fields->position_y_ [base_index + i] = *pos_y++;
        smem_fields->position_z_ [base_index + i] = *pos_z++;
        smem_fields->velocity_x_ [base_index + i] = *vel_x++;
        smem_fields->velocity_y_ [base_index + i] = *vel_y++;
        smem_fields->velocity_z_ [base_index + i] = *vel_z++;
        smem_fields->density_    [base_index + i] = *density++;
        smem_fields->temperature_[base_index + i] = *temperature++;
    }

    if (count < kMaxParticlesInCell)
        smem_fields->position_x_[base_index + i] = kInvalidPos;
}

__device__ void ReadPosAndField(ushort3* smem_pos, uint16_t* smem_��,
                                const FlipParticles& particles,
                                const uint16_t* field, uint grid_x, uint grid_y,
                                uint grid_z, uint smem_i, uint3 volume_size)
{
    uint32_t* p_index = particles.particle_index_;
    uint32_t* p_count = particles.particle_count_;

    int cell = (grid_z * volume_size.y + grid_y) * volume_size.x + grid_x;
    int count = p_count[cell];
    int index = p_index[cell];

    const uint16_t* pos_x   = particles.position_x_ + index;
    const uint16_t* pos_y   = particles.position_y_ + index;
    const uint16_t* pos_z   = particles.position_z_ + index;
    const uint16_t* f       = field + index;

    int i = 0;
    for (;i < count; i++) {
        smem_pos[smem_i * kMaxParticlesInCell + i] = make_ushort3(*pos_x++, *pos_y++, *pos_z++);
        smem_��  [smem_i * kMaxParticlesInCell + i] = *f++;
    }

    if (count < kMaxParticlesInCell)
        smem_pos[smem_i * kMaxParticlesInCell + i].x = kInvalidPos;
}

__device__ void FlipReadBlockAndHalo_32x6_all(ParticleFields* smem_fields,
                                              const FlipParticles& particles,
                                              uint grid_z, uint thread_x,
                                              uint thread_y, uint3 volume_size)
{
    uint linear_index = thread_y * blockDim.x + thread_x;

    const uint kSmemWidth = 48;

    uint smem_x  = linear_index % kSmemWidth;
    uint smem_y1 = linear_index / kSmemWidth;
    uint smem_y2 = smem_y1 + 4;

    int grid_x  = static_cast<int>(blockIdx.x * blockDim.x + smem_x)  - 8;
    int grid_y1 = static_cast<int>(blockIdx.y * blockDim.y + smem_y1) - 1;
    int grid_y2 = static_cast<int>(blockIdx.y * blockDim.y + smem_y2) - 1;

    uint16_t* smem_pos_x = smem_fields->position_x_;

    // Tightening the scope of memory read will not notablely affect the
    // performance, because the 8(one-side) extra units are actually the
    // minimum amount of data size for fetching the halo.
    if (grid_x < 0 || grid_x >= volume_size.x) {
        smem_pos_x[ linear_index        * kMaxParticlesInCell] = kInvalidPos;
        smem_pos_x[(linear_index + 192) * kMaxParticlesInCell] = kInvalidPos;
    } else {
        if (grid_y1 >= 0 && grid_y1 < volume_size.y)
            ReadPosAndField_all(smem_fields, particles, grid_x, grid_y1, grid_z, linear_index, volume_size);
        else
            smem_pos_x[linear_index * kMaxParticlesInCell] = kInvalidPos;

        if (grid_y2 >= 0 && grid_y2 < volume_size.y)
            ReadPosAndField_all(smem_fields, particles, grid_x, grid_y2, grid_z, linear_index + 192, volume_size);
        else
            smem_pos_x[(linear_index + 192) * kMaxParticlesInCell] = kInvalidPos;
    }
}

__device__ void FlipReadBlockAndHalo_32x6(ushort3* smem_pos, uint16_t* smem_��,
                                          const FlipParticles& particles,
                                          const uint16_t* field, uint grid_z,
                                          uint thread_x, uint thread_y,
                                          uint3 volume_size)
{
    uint linear_index = thread_y * blockDim.x + thread_x;

    const uint kSmemWidth = 48;

    uint smem_x  = linear_index % kSmemWidth;
    uint smem_y1 = linear_index / kSmemWidth;
    uint smem_y2 = smem_y1 + 4;

    int grid_x  = static_cast<int>(blockIdx.x * blockDim.x + smem_x)  - 8;
    int grid_y1 = static_cast<int>(blockIdx.y * blockDim.y + smem_y1) - 1;
    int grid_y2 = static_cast<int>(blockIdx.y * blockDim.y + smem_y2) - 1;

    if (grid_x < 0 || grid_x >= volume_size.x) {
        smem_pos[ linear_index        * kMaxParticlesInCell].x = kInvalidPos;
        smem_pos[(linear_index + 192) * kMaxParticlesInCell].x = kInvalidPos;
    } else {
        if (grid_y1 >= 0 && grid_y1 < volume_size.y)
            ReadPosAndField(smem_pos, smem_��, particles, field, grid_x, grid_y1, grid_z, linear_index, volume_size);
        else
            smem_pos[linear_index * kMaxParticlesInCell].x = kInvalidPos;

        if (grid_y2 >= 0 && grid_y2 < volume_size.y)
            ReadPosAndField(smem_pos, smem_��, particles, field, grid_x, grid_y2, grid_z, linear_index + 192, volume_size);
        else
            smem_pos[(linear_index + 192) * kMaxParticlesInCell].x = kInvalidPos;
    }
}

__device__ void ComputeWeightedAverage_smem_All(
    FieldWeightAndWeightedSum* w, const ParticleFields* smem_fields,
    uint strided, float3 p0)
{
    const uint16_t* pos_x =       smem_fields->position_x_  + strided;
    const uint16_t* pos_y =       smem_fields->position_y_  + strided;
    const uint16_t* pos_z =       smem_fields->position_z_  + strided;
    const uint16_t* vel_x =       smem_fields->velocity_x_  + strided;
    const uint16_t* vel_y =       smem_fields->velocity_y_  + strided;
    const uint16_t* vel_z =       smem_fields->velocity_z_  + strided;
    const uint16_t* density =     smem_fields->density_     + strided;
    const uint16_t* temperature = smem_fields->temperature_ + strided;

    for (int i = 0; i < kMaxParticlesInCell; i++) {
        float x = __half2float(*pos_x++);
        if (x < 0.0f) // Empty cell.
            return;

        float y = __half2float(*pos_y++);
        float z = __half2float(*pos_z++);

        float weight_vel_x = DistanceWeight(x, y, z, p0.x - 0.5f, p0.y,        p0.z);
        float weight_vel_y = DistanceWeight(x, y, z, p0.x,        p0.y - 0.5f, p0.z);
        float weight_vel_z = DistanceWeight(x, y, z, p0.x,        p0.y,        p0.z - 0.5f);
        float weight =       DistanceWeight(x, y, z, p0.x,        p0.y,        p0.z);

        w->weight_vel_x_       += weight_vel_x;
        w->weight_vel_y_       += weight_vel_y;
        w->weight_vel_z_       += weight_vel_z;
        w->weight_density_     += weight;
        w->weight_temperature_ += weight;

        w->sum_vel_x_       += weight_vel_x * __half2float(*(vel_x       + i));
        w->sum_vel_y_       += weight_vel_y * __half2float(*(vel_y       + i));
        w->sum_vel_z_       += weight_vel_z * __half2float(*(vel_z       + i));
        w->sum_density_     += weight       * __half2float(*(density     + i));
        w->sum_temperature_ += weight       * __half2float(*(temperature + i));
    }
}

__device__ void ComputeWeightedAverage_smem(float* total_value,
                                            float* total_weight,
                                            const ushort3* smem_pos, float3 p0,
                                            const uint16_t* smem_��)
{
    for (int i = 0; i < kMaxParticlesInCell; i++) {
        const ushort3* pos = smem_pos + i;
        float x = __half2float(pos->x);
        if (x < 0.0f)
            return;

        float y = __half2float(pos->y);
        float z = __half2float(pos->z);

        float weight = DistanceWeight(x, y, z, p0.x, p0.y, p0.z);

        *total_weight += weight;
        *total_value += weight * __half2float(*(smem_�� + i));
    }
}

__device__ inline void SwitchBuffers_all(ParticleFields* fields_ping,
                                         ParticleFields* fields_pong)
{
    __syncthreads();

    ParticleFields temp = *fields_ping;
    *fields_ping = *fields_pong;
    *fields_pong = temp;
}

__device__ inline void SwitchBuffers(ushort3** pos_ping, ushort3** pos_pong,
                                     uint16_t** ��_ping, uint16_t** ��_pong)
{
    __syncthreads();

    ushort3* pos_temp = *pos_ping;
    *pos_ping = *pos_pong;
    *pos_pong = pos_temp;

    uint16_t* ��_temp = *��_ping;
    *��_ping = *��_pong;
    *��_pong = ��_temp;
}

__device__ void SaveToSurface(const FieldWeightAndWeightedSum& w, uint x,
                              uint y, uint z)
{
    uint16_t r1 = __float2half_rn(w.sum_vel_x_ / w.weight_vel_x_);
    surf3Dwrite(r1, surf_x, x * sizeof(r1), y, z, cudaBoundaryModeTrap);
    uint16_t r2 = __float2half_rn(w.sum_vel_y_ / w.weight_vel_y_);
    surf3Dwrite(r2, surf_y, x * sizeof(r2), y, z, cudaBoundaryModeTrap);
    uint16_t r3 = __float2half_rn(w.sum_vel_z_ / w.weight_vel_z_);
    surf3Dwrite(r3, surf_z, x * sizeof(r3), y, z, cudaBoundaryModeTrap);
    uint16_t r4 = __float2half_rn(w.sum_density_ / w.weight_density_);
    surf3Dwrite(r4, surf_d, x * sizeof(r4), y, z, cudaBoundaryModeTrap);
    uint16_t r5 = __float2half_rn(w.sum_temperature_ / w.weight_temperature_);
    surf3Dwrite(r5, surf_t, x * sizeof(r5), y, z, cudaBoundaryModeTrap);
}

// =============================================================================

__global__ void TransferToGridKernel(FlipParticles particles, uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x - 1 || y >= volume_size.y - 1 || z >= volume_size.z - 1)
        return;

    if (x < 1 || y < 1 || z < 1)
        return;

    uint*     p_index     = particles.particle_index_;
    uint*     p_count     = particles.particle_count_;
    uint16_t* pos_x       = particles.position_x_;
    uint16_t* pos_y       = particles.position_y_;
    uint16_t* pos_z       = particles.position_z_;
    uint16_t* vel_x       = particles.velocity_x_;
    uint16_t* vel_y       = particles.velocity_y_;
    uint16_t* vel_z       = particles.velocity_z_;
    uint16_t* density     = particles.density_;
    uint16_t* temperature = particles.temperature_;

    float weight_vel_x       = 0.0001f;
    float weight_vel_y       = 0.0001f;
    float weight_vel_z       = 0.0001f;
    float weight_density     = 0.0001f;
    float weight_temperature = 0.0001f;
    float avg_vel_x          = 0.0f;
    float avg_vel_y          = 0.0f;
    float avg_vel_z          = 0.0f;
    float avg_density        = 0.0f;
    float avg_temperature    = 0.0f;

    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++)
            for (int k = -1; k <= 1; k++) {
        int3 pos = make_int3(x + k, y + j, z + i);
        int cell = (pos.z * volume_size.y + pos.y) * volume_size.x + pos.x;
        int count = p_count[cell];
        if (!count)
            continue;

        int index = p_index[cell];
        float3 coord = make_float3(x, y, z) + 0.5f;
        ComputeWeightedAverage(&avg_vel_x,       &weight_vel_x,       pos_x + index, pos_y + index, pos_z + index, coord + make_float3(-0.5f, 0.0f, 0.0f), vel_x + index,       count);
        ComputeWeightedAverage(&avg_vel_y,       &weight_vel_y,       pos_x + index, pos_y + index, pos_z + index, coord + make_float3(0.0f, -0.5f, 0.0f), vel_y + index,       count);
        ComputeWeightedAverage(&avg_vel_z,       &weight_vel_z,       pos_x + index, pos_y + index, pos_z + index, coord + make_float3(0.0f, 0.0f, -0.5f), vel_z + index,       count);

        // TODO: Share the weight data of density and temperature.
        ComputeWeightedAverage(&avg_density,     &weight_density,     pos_x + index, pos_y + index, pos_z + index, coord,                                  density + index,     count);
        ComputeWeightedAverage(&avg_temperature, &weight_temperature, pos_x + index, pos_y + index, pos_z + index, coord,                                  temperature + index, count);
    }

    uint16_t r_x = __float2half_rn(avg_vel_x       / weight_vel_x      );
    uint16_t r_y = __float2half_rn(avg_vel_y       / weight_vel_y      );
    uint16_t r_z = __float2half_rn(avg_vel_z       / weight_vel_z      );
    uint16_t r_d = __float2half_rn(avg_density     / weight_density    );
    uint16_t r_t = __float2half_rn(avg_temperature / weight_temperature);
    
    surf3Dwrite(r_x, surf_x, x * sizeof(r_x), y, z, cudaBoundaryModeTrap);
    surf3Dwrite(r_y, surf_y, x * sizeof(r_y), y, z, cudaBoundaryModeTrap);
    surf3Dwrite(r_z, surf_z, x * sizeof(r_z), y, z, cudaBoundaryModeTrap);
    surf3Dwrite(r_d, surf_d, x * sizeof(r_d), y, z, cudaBoundaryModeTrap);
    surf3Dwrite(r_t, surf_t, x * sizeof(r_t), y, z, cudaBoundaryModeTrap);

    // TODO: Diffuse the field if |total_weight| is too small(a hole near the
    //       spot).
}

__global__ void TransferFieldToGridKernel(FlipParticles particles,
                                          uint16_t* field, float3 offset,
                                          uint3 volume_size)
{
    int x = VolumeX();
    int y = VolumeY();
    int z = VolumeZ();

    if (x >= volume_size.x - 1 || y >= volume_size.y - 1 ||
            z >= volume_size.z - 1)
        return;

    if (x < 1 || y < 1 || z < 1)
        return;

    uint*     p_index = particles.particle_index_;
    uint*     p_count = particles.particle_count_;
    uint16_t* pos_x   = particles.position_x_;
    uint16_t* pos_y   = particles.position_y_;
    uint16_t* pos_z   = particles.position_z_;

    float weight = 0.0001f;
    float avg    = 0.0f;

    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++)
            for (int k = -1; k <= 1; k++) {
        int3 pos = make_int3(x + k, y + j, z + i);
        int cell = (pos.z * volume_size.y + pos.y) * volume_size.x + pos.x;
        int count = p_count[cell];
        if (!count)
            continue;

        int index = p_index[cell];
        float3 coord = make_float3(x, y, z) + 0.5f;
        ComputeWeightedAverage(&avg, &weight, pos_x + index, pos_y + index,
                               pos_z + index, coord + offset, field + index,
                               count);
    }

    uint16_t r = __float2half_rn(avg / weight);
    surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
}

__device__ void ComputeWeightedAverage_smem(float* total_value,
                                            float* total_weight,
                                            const ushort3* smem_pos, float3 p0,
                                            const uint16_t* smem_��, int count)
{
    for (int i = 0; i < count; i++) {
        const ushort3* pos = smem_pos + i;
        float x = __half2float(pos->x);
        float y = __half2float(pos->y);
        float z = __half2float(pos->z);

        float weight = DistanceWeight(x, y, z, p0.x, p0.y, p0.z);

        *total_weight += weight;
        *total_value += weight * __half2float(*(smem_�� + i));
    }
}

__global__ void TransferFieldToGridKernel_smem_overlap(FlipParticles particles,
                                                       uint16_t* field,
                                                       float3 offset,
                                                       uint3 volume_size)
{
    // Use fp16 for saving shared memory.
    // Total shared memory usage:
    // 6 * 384 * 4 * 2 + 2 * 384 * 4 * 2 = 24576.
    __shared__ ushort3  smem_pos1[384 * kMaxParticlesInCell];
    __shared__ ushort3  smem_pos2[384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_��1  [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_��2  [384 * kMaxParticlesInCell];

    ushort3*  smem_pos_ping = smem_pos1;
    ushort3*  smem_pos_pong = smem_pos2;
    uint16_t* smem_��_ping   = smem_��1;
    uint16_t* smem_��_pong   = smem_��2;

    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;
    const uint bw = blockDim.x + 16;

    int x = VolumeX();
    int y = VolumeY();

    if (x >= volume_size.x || y >= volume_size.y)
        return;

    FlipReadBlockAndHalo_32x6(smem_pos_ping, smem_��_ping, particles, field, 0,
                              tx, ty, volume_size);
    SwitchBuffers(&smem_pos_ping, &smem_pos_pong, &smem_��_ping, &smem_��_pong);

    const float kInitialWeight = 0.0001f;

    float avg_near    = 0.0f;
    float weight_near = kInitialWeight;
    float avg_mid     = 0.0f;
    float weight_mid  = kInitialWeight;
    float avg_far     = 0.0f;
    float weight_far  = kInitialWeight;

    float3 coord =               make_float3(x, y, 0) + 0.5f + offset;
    float3 coord_temp =  coord + make_float3(0, 0, 1);
    float3 coord_temp2 = coord + make_float3(0, 0, 2);
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
        uint smem_i = (ty + i + 1) * bw + tx + j + 8;
        uint strided = smem_i * kMaxParticlesInCell;
        ComputeWeightedAverage_smem(&avg_near, &weight_near,
                                    smem_pos_pong + strided, coord,
                                    smem_��_pong + strided);
        ComputeWeightedAverage_smem(&avg_mid, &weight_mid,
                                    smem_pos_pong + strided, coord_temp,
                                    smem_��_pong + strided);
    }

    FlipReadBlockAndHalo_32x6(smem_pos_ping, smem_��_ping, particles, field, 1,
                              tx, ty, volume_size);

    // For each plane, we calculate the weighted average value of the
    // farthest/middle/nearest plane of the last/current/next point, so as to
    // reuse all of the position and field data of a plane.
    uint z = 1;
    for (; z < volume_size.z - 1; z++) {
        SwitchBuffers(&smem_pos_ping, &smem_pos_pong, &smem_��_ping, &smem_��_pong);

        for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
            uint smem_i = (ty + i + 1) * bw + tx + j + 8;
            uint strided = smem_i * kMaxParticlesInCell;
            ComputeWeightedAverage_smem(&avg_near, &weight_near,
                                        smem_pos_pong + strided, coord,
                                        smem_��_pong + strided);
            ComputeWeightedAverage_smem(&avg_mid, &weight_mid,
                                        smem_pos_pong + strided, coord_temp,
                                        smem_��_pong + strided);
            ComputeWeightedAverage_smem(&avg_far, &weight_far,
                                        smem_pos_pong + strided, coord_temp2,
                                        smem_��_pong + strided);
        }
        uint16_t r = __float2half_rn(avg_near / weight_near);
        surf3Dwrite(r, surf, x * sizeof(r), y, z - 1, cudaBoundaryModeTrap);

        avg_near    = avg_mid;
        weight_near = weight_mid;
        avg_mid     = avg_far;
        weight_mid  = weight_far;
        avg_far     = 0.0f;
        weight_far  = kInitialWeight;

        coord.z       += 1.0f;
        coord_temp.z  += 1.0f;
        coord_temp2.z += 1.0f;

        FlipReadBlockAndHalo_32x6(smem_pos_ping, smem_��_ping, particles,
                                  field, z + 1, tx, ty, volume_size);
    }
    __syncthreads();

    coord = make_float3(x, y, z - 1) + 0.5f;
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
        uint smem_i = (ty + i + 1) * bw + tx + j + 8;
        uint strided = smem_i * kMaxParticlesInCell;
        ComputeWeightedAverage_smem(&avg_near, &weight_near, smem_pos_ping + strided, coord, smem_��_ping + strided);
        ComputeWeightedAverage_smem(&avg_mid, &weight_mid, smem_pos_ping + strided, coord_temp, smem_��_ping + strided);
    }
    uint16_t r = __float2half_rn(avg_near / weight_near);
    surf3Dwrite(r, surf, x * sizeof(r), y, z - 1, cudaBoundaryModeTrap);
    uint16_t r2 = __float2half_rn(avg_mid / weight_mid);
    surf3Dwrite(r2, surf, x * sizeof(r2), y, z, cudaBoundaryModeTrap);
}

__global__ void TransferFieldToGridKernel_smem(FlipParticles particles,
                                               uint16_t* field, float3 offset,
                                               uint3 volume_size)
{
    // Use fp16 for saving shared memory.
    // Total shared memory usage:
    // 6 * 384 * 4 * 2 + 2 * 384 * 4 * 2 = 24576.
    __shared__ ushort3  smem_pos1[384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_��1  [384 * kMaxParticlesInCell];

    ushort3*  smem_pos_ping = smem_pos1;
    ushort3*  smem_pos_pong = smem_pos1;
    uint16_t* smem_��_ping   = smem_��1;
    uint16_t* smem_��_pong   = smem_��1;

    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;
    const uint bw = blockDim.x + 16;

    int x = VolumeX();
    int y = VolumeY();

    if (x >= volume_size.x || y >= volume_size.y)
        return;

    FlipReadBlockAndHalo_32x6(smem_pos_ping, smem_��_ping, particles, field, 0,
                              tx, ty, volume_size);
    SwitchBuffers(&smem_pos_ping, &smem_pos_pong, &smem_��_ping, &smem_��_pong);

    const float kInitialWeight = 0.0001f;

    float avg_near    = 0.0f;
    float weight_near = kInitialWeight;
    float avg_mid     = 0.0f;
    float weight_mid  = kInitialWeight;
    float avg_far     = 0.0f;
    float weight_far  = kInitialWeight;

    float3 coord =               make_float3(x, y, 0) + 0.5f + offset;
    float3 coord_temp =  coord + make_float3(0, 0, 1);
    float3 coord_temp2 = coord + make_float3(0, 0, 2);
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
        uint smem_i = (ty + i + 1) * bw + tx + j + 8;
        uint strided = smem_i * kMaxParticlesInCell;
        ComputeWeightedAverage_smem(&avg_near, &weight_near,
                                    smem_pos_pong + strided, coord,
                                    smem_��_pong + strided);
        ComputeWeightedAverage_smem(&avg_mid, &weight_mid,
                                    smem_pos_pong + strided, coord_temp,
                                    smem_��_pong + strided);
    }

    __syncthreads();
    FlipReadBlockAndHalo_32x6(smem_pos_ping, smem_��_ping, particles, field, 1,
                              tx, ty, volume_size);

    // For each plane, we calculate the weighted average value of the
    // farthest/middle/nearest plane of the last/current/next point, so as to
    // reuse all of the position and field data of a plane.
    uint z = 1;
    for (; z < volume_size.z - 1; z++) {
        SwitchBuffers(&smem_pos_ping, &smem_pos_pong, &smem_��_ping,
                      &smem_��_pong);

        for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
            uint smem_i = (ty + i + 1) * bw + tx + j + 8;
            uint strided = smem_i * kMaxParticlesInCell;
            ComputeWeightedAverage_smem(&avg_near, &weight_near,
                                        smem_pos_pong + strided, coord,
                                        smem_��_pong + strided);
            ComputeWeightedAverage_smem(&avg_mid, &weight_mid,
                                        smem_pos_pong + strided, coord_temp,
                                        smem_��_pong + strided);
            ComputeWeightedAverage_smem(&avg_far, &weight_far,
                                        smem_pos_pong + strided, coord_temp2,
                                        smem_��_pong + strided);
        }
        uint16_t r = __float2half_rn(avg_near / weight_near);
        surf3Dwrite(r, surf, x * sizeof(r), y, z - 1, cudaBoundaryModeTrap);

        avg_near    = avg_mid;
        weight_near = weight_mid;
        avg_mid     = avg_far;
        weight_mid  = weight_far;
        avg_far     = 0.0f;
        weight_far  = kInitialWeight;

        coord.z       += 1.0f;
        coord_temp.z  += 1.0f;
        coord_temp2.z += 1.0f;

        __syncthreads();
        FlipReadBlockAndHalo_32x6(smem_pos_ping, smem_��_ping, particles, field,
                                  z + 1, tx, ty, volume_size);
    }
    __syncthreads();

    coord = make_float3(x, y, z - 1) + 0.5f;
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
        uint smem_i = (ty + i + 1) * bw + tx + j + 8;
        uint strided = smem_i * kMaxParticlesInCell;
        ComputeWeightedAverage_smem(&avg_near, &weight_near,
                                    smem_pos_ping + strided, coord,
                                    smem_��_ping + strided);
        ComputeWeightedAverage_smem(&avg_mid, &weight_mid,
                                    smem_pos_ping + strided, coord_temp,
                                    smem_��_ping + strided);
    }
    uint16_t r = __float2half_rn(avg_near / weight_near);
    surf3Dwrite(r, surf, x * sizeof(r), y, z - 1, cudaBoundaryModeTrap);
    uint16_t r2 = __float2half_rn(avg_mid / weight_mid);
    surf3Dwrite(r2, surf, x * sizeof(r2), y, z, cudaBoundaryModeTrap);
}

__global__ void TransferToGridKernel_smem_overlap(FlipParticles particles,
                                                  uint3 volume_size)
{
    // Use fp16 for saving shared memory.
    // Total shared memory usage:
    // 2 * 384 * 4 * 16 = 49152.
    __shared__ uint16_t smem_pos_x1      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_pos_x2      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_pos_y1      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_pos_y2      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_pos_z1      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_pos_z2      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_vel_x1      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_vel_x2      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_vel_y1      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_vel_y2      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_vel_z1      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_vel_z2      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_density1    [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_density2    [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_temperature1[384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_temperature2[384 * kMaxParticlesInCell];

    ParticleFields ping = {
        smem_pos_x1,
        smem_pos_y1,
        smem_pos_z1,
        smem_vel_x1,
        smem_vel_y1,
        smem_vel_z1,
        smem_density1,
        smem_temperature1,
    };

    ParticleFields pong = {
        smem_pos_x2,
        smem_pos_y2,
        smem_pos_z2,
        smem_vel_x2,
        smem_vel_y2,
        smem_vel_z2,
        smem_density2,
        smem_temperature2,
    };

    ParticleFields* smem_ping = &ping;
    ParticleFields* smem_pong = &pong;

    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;
    const uint bw = blockDim.x + 16;

    int x = VolumeX();
    int y = VolumeY();

    if (x >= volume_size.x || y >= volume_size.y)
        return;

    FlipReadBlockAndHalo_32x6_all(smem_ping, particles, 0, tx, ty, volume_size);
    SwitchBuffers_all(smem_ping, smem_pong);

    FieldWeightAndWeightedSum w_near;
    FieldWeightAndWeightedSum w_mid;
    FieldWeightAndWeightedSum w_far;
    ResetWeight(&w_near);
    ResetWeight(&w_mid);
    ResetWeight(&w_far);

    float3 coord       =         make_float3(x, y, 0) + 0.5f;
    float3 coord_temp  = coord + make_float3(0, 0, 1);
    float3 coord_temp2 = coord + make_float3(0, 0, 2);
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
        uint smem_i = (ty + i + 1) * bw + tx + j + 8;
        uint strided = smem_i * kMaxParticlesInCell;
        ComputeWeightedAverage_smem_All(&w_near, smem_pong, strided, coord);
        ComputeWeightedAverage_smem_All(&w_mid,  smem_pong, strided, coord_temp);
    }

    FlipReadBlockAndHalo_32x6_all(smem_ping, particles, 1, tx, ty, volume_size);

    // For each plane, we calculate the weighted average value of the
    // farthest/middle/nearest plane of the last/current/next point, so as to
    // reuse all of the position and field data of a plane.
    uint z = 1;
    for (; z < volume_size.z - 1; z++) {
        SwitchBuffers_all(smem_ping, smem_pong);

        for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
            uint smem_i = (ty + i + 1) * bw + tx + j + 8;
            uint strided = smem_i * kMaxParticlesInCell;
            ComputeWeightedAverage_smem_All(&w_near, smem_pong, strided, coord);
            ComputeWeightedAverage_smem_All(&w_mid,  smem_pong, strided, coord_temp);
            ComputeWeightedAverage_smem_All(&w_far,  smem_pong, strided, coord_temp2);
        }
        SaveToSurface(w_near, x, y, z - 1);

        w_near = w_mid;
        w_mid = w_far;
        ResetWeight(&w_far);

        coord.z       += 1.0f;
        coord_temp.z  += 1.0f;
        coord_temp2.z += 1.0f;

        FlipReadBlockAndHalo_32x6_all(smem_ping, particles, z + 1, tx, ty, volume_size);
    }
    __syncthreads();

    coord = make_float3(x, y, z - 1) + 0.5f;
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
        uint smem_i = (ty + i + 1) * bw + tx + j + 8;
        uint strided = smem_i * kMaxParticlesInCell;
        ComputeWeightedAverage_smem_All(&w_near, smem_ping, strided, coord);
        ComputeWeightedAverage_smem_All(&w_mid,  smem_ping, strided, coord_temp);
    }
    SaveToSurface(w_near, x, y, z - 1);
    SaveToSurface(w_mid,  x, y, z);
}

// TODO: We can save shared memory by storing the weights instead of positions.
// TODO: Bank conflict optimization.
//
// As using shared memory to reduce the latency of data access, there are some
// factors becoming our major concern:
// 1. fp16 or fp32;
// 2. field by field or all fields in one time;
// 3. 8/6/4 particles per cell;
// 4. ping-pong scheme to reduce the data dependency between steps, so that
//    the hardware can overlap the arithmetic operations and the next-step 
//    memory transaction.
//
// This kernel is so large that uses 96 registers by default, which means on a
// Maxwell gpu the number of blocks is limited to 3:
//    64K registers / (192 threads * 96) = 3.55
// So I manually configure the maxrregcount to 80 to increase the occupancy.
//
// Kepler gpu is not affected by this issue since its thread register limit
// is 63.
//
// TODO: A big issue of this kernel is that computation and memory-fetch
//       overlapping is shutdown due to the lack of shared memory. The only
//       hope is a high occupancy would hide the latency.
//
// This kernel is fast, but also memory-intensive, that not suitable for
// some older gpus such as Kepler or Fermi.
//
// TODO: Further reduce the shared-memory allocation for higher occupancy/
//       low bank conflict.
__global__ void TransferToGridKernel_smem(FlipParticles particles,
                                          uint3 volume_size)
{
    // Use fp16 for saving shared memory.
    // Total shared memory usage:
    // 2 * 384 * 4 * 8 = 24576.
    __shared__ uint16_t smem_pos_x1      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_pos_y1      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_pos_z1      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_vel_x1      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_vel_y1      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_vel_z1      [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_density1    [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_temperature1[384 * kMaxParticlesInCell];

    ParticleFields smem = {
        smem_pos_x1,
        smem_pos_y1,
        smem_pos_z1,
        smem_vel_x1,
        smem_vel_y1,
        smem_vel_z1,
        smem_density1,
        smem_temperature1,
    };

    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;
    const uint bw = blockDim.x + 16;

    int x = VolumeX();
    int y = VolumeY();

    if (x >= volume_size.x || y >= volume_size.y)
        return;

    FlipReadBlockAndHalo_32x6_all(&smem, particles, 0, tx, ty, volume_size);
    __syncthreads();

    FieldWeightAndWeightedSum w_near;
    FieldWeightAndWeightedSum w_mid;
    FieldWeightAndWeightedSum w_far;
    ResetWeight(&w_near);
    ResetWeight(&w_mid);
    ResetWeight(&w_far);

    float3 coord       =         make_float3(x, y, 0) + 0.5f;
    float3 coord_temp  = coord + make_float3(0, 0, 1);
    float3 coord_temp2 = coord + make_float3(0, 0, 2);
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
        uint smem_i = (ty + i + 1) * bw + tx + j + 8;
        uint strided = smem_i * kMaxParticlesInCell;
        ComputeWeightedAverage_smem_All(&w_near, &smem, strided, coord);
        ComputeWeightedAverage_smem_All(&w_mid,  &smem, strided, coord_temp);
    }

    __syncthreads();
    FlipReadBlockAndHalo_32x6_all(&smem, particles, 1, tx, ty, volume_size);

    // For each plane, we calculate the weighted average value of the
    // farthest/middle/nearest plane of the last/current/next point, so as to
    // reuse all of the position and field data of a plane.
    uint z = 1;
    for (; z < volume_size.z - 1; z++) {
        __syncthreads();

        for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
            uint smem_i = (ty + i + 1) * bw + tx + j + 8;
            uint strided = smem_i * kMaxParticlesInCell;
            ComputeWeightedAverage_smem_All(&w_near, &smem, strided, coord);
            ComputeWeightedAverage_smem_All(&w_mid,  &smem, strided, coord_temp);
            ComputeWeightedAverage_smem_All(&w_far,  &smem, strided, coord_temp2);
        }
        SaveToSurface(w_near, x, y, z - 1);

        w_near = w_mid;
        w_mid = w_far;
        ResetWeight(&w_far);

        coord.z       += 1.0f;
        coord_temp.z  += 1.0f;
        coord_temp2.z += 1.0f;

        __syncthreads();
        FlipReadBlockAndHalo_32x6_all(&smem, particles, z + 1, tx, ty, volume_size);
    }
    __syncthreads();

    coord = make_float3(x, y, z - 1) + 0.5f;
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
        uint smem_i = (ty + i + 1) * bw + tx + j + 8;
        uint strided = smem_i * kMaxParticlesInCell;
        ComputeWeightedAverage_smem_All(&w_near, &smem, strided, coord);
        ComputeWeightedAverage_smem_All(&w_mid,  &smem, strided, coord_temp);
    }
    SaveToSurface(w_near, x, y, z - 1);
    SaveToSurface(w_mid,  x, y, z);
}

// =============================================================================

namespace kern_launcher
{
void TransferToGrid_smem(cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z,
                         cudaArray* density, cudaArray* temperature,
                         const FlipParticles& particles, uint3 volume_size,
                         BlockArrangement* ba)
{
    uint16_t* fields[] = {
        particles.velocity_x_,
        particles.velocity_y_,
        particles.velocity_z_,
        particles.density_,
        particles.temperature_
    };

    float3 offsets[] = {
        make_float3(-0.5f, 0.0f, 0.0f),
        make_float3(0.0f, -0.5f, 0.0f),
        make_float3(0.0f, 0.0f, -0.5f),
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 0.0f),
    };

    cudaArray* surfs[] = {
        vel_x,
        vel_y,
        vel_z,
        density,
        temperature,
    };

    dim3 block;
    dim3 grid;
    int smem = 1;
    if (smem) {
        dim3 block(32, 6, 1);
        dim3 grid((volume_size.x + block.x - 1) / block.x,
                  (volume_size.y + block.y - 1) / block.y,
                  1);

        for (int i = 0; i < sizeof(fields) / sizeof(*fields); i++) {
            if (BindCudaSurfaceToArray(&surf, surfs[i]) != cudaSuccess)
                return;

            TransferFieldToGridKernel_smem<<<grid, block>>>(particles,
                                                            fields[i],
                                                            offsets[i],
                                                            volume_size);
            DCHECK_KERNEL();
        }
    } else {
        ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
        for (int i = 0; i < sizeof(fields) / sizeof(*fields); i++) {
            if (BindCudaSurfaceToArray(&surf, surfs[i]) != cudaSuccess)
                return;

            TransferFieldToGridKernel<<<grid, block>>>(particles, fields[i],
                                                       offsets[i], volume_size);
            DCHECK_KERNEL();
        }
    }
}

void TransferToGridOpt1(cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z,
                        cudaArray* density, cudaArray* temperature,
                        const FlipParticles& particles, uint3 volume_size,
                        BlockArrangement* ba)
{
    /*
    1. use shared memory + 2.5d to calculate the weighted_avg + total_weight,
       but only consider the first particle within a cell.
    2. repeat the same procedure, but take the 2nd, 3rd, 4th... particle into
       account.
    3. update all fields = weighted_avg / total_weight.
    */
}

void TransferToGrid(cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z,
                    cudaArray* density, cudaArray* temperature,
                    const FlipParticles& particles, uint3 volume_size,
                    BlockArrangement* ba)
{
    bool field_by_field = false;
    if (field_by_field) {
        TransferToGrid_smem(vel_x, vel_y, vel_z, density, temperature,
                            particles, volume_size, ba);
    } else {
        if (BindCudaSurfaceToArray(&surf_x, vel_x) != cudaSuccess)
            return;

        if (BindCudaSurfaceToArray(&surf_y, vel_y) != cudaSuccess)
            return;

        if (BindCudaSurfaceToArray(&surf_z, vel_z) != cudaSuccess)
            return;

        if (BindCudaSurfaceToArray(&surf_d, density) != cudaSuccess)
            return;

        if (BindCudaSurfaceToArray(&surf_t, temperature) != cudaSuccess)
            return;

        bool smem = true;
        if (smem) {
            dim3 block(32, 6, 1);
            dim3 grid((volume_size.x + block.x - 1) / block.x,
                      (volume_size.y + block.y - 1) / block.y,
                      1);
        
            TransferToGridKernel_smem<<<grid, block>>>(particles, volume_size);
            DCHECK_KERNEL();
        } else {
            dim3 block;
            dim3 grid;
            ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
            TransferToGridKernel<<<grid, block>>>(particles, volume_size);
            DCHECK_KERNEL();
        }
    }
}
}
