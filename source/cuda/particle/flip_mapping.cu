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
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_xp;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_yp;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_zp;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_d;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_t;

const uint32_t kMaxParticlesInCell = 6;

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

__device__ void ReadPosAndField(ushort3* smem_pos, uint16_t* smem_¦Õ,
                                uint8_t* smem_count,
                                const FlipParticles& particles,
                                const uint16_t* field, uint grid_x, uint grid_y,
                                uint grid_z, uint smem_i, uint3 volume_size)
{
    uint32_t* p_index = particles.particle_index_;
    uint32_t* p_count = particles.particle_count_;
    uint16_t* pos_x   = particles.position_x_;
    uint16_t* pos_y   = particles.position_y_;
    uint16_t* pos_z   = particles.position_z_;

    int cell = (grid_z * volume_size.y + grid_y) * volume_size.x + grid_x;
    int count = p_count[cell];

    smem_count[smem_i] = static_cast<uint8_t>(count);

    int index = p_index[cell];
    pos_x += index;
    pos_y += index;
    pos_z += index;
    field += index;

    for (int i = 0; i < count; i++) {
        smem_pos[smem_i * kMaxParticlesInCell + i] = make_ushort3(*(pos_x + i), *(pos_y + i), *(pos_z + i));
        smem_¦Õ  [smem_i * kMaxParticlesInCell + i] = *(field + i);
    }
}

__device__ void FlipReadBlockAndHalo_32x6(ushort3* smem_pos, uint16_t* smem_¦Õ,
                                          uint8_t* smem_count,
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

    // TODO: Further tighten the scope of memory read?
    if (grid_x < 0 || grid_x >= volume_size.x) {
        smem_count[linear_index] = 0;
        smem_count[linear_index + 192] = 0;
    } else {
        if (grid_y1 >= 0 && grid_y1 < volume_size.y)
            ReadPosAndField(smem_pos, smem_¦Õ, smem_count, particles, field,
                            grid_x, grid_y1, grid_z, linear_index, volume_size);
        else
            smem_count[linear_index] = 0;

        if (grid_y2 >= 0 && grid_y2 < volume_size.y)
            ReadPosAndField(smem_pos, smem_¦Õ, smem_count, particles, field,
                            grid_x, grid_y2, grid_z, linear_index + 192,
                            volume_size);
        else
            smem_count[linear_index + 192] = 0;
    }
}

__device__ void ComputeWeightedAverage_smem(float* total_value,
                                            float* total_weight,
                                            const ushort3* smem_pos, float3 p0,
                                            const uint16_t* smem_¦Õ, int count)
{
    for (int i = 0; i < count; i++) {
        const ushort3* pos = smem_pos + i;
        float x = __half2float(pos->x);
        float y = __half2float(pos->y);
        float z = __half2float(pos->z);

        float weight = DistanceWeight(x, y, z, p0.x, p0.y, p0.z);

        *total_weight += weight;
        *total_value += weight * __half2float(*(smem_¦Õ + i));
    }
}

__device__ inline void AdvanceBuffer(ushort3** pos_ping, ushort3** pos_pong,
                                     uint16_t** ¦Õ_ping, uint16_t** ¦Õ_pong,
                                     uint8_t** count_ping, uint8_t** count_pong)
{
    __syncthreads();

    ushort3* pos_temp = *pos_ping;
    *pos_ping = *pos_pong;
    *pos_pong = pos_temp;

    uint16_t* ¦Õ_temp = *¦Õ_ping;
    *¦Õ_ping = *¦Õ_pong;
    *¦Õ_pong = ¦Õ_temp;

    uint8_t* count_temp = *count_ping;
    *count_ping = *count_pong;
    *count_pong = count_temp;
}

// TODO: We can save shared memory by storing the weights instead of positions.
// TODO: Bank conflict optimization.
// TODO: Use negative-position value to substitute |smem_count|.
// TODO: Expand the plane size if shared memory is greater than 48KB.
//
// As using shared memory to reduce the latency of data access, there are some
// factors becoming our major concern:
// 1. fp16 or fp32;
// 2. field by field or all fields in one time;
// 3. 8/6/4 particles per cell;
// 4. ping-pong scheme to reduce the data dependency between steps, so that
//    the hardware can overlap the arithmetic operations and the next-step 
//    memory transaction.
__global__ void TransferFieldToGridKernel_smem(FlipParticles particles,
                                               uint16_t* field, float3 offset,
                                               uint3 volume_size)
{
    // Use fp16 for saving shared memory.
    // Total shared memory usage:
    // 6 * 384 * 6 * 2 + 2 * 384 * 6 * 2 + 384 * 2 = 37632.
    __shared__ ushort3  smem_pos1  [384 * kMaxParticlesInCell];
    __shared__ ushort3  smem_pos2  [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_¦Õ1    [384 * kMaxParticlesInCell];
    __shared__ uint16_t smem_¦Õ2    [384 * kMaxParticlesInCell];
    __shared__ uint8_t  smem_count1[384];
    __shared__ uint8_t  smem_count2[384];

    ushort3*  smem_pos_ping   = smem_pos1;
    ushort3*  smem_pos_pong   = smem_pos2;
    uint16_t* smem_¦Õ_ping     = smem_¦Õ1;
    uint16_t* smem_¦Õ_pong     = smem_¦Õ2;
    uint8_t*  smem_count_ping = smem_count1;
    uint8_t*  smem_count_pong = smem_count2;

    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;
    const uint bw = blockDim.x + 16;

    int x = VolumeX();
    int y = VolumeY();

    if (x >= volume_size.x || y >= volume_size.y)
        return;

    FlipReadBlockAndHalo_32x6(smem_pos_ping, smem_¦Õ_ping, smem_count_ping,
                              particles, field, 0, tx, ty, volume_size);
    AdvanceBuffer(&smem_pos_ping, &smem_pos_pong, &smem_¦Õ_ping, &smem_¦Õ_pong,
                  &smem_count_ping, &smem_count_pong);

    const float kInitialWeight = 0.0001f;

    float avg_near    = 0.0f;
    float weight_near = kInitialWeight;
    float avg_mid     = 0.0f;
    float weight_mid  = kInitialWeight;
    float avg_far     = 0.0f;
    float weight_far  = kInitialWeight;

    float3 coord = make_float3(x, y, 0) + 0.5f + offset;
    float3 coord_temp = coord + make_float3(0, 0, 1);
    float3 coord_temp2 = coord + make_float3(0, 0, 2);
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
        uint smem_i = (ty + i + 1) * bw + tx + j + 8;
        uint strided = smem_i * kMaxParticlesInCell;
        ComputeWeightedAverage_smem(&avg_near, &weight_near,
                                    smem_pos_pong + strided, coord,
                                    smem_¦Õ_pong + strided,
                                    smem_count_pong[smem_i]);

        ComputeWeightedAverage_smem(&avg_mid, &weight_mid,
                                    smem_pos_pong + strided, coord_temp,
                                    smem_¦Õ_pong + strided,
                                    smem_count_pong[smem_i]);
    }

    FlipReadBlockAndHalo_32x6(smem_pos_ping, smem_¦Õ_ping, smem_count_ping,
                              particles, field, 1, tx, ty, volume_size);

    // For each plane, we calculate the weighted average value of the
    // farthest/middle/nearest plane of the last/current/next point, so as to
    // reuse all of the position and field data of a plane.
    uint z = 1;
    for (; z < volume_size.z - 1; z++) {
        AdvanceBuffer(&smem_pos_ping, &smem_pos_pong, &smem_¦Õ_ping,
                      &smem_¦Õ_pong, &smem_count_ping, &smem_count_pong);

        for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
            uint smem_i = (ty + i + 1) * bw + tx + j + 8;
            uint strided = smem_i * kMaxParticlesInCell;
            ComputeWeightedAverage_smem(&avg_near, &weight_near,
                                        smem_pos_pong + strided, coord,
                                        smem_¦Õ_pong + strided,
                                        smem_count_pong[smem_i]);
            ComputeWeightedAverage_smem(&avg_mid, &weight_mid,
                                        smem_pos_pong + strided, coord_temp,
                                        smem_¦Õ_pong + strided,
                                        smem_count_pong[smem_i]);
            ComputeWeightedAverage_smem(&avg_far, &weight_far,
                                        smem_pos_pong + strided, coord_temp2,
                                        smem_¦Õ_pong + strided,
                                        smem_count_pong[smem_i]);
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

        FlipReadBlockAndHalo_32x6(smem_pos_ping, smem_¦Õ_ping, smem_count_ping,
                                  particles, field, z + 1, tx, ty, volume_size);
    }
    __syncthreads();

    coord = make_float3(x, y, z - 1) + 0.5f;
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
        uint smem_i = (ty + i + 1) * bw + tx + j + 8;
        uint strided = smem_i * kMaxParticlesInCell;
        ComputeWeightedAverage_smem(&avg_near, &weight_near,
                                    smem_pos_ping + strided, coord,
                                    smem_¦Õ_ping + strided,
                                    smem_count_ping[smem_i]);
        ComputeWeightedAverage_smem(&avg_mid, &weight_mid,
                                    smem_pos_ping + strided, coord_temp,
                                    smem_¦Õ_ping + strided,
                                    smem_count_ping[smem_i]);
    }
    uint16_t r = __float2half_rn(avg_near / weight_near);
    surf3Dwrite(r, surf, x * sizeof(r), y, z - 1, cudaBoundaryModeTrap);
    uint16_t r2 = __float2half_rn(avg_mid / weight_mid);
    surf3Dwrite(r2, surf, x * sizeof(r2), y, z, cudaBoundaryModeTrap);
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
    bool field_by_field = true;
    if (field_by_field) {
        TransferToGrid_smem(vel_x, vel_y, vel_z, density, temperature,
                            particles, volume_size, ba);
        return;
    }

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

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    TransferToGridKernel<<<grid, block>>>(particles, volume_size);
    DCHECK_KERNEL();
}
}
