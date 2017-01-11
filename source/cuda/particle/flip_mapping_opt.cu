//
// Hypermorph - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Hypermorph license (* see part 1 below)
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

#include "cuda/block_arrangement.h"
#include "cuda/cuda_common_host.h"
#include "cuda/cuda_common_kern.h"
#include "cuda/cuda_debug.h"
#include "cuda/particle/flip_common.cuh"
#include "flip.h"

surface<void, cudaSurfaceType3D> surf_x;
surface<void, cudaSurfaceType3D> surf_y;
surface<void, cudaSurfaceType3D> surf_z;
surface<void, cudaSurfaceType3D> surf_d;
surface<void, cudaSurfaceType3D> surf_t;

namespace flip
{
__device__ inline float WeightKernel(float p, float p0)
{
    return 1.0f - __saturatef(fabsf(p - p0));
}

__device__ float DistanceWeight(const float3& p, const float3& p0)
{
    return WeightKernel(p.x, p0.x) * WeightKernel(p.y, p0.y) * WeightKernel(p.z, p0.z);
}

__device__ void CommitToTexture(const FlipParticles& particles,
                                const int smem_slice_size,
                                const int block_width,
                                const float* smem_vx_wsum,
                                const float* smem_vy_wsum,
                                const float* smem_vz_wsum,
                                const float* smem_d_wsum,
                                const float* smem_t_wsum,
                                const float* smem_vx_weight,
                                const float* smem_vy_weight,
                                const float* smem_vz_weight,
                                const float* smem_c_weight,
                                const int2& block_base_grid_coord, int z)
{
    int thread_id = threadIdx.x;
    if (thread_id > smem_slice_size)
        return;

    int x = block_base_grid_coord.x + thread_id % block_width;
    int y = block_base_grid_coord.y + thread_id / block_width;
    const float kInitialWeight = 0.00001f;

    uint16_t rx = __float2half_rn(smem_vx_wsum[thread_id] / (smem_vx_weight[thread_id] + kInitialWeight));
    surf3Dwrite(rx, surf_x, x * sizeof(rx), y, z, cudaBoundaryModeTrap);
    uint16_t ry = __float2half_rn(smem_vy_wsum[thread_id] / (smem_vy_weight[thread_id] + kInitialWeight));
    surf3Dwrite(ry, surf_y, x * sizeof(ry), y, z, cudaBoundaryModeTrap);
    uint16_t rz = __float2half_rn(smem_vz_wsum[thread_id] / (smem_vz_weight[thread_id] + kInitialWeight));
    surf3Dwrite(rz, surf_z, x * sizeof(rz), y, z, cudaBoundaryModeTrap);
    uint16_t rd = __float2half_rn(smem_d_wsum[thread_id] / (smem_c_weight[thread_id] + kInitialWeight));
    surf3Dwrite(rd, surf_d, x * sizeof(rd), y, z, cudaBoundaryModeTrap);
    uint16_t rt = __float2half_rn(smem_t_wsum[thread_id] / (smem_c_weight[thread_id] + kInitialWeight));
    surf3Dwrite(rt, surf_t, x * sizeof(rt), y, z, cudaBoundaryModeTrap);
}

__device__ void VoxelizeParticle(float* smem_vx_wsum, float* smem_vy_wsum,
                                 float* smem_vz_wsum, float* smem_d_wsum,
                                 float* smem_t_wsum, float* smem_vx_weight,
                                 float* smem_vy_weight, float* smem_vz_weight,
                                 float* smem_c_weight, const int block_width,
                                 const int block_height, const int halo_size, 
                                 const FlipParticles& particles,
                                 int particle_id, const int2& slice_coord,
                                 int z, int sz_offset)
{
    float3 coord = flip::Position(particles, particle_id);
    float3 corner = floorf(coord);

    // Classify the particle by its situation of adjacency.
    float3 offset = floorf((coord - corner) * 2.0f);
    int3 offset_i = Float2Int(offset);

    float3 center = corner + 0.5f;
    int3 base_block_coord = make_int3(slice_coord.x - halo_size, slice_coord.y - halo_size, z - halo_size);
    for (int i = -1; i < 1; i++) for (int j = -1; j < 1; j++) for (int k = -1; k < 1; k++) {
        // Careful of the sign of the numbers.
        int3 iter_offset_i = make_int3(k, j, i) + offset_i;
        int3 block_coord = base_block_coord + iter_offset_i;

        int sz = iter_offset_i.z + sz_offset;
        if (sz > 2)
            sz -= 3;

        int si = block_coord.y * block_width + block_coord.x;
        si += sz * block_width * block_height;

        if (si < 0 || si >= block_width * block_height * 3) // TODO: Use constant.
            continue;

        float3 iter_offset = make_float3(k, j, i);
        float3 ref_point = center + iter_offset + offset;
        float3 ref_point_x = center - iter_offset + make_float3(1.0f,     offset.y, offset.z);
        float3 ref_point_y = center - iter_offset + make_float3(offset.x, 1.0f,     offset.z);
        float3 ref_point_z = center - iter_offset + make_float3(offset.x, offset.y, 1.0f);

        float w  = DistanceWeight(coord, ref_point);
        float wx = DistanceWeight(coord, ref_point_x);
        float wy = DistanceWeight(coord, ref_point_y);
        float wz = DistanceWeight(coord, ref_point_z);

        atomicAdd(smem_c_weight  + si, w);
        atomicAdd(smem_vx_weight + si, wx);
        atomicAdd(smem_vy_weight + si, wy);
        atomicAdd(smem_vz_weight + si, wz);

        float vx = __half2float(particles.velocity_x_[particle_id]);
        float vy = __half2float(particles.velocity_y_[particle_id]);
        float vz = __half2float(particles.velocity_z_[particle_id]);

        atomicAdd(smem_vx_wsum + si, wx * vx);
        atomicAdd(smem_vy_wsum + si, wy * vy);
        atomicAdd(smem_vz_wsum + si, wz * vz);

        float density = __half2float(particles.density_[particle_id]);
        atomicAdd(smem_d_wsum + si, w * density);

        float temperature = __half2float(particles.temperature_[particle_id]);
        atomicAdd(smem_t_wsum + si, w * temperature);
    }
}

__device__ void ProcessSegment(float* smem_vx_wsum, float* smem_vy_wsum,
                               float* smem_vz_wsum, float* smem_d_wsum,
                               float* smem_t_wsum, float* smem_vx_weight,
                               float* smem_vy_weight, float* smem_vz_weight,
                               float* smem_c_weight, const int block_width,
                               const int block_height,const int slice_width,
                               const int halo_size, const uint3& volume_size,
                               const int2& block_base_grid_coord,
                               const int block_linear_cell_id, const int z,
                               const FlipParticles& particles, int sz_offset)
{
    int2 slice_coord = make_int2(block_linear_cell_id % slice_width,
                                 block_linear_cell_id / slice_width);
    int2 grid_coord = block_base_grid_coord - halo_size + slice_coord;
    if (grid_coord.x < 0 || grid_coord.x >= volume_size.x ||
            grid_coord.y < 0 || grid_coord.y >= volume_size.y)
        return;

    int particle_id = ParticleIndex(
        (z * volume_size.y + grid_coord.y) * volume_size.x + grid_coord.x);
    particle_id += threadIdx.x % kMaxNumParticlesPerCell;

    // Active particles should be consecutive.
    if (IsParticleUndefined(particles.position_x_[particle_id]))
        return;

    VoxelizeParticle(smem_vx_wsum, smem_vy_wsum, smem_vz_wsum, smem_d_wsum,
                     smem_t_wsum, smem_vx_weight, smem_vy_weight,
                     smem_vz_weight, smem_c_weight, block_width, block_height,
                     halo_size,  particles, particle_id, slice_coord, z,
                     sz_offset);
}

__global__ void TransferToGridLopKernel(FlipParticles particles,
                                        uint3 volume_size)
{
    const int block_width = 32;
    const int block_height = 8;
    const int halo_size = 1;
    const int slice_width = block_width + halo_size * 2;
    const int slice_height = block_height + halo_size * 2;
    const int segment_count = 4;
    const int segment_stride = slice_width * slice_height / segment_count;
    const int smem_slice_size = block_width * block_height;

    __shared__ float smem_vx_wsum  [smem_slice_size * 3];
    __shared__ float smem_vy_wsum  [smem_slice_size * 3];
    __shared__ float smem_vz_wsum  [smem_slice_size * 3];
    __shared__ float smem_d_wsum   [smem_slice_size * 3];
    __shared__ float smem_t_wsum   [smem_slice_size * 3];
    __shared__ float smem_vx_weight[smem_slice_size * 3];
    __shared__ float smem_vy_weight[smem_slice_size * 3];
    __shared__ float smem_vz_weight[smem_slice_size * 3];
    __shared__ float smem_c_weight [smem_slice_size * 3];

    int thread_id = threadIdx.x;
    if (thread_id >= segment_stride)
        return; // The number of abandoned threads are always less than that
                // of a warp, so that we could just omit the thread
                // synchronization within them.

    int block_linear_cell_id = thread_id / kMaxNumParticlesPerCell;
    int2 block_base_grid_coord = make_int2(block_width * blockIdx.x,
                                           block_height * blockIdx.y);

    int z = 0;
    int sz_offset = 3;
    for (int seg = 0; seg < segment_count; seg++) {
        ProcessSegment(smem_vx_wsum, smem_vy_wsum, smem_vz_wsum,
                        smem_d_wsum, smem_t_wsum, smem_vx_weight,
                        smem_vy_weight, smem_vz_weight, smem_c_weight,
                        block_width, block_height, slice_width, halo_size,
                        volume_size, block_base_grid_coord,
                        block_linear_cell_id, z, particles, sz_offset);
        block_linear_cell_id += segment_stride;
    }

    for (z = 1; z < volume_size.z; z++) {
        if (++sz_offset > 3)
            sz_offset = 1;

        block_base_grid_coord = make_int2(block_width * blockIdx.x,
                                          block_height * blockIdx.y);
        for (int seg = 0; seg < segment_count; seg++) {
            ProcessSegment(smem_vx_wsum, smem_vy_wsum, smem_vz_wsum,
                           smem_d_wsum, smem_t_wsum, smem_vx_weight,
                           smem_vy_weight, smem_vz_weight, smem_c_weight,
                           block_width, block_height, slice_width, halo_size,
                           volume_size, block_base_grid_coord,
                           block_linear_cell_id, z, particles, sz_offset);
            block_linear_cell_id += segment_stride;
        }

        int so = (sz_offset - 1) * smem_slice_size;
        __syncthreads();
        CommitToTexture(particles, smem_slice_size, block_width,
                        smem_vx_wsum + so, smem_vy_wsum + so, smem_vz_wsum + so,
                        smem_d_wsum + so, smem_t_wsum + so, smem_vx_weight + so,
                        smem_vy_weight + so, smem_vz_weight + so,
                        smem_c_weight + so, block_base_grid_coord, z - 1);
    }
    if (++sz_offset > 3)
        sz_offset = 1;

    int so = (sz_offset - 1) * smem_slice_size;
    CommitToTexture(particles, smem_slice_size, block_width,
                    smem_vx_wsum + so, smem_vy_wsum + so, smem_vz_wsum + so,
                    smem_d_wsum + so, smem_t_wsum + so, smem_vx_weight + so,
                    smem_vy_weight + so, smem_vz_weight + so,
                    smem_c_weight + so, block_base_grid_coord, z - 1);
}
}

namespace kern_launcher
{
void TransferToGridOpt(cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z,
                       cudaArray* density, cudaArray* temperature,
                       const FlipParticles& particles,
                       uint3 volume_size, BlockArrangement* ba)
{
    assert(
        static_cast<uint>(particles.num_of_particles_) >=
            volume_size.x * volume_size.y * volume_size.z *
                kMaxNumParticlesPerCell);

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

    const int block_width = 32;
    const int block_height = 8;
    dim3 block(block_width * block_height, 1, 1);
    dim3 grid((volume_size.x + block_width - 1) / block_width,
              (volume_size.y + block_height - 1) / block_height,
              1);

    int sm_size = ba->GetSharedMemPerSMInKB();
    if (sm_size >= 96)
        flip::TransferToGridLopKernel<<<grid, block>>>(particles, volume_size);

    DCHECK_KERNEL();
}
}
