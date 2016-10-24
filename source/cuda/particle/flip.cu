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
#include "flip.h"

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

const uint32_t kCellUndefined = static_cast<uint32_t>(-1);
const uint32_t kMaxNumParticlesPerCell = 4;
const uint32_t kMinNumParticlesPerCell = 2;
const uint32_t kMaxNumSamplesForOneTime = 3;

__device__ bool IsCellUndefined(uint cell_index)
{
    return cell_index == kCellUndefined;
}

__device__ void SetUndefined(uint* cell_index)
{
    *cell_index = kCellUndefined;
}

__device__ void FreeParticle(const FlipParticles& p, uint i)
{
    SetUndefined(&p.cell_index_[i]);

    // Assign an invalid position value to indicate the binding kernel to
    // treat it as a free particle.
    p.position_x_[i] = __float2half_rn(-1.0f);
}

__device__ bool IsStopped(float v_x, float v_y, float v_z)
{
    // To determine the time to recycle particles.
    const float v_¦Å = 0.0001f;
    return !(v_x > v_¦Å || v_x < -v_¦Å || v_y > v_¦Å || v_y < -v_¦Å ||
             v_z > v_¦Å || v_z < -v_¦Å);
}

__device__ bool IsCellActive(float v_x, float v_y, float v_z, float density,
                             float temperature)
{
    const float kEpsilon = 0.0001f;
    return !IsStopped(v_x, v_y, v_z) || density > kEpsilon ||
            temperature > kEpsilon;
}

// NOTE: Assuming never overflows/underflows.
template <int Increment>
__device__ uint8_t AtomicIncrementUint8(uint8_t* addr)
{
    uint r = 0;
    uint* base_addr =
        reinterpret_cast<uint*>(reinterpret_cast<size_t>(addr) & ~3);
    switch (reinterpret_cast<size_t>(addr) & 3) {
        case 0:
            r = atomicAdd(base_addr, static_cast<uint>(Increment));
            return static_cast<uint8_t>(r & 0xFF);
        case 1:
            r = atomicAdd(base_addr, static_cast<uint>(Increment) << 8);
            return static_cast<uint8_t>((r >> 8) & 0xFF);
        case 2:
            r = atomicAdd(base_addr, static_cast<uint>(Increment) << 16);
            return static_cast<uint8_t>((r >> 16) & 0xFF);
        case 3:
            r = atomicAdd(base_addr, static_cast<uint>(Increment) << 24);
            return static_cast<uint8_t>((r >> 24) & 0xFF);
    }

    return 0;
}

__device__ uint Tausworthe(uint z, int s1, int s2, int s3, uint M)
{
    uint b = (((z << s1) ^ z) >> s2);
    return (((z & M) << s3) ^ b);
}

__device__ float3 RandomCoord(uint* random_seed)
{
    uint seed = *random_seed;
    uint seed0 = Tausworthe(seed,  (blockIdx.x  + 1) & 0xF, (blockIdx.y  + 2) & 0xF, (blockIdx.z  + 3) & 0xF, 0xFFFFFFFE);
    uint seed1 = Tausworthe(seed0, (threadIdx.x + 1) & 0xF, (threadIdx.y + 2) & 0xF, (threadIdx.z + 3) & 0xF, 0xFFFFFFF8);
    uint seed2 = Tausworthe(seed1, (threadIdx.y + 1) & 0xF, (threadIdx.z + 2) & 0xF, (threadIdx.x + 3) & 0xF, 0xFFFFFFF0);
    uint seed3 = Tausworthe(seed2, (threadIdx.z + 1) & 0xF, (threadIdx.x + 2) & 0xF, (threadIdx.y + 3) & 0xF, 0xFFFFFFE0);

    float rand_x = (seed1 & 127) / 129.5918f - 0.49f;
    float rand_y = (seed2 & 127) / 129.5918f - 0.49f;
    float rand_z = (seed3 & 127) / 129.5918f - 0.49f;

    *random_seed = seed3;
    return make_float3(rand_x, rand_y, rand_z);
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

// =============================================================================

// Active particles should be consecutive, but could be freed during the
// routine.
__global__ void AdvectParticlesKernel(FlipParticles particles,
                                      uint3 volume_size,
                                      float time_step_over_cell_size)
{
    FlipParticles& p = particles;

    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= *p.num_of_actives_) // Maybe dynamic parallelism is a better
                                 // choice.
        return;

    float v_x = __half2float(p.velocity_x_[i]) * 0.99f;
    float v_y = __half2float(p.velocity_y_[i]) * 0.99f;
    float v_z = __half2float(p.velocity_z_[i]) * 0.99f;

    if (IsStopped(v_x, v_y, v_z)) {
        // We don't need the number of active particles until the sorting is
        // done.
        return;
    }

    float x = __half2float(p.position_x_[i]);
    float y = __half2float(p.position_y_[i]);
    float z = __half2float(p.position_z_[i]);

    // TODO: Keep the same boundary conditions as the grid.
    x += v_x * time_step_over_cell_size;
    y += v_y * time_step_over_cell_size;
    z += v_z * time_step_over_cell_size;

    if (x >= 0 && x < volume_size.x && y >= 0 && y < volume_size.y && z >= 0 &&
            z < volume_size.z) {
        p.position_x_[i] = __float2half_rn(x);
        p.position_y_[i] = __float2half_rn(y);
        p.position_z_[i] = __float2half_rn(z);

        int xi = static_cast<int>(x);
        int yi = static_cast<int>(y);
        int zi = static_cast<int>(z);

        uint cell_index = (zi * volume_size.y + yi) * volume_size.x + xi;
        p.cell_index_[i] = cell_index;
    } else {
        FreeParticle(particles, i);
    }
}

// Active particles should be consecutive, but could be freed during the
// routine.
__global__ void AdvectParticlesHighOrderKernel(FlipParticles particles,
                                               uint3 volume_size,
                                               float time_step_over_cell_size)
{
    FlipParticles& p = particles;

    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= *p.num_of_actives_) // Maybe dynamic parallelism is a better
                                 // choice.
        return;

    float v_x = __half2float(p.velocity_x_[i]);
    float v_y = __half2float(p.velocity_y_[i]);
    float v_z = __half2float(p.velocity_z_[i]);

    if (IsStopped(v_x, v_y, v_z)) {
        // Don't eliminate the particle. It may contains density/temperature
        // information.
        //
        // We don't need the number of active particles until the sorting is
        // done.
        return;
    }

    float x = __half2float(p.position_x_[i]);
    float y = __half2float(p.position_y_[i]);
    float z = __half2float(p.position_z_[i]);

    // TODO: Velocity dissipation.
    // TODO: Keep the same boundary conditions as the grid.
    float mid_x = x + 0.5f * time_step_over_cell_size * v_x + 0.5f;
    float mid_y = y + 0.5f * time_step_over_cell_size * v_y + 0.5f;
    float mid_z = z + 0.5f * time_step_over_cell_size * v_z + 0.5f;

    float v_x2 = tex3D(tex_x, mid_x + 0.5f, mid_y,        mid_z);
    float v_y2 = tex3D(tex_y, mid_x,        mid_y + 0.5f, mid_z);
    float v_z2 = tex3D(tex_z, mid_x,        mid_y,        mid_z + 0.5f);

    float mid_x2 = x + 0.75f * time_step_over_cell_size * v_x2 + 0.5f;
    float mid_y2 = y + 0.75f * time_step_over_cell_size * v_y2 + 0.5f;
    float mid_z2 = z + 0.75f * time_step_over_cell_size * v_z2 + 0.5f;

    float v_x3 = tex3D(tex_x, mid_x2 + 0.5f, mid_y2,        mid_z2);
    float v_y3 = tex3D(tex_y, mid_x2,        mid_y2 + 0.5f, mid_z2);
    float v_z3 = tex3D(tex_z, mid_x2,        mid_y2,        mid_z2 + 0.5f);

    float c1 = 2.0f / 9.0f * time_step_over_cell_size;
    float c2 = 3.0f / 9.0f * time_step_over_cell_size;
    float c3 = 4.0f / 9.0f * time_step_over_cell_size;

    float pos_x = x + c1 * v_x + c2 * v_x2 + c3 * v_x3;
    float pos_y = y + c1 * v_y + c2 * v_y2 + c3 * v_y3;
    float pos_z = z + c1 * v_z + c2 * v_z2 + c3 * v_z3;

    if (x >= 0 && x < volume_size.x && y >= 0 && y < volume_size.y && z >= 0 &&
            z < volume_size.z) {
        p.position_x_[i] = __float2half_rn(pos_x);
        p.position_y_[i] = __float2half_rn(pos_y);
        p.position_z_[i] = __float2half_rn(pos_z);

        int xi = static_cast<int>(x);
        int yi = static_cast<int>(y);
        int zi = static_cast<int>(z);

        uint cell_index = (zi * volume_size.y + yi) * volume_size.x + xi;
        p.cell_index_[i] = cell_index;
    } else {
        FreeParticle(particles, i);
    }
}

// Fields should be reset: particle_count, in_cell_index
// Fields should be available: cell_index.
// Active particles may *NOT* be consecutive.
__global__ void BindParticlesToCellsKernel(FlipParticles particles,
                                           uint3 volume_size)
{
    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= particles.num_of_particles_)
        return;

    uint cell_index = particles.cell_index_[i];
    if (IsCellUndefined(cell_index))
        return;

    // TODO: Free particles in resample kernel?
    uint* p_count = particles.particle_count_;
    if (p_count[cell_index] >= kMaxNumParticlesPerCell) {
        FreeParticle(particles, i);
    } else {
        uint old_count = atomicAdd(p_count + cell_index, 1);
        if (old_count >= kMaxNumParticlesPerCell) {
            atomicAdd(p_count + cell_index, static_cast<uint>(-1));
            FreeParticle(particles, i);
        } else {
            particles.in_cell_index_[i] = old_count;
        }
    }
}

__global__ void CalculateNumberOfActiveParticles(FlipParticles particles,
                                                 int last_cell_index)
{
    *particles.num_of_actives_ =
        particles.particle_index_[last_cell_index] +
        particles.particle_count_[last_cell_index];
}

// Fields should be available: cell_index, particle_count, particle_index.
__global__ void EmitParticlesKernel(FlipParticles particles,
                                    float3 center_point, float3 hotspot,
                                    float radius, float density,
                                    float temperature, uint random_seed,
                                    uint3 volume_size)
{
    uint x = VolumeX();
    uint y = 1 + threadIdx.y;
    uint z = VolumeZ();

    if (x >= volume_size.x || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float2 diff =
        make_float2(coord.x, coord.z) - make_float2(hotspot.x, hotspot.z);
    float d = hypotf(diff.x, diff.y);
    if (d >= radius)
        return;

    uint cell_index = (z * volume_size.y + y) * volume_size.x + x;
    int count = particles.particle_count_[cell_index];
    if (!count) {
        int new_particles = kMaxNumSamplesForOneTime;
        int base_index = atomicAdd(particles.num_of_actives_, new_particles);
        if (base_index + new_particles > particles.num_of_particles_) {
            atomicAdd(particles.num_of_actives_, -new_particles);
            return; // Not enough free particles.
        }

        particles.particle_count_[cell_index] += new_particles;
        uint seed = random_seed;
        for (int i = 0; i < new_particles; i++) {
            float3 pos = coord + RandomCoord(&seed);

            int index = base_index + i;

            // Not necessary to initialize the in_cell_index field.
            // Particle-cell mapping will be done in the binding kernel.

            // Assign a valid value to |cell_index_| to activate this particle.
            particles.cell_index_ [index] = cell_index;
            particles.position_x_ [index] = __float2half_rn(pos.x);
            particles.position_y_ [index] = __float2half_rn(pos.y);
            particles.position_z_ [index] = __float2half_rn(pos.z);
            particles.velocity_x_ [index] = 0;
            particles.velocity_y_ [index] = 0;
            particles.velocity_z_ [index] = 0;
            particles.density_    [index] = __float2half_rn(density);
            particles.temperature_[index] = __float2half_rn(temperature);
        }
    } else {
        uint p_index = particles.particle_index_[cell_index];
        for (int i = 0; i < count; i++) {
            // TOOD: Reset velocity to 0?
            particles.density_    [p_index + i] = __float2half_rn(density);
            particles.temperature_[p_index + i] = __float2half_rn(temperature);
        }
    }
}

// Should be invoked *BEFORE* resample kernel. Please read the comments of
// ResampleKernel().
// Active particles should be consecutive.
__global__ void InterpolateDeltaVelocityKernel(uint16_t* vel_x, uint16_t* vel_y,
                                               uint16_t* vel_z,
                                               const uint16_t* pos_x,
                                               const uint16_t* pos_y,
                                               const uint16_t* pos_z,
                                               int* num_of_active_particles)
{
    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= *num_of_active_particles) // Maybe dynamic parallelism is a better
                                       // choice.
        return;

    // Already constrained by |num_of_active_particles|.
    //
    //if (IsCellUndefined(cell_index[i]))
    //    return;

    float x = __half2float(pos_x[i]) + 0.5f;
    float y = __half2float(pos_y[i]) + 0.5f;
    float z = __half2float(pos_z[i]) + 0.5f;

    float v_x =  tex3D(tex_x,  x + 0.5f, y,        z);
    float v_y =  tex3D(tex_y,  x,        y + 0.5f, z);
    float v_z =  tex3D(tex_z,  x,        y,        z + 0.5f);

    float v_xp = tex3D(tex_xp, x + 0.5f, y,        z);
    float v_yp = tex3D(tex_yp, x,        y + 0.5f, z);
    float v_zp = tex3D(tex_zp, x,        y,        z + 0.5f);

    float ¦Ä_x = v_xp - v_x;
    float ¦Ä_y = v_yp - v_y;
    float ¦Ä_z = v_zp - v_z;

    vel_x[i] = __float2half_rn(__half2float(vel_x[i]) + ¦Ä_x);
    vel_y[i] = __float2half_rn(__half2float(vel_y[i]) + ¦Ä_y);
    vel_z[i] = __float2half_rn(__half2float(vel_z[i]) + ¦Ä_z);
}

// Should be invoked *AFTER* interpolation kernel. Since the newly inserted
// particles sample the new velocity filed, they don't need any correction.
//
// One should be very careful designing the mechanism of re-sampling: an
// important difference between particles and grid is that once a particle is
// created, its density and temperature are not gonna change during its life
// time(except decaying).
//
// Fields should be available: cell_index, particle_count.
// Active particles should be consecutive.
__global__ void ResampleKernel(FlipParticles particles, uint random_seed,
                               uint3 volume_size)
{
    int free_particles =
        particles.num_of_particles_ - *particles.num_of_actives_;
    if (free_particles < kMaxNumSamplesForOneTime)
        return; // No more free particles.

    uint x = VolumeX();
    uint y = VolumeY();
    uint z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    uint cell_index = (z * volume_size.y + y) * volume_size.x + x;
    int count = particles.particle_count_[cell_index];

    // Scan for all undersampled cells, and try to insert new particles.
    if (count > kMinNumParticlesPerCell)
        return;

    // CAUTION: All the physics variables, except velocity, should always be
    //          updated directly to the particles, or these changes might never
    //          get a chance to be applied to the particles, since the re-sample
    //          kernel only concerns about the cells that not having sufficient
    //          particles.
    int needed = min(kMaxNumParticlesPerCell - count, kMaxNumSamplesForOneTime);
    if (needed <= 0)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;

    float v_x =         tex3D(tex_x, coord.x + 0.5f, coord.y,        coord.z);
    float v_y =         tex3D(tex_y, coord.x,        coord.y + 0.5f, coord.z);
    float v_z =         tex3D(tex_z, coord.x,        coord.y,        coord.z + 0.5f);
    float density =     tex3D(tex_d, coord.x,        coord.y,        coord.z);
    float temperature = tex3D(tex_t, coord.x,        coord.y,        coord.z);

    if (!IsCellActive(v_x, v_y, v_z, density, temperature)) {
        // FIXME: Recycle inactive particles.
        return;
    }

    int base_index = atomicAdd(particles.num_of_actives_, needed);
    if (base_index + needed > particles.num_of_particles_) {
        atomicAdd(particles.num_of_actives_, -needed);
        return; // Not enough free particles.
    }

    // Reseed particles.
    uint seed = random_seed;
    for (int i = 0; i < needed; i++) {
        float3 pos = coord + RandomCoord(&seed);

        // TODO: Accelerate with share memory.
        v_x         = tex3D(tex_x, pos.x + 0.5f, pos.y,        pos.z);
        v_y         = tex3D(tex_y, pos.x,        pos.y + 0.5f, pos.z);
        v_z         = tex3D(tex_z, pos.x,        pos.y,        pos.z + 0.5f);
        density     = tex3D(tex_d, pos.x,        pos.y,        pos.z);
        temperature = tex3D(tex_t, pos.x,        pos.y,        pos.z);

        int index = base_index + i;

        // Not necessary to initialize the in_cell_index field.
        // Particle-cell mapping will be done in the binding kernel.

        // Assign a valid value to |cell_index_| to activate this particle.
        particles.cell_index_ [index] = cell_index;
        particles.position_x_ [index] = __float2half_rn(pos.x);
        particles.position_y_ [index] = __float2half_rn(pos.y);
        particles.position_z_ [index] = __float2half_rn(pos.z);
        particles.velocity_x_ [index] = __float2half_rn(v_x);
        particles.velocity_y_ [index] = __float2half_rn(v_y);
        particles.velocity_z_ [index] = __float2half_rn(v_z);
        particles.density_    [index] = __float2half_rn(density);
        particles.temperature_[index] = __float2half_rn(temperature);
    }
}

__global__ void ResetParticlesKernel(FlipParticles particles)
{
    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= particles.num_of_particles_)
        return;

    FreeParticle(particles, i);
    particles.in_cell_index_ = 0;
    particles.velocity_x_ = 0;
    particles.velocity_y_ = 0;
    particles.velocity_z_ = 0;
    particles.position_x_ = 0;
    particles.position_y_ = 0;
    particles.position_z_ = 0;
    particles.density_ = 0;
    particles.temperature_ = 0;

    if (i == 0)
        *particles.num_of_actives_ = 0;
}

// Fields should be available: cell_index, in_cell_index
// Active particles may *NOT* be consecutive.
template <typename Type>
__global__ void SortFieldKernel(Type* field_np1, Type* field,
                                uint32_t* cell_index, uint8_t* in_cell_index,
                                uint32_t* particle_index, uint num_of_particles,
                                uint3 volume_size)
{
    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= num_of_particles)
        return;

    if (!IsCellUndefined(cell_index[i])) {
        uint sort_index = particle_index[cell_index[i]] + in_cell_index[i];
        field_np1[sort_index] = field[i];
    }
}

// Fields should be available: cell_index, in_cell_index
// Active particles may *NOT* be consecutive.
__global__ void SortParticlesKernel(FlipParticles p_dst, FlipParticles p_src,
                                    uint3 volume_size)
{
    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= p_dst.num_of_particles_)
        return;

    if (i == 0) {
        // We need the number of active particles for allocation in the next
        // frame.
        int last_cell = volume_size.x * volume_size.y * volume_size.z - 1;
        *p_dst.num_of_actives_ =
            p_src.particle_index_[last_cell] + p_src.particle_count_[last_cell];
    }

    uint cell_index = p_src.cell_index_[i];
    uint in_cell = p_src.in_cell_index_[i];
    if (!IsCellUndefined(cell_index)) {
        uint sort_index = p_src.particle_index_[cell_index] + in_cell;

        p_dst.position_x_ [sort_index] = p_src.position_x_[i];
        p_dst.position_y_ [sort_index] = p_src.position_y_[i];
        p_dst.position_z_ [sort_index] = p_src.position_z_[i];
        p_dst.velocity_x_ [sort_index] = p_src.velocity_x_[i];
        p_dst.velocity_y_ [sort_index] = p_src.velocity_y_[i];
        p_dst.velocity_z_ [sort_index] = p_src.velocity_z_[i];
        p_dst.density_    [sort_index] = p_src.density_[i];
        p_dst.temperature_[sort_index] = p_src.temperature_[i];
    }
}

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
        ComputeWeightedAverage(&avg_vel_x,       &weight_vel_x,   pos_x + index, pos_y + index, pos_z + index, coord + make_float3(-0.5f, 0.0f, 0.0f), vel_x +   index, count);
        ComputeWeightedAverage(&avg_vel_y,       &weight_vel_y,   pos_x + index, pos_y + index, pos_z + index, coord + make_float3(0.0f, -0.5f, 0.0f), vel_y +   index, count);
        ComputeWeightedAverage(&avg_vel_z,       &weight_vel_z,   pos_x + index, pos_y + index, pos_z + index, coord + make_float3(0.0f, 0.0f, -0.5f), vel_z +   index, count);
        ComputeWeightedAverage(&avg_density,     &weight_density, pos_x + index, pos_y + index, pos_z + index, coord,                                  density + index, count);
        ComputeWeightedAverage(&avg_temperature, &weight_temperature, pos_x + index, pos_y + index, pos_z + index, coord, temperature + index, count);
    }

    uint16_t r_x = __float2half_rn(avg_vel_x       / weight_vel_x      );
    uint16_t r_y = __float2half_rn(avg_vel_y       / weight_vel_y      );
    uint16_t r_z = __float2half_rn(avg_vel_z       / weight_vel_z      );
    uint16_t r_d = __float2half_rn(avg_density     / weight_density     * 1.0f);
    uint16_t r_t = __float2half_rn(avg_temperature / weight_temperature * 1.0f);
    
    surf3Dwrite(r_x, surf_x, x * sizeof(r_x), y, z, cudaBoundaryModeTrap);
    surf3Dwrite(r_y, surf_y, x * sizeof(r_y), y, z, cudaBoundaryModeTrap);
    surf3Dwrite(r_z, surf_z, x * sizeof(r_z), y, z, cudaBoundaryModeTrap);
    surf3Dwrite(r_d, surf_d, x * sizeof(r_d), y, z, cudaBoundaryModeTrap);
    surf3Dwrite(r_t, surf_t, x * sizeof(r_t), y, z, cudaBoundaryModeTrap);

    // TODO: Diffuse the field if |total_weight| is too small(a hole near the
    //       spot).
}

// =============================================================================

namespace kern_launcher
{
void AdvectParticles(const FlipParticles& particles, cudaArray* vel_x,
                     cudaArray* vel_y, cudaArray* vel_z, float time_step,
                     float cell_size, uint3 volume_size, BlockArrangement* ba)
{
    dim3 block;
    dim3 grid;
    ba->ArrangeLinear(&grid, &block, particles.num_of_particles_);

    int high_order = 1;
    if (high_order) {
        auto bound_x = BindHelper::Bind(&tex_x, vel_x, false,
                                        cudaFilterModeLinear,
                                        cudaAddressModeClamp);
        if (bound_x.error() != cudaSuccess)
            return;

        auto bound_y = BindHelper::Bind(&tex_y, vel_y, false,
                                        cudaFilterModeLinear,
                                        cudaAddressModeClamp);
        if (bound_y.error() != cudaSuccess)
            return;

        auto bound_z = BindHelper::Bind(&tex_z, vel_z, false,
                                        cudaFilterModeLinear,
                                        cudaAddressModeClamp);
        if (bound_z.error() != cudaSuccess)
            return;

        AdvectParticlesHighOrderKernel<<<grid, block>>>(particles, volume_size,
                                                        time_step / cell_size);
    } else {
        AdvectParticlesKernel<<<grid, block>>>(particles, volume_size,
                                               time_step / cell_size);
    }

    DCHECK_KERNEL();
}

void BindParticlesToCells(const FlipParticles& particles, uint3 volume_size,
                          BlockArrangement* ba)
{
    uint num_of_cells = volume_size.x * volume_size.y * volume_size.z;
    cudaError_t e = cudaMemsetAsync(
        particles.in_cell_index_, 0,
        particles.num_of_particles_ * sizeof(*particles.in_cell_index_));
    assert(e == cudaSuccess);
    if (e != cudaSuccess)
        return;
    
    e = cudaMemsetAsync(particles.particle_count_, 0,
                        num_of_cells * sizeof(*particles.particle_count_));
    assert(e == cudaSuccess);
    if (e != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangeLinear(&grid, &block, particles.num_of_particles_);
    BindParticlesToCellsKernel<<<grid, block>>>(particles, volume_size);
    DCHECK_KERNEL();
}

void EmitParticles(const FlipParticles& particles, float3 center_point,
                   float3 hotspot, float radius, float density,
                   float temperature, uint random_seed, uint3 volume_size,
                   BlockArrangement* ba)
{
    const int kHeatLayerThickness = 3;
    dim3 block(volume_size.x, kHeatLayerThickness, 1);
    dim3 grid;
    ba->ArrangeGrid(&grid, block, volume_size);
    grid.y = 1;
    EmitParticlesKernel<<<grid, block>>>(particles, center_point, hotspot,
                                         radius, density, temperature,
                                         random_seed, volume_size);
    DCHECK_KERNEL();
}

void InterpolateDeltaVelocity(const FlipParticles& particles, cudaArray* vnp1_x,
                              cudaArray* vnp1_y, cudaArray* vnp1_z,
                              cudaArray* vn_x, cudaArray* vn_y, cudaArray* vn_z,
                              BlockArrangement* ba)
{
    auto bound_xp = BindHelper::Bind(&tex_xp, vnp1_x, false,
                                     cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_xp.error() != cudaSuccess)
        return;

    auto bound_yp = BindHelper::Bind(&tex_yp, vnp1_y, false,
                                     cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_yp.error() != cudaSuccess)
        return;

    auto bound_zp = BindHelper::Bind(&tex_zp, vnp1_z, false,
                                     cudaFilterModeLinear,
                                     cudaAddressModeClamp);
    if (bound_zp.error() != cudaSuccess)
        return;

    auto bound_x = BindHelper::Bind(&tex_x, vn_x, false,
                                    cudaFilterModeLinear, cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, vn_y, false,
                                    cudaFilterModeLinear, cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, vn_z, false,
                                    cudaFilterModeLinear, cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangeLinear(&grid, &block, particles.num_of_particles_);
    InterpolateDeltaVelocityKernel<<<grid, block>>>(particles.velocity_x_,
                                                    particles.velocity_y_,
                                                    particles.velocity_z_,
                                                    particles.position_x_,
                                                    particles.position_y_,
                                                    particles.position_z_,
                                                    particles.num_of_actives_);
    DCHECK_KERNEL();
}

void Resample(const FlipParticles& particles, cudaArray* vel_x,
              cudaArray* vel_y, cudaArray* vel_z, cudaArray* density,
              cudaArray* temperature, uint random_seed, uint3 volume_size,
              BlockArrangement* ba)
{
    auto bound_x = BindHelper::Bind(&tex_x, vel_x, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_x.error() != cudaSuccess)
        return;

    auto bound_y = BindHelper::Bind(&tex_y, vel_y, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_y.error() != cudaSuccess)
        return;

    auto bound_z = BindHelper::Bind(&tex_z, vel_z, false, cudaFilterModeLinear,
                                    cudaAddressModeClamp);
    if (bound_z.error() != cudaSuccess)
        return;

    auto bound_d = BindHelper::Bind(&tex_d, density, false,
                                    cudaFilterModeLinear, cudaAddressModeClamp);
    if (bound_d.error() != cudaSuccess)
        return;

    auto bound_t = BindHelper::Bind(&tex_t, temperature, false,
                                    cudaFilterModeLinear, cudaAddressModeClamp);
    if (bound_t.error() != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangePrefer3dLocality(&block, &grid, volume_size);
    ResampleKernel<<<grid, block>>>(particles, random_seed, volume_size);
    DCHECK_KERNEL();
}

void ResetParticles(const FlipParticles& particles, uint3 volume_size,
                    BlockArrangement* ba)
{
    dim3 block;
    dim3 grid;
    ba->ArrangeLinear(&grid, &block, particles.num_of_particles_);
    ResetParticlesKernel<<<grid, block>>>(particles);

    uint num_of_cells = volume_size.x * volume_size.y * volume_size.z;
    cudaMemsetAsync(particles.particle_index_, 0,
                    num_of_cells * sizeof(*particles.particle_index_));
    cudaMemsetAsync(particles.particle_count_, 0,
                    num_of_cells * sizeof(*particles.particle_count_));
    DCHECK_KERNEL();
}

void FastSort(FlipParticles particles, uint3 volume_size, BlockArrangement* ba)
{
    FlipParticles& p_dst = particles; // FIXME
    FlipParticles& p_src = particles;

    uint num_of_cells = volume_size.x * volume_size.y * volume_size.z;
    cudaError_t e = cudaMemcpyAsync(
        p_dst.particle_count_, p_src.particle_count_,
        num_of_cells * sizeof(*p_dst.particle_count_),
        cudaMemcpyDeviceToDevice);
    assert(e == cudaSuccess);
    if (e != cudaSuccess)
        return;

    e = cudaMemcpyAsync(p_dst.particle_index_, p_src.particle_index_,
                        num_of_cells * sizeof(*p_dst.particle_index_),
                        cudaMemcpyDeviceToDevice);
    assert(e == cudaSuccess);
    if (e != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangeLinear(&grid, &block, p_dst.num_of_particles_);
    SortParticlesKernel<<<grid, block>>>(p_dst, p_src, volume_size);
    DCHECK_KERNEL();
}

void SortParticles(FlipParticles particles, int* num_active_particles,
                   uint16_t* aux, uint3 volume_size, BlockArrangement* ba,
                   AuxBufferManager* bm)
{
    bool fast_sort = false;
    if (fast_sort) {
        FastSort(particles, volume_size, ba);
        return;
    }

    dim3 block;
    dim3 grid;
    ba->ArrangeLinear(&grid, &block, particles.num_of_particles_);

    uint16_t* fields[] = {
        particles.position_x_,
        particles.position_y_,
        particles.position_z_,
        particles.velocity_x_,
        particles.velocity_y_,
        particles.velocity_z_,
        particles.density_,
        particles.temperature_
    };

    for (int i = 0; i < sizeof(fields) / sizeof(*fields); i++) {
        SortFieldKernel<<<grid, block>>>(aux, fields[i],
                                         particles.cell_index_,
                                         particles.in_cell_index_,
                                         particles.particle_index_,
                                         particles.num_of_particles_,
                                         volume_size);
        DCHECK_KERNEL();

        cudaError_t e = cudaMemcpyAsync(
            fields[i], aux, particles.num_of_particles_ * sizeof(*fields[i]),
            cudaMemcpyDeviceToDevice);
        assert(e == cudaSuccess);
        if (e != cudaSuccess)
            return;
    }

    // Sort index fields.
    std::unique_ptr<uint32_t, std::function<void(void*)>> buf(
        reinterpret_cast<uint32_t*>(
            bm->Allocate(particles.num_of_particles_ * sizeof(uint32_t))),
        [&bm](void* p) { bm->Free(p); });
    SortFieldKernel<<<grid, block>>>(buf.get(), particles.cell_index_,
                                     particles.cell_index_,
                                     particles.in_cell_index_,
                                     particles.particle_index_,
                                     particles.num_of_particles_, volume_size);
    DCHECK_KERNEL();

    SortFieldKernel<<<grid, block>>>(reinterpret_cast<uint8_t*>(aux),
                                     particles.in_cell_index_,
                                     particles.cell_index_,
                                     particles.in_cell_index_,
                                     particles.particle_index_,
                                     particles.num_of_particles_, volume_size);
    DCHECK_KERNEL();

    cudaError_t e = cudaMemcpyAsync(
        particles.cell_index_, buf.get(),
        particles.num_of_particles_ * sizeof(*buf), cudaMemcpyDeviceToDevice);
    assert(e == cudaSuccess);
    if (e != cudaSuccess)
        return;

    e = cudaMemcpyAsync(
        particles.in_cell_index_, aux,
        particles.num_of_particles_ * sizeof(uint8_t),
        cudaMemcpyDeviceToDevice);
    assert(e == cudaSuccess);
    if (e != cudaSuccess)
        return;

    int last_cell_index = volume_size.x * volume_size.y * volume_size.z - 1;
    CalculateNumberOfActiveParticles<<<1, 1>>>(particles, last_cell_index);
    DCHECK_KERNEL();

    cudaMemcpyAsync(num_active_particles, particles.num_of_actives_,
                    sizeof(*num_active_particles), cudaMemcpyDeviceToHost);
}

void TransferToGrid(cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z,
                    cudaArray* density, cudaArray* temperature,
                    const FlipParticles& particles, uint3 volume_size,
                    BlockArrangement* ba)
{
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
