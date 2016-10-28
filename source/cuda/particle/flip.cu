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

const uint32_t kMaxNumParticlesPerCell = 4;
const uint32_t kMinNumParticlesPerCell = 2;
const uint32_t kMaxNumSamplesForOneTime = 3;

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

// =============================================================================

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

    float x = __half2float(pos_x[i]);
    float y = __half2float(pos_y[i]);
    float z = __half2float(pos_z[i]);

    float v_x =  tex3D(tex_x,  x + 0.5f, y,        z);
    float v_y =  tex3D(tex_y,  x,        y + 0.5f, z);
    float v_z =  tex3D(tex_z,  x,        y,        z + 0.5f);

    float v_xp = tex3D(tex_xp, x + 0.5f, y,        z);
    float v_yp = tex3D(tex_yp, x,        y + 0.5f, z);
    float v_zp = tex3D(tex_zp, x,        y,        z + 0.5f);

    float ¦Ä_x = v_xp - v_x;
    float ¦Ä_y = v_yp - v_y;
    float ¦Ä_z = v_zp - v_z;

    // v_np1 = (1 - ¦Á) * v_n_pic + ¦Á * v_n_flip.
    // We are using ¦Á = 1.
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

        // TODO: Accelerate with shared memory.
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
__global__ void SortParticlesKernel(FlipParticles p_aux, FlipParticles p_src,
                                    int last_cell_index, uint3 volume_size)
{
    uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= p_src.num_of_particles_)
        return;

    if (i == 0) {
        // We need the number of active particles for allocation in the next
        // frame.
        *p_src.num_of_actives_ =
            p_src.particle_index_[last_cell_index] +
            p_src.particle_count_[last_cell_index];
    }

    uint cell_index = p_src.cell_index_[i];
    uint in_cell    = p_src.in_cell_index_[i];
    if (!IsCellUndefined(cell_index)) {
        uint sort_index = p_src.particle_index_[cell_index] + in_cell;

        p_aux.cell_index_   [sort_index] = p_src.cell_index_[i];
        p_aux.in_cell_index_[sort_index] = p_src.in_cell_index_[i];
        p_aux.position_x_   [sort_index] = p_src.position_x_[i];
        p_aux.position_y_   [sort_index] = p_src.position_y_[i];
        p_aux.position_z_   [sort_index] = p_src.position_z_[i];
        p_aux.velocity_x_   [sort_index] = p_src.velocity_x_[i];
        p_aux.velocity_y_   [sort_index] = p_src.velocity_y_[i];
        p_aux.velocity_z_   [sort_index] = p_src.velocity_z_[i];
        p_aux.density_      [sort_index] = p_src.density_[i];
        p_aux.temperature_  [sort_index] = p_src.temperature_[i];
    }
}

// =============================================================================

namespace kern_launcher
{
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
    const int kHeatLayerThickness = 2;
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

void FastSort(FlipParticles particles, FlipParticles aux,
              uint3 volume_size, BlockArrangement* ba)
{
    FlipParticles& p_src = particles;
    FlipParticles& p_aux = aux;
    int last_cell_index = volume_size.x * volume_size.y * volume_size.z - 1;

    dim3 block;
    dim3 grid;
    ba->ArrangeLinear(&grid, &block, p_src.num_of_particles_);
    SortParticlesKernel<<<grid, block>>>(p_aux, p_src, last_cell_index,
                                         volume_size);
    DCHECK_KERNEL();
}

void SortParticles(FlipParticles particles, int* num_active_particles,
                   FlipParticles aux, uint3 volume_size,
                   BlockArrangement* ba)
{
    if (aux.velocity_x_) {
        FastSort(particles, aux, volume_size, ba);
    } else {
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
            SortFieldKernel<<<grid, block>>>(aux.position_x_, fields[i],
                                             particles.cell_index_,
                                             particles.in_cell_index_,
                                             particles.particle_index_,
                                             particles.num_of_particles_,
                                             volume_size);
            DCHECK_KERNEL();

            cudaError_t e = cudaMemcpyAsync(
                fields[i], aux.position_x_,
                particles.num_of_particles_ * sizeof(*fields[i]),
                cudaMemcpyDeviceToDevice);
            assert(e == cudaSuccess);
            if (e != cudaSuccess)
                return;
        }

        // Sort index fields.
        SortFieldKernel<<<grid, block>>>(aux.cell_index_, particles.cell_index_,
                                         particles.cell_index_,
                                         particles.in_cell_index_,
                                         particles.particle_index_,
                                         particles.num_of_particles_,
                                         volume_size);
        DCHECK_KERNEL();

        SortFieldKernel<<<grid, block>>>(aux.in_cell_index_,
                                         particles.in_cell_index_,
                                         particles.cell_index_,
                                         particles.in_cell_index_,
                                         particles.particle_index_,
                                         particles.num_of_particles_,
                                         volume_size);
        DCHECK_KERNEL();

        cudaError_t e = cudaMemcpyAsync(
            particles.cell_index_, aux.cell_index_,
            particles.num_of_particles_ * sizeof(*particles.cell_index_),
            cudaMemcpyDeviceToDevice);
        assert(e == cudaSuccess);
        if (e != cudaSuccess)
            return;

        e = cudaMemcpyAsync(
            particles.in_cell_index_, aux.in_cell_index_,
            particles.num_of_particles_ * sizeof(*particles.in_cell_index_),
            cudaMemcpyDeviceToDevice);
        assert(e == cudaSuccess);
        if (e != cudaSuccess)
            return;

        int last_cell_index = volume_size.x * volume_size.y * volume_size.z - 1;
        CalculateNumberOfActiveParticles<<<1, 1>>>(particles, last_cell_index);
        DCHECK_KERNEL();
    }

    cudaMemcpyAsync(num_active_particles, particles.num_of_actives_,
                    sizeof(*num_active_particles), cudaMemcpyDeviceToHost);
}
}
