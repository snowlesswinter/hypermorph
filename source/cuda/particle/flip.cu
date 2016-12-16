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
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "cuda/aux_buffer_manager.h"
#include "cuda/block_arrangement.h"
#include "cuda/cuda_common_host.h"
#include "cuda/cuda_common_kern.h"
#include "cuda/cuda_debug.h"
#include "cuda/fluid_impulse.h"
#include "cuda/particle/flip_common.cuh"
#include "flip.h"
#include "random.cuh"

texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_x;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_y;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_z;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_d;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> tex_t;

namespace
{
__device__ bool IsCellActive(float v_x, float v_y, float v_z, float density,
                             float temperature)
{
    const float kEpsilon = 0.0001f;
    return !IsStopped(v_x, v_y, v_z) || density > kEpsilon ||
            temperature > kEpsilon;
}

__device__ bool IsThereEnoughFreeParticles(FlipParticles& particles, int needed)
{
    // As we don't need to compact the particles, the number of active particles
    // becomes less concerned by us.
    bool limited_particles = false;
    if (!limited_particles)
        return true;

    int total_count = atomicAdd(particles.num_of_actives_, needed);
    if (total_count + needed > particles.num_of_particles_) {
        atomicAdd(particles.num_of_actives_, -needed);
        return false;
    }

    return true;
}

struct HorizontalEmission
{
    __device__ static bool OutsideVolume(uint x, uint y, uint z,
                                         const uint3& volume_size)
    {
        return y >= volume_size.y || z >= volume_size.z;
    }
    __device__ static float CalculateRadius(const float3& coord,
                                            const float3& center,
                                            const float3& hotspot)
    {
        float2 diff =
            make_float2(coord.y, coord.z) - make_float2(center.y, center.z);
        return hypotf(diff.x, diff.y);
    }
    __device__ static void SetVelX(uint16_t* vel_x, const float3 velocity)
    {
        *vel_x = __float2half_rn(velocity.x);
    }
};

struct VerticalEmission
{
    __device__ static bool OutsideVolume(uint x, uint y, uint z,
                                         const uint3& volume_size)
    {
        return x >= volume_size.x || z >= volume_size.z;
    }
    __device__ static float CalculateRadius(const float3& coord,
                                            const float3& center,
                                            const float3& hotspot)
    {
        float2 diff =
            make_float2(coord.x, coord.z) - make_float2(hotspot.x, hotspot.z);
        return hypotf(diff.x, diff.y);
    }
    __device__ static void SetVelX(uint16_t* vel_x, const float3 velocity)
    {
    }
};

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

// =============================================================================

// Fields should be available: particle_count.
template <typename Emission>
__global__ void EmitFlipParticlesKernel(FlipParticles particles, float3 center,
                                        float3 hotspot, float radius,
                                        float density, float temperature,
                                        float3 velocity, uint random_seed,
                                        uint3 volume_size)
{
    uint x = VolumeX();
    uint y = VolumeY();
    uint z = VolumeZ();

    if (Emission::OutsideVolume(x, y, z, volume_size))
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float d = Emission::CalculateRadius(coord, center, hotspot);
    if (d >= radius)
        return;

    uint cell_index = LinearIndexVolume(x, y, z, volume_size);
    int count = particles.particle_count_[cell_index];
    if (!count) {
        int new_particles = kMaxNumSamplesForOneTime;
        if (!IsThereEnoughFreeParticles(particles, new_particles))
            return;

        particles.particle_count_[cell_index] = new_particles;
        uint seed = random_seed + cell_index;
        for (int i = 0; i < new_particles; i++) {
            float3 pos = coord + RandomCoordCube(&seed);

            int index = cell_index * kMaxNumParticlesPerCell + i;

            particles.position_x_ [index] = __float2half_rn(pos.x);
            particles.position_y_ [index] = __float2half_rn(pos.y);
            particles.position_z_ [index] = __float2half_rn(pos.z);
            particles.velocity_x_ [index] = 0;
            particles.velocity_y_ [index] = 0;
            particles.velocity_z_ [index] = 0;
            particles.density_    [index] = __float2half_rn(density);
            particles.temperature_[index] = __float2half_rn(temperature);

            Emission::SetVelX(&particles.velocity_x_[index], velocity);
        }
    } else {
        uint p_index = cell_index * kMaxNumParticlesPerCell;
        for (int i = 0; i < count; i++) {
            particles.density_    [p_index + i] = __float2half_rn(density);
            particles.temperature_[p_index + i] = __float2half_rn(temperature);

            Emission::SetVelX(&particles.velocity_x_[p_index + i], velocity);
        }
    }
}

// Fields should be available: particle_count.
__global__ void EmitFlipParticlesFromSphereKernel(FlipParticles particles,
                                                  float3 center, float radius,
                                                  float density,
                                                  float temperature,
                                                  float velocity,
                                                  uint random_seed,
                                                  uint3 volume_size)
{
    uint x = VolumeX();
    uint y = VolumeY();
    uint z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    float3 coord = make_float3(x, y, z) + 0.5f;
    float3 diff = coord - center;
    float d = norm3df(diff.x, diff.y, diff.z);
    if (d >= radius)
        return;

    uint cell_index = LinearIndexVolume(x, y, z, volume_size);
    int count = particles.particle_count_[cell_index];
    if (!count) {
        int new_particles = kMaxNumSamplesForOneTime;
        if (!IsThereEnoughFreeParticles(particles, new_particles))
            return;

        particles.particle_count_[cell_index] = new_particles;
        uint seed = random_seed + cell_index;
        for (int i = 0; i < new_particles; i++) {
            float3 pos = coord + RandomCoordCube(&seed);

            int index = cell_index * kMaxNumParticlesPerCell + i;

            float3 dir = pos - center;
            float3 vel = normalize(dir) * velocity;

            // Not necessary to initialize the in_cell_index field.
            // Particle-cell mapping will be done in the binding kernel.

            particles.position_x_ [index] = __float2half_rn(pos.x);
            particles.position_y_ [index] = __float2half_rn(pos.y);
            particles.position_z_ [index] = __float2half_rn(pos.z);
            particles.velocity_x_ [index] = __float2half_rn(vel.x);
            particles.velocity_y_ [index] = __float2half_rn(vel.y);
            particles.velocity_z_ [index] = __float2half_rn(vel.z);
            particles.density_    [index] = __float2half_rn(density);
            particles.temperature_[index] = __float2half_rn(temperature);
        }
    } else {
        uint p_index = cell_index * kMaxNumParticlesPerCell;
        for (int i = 0; i < count; i++) {
            float pos_x = __half2float(particles.position_x_[p_index + i]);
            float pos_y = __half2float(particles.position_y_[p_index + i]);
            float pos_z = __half2float(particles.position_z_[p_index + i]);
            float3 pos = make_float3(pos_x, pos_y, pos_z);

            float3 dir = pos - center;
            float3 vel = normalize(dir) * velocity;

            particles.velocity_x_ [p_index + i] = __float2half_rn(vel.x);
            particles.velocity_y_ [p_index + i] = __float2half_rn(vel.y);
            particles.velocity_z_ [p_index + i] = __float2half_rn(vel.z);
            particles.density_    [p_index + i] = __float2half_rn(density);
            particles.temperature_[p_index + i] = __float2half_rn(temperature);
        }
    }
}

// The new particles sample the velocity of the last step. Please see the
// comments of FLIP particle advection kernel.
//
// One should be very careful designing the mechanism of re-sampling: an
// important difference between particles and grid is that once a particle is
// created, its density and temperature are not gonna change during its life
// time(except decaying).
//
// Fields should be available: particle_count.
// Active particles are always *NOT* consecutive.
__global__ void ResampleKernel(FlipParticles particles, uint random_seed,
                               uint3 volume_size)
{
    uint x = VolumeX();
    uint y = VolumeY();
    uint z = VolumeZ();

    if (x >= volume_size.x || y >= volume_size.y || z >= volume_size.z)
        return;

    uint cell_index = LinearIndexVolume(x, y, z, volume_size);
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

    // Used to be: IsCellActive(center_vel, center_density, center_temperature).
    // But that is not correct. We should not judge all the particles only by
    // the state right at the center of the cell.

    if (!IsThereEnoughFreeParticles(particles, needed))
        return;

    // FIXME: Rectify particle count.

    float3 coord = make_float3(x, y, z) + 0.5f;

    // Reseed particles.
    uint seed = random_seed + cell_index;
    for (int i = 0; i < needed; i++) {
        float3 pos = coord + RandomCoordCube(&seed);

        // TODO: Accelerate with shared memory.
        float3 v           = LoadVel(tex_x, tex_y, tex_z, pos);
        float  density     = tex3D  (tex_d, pos.x, pos.y, pos.z);
        float  temperature = tex3D  (tex_t, pos.x, pos.y, pos.z);

        int index = cell_index * kMaxNumParticlesPerCell + count + i;

        particles.position_x_ [index] = __float2half_rn(pos.x);
        particles.position_y_ [index] = __float2half_rn(pos.y);
        particles.position_z_ [index] = __float2half_rn(pos.z);
        particles.velocity_x_ [index] = __float2half_rn(v.x);
        particles.velocity_y_ [index] = __float2half_rn(v.y);
        particles.velocity_z_ [index] = __float2half_rn(v.z);
        particles.density_    [index] = __float2half_rn(density);
        particles.temperature_[index] = __float2half_rn(temperature);
    }
}

__global__ void ResetParticlesKernel(FlipParticles particles)
{
    uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i >= particles.num_of_particles_)
        return;

    particles.velocity_x_ [i] = 0;
    particles.velocity_y_ [i] = 0;
    particles.velocity_z_ [i] = 0;
    particles.position_x_ [i] = 0;
    particles.position_y_ [i] = 0;
    particles.position_z_ [i] = 0;
    particles.density_    [i] = 0;
    particles.temperature_[i] = 0;
    FreeParticle(particles, i);

    if (i == 0)
        *particles.num_of_actives_ = 0;
}

// Fields should be reset: particle_count
// Active particles may *NOT* be consecutive.
__global__ void SortParticlesKernel(FlipParticles p_aux, FlipParticles p_src,
                                    float time_step, float velocity_dissipation,
                                    float density_dissipation,
                                    float temperature_dissipation,
                                    uint3 volume_size)
{
    uint i = LinearIndex();
    if (i >= p_src.num_of_particles_)
        return;

    uint16_t xh = p_src.position_x_[i];
    uint16_t yh = p_src.position_y_[i];
    uint16_t zh = p_src.position_z_[i];

    if (IsCellUndefined(xh))
        return;

    int cell_index = CellIndex(xh, yh, zh, volume_size);
    uint* p_count = p_src.particle_count_;
    if (p_count[cell_index] < kMaxNumParticlesPerCell) {
        uint old_count = atomicAdd(p_count + cell_index, 1);
        if (old_count >= kMaxNumParticlesPerCell) {
            // TODO: Could leave the count to be larger than
            //       kMaxNumParticlesPerCell, and clamp it in the mapping
            //       kernel. But the active number calculation is a problem.
            atomicAdd(p_count + cell_index, static_cast<uint>(-1));
        } else {
            uint sort_index = cell_index * kMaxNumParticlesPerCell + old_count;

            p_aux.position_x_ [sort_index] = xh;
            p_aux.position_y_ [sort_index] = yh;
            p_aux.position_z_ [sort_index] = zh;
            p_aux.velocity_x_ [sort_index] = __float2half_rn((1.0f - velocity_dissipation    * time_step) * __half2float(p_src.velocity_x_[i]));
            p_aux.velocity_y_ [sort_index] = __float2half_rn((1.0f - velocity_dissipation    * time_step) * __half2float(p_src.velocity_y_[i]));
            p_aux.velocity_z_ [sort_index] = __float2half_rn((1.0f - velocity_dissipation    * time_step) * __half2float(p_src.velocity_z_[i]));
            p_aux.density_    [sort_index] = __float2half_rn((1.0f - density_dissipation     * time_step) * __half2float(p_src.density_[i]));
            p_aux.temperature_[sort_index] = __float2half_rn((1.0f - temperature_dissipation * time_step) * __half2float(p_src.temperature_[i]));
        }
    }
}
} // Anonymous namespace.

struct SchemeDefault
{
    __host__ SchemeDefault(uint32_t* p_count)
        : p_count_(p_count)
    {
    }
    __device__ int Load(uint i, uint row_stride, uint slice_stride)
    {
        return p_count_[i];
    }
    __device__ void Save(int* dest, int result)
    {
        *dest = result;
    }

    uint32_t* p_count_;
};

#include "../volume_reduction.cuh"

// =============================================================================

namespace kern_launcher
{
void EmitFlipParticles(const FlipParticles& particles, float3 center,
                       float3 hotspot, float radius, float density,
                       float temperature, float3 velocity, FluidImpulse impulse,
                       uint random_seed, uint3 volume_size,
                       BlockArrangement* ba)
{

    switch (impulse) {
        case IMPULSE_HOT_FLOOR: {
            const float kHeatLayerThickness = 0.025f * volume_size.y;
            uint3 actual_size = volume_size;
            actual_size.y = static_cast<uint>(std::ceil(kHeatLayerThickness));

            dim3 grid;
            dim3 block;
            ba->ArrangeRowScan(&grid, &block, actual_size);
            EmitFlipParticlesKernel<VerticalEmission><<<grid, block>>>(
                particles, center, hotspot, radius, density, temperature,
                velocity, random_seed, volume_size);
            break;
        }
        case IMPULSE_SPHERE: {
            uint3 actual_size = volume_size;
            actual_size.y = static_cast<uint>(std::ceil(radius + center.y));

            dim3 grid;
            dim3 block;
            ba->ArrangeRowScan(&grid, &block, actual_size);
            EmitFlipParticlesFromSphereKernel<<<grid, block>>>(
                particles, center, radius, density, temperature, velocity.x,
                random_seed, volume_size);
            break;
        }
        case IMPULSE_BUOYANT_JET: {
            const float kHeatLayerThickness = 0.02f * volume_size.x;
            uint3 actual_size = volume_size;
            actual_size.x = static_cast<uint>(std::ceil(kHeatLayerThickness));

            dim3 grid;
            dim3 block;
            ba->ArrangeRowScan(&grid, &block, actual_size);
            EmitFlipParticlesKernel<HorizontalEmission><<<grid, block>>>(
                particles, center, hotspot, radius, density, temperature,
                velocity, random_seed, volume_size);
            break;
        }
    }
    
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

    dim3 grid;
    dim3 block;
    ba->ArrangePrefer3dLocality(&grid, &block, volume_size);
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
    cudaMemsetAsync(particles.particle_count_, 0,
                    num_of_cells * sizeof(*particles.particle_count_));
    DCHECK_KERNEL();
}

void SortParticles(FlipParticles particles, int* num_active_particles,
                   FlipParticles aux, float time_step,
                   float velocity_dissipation, float density_dissipation,
                   float temperature_dissipation, uint3 volume_size,
                   BlockArrangement* ba, AuxBufferManager* bm)
{
    // Reset all particles in |aux| to undefined.
    thrust::device_ptr<uint16_t> v(aux.position_x_);
    thrust::fill(v, v + aux.num_of_particles_, kInvalidPos);

    FlipParticles& p_src = particles;
    FlipParticles& p_aux = aux;

    uint num_of_cells = volume_size.x * volume_size.y * volume_size.z;
    cudaError_t e = cudaMemsetAsync(
        p_src.particle_count_, 0,
        num_of_cells * sizeof(*p_src.particle_count_));
    assert(e == cudaSuccess);
    if (e != cudaSuccess)
        return;

    dim3 block;
    dim3 grid;
    ba->ArrangeLinear(&grid, &block, p_src.num_of_particles_);
    SortParticlesKernel<<<grid, block>>>(p_aux, p_src, time_step,
                                         velocity_dissipation,
                                         density_dissipation,
                                         temperature_dissipation, volume_size);
    DCHECK_KERNEL();

    SchemeDefault scheme(p_src.particle_count_);
    ReduceVolume<int>(p_src.num_of_actives_, scheme, volume_size, ba, bm);
    DCHECK_KERNEL();

    cudaMemcpyAsync(num_active_particles, particles.num_of_actives_,
                    sizeof(*num_active_particles), cudaMemcpyDeviceToHost);
}
}
