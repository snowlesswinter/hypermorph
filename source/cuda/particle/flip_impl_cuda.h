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

#ifndef _FLIP_IMPL_CUDA_H_
#define _FLIP_IMPL_CUDA_H_

#include "third_party/glm/fwd.hpp"

#include "cuda/fluid_impulse.h"

struct cudaArray;
struct FlipParticles;
class AuxBufferManager;
class BlockArrangement;
class CudaVolume;
class RandomHelper;
class FlipImplCuda
{
public:
    class Observer
    {
    public:
        virtual void OnEmitted() = 0;
        virtual void OnVelocityInterpolated() = 0;
        virtual void OnResampled() = 0;
        virtual void OnAdvected() = 0;
        virtual void OnCellBound() = 0;
        virtual void OnPrefixSumCalculated() = 0;
        virtual void OnSorted() = 0;
        virtual void OnTransferred() = 0;
    };

    FlipImplCuda(Observer* observer, BlockArrangement* ba, AuxBufferManager* bm,
                 RandomHelper* rand);
    ~FlipImplCuda();

    void Advect(const FlipParticles& particles, int* num_active_particles,
                const FlipParticles& aux, cudaArray* vnp1_x, cudaArray* vnp1_y,
                cudaArray* vnp1_z, cudaArray* vn_x, cudaArray* vn_y,
                cudaArray* vn_z, cudaArray* density, cudaArray* temperature,
                float time_step, float velocity_dissipation,
                float density_dissipation, float temperature_dissipation,
                const glm::ivec3& volume_size);
    void Emit(const FlipParticles& particles, const glm::vec3& center_point,
              const glm::vec3& hotspot, float radius, float density,
              float temperature, const glm::vec3& velocity,
              const glm::ivec3& volume_size);
    void Reset(const FlipParticles& particles, const glm::ivec3& volume_size);

    void set_cell_size(float cell_size) { cell_size_ = cell_size; }
    void set_fluid_impulse(FluidImpulse i) { impulse_ = i; }

private:
    void CompactParticles(FlipParticles* particles, int* num_active_particles,
                          const FlipParticles& aux,
                          const glm::ivec3& volume_size);

    Observer* observer_;
    BlockArrangement* ba_;
    AuxBufferManager* bm_;
    RandomHelper* rand_;
    float cell_size_;
    FluidImpulse impulse_;
};

#endif // _FLIP_IMPL_CUDA_H_