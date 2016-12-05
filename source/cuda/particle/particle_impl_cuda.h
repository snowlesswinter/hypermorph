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

#ifndef _PARTICLE_IMPL_CUDA_H_
#define _PARTICLE_IMPL_CUDA_H_

#include "third_party/glm/fwd.hpp"

#include "cuda/fluid_impulse.h"

struct cudaArray;
class AuxBufferManager;
class BlockArrangement;
class RandomHelper;
class ParticleImplCuda
{
public:
    class Observer
    {
    public:
        virtual void OnEmitted() = 0;
        virtual void OnAdvected() = 0;
    };

    ParticleImplCuda(Observer* observer, BlockArrangement* ba,
                     AuxBufferManager* bm, RandomHelper* rand);
    ~ParticleImplCuda();

    void Advect(uint16_t* pos_x, uint16_t* pos_y, uint16_t* pos_z,
                uint16_t* density, uint16_t* life, int num_of_particles,
                cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z,
                float time_step, const glm::ivec3& volume_size);
    void Emit(uint16_t* pos_x, uint16_t* pos_y, uint16_t* pos_z,
              uint16_t* density, uint16_t* life, int* tail,
              int num_of_particles, int num_to_emit, const glm::vec3& location,
              float radius, float density_value);
    void Reset(uint16_t* life, int num_of_particles);

    void set_cell_size(float cell_size) { cell_size_ = cell_size; }
    void set_fluid_impulse(FluidImpulse i) { impulse_ = i; }
    void set_outflow(bool outflow) { outflow_ = outflow; }

private:
    Observer* observer_;
    BlockArrangement* ba_;
    AuxBufferManager* bm_;
    RandomHelper* rand_;
    float cell_size_;
    FluidImpulse impulse_;
    bool outflow_;
};

#endif // _PARTICLE_IMPL_CUDA_H_