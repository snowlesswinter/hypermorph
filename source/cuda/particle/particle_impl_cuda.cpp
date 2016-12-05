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

#include "particle_impl_cuda.h"

#include <cassert>

#include "third_party/opengl/glew.h"

#include <cuda_runtime.h>
#include <helper_math.h>

#include "cuda/aux_buffer_manager.h"
#include "cuda/kernel_launcher.h"
#include "cuda/random_helper.h"
#include "flip.h"
#include "third_party/glm/vec3.hpp"

namespace
{
uint3 FromGlmVector(const glm::ivec3& v)
{
    return make_uint3(static_cast<uint>(v.x), static_cast<uint>(v.y),
                      static_cast<uint>(v.z));
}
} // Anonymous namespace.

ParticleImplCuda::ParticleImplCuda(Observer* observer, BlockArrangement* ba,
                                   AuxBufferManager* bm, RandomHelper* rand)
    : observer_(observer)
    , ba_(ba)
    , bm_(bm)
    , rand_(rand)
    , cell_size_(0.15f)
    , impulse_(IMPULSE_HOT_FLOOR)
    , outflow_(false)
{
}

ParticleImplCuda::~ParticleImplCuda()
{

}

void ParticleImplCuda::Advect(uint16_t* pos_x, uint16_t* pos_y, uint16_t* pos_z,
                              uint16_t* density, uint16_t* life,
                              int num_of_particles, cudaArray* vel_x,
                              cudaArray* vel_y, cudaArray* vel_z,
                              float time_step, const glm::ivec3& volume_size)
{
    kern_launcher::AdvectParticles(pos_x, pos_y, pos_z, density, life,
                                   num_of_particles, vel_x, vel_y, vel_z,
                                   time_step, cell_size_, outflow_,
                                   FromGlmVector(volume_size), ba_);
    observer_->OnAdvected();
}

void ParticleImplCuda::Emit(uint16_t* pos_x, uint16_t* pos_y, uint16_t* pos_z,
                            uint16_t* density, uint16_t* life, int* tail,
                            int num_of_particles, int num_to_emit,
                            const glm::vec3& location, float radius,
                            float density_value)
{
    kern_launcher::EmitParticles(
        pos_x, pos_y, pos_z, density, life, tail, num_of_particles, num_to_emit,
        make_float3(location.x, location.y, location.z), radius, density_value,
        rand_->Iterate(), ba_);

    observer_->OnEmitted();
}

void ParticleImplCuda::Reset(uint16_t* life, int num_of_particles)
{
    //kern_launcher::ResetParticles(particles, FromGlmVector(volume_size), ba_);
}
