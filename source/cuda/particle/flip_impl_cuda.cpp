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

#include "flip_impl_cuda.h"

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

FlipImplCuda::FlipImplCuda(Observer* observer, BlockArrangement* ba,
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

FlipImplCuda::~FlipImplCuda()
{

}

void FlipImplCuda::Advect(const FlipParticles& particles,
                          int* num_active_particles,
                          const FlipParticles& aux, cudaArray* vnp1_x,
                          cudaArray* vnp1_y, cudaArray* vnp1_z, cudaArray* vn_x,
                          cudaArray* vn_y, cudaArray* vn_z, cudaArray* density,
                          cudaArray* temperature, float time_step,
                          float velocity_dissipation, float density_dissipation,
                          float temperature_dissipation,
                          const glm::ivec3& volume_size)
{
    // Sample the last step's velocity.
    kern_launcher::Resample(particles, vn_x, vn_y, vn_z, density, temperature,
                            rand_->Iterate(),
                            FromGlmVector(volume_size), ba_);
    observer_->OnResampled();

    kern_launcher::AdvectFlipParticles(particles, vnp1_x, vnp1_y, vnp1_z, vn_x,
                                       vn_y, vn_z, time_step, cell_size_,
                                       outflow_, FromGlmVector(volume_size),
                                       ba_);
    observer_->OnAdvected();

    kern_launcher::SortParticles(particles, num_active_particles, aux,
                                 time_step, velocity_dissipation,
                                 density_dissipation, temperature_dissipation,
                                 FromGlmVector(volume_size), ba_, bm_);
    observer_->OnSorted();

    FlipParticles p = particles;
    p.position_x_  = aux.position_x_;
    p.position_y_  = aux.position_y_;
    p.position_z_  = aux.position_z_;
    p.velocity_x_  = aux.velocity_x_;
    p.velocity_y_  = aux.velocity_y_;
    p.velocity_z_  = aux.velocity_z_;
    p.density_     = aux.density_;
    p.temperature_ = aux.temperature_;

    kern_launcher::TransferToGrid(vn_x, vn_y, vn_z, density, temperature,
                                  p, particles, FromGlmVector(volume_size),
                                  ba_);
//     kern_launcher::TransferToGridOpt(vn_x, vn_y, vn_z, density, temperature,
//                                      p, FromGlmVector(volume_size), ba_);
    observer_->OnTransferred();
}

void FlipImplCuda::Emit(const FlipParticles& particles,
                        const glm::vec3& center_point, const glm::vec3& hotspot,
                        float radius, float density, float temperature,
                        const glm::vec3& velocity,
                        const glm::ivec3& volume_size)
{
    kern_launcher::EmitFlipParticles(
        particles, make_float3(center_point.x, center_point.y, center_point.z),
        make_float3(hotspot.x, hotspot.y, hotspot.z), radius, density,
        temperature, make_float3(velocity.x, velocity.y, velocity.z),
        impulse_, rand_->Iterate(), FromGlmVector(volume_size), ba_);

    observer_->OnEmitted();
}

void FlipImplCuda::Reset(const FlipParticles& particles,
                         const glm::ivec3& volume_size)
{
    kern_launcher::ResetParticles(particles, FromGlmVector(volume_size), ba_);
}
