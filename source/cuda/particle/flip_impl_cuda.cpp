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

FlipImplCuda::FlipImplCuda(BlockArrangement* ba, AuxBufferManager* bm,
                           RandomHelper* rand)
    : ba_(ba)
    , bm_(bm)
    , rand_(rand)
    , cell_size_(0.15f)
{

}

FlipImplCuda::~FlipImplCuda()
{

}

void FlipImplCuda::Advect(const FlipParticles& particles, uint16_t* aux,
                          cudaArray* vnp1_x,
                          cudaArray* vnp1_y, cudaArray* vnp1_z, cudaArray* vn_x,
                          cudaArray* vn_y, cudaArray* vn_z, cudaArray* density,
                          cudaArray* temperature, float time_step,
                          const glm::ivec3& volume_size)
{
    kern_launcher::InterpolateDeltaVelocity(particles, vnp1_x, vnp1_y, vnp1_z,
                                            vn_x, vn_y, vn_z, ba_);
    kern_launcher::Resample(particles, vnp1_x, vnp1_y, vnp1_z, density,
                            temperature, rand_->Iterate(),
                            FromGlmVector(volume_size), ba_);
    kern_launcher::AdvectParticles(particles, vnp1_x, vnp1_y, vnp1_z, time_step,
                                   cell_size_, FromGlmVector(volume_size), ba_);
    CompactParticles(particles, aux, volume_size);
    kern_launcher::TransferToGrid(vn_x, vn_y, vn_z, density, temperature,
                                  particles, FromGlmVector(volume_size), ba_);
}

void FlipImplCuda::Reset(const FlipParticles& particles)
{
    kern_launcher::ResetParticles(particles, ba_);
}

void FlipImplCuda::CompactParticles(const FlipParticles& particles,
                                    uint16_t* aux,
                                    const glm::ivec3& volume_size)
{
    uint num_of_cells = volume_size.x * volume_size.y * volume_size.z;
    kern_launcher::BindParticlesToCells(particles, FromGlmVector(volume_size),
                                        ba_);
    kern_launcher::BuildCellOffsets(particles.particle_index_,
                                    particles.particle_count_, num_of_cells,
                                    ba_, bm_);
    kern_launcher::SortParticles(particles, aux, FromGlmVector(volume_size),
                                 ba_, bm_);
}
