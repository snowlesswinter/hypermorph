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

struct cudaArray;
struct FlipParticles;
class AuxBufferManager;
class BlockArrangement;
class CudaVolume;
class RandomHelper;
class FlipImplCuda
{
public:
    FlipImplCuda(BlockArrangement* ba, AuxBufferManager* bm,
                 RandomHelper* rand);
    ~FlipImplCuda();

    void Advect(const FlipParticles& p_next, const FlipParticles& p_cur,
                cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z,
                cudaArray* density, cudaArray* temperature, cudaArray* delta_x,
                cudaArray* delta_y, cudaArray* delta_z, float time_step,
                const glm::ivec3& volume_size);
    void Reset(const FlipParticles& particles);

    void set_cell_size(float cell_size) { cell_size_ = cell_size; }

private:
    void CompactParticles(const FlipParticles& p_cur,
                          const FlipParticles& p_next,
                          const glm::ivec3& volume_size);

    BlockArrangement* ba_;
    AuxBufferManager* bm_;
    RandomHelper* rand_;
    float cell_size_;
};

#endif // _FLIP_IMPL_CUDA_H_