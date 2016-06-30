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

#ifndef _FLIP_H_
#define _FLIP_H_

#include "third_party/opengl/glew.h"

#include <cuda_runtime.h>
#include <driver_types.h>
#include <helper_cuda.h>
#include <helper_math.h>

struct FlipParticles
{
    static const uint kCellUndefined = static_cast<uint>(-1);
    static const uint kMaxNumParticlesPerCell = 6;
    static const uint kMinNumParticlesPerCell = 3;

    static __device__ bool IsCellUndefined(uint cell_index);
    static __device__ void SetUndefined(uint* cell_index);
    static __device__ void FreeParticle(const FlipParticles& p, uint i);
    static __device__ bool IsStopped(float v_x, float v_y, float v_z);

    uint* particle_index_;          // Cell index -> particle index.
    uint* cell_index_;              // Particle index -> cell index.
    uint8_t* in_cell_index_;        // Particle index -> in-cell index.
    uint8_t* particle_count_;       // Cell index -> # particles in cell.
    uint16_t* position_x_;
    uint16_t* position_y_;
    uint16_t* position_z_;
    uint16_t* velocity_x_;
    uint16_t* velocity_y_;
    uint16_t* velocity_z_;
    uint16_t* density_;
    uint16_t* temperature_;
    int* num_of_active_particles_;
    int num_of_particles_;
};

#endif // _FLIP_H_