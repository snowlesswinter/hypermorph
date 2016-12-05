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

#include "stdafx.h"
#include "particles.h"

#include "cuda_host/cuda_main.h"
#include "graphics_mem_piece.h"
#include "graphics_volume_group.h"

template <typename T>
bool InitParticleField(T* field, GraphicsLib lib, int n)
{
    T r = std::make_shared<T::element_type>(lib);
    if (r->Create(n)) {
        *field = r;
        return true;
    }

    return false;
}

Particles::Particles(int max_num_particles)
    : max_num_particles_(max_num_particles)
    , position_x_()
    , position_y_()
    , position_z_()
    , density_()
    , life_()
    , tail_()
{
}

Particles::~Particles()
{
}

GraphicsMemPiece* Particles::GetActiveParticleCountMemPiece()
{
    return nullptr;
}

GraphicsVolume* Particles::GetDensityVolume()
{
    return nullptr;
}

GraphicsVolume3* Particles::GetVelocityField()
{
    return nullptr;
}

GraphicsLinearMemU16* Particles::GetParticleDensityField()
{
    return density_.get();
}

GraphicsLinearMemU16* Particles::GetParticlePosXField()
{
    return position_x_.get();
}

GraphicsLinearMemU16* Particles::GetParticlePosYField()
{
    return position_y_.get();
}

GraphicsLinearMemU16* Particles::GetParticlePosZField()
{
    return position_z_.get();
}

GraphicsLinearMemU16* Particles::GetParticleTemperatureField()
{
    return nullptr;
}

GraphicsVolume* Particles::GetTemperatureVolume()
{
    return nullptr;
}

void Particles::Advect(float time_step, const GraphicsVolume3* velocity_field)
{
    CudaMain::Instance()->MoveParticles(position_x_->cuda_linear_mem(),
                                        position_y_->cuda_linear_mem(),
                                        position_z_->cuda_linear_mem(),
                                        density_->cuda_linear_mem(),
                                        life_->cuda_linear_mem(),
                                        max_num_particles_,
                                        velocity_field->x()->cuda_volume(),
                                        velocity_field->y()->cuda_volume(),
                                        velocity_field->z()->cuda_volume(),
                                        time_step);
}

void Particles::Emit(const glm::vec3& location, float radius, float density)
{
    CudaMain::Instance()->EmitParticles(position_x_->cuda_linear_mem(),
                                        position_y_->cuda_linear_mem(),
                                        position_z_->cuda_linear_mem(),
                                        density_->cuda_linear_mem(),
                                        life_->cuda_linear_mem(),
                                        tail_->cuda_mem_piece(),
                                        max_num_particles_, 5000, location,
                                        radius, density);
}

bool Particles::Initialize(GraphicsLib lib)
{
    bool result = true;
    int n = max_num_particles_;
    result &= InitParticleField(&position_x_, lib, n);
    result &= InitParticleField(&position_y_, lib, n);
    result &= InitParticleField(&position_z_, lib, n);
    result &= InitParticleField(&density_, lib, n);
    result &= InitParticleField(&life_, lib, n);

    tail_ = std::make_shared<GraphicsMemPiece>(lib);
    result &= tail_->Create(sizeof(int));
    return result;
}
