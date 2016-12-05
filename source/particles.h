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

#ifndef _PARTICLES_H_
#define _PARTICLES_H_

#include <memory>

#include "fluid_solver/fluid_buffer_owner.h"
#include "graphics_lib_enum.h"
#include "graphics_linear_mem.h"
#include "third_party/glm/fwd.hpp"

class GraphicsMemPiece;
class GraphicsVolume3;
class Particles : public FluidBufferOwner
{
public:
    Particles(int max_num_particles);
    virtual ~Particles();

    // Overridden from FluidBufferOwner:
    virtual GraphicsMemPiece* GetActiveParticleCountMemPiece() override;
    virtual GraphicsVolume* GetDensityVolume() override;
    virtual GraphicsVolume3* GetVelocityField() override;
    virtual GraphicsLinearMemU16* GetParticleDensityField() override;
    virtual GraphicsLinearMemU16* GetParticlePosXField() override;
    virtual GraphicsLinearMemU16* GetParticlePosYField() override;
    virtual GraphicsLinearMemU16* GetParticlePosZField() override;
    virtual GraphicsLinearMemU16* GetParticleTemperatureField() override;
    virtual GraphicsVolume* GetTemperatureVolume() override;

    void Advect(float time_step, const GraphicsVolume3* velocity_field);
    void Emit(const glm::vec3& location, float radius, float density);
    bool Initialize(GraphicsLib lib);

private:
    int max_num_particles_;
    std::shared_ptr<GraphicsLinearMemU16> position_x_;
    std::shared_ptr<GraphicsLinearMemU16> position_y_;
    std::shared_ptr<GraphicsLinearMemU16> position_z_;
    std::shared_ptr<GraphicsLinearMemU16> density_;
    std::shared_ptr<GraphicsLinearMemU16> life_;
    std::shared_ptr<GraphicsMemPiece> tail_;
};

#endif // _FLIP_FLUID_SOLVER_H_