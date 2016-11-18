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

#ifndef _FLUID_SIMULATOR_H_
#define _FLUID_SIMULATOR_H_

#include <memory>

#include "graphics_lib_enum.h"
#include "graphics_volume_group.h"
#include "poisson_solver/poisson_solver_enum.h"
#include "third_party/glm/vec3.hpp"

class FluidBufferOwner;
class FluidSolver;
class FluidUnittest;
class GraphicsVolume;
class OpenBoundaryMultigridPoissonSolver;
class PoissonCore;
class PoissonSolver;
class FluidSimulator
{
public:
    FluidSimulator();
    ~FluidSimulator();

    bool Init();
    void Reset();
    bool IsImpulsing() const;
    void NotifyConfigChanged();
    void StartImpulsing(float x, float y);
    void StopImpulsing();
    void Update(float delta_time, double seconds_elapsed, int frame_count);
    void UpdateImpulsing(float x, float y);

    FluidBufferOwner* buf_owner() const { return buf_owner_; }
    void set_solver_choice(PoissonSolverEnum ps) { solver_choice_ = ps; }
    void set_diagnosis(int diagnosis);
    GraphicsLib graphics_lib() const { return graphics_lib_; }
    void set_graphics_lib(GraphicsLib lib) { graphics_lib_ = lib; }
    void set_grid_size(const glm::ivec3& size) { grid_size_ = size; }

private:
    PoissonSolver* GetPressureSolver();
    FluidSolver* GetFluidSolver();

    void SetFluidProperties(FluidSolver* fluid_solver);
    
    glm::ivec3 grid_size_;
    int data_byte_width_;
    GraphicsLib graphics_lib_;
    std::unique_ptr<FluidSolver> fluid_solver_;
    FluidBufferOwner* buf_owner_;
    PoissonSolverEnum solver_choice_;
    std::unique_ptr<PoissonCore> multigrid_core_;
    std::unique_ptr<PoissonSolver> pressure_solver_;
    std::unique_ptr<PoissonSolver> psi_solver_;
    std::shared_ptr<glm::vec2> manual_impulse_;
};

#endif // _FLUID_SIMULATOR_H_