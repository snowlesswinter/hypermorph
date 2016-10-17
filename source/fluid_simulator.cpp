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

#include "stdafx.h"
#include "fluid_simulator.h"

#include <cassert>

#include "cuda_host/cuda_main.h"
#include "cuda_host/cuda_volume.h"
#include "fluid_config.h"
#include "fluid_solver/flip_fluid_solver.h"
#include "fluid_solver/grid_fluid_solver.h"
#include "graphics_volume.h"
#include "metrics.h"
#include "opengl/gl_volume.h"
#include "poisson_solver/full_multigrid_poisson_solver.h"
#include "poisson_solver/poisson_core_cuda.h"
#include "poisson_solver/poisson_core_glsl.h"
#include "poisson_solver/multigrid_poisson_solver.h"
#include "poisson_solver/open_boundary_multigrid_poisson_solver.h"
#include "poisson_solver/preconditioned_conjugate_gradient.h"
#include "third_party/glm/vec2.hpp"
#include "third_party/opengl/glew.h"
#include "utility.h"

const float kMaxTimeStep = 0.3f;
const glm::vec3 kImpulsePosition(0.5f, 0.05f, 0.5f);

FluidSimulator::FluidSimulator()
    : grid_size_(128)
    , cell_size_(0.15f)
    , data_byte_width_(2)
    , graphics_lib_(GRAPHICS_LIB_CUDA)
    , fluid_solver_()
    , solver_choice_(POISSON_SOLVER_FULL_MULTI_GRID)
    , multigrid_core_()
    , pressure_solver_()
    , psi_solver_()
    , density_()
    , manual_impulse_()
{
}

FluidSimulator::~FluidSimulator()
{
}

bool FluidSimulator::Init()
{
    glm::uvec3 grid_size = FluidConfig::Instance()->grid_size();

    int width  = static_cast<int>(grid_size.x);
    int height = static_cast<int>(grid_size.y);
    int depth  = static_cast<int>(grid_size.z);

    density_ = std::make_shared<GraphicsVolume>(graphics_lib_);

    bool result = density_->Create(width, height, depth, 1, 2, 0);
    assert(result);
    if (!result)
        return false;

    PoissonSolver* pressure_solver = GetPressureSolver();
    FluidSolver* fluid_solver = GetFluidSolver();

    FluidSolver::FluidProperties properties;
    properties.ambient_temperature_ =
        FluidConfig::Instance()->ambient_temperature();
    properties.buoyancy_coef_ = FluidConfig::Instance()->smoke_buoyancy();
    properties.density_dissipation_ =
        FluidConfig::Instance()->density_dissipation();
    properties.temperature_dissipation_ =
        FluidConfig::Instance()->temperature_dissipation();
    properties.velocity_dissipation_ =
        FluidConfig::Instance()->velocity_dissipation();
    properties.vorticity_confinement_ =
        FluidConfig::Instance()->vorticity_confinement();
    properties.weight_ =
        FluidConfig::Instance()->smoke_weight();

    fluid_solver->SetProperties(properties);
    if (!fluid_solver->Initialize(graphics_lib_, width, height, depth))
        return false;

    fluid_solver->SetPressureSolver(pressure_solver);
    return true;
}

void FluidSimulator::Reset()
{
    density_->Clear();
    fluid_solver_->Reset();
}

bool FluidSimulator::IsImpulsing() const
{
    return !!manual_impulse_;
}

void FluidSimulator::NotifyConfigChanged()
{
    CudaMain::Instance()->SetCellSize(FluidConfig::Instance()->cell_size());
    CudaMain::Instance()->SetStaggered(FluidConfig::Instance()->staggered());
    CudaMain::Instance()->SetMidPoint(FluidConfig::Instance()->mid_point());
    CudaMain::Instance()->SetOutflow(FluidConfig::Instance()->outflow());
    CudaMain::Instance()->SetAdvectionMethod(
        FluidConfig::Instance()->advection_method());
    CudaMain::Instance()->SetFluidImpulse(
        FluidConfig::Instance()->fluid_impluse());
}

void FluidSimulator::StartImpulsing(float x, float y)
{
    manual_impulse_.reset(new glm::vec2(x, y));
}

void FluidSimulator::StopImpulsing()
{
    manual_impulse_.reset();
}

void FluidSimulator::Update(float delta_time, double seconds_elapsed,
                            int frame_count)
{
    int debug = 0;
    if (debug) {
        delta_time = 0.0f;
        seconds_elapsed = 0.0f;
        frame_count = 1;
    }

    float fixed_time_step = FluidConfig::Instance()->fixed_time_step();
    float proper_delta_time = fixed_time_step > 0.0f ?
        fixed_time_step : std::min(delta_time, kMaxTimeStep);

    float radius_factor = FluidConfig::Instance()->splat_radius_factor();
    float time_stretch = FluidConfig::Instance()->time_stretch() + 0.00001f;
    float impulse_density = FluidConfig::Instance()->impulse_density();
    float impulse_temperature = FluidConfig::Instance()->impulse_temperature();

    glm::vec3 pos = kImpulsePosition * glm::vec3(grid_size_);
    double ¦Ð = 3.1415926;
    float splat_radius = grid_size_.x * radius_factor;
    float sin_factor =
        static_cast<float>(sin(seconds_elapsed / time_stretch * 2.0 * ¦Ð));
    float cos_factor =
        static_cast<float>(cos(seconds_elapsed / time_stretch * 2.0 * ¦Ð));
    float hotspot_x = cos_factor * splat_radius * 0.8f + pos.x;
    float hotspot_z = sin_factor * splat_radius * 0.8f + pos.z;
    glm::vec3 hotspot(hotspot_x, 0.0f, hotspot_z);

    bool do_impulse = false;
    if (manual_impulse_) {
        hotspot = glm::vec3(0.5f * grid_size_.x * (manual_impulse_->x + 1.0f),
                            0.0f,
                            0.5f * grid_size_.z * (manual_impulse_->y + 1.0f));
        do_impulse = true;
    } else if (FluidConfig::Instance()->auto_impulse()) {
        do_impulse = true;
    }

    CudaMain::FluidImpulse impulse = FluidConfig::Instance()->fluid_impluse();
    if (impulse == CudaMain::IMPULSE_BUOYANT_JET) {
        pos.x = pos.y;
        pos.y = splat_radius + 2;
    }

    if (do_impulse)
        fluid_solver_->Impulse(density_.get(), splat_radius, pos, hotspot,
                               impulse_density, impulse_temperature);

    fluid_solver_->Solve(density_.get(), proper_delta_time);
}

void FluidSimulator::UpdateImpulsing(float x, float y)
{
    if (manual_impulse_) {
        *manual_impulse_ = glm::vec2(x, y);
    }
}

void FluidSimulator::set_diagnosis(int diagnosis)
{
    fluid_solver_->SetDiagnosis(diagnosis);
}

PoissonSolver* FluidSimulator::GetPressureSolver()
{
    if (!multigrid_core_) {
        if (graphics_lib_ == GRAPHICS_LIB_CUDA)
            multigrid_core_.reset(new PoissonCoreCuda());
        else
            multigrid_core_.reset(new PoissonCoreGlsl());
    }

    int num_iterations = 0;
    int num_nested_iterations =
        FluidConfig::Instance()->num_multigrid_iterations();
    switch (solver_choice_) {
        case POISSON_SOLVER_JACOBI:
        case POISSON_SOLVER_GAUSS_SEIDEL:
        case POISSON_SOLVER_DAMPED_JACOBI: {
            //num_iterations =
            //    FluidConfig::Instance()->num_jacobi_iterations();
            //DampedJacobi(pressure, divergence, cell_size,
            //             num_iterations);
            break;
        }
        case POISSON_SOLVER_MULTI_GRID: {
            if (!pressure_solver_) {
                pressure_solver_.reset(
                    new MultigridPoissonSolver(multigrid_core_.get()));
                pressure_solver_->Initialize(grid_size_.x, grid_size_.y,
                                             grid_size_.z, data_byte_width_,
                                             32);
            }
            num_iterations =
                FluidConfig::Instance()->num_multigrid_iterations();
            break;
        }
        case POISSON_SOLVER_FULL_MULTI_GRID: {
            if (!pressure_solver_) {
                pressure_solver_.reset(
                    new FullMultigridPoissonSolver(multigrid_core_.get()));
                pressure_solver_->Initialize(grid_size_.x, grid_size_.y,
                                             grid_size_.z, data_byte_width_,
                                             32);
            }
            num_iterations =
                FluidConfig::Instance()->num_full_multigrid_iterations();
            break;
        }
        case POISSON_SOLVER_MULTI_GRID_PRECONDITIONED_CONJUGATE_GRADIENT: {
            if (!pressure_solver_) {
                pressure_solver_.reset(
                    new PreconditionedConjugateGradient(multigrid_core_.get()));
                pressure_solver_->Initialize(grid_size_.x, grid_size_.y,
                                             grid_size_.z, data_byte_width_,
                                             32);
            }
            num_iterations =
                FluidConfig::Instance()->num_mgpcg_iterations();
            break;
        }
        default: {
            break;
        }
    }

    if (pressure_solver_)
        pressure_solver_->SetNestedSolverIterations(num_nested_iterations);

    return pressure_solver_.get();
}

std::shared_ptr<GraphicsVolume> FluidSimulator::GetDensityField() const
{
    return density_;
}

FluidSolver* FluidSimulator::GetFluidSolver()
{
    if (!fluid_solver_) {
        fluid_solver_.reset(new FlipFluidSolver());
    }

    return fluid_solver_.get();
}
