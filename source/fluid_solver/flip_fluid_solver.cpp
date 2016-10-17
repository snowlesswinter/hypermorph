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
#include "flip_fluid_solver.h"

#include "cuda_host/cuda_main.h"
#include "cuda_host/cuda_volume.h"
#include "graphics_linear_mem.h"
#include "graphics_mem_piece.h"
#include "graphics_volume.h"
#include "graphics_volume_group.h"
#include "metrics.h"
#include "poisson_solver/poisson_solver.h"
#include "third_party/glm/vec2.hpp"
#include "third_party/glm/vec3.hpp"

namespace
{
enum DiagnosisTarget
{
    DIAG_NONE,
    DIAG_VELOCITY,
    DIAG_PRESSURE,
    DIAG_CURL,
    DIAG_DELTA_VORT,
    DIAG_PSI,

    NUM_DIAG_TARGETS
};

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

template <typename U, typename V>
void SetCudaParticles(U* cuda_p, const V& p)
{
    cuda_p->particle_index_   = p->particle_index_->cuda_linear_mem();
    cuda_p->cell_index_       = p->cell_index_->cuda_linear_mem();
    cuda_p->in_cell_index_    = p->in_cell_index_->cuda_linear_mem();
    cuda_p->particle_count_   = p->particle_count_->cuda_linear_mem();
    cuda_p->position_x_       = p->position_x_->cuda_linear_mem();
    cuda_p->position_y_       = p->position_y_->cuda_linear_mem();
    cuda_p->position_z_       = p->position_z_->cuda_linear_mem();
    cuda_p->velocity_x_       = p->velocity_x_->cuda_linear_mem();
    cuda_p->velocity_y_       = p->velocity_y_->cuda_linear_mem();
    cuda_p->velocity_z_       = p->velocity_z_->cuda_linear_mem();
    cuda_p->density_          = p->density_->cuda_linear_mem();
    //cuda_p->temperature_      = p->temperature_->cuda_linear_mem();
    cuda_p->num_of_actives_   = p->num_of_actives_->cuda_mem_piece();
    cuda_p->num_of_particles_ = p->num_of_particles_;
}

const int kNumParticles = 1000000;
} // Anonymous namespace

struct FlipFluidSolver::FlipParticles
{
    // TODO: Two set of particles can share some fields.
    std::shared_ptr<GraphicsLinearMemU32> particle_index_;
    std::shared_ptr<GraphicsLinearMemU32> cell_index_;
    std::shared_ptr<GraphicsLinearMemU32> particle_count_;
    std::shared_ptr<GraphicsLinearMemU8>  in_cell_index_;
    std::shared_ptr<GraphicsLinearMemU16> position_x_;
    std::shared_ptr<GraphicsLinearMemU16> position_y_;
    std::shared_ptr<GraphicsLinearMemU16> position_z_;
    std::shared_ptr<GraphicsLinearMemU16> velocity_x_;
    std::shared_ptr<GraphicsLinearMemU16> velocity_y_;
    std::shared_ptr<GraphicsLinearMemU16> velocity_z_;
    std::shared_ptr<GraphicsLinearMemU16> density_;
    std::shared_ptr<GraphicsLinearMemU16> temperature_;
    std::shared_ptr<GraphicsMemPiece>     num_of_actives_;
    int                                   num_of_particles_;

    FlipParticles(GraphicsLib lib)
        : particle_index_(std::make_shared<GraphicsLinearMemU32>(lib))
        , cell_index_(std::make_shared<GraphicsLinearMemU32>(lib))
        , particle_count_(std::make_shared<GraphicsLinearMemU32>(lib))
        , in_cell_index_(std::make_shared<GraphicsLinearMemU8>(lib))
        , position_x_(std::make_shared<GraphicsLinearMemU16>(lib))
        , position_y_(std::make_shared<GraphicsLinearMemU16>(lib))
        , position_z_(std::make_shared<GraphicsLinearMemU16>(lib))
        , velocity_x_(std::make_shared<GraphicsLinearMemU16>(lib))
        , velocity_y_(std::make_shared<GraphicsLinearMemU16>(lib))
        , velocity_z_(std::make_shared<GraphicsLinearMemU16>(lib))
        , density_(std::make_shared<GraphicsLinearMemU16>(lib))
        , temperature_(std::make_shared<GraphicsLinearMemU16>(lib))
        , num_of_actives_(std::make_shared<GraphicsMemPiece>(lib))
        , num_of_particles_(0)
    {
    }
};

FlipFluidSolver::FlipFluidSolver()
    : FluidSolver()
    , graphics_lib_(GRAPHICS_LIB_CUDA)
    , grid_size_(128)
    , pressure_solver_(nullptr)
    , particles_(new FlipParticles(graphics_lib_))
    , aux_()
    , frame_(0)
{

}

FlipFluidSolver::~FlipFluidSolver()
{

}

void FlipFluidSolver::Impulse(GraphicsVolume* density, float splat_radius,
                              const glm::vec3& impulse_position,
                              const glm::vec3& hotspot, float impulse_density,
                              float impulse_temperature)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ApplyImpulseDensity(density->cuda_volume(),
                                                  impulse_position, hotspot,
                                                  splat_radius,
                                                  impulse_density);
        CudaMain::Instance()->ApplyImpulse(temperature_->cuda_volume(),
                                           temperature_->cuda_volume(),
                                           impulse_position, hotspot,
                                           splat_radius, impulse_temperature);
    }
}

bool FlipFluidSolver::Initialize(GraphicsLib graphics_lib, int width,
                                 int height, int depth)
{
    velocity_ = std::make_shared<GraphicsVolume3>(graphics_lib_);
    velocity_prev_ = std::make_shared<GraphicsVolume3>(graphics_lib_);
    temperature_ = std::make_shared<GraphicsVolume>(graphics_lib_);
    general1a_ = std::make_shared<GraphicsVolume>(graphics_lib_);
    general1b_ = std::make_shared<GraphicsVolume>(graphics_lib_);

    grid_size_ = glm::ivec3(width, height, depth);

    bool result = velocity_->Create(width, height, depth, 1, 2, 0);
    assert(result);
    if (!result)
        return false;

    result = velocity_prev_->Create(width, height, depth, 1, 2, 0);
    assert(result);
    if (!result)
        return false;

    result = temperature_->Create(width, height, depth, 1, 2, 0);
    assert(result);
    if (!result)
        return false;

    result = general1a_->Create(width, height, depth, 1, 2, 0);
    assert(result);
    if (!result)
        return false;

    result = general1b_->Create(width, height, depth, 1, 2, 0);
    assert(result);
    if (!result)
        return false;

    int cell_count = grid_size_.x * grid_size_.y * grid_size_.z;
    result = InitParticles(particles_.get(), graphics_lib_, cell_count);
    assert(result);
    if (!result)
        return false;

    result = InitParticleField(&aux_, graphics_lib, kNumParticles);
    assert(result);
    if (!result)
        return false;

    Reset();
    return true;
}

void FlipFluidSolver::Reset()
{
    if (temperature_)
        temperature_->Clear();

    if (general1a_)
        general1a_->Clear();

    if (general1b_)
        general1b_->Clear();

    if (velocity_ && *velocity_) {
        velocity_->x()->Clear();
        velocity_->y()->Clear();
        velocity_->z()->Clear();
    }

    if (velocity_prev_ && *velocity_prev_) {
        velocity_prev_->x()->Clear();
        velocity_prev_->y()->Clear();
        velocity_prev_->z()->Clear();
    }

    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::FlipParticles p;
        SetCudaParticles(&p, particles_);
        CudaMain::Instance()->ResetParticles(&p);
    }

    frame_ = 0;
    Metrics::Instance()->Reset();
}

void FlipFluidSolver::SetDiagnosis(int diagnosis)
{

}

void FlipFluidSolver::SetPressureSolver(PoissonSolver* solver)
{
    pressure_solver_ = solver;
}

void FlipFluidSolver::Solve(GraphicsVolume* density, float delta_time)
{
    Metrics::Instance()->OnFrameUpdateBegins();

    // Calculate divergence.
    ComputeDivergence(general1a_);
    Metrics::Instance()->OnDivergenceComputed();

    // Solve pressure-velocity Poisson equation
    SolvePressure(general1b_, general1a_, 1);
    Metrics::Instance()->OnPressureSolved();

    // Rectify velocity via the gradient of pressure
    SubtractGradient(general1b_);
    Metrics::Instance()->OnVelocityRectified();

    // Advect density and temperature
    AdvectTemperature(delta_time);
    Metrics::Instance()->OnTemperatureAvected();

    MoveParticles(density, delta_time);
    Metrics::Instance()->OnVelocityAvected();

    // TODO: Apply buoyancy directly inside the move particles kernel.

    // Apply buoyancy and gravity
    ApplyBuoyancy(*density, delta_time);
    Metrics::Instance()->OnBuoyancyApplied();

    CudaMain::Instance()->RoundPassed(frame_++);
}

bool FlipFluidSolver::InitParticles(FlipParticles* particles, GraphicsLib lib,
                                    int cell_count)
{
    bool result = true;
    int n = kNumParticles;
    particles->num_of_particles_ = n;
    particles->num_of_actives_->Create(sizeof(int));

    result &= InitParticleField(&particles->particle_index_, lib, cell_count);
    result &= InitParticleField(&particles->cell_index_,     lib, n);
    result &= InitParticleField(&particles->in_cell_index_,  lib, n);
    result &= InitParticleField(&particles->particle_count_, lib, cell_count);
    result &= InitParticleField(&particles->position_x_,     lib, n);
    result &= InitParticleField(&particles->position_y_,     lib, n);
    result &= InitParticleField(&particles->position_z_,     lib, n);
    result &= InitParticleField(&particles->velocity_x_,     lib, n);
    result &= InitParticleField(&particles->velocity_y_,     lib, n);
    result &= InitParticleField(&particles->velocity_z_,     lib, n);
    result &= InitParticleField(&particles->density_,        lib, n);
    //result &= InitParticleField(&particles->temperature_,    lib, n);

    return result;
}

void FlipFluidSolver::AdvectTemperature(float delta_time)
{
    float temperature_dissipation = GetProperties().temperature_dissipation_;
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->AdvectField(general1a_->cuda_volume(),
                                          temperature_->cuda_volume(),
                                          velocity_->x()->cuda_volume(),
                                          velocity_->y()->cuda_volume(),
                                          velocity_->z()->cuda_volume(),
                                          general1b_->cuda_volume(),
                                          delta_time, temperature_dissipation);
    }

    std::swap(temperature_, general1a_);
}

void FlipFluidSolver::ApplyBuoyancy(const GraphicsVolume& density,
                                    float delta_time)
{
    float smoke_weight = GetProperties().weight_;
    float ambient_temperature = GetProperties().ambient_temperature_;
    float buoyancy_coef = GetProperties().buoyancy_coef_;
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ApplyBuoyancy(velocity_->x()->cuda_volume(),
                                            velocity_->y()->cuda_volume(),
                                            velocity_->z()->cuda_volume(),
                                            velocity_prev_->x()->cuda_volume(),
                                            velocity_prev_->y()->cuda_volume(),
                                            velocity_prev_->z()->cuda_volume(),
                                            temperature_->cuda_volume(),
                                            density.cuda_volume(), delta_time,
                                            ambient_temperature,
                                            buoyancy_coef, smoke_weight);
    }
}

void FlipFluidSolver::ComputeDivergence(
    std::shared_ptr<GraphicsVolume> divergence)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ComputeDivergence(divergence->cuda_volume(),
                                                velocity_->x()->cuda_volume(),
                                                velocity_->y()->cuda_volume(),
                                                velocity_->z()->cuda_volume());
    }
}

void FlipFluidSolver::MoveParticles(GraphicsVolume* density, float delta_time)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::FlipParticles p;
        SetCudaParticles(&p, particles_);
        CudaMain::Instance()->MoveParticles(&p, aux_->cuda_linear_mem(),
                                            velocity_->x()->cuda_volume(),
                                            velocity_->y()->cuda_volume(),
                                            velocity_->z()->cuda_volume(),
                                            velocity_prev_->x()->cuda_volume(),
                                            velocity_prev_->y()->cuda_volume(), 
                                            velocity_prev_->z()->cuda_volume(),
                                            density->cuda_volume(),
                                            temperature_->cuda_volume(),
                                            delta_time);
    }
}

void FlipFluidSolver::SolvePressure(std::shared_ptr<GraphicsVolume> pressure,
                                    std::shared_ptr<GraphicsVolume> divergence,
                                    int num_iterations)
{
    if (pressure_solver_) {
        pressure_solver_->SetDiagnosis(diagnosis_ == DIAG_PRESSURE);
        pressure_solver_->Solve(pressure, divergence, num_iterations);
    }
}

void FlipFluidSolver::SubtractGradient(std::shared_ptr<GraphicsVolume> pressure)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->SubtractGradient(velocity_->x()->cuda_volume(),
                                               velocity_->y()->cuda_volume(),
                                               velocity_->z()->cuda_volume(),
                                               pressure->cuda_volume());
    }
}
