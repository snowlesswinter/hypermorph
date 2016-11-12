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
    cuda_p->particle_index_   = p->particle_index_ ? p->particle_index_->cuda_linear_mem() : nullptr;
    cuda_p->cell_index_       = p->cell_index_->cuda_linear_mem();
    cuda_p->in_cell_index_    = p->in_cell_index_->cuda_linear_mem();
    cuda_p->particle_count_   = p->particle_count_ ? p->particle_count_->cuda_linear_mem() : nullptr;
    cuda_p->position_x_       = p->position_x_->cuda_linear_mem();
    cuda_p->position_y_       = p->position_y_->cuda_linear_mem();
    cuda_p->position_z_       = p->position_z_->cuda_linear_mem();
    cuda_p->velocity_x_       = p->velocity_x_->cuda_linear_mem();
    cuda_p->velocity_y_       = p->velocity_y_->cuda_linear_mem();
    cuda_p->velocity_z_       = p->velocity_z_->cuda_linear_mem();
    cuda_p->density_          = p->density_->cuda_linear_mem();
    cuda_p->temperature_      = p->temperature_->cuda_linear_mem();
    cuda_p->num_of_actives_   = p->num_of_actives_ ? p->num_of_actives_->cuda_mem_piece() : nullptr;
    cuda_p->num_of_particles_ = p->num_of_particles_;
}
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
        : particle_index_()
        , cell_index_()
        , particle_count_()
        , in_cell_index_()
        , position_x_()
        , position_y_()
        , position_z_()
        , velocity_x_()
        , velocity_y_()
        , velocity_z_()
        , density_()
        , temperature_()
        , num_of_actives_()
        , num_of_particles_(0)
    {
    }
};

FlipFluidSolver::FlipFluidSolver(int max_num_particles)
    : FluidSolver()
    , FluidBufferOwner()
    , graphics_lib_(GRAPHICS_LIB_CUDA)
    , grid_size_(128)
    , max_num_particles_(max_num_particles)
    , pressure_solver_(nullptr)
    , diagnosis_(DIAG_NONE)
    , velocity_()
    , velocity_prev_()
    , density_()
    , temperature_()
    , general1a_()
    , general1b_()
    , diagnosis_volume_()
    , particles_(new FlipParticles(graphics_lib_))
    , particles_aux_(new FlipParticles(graphics_lib_))
    , frame_(0)
    , num_active_particles_(0)
{
}

FlipFluidSolver::~FlipFluidSolver()
{
}

void FlipFluidSolver::Impulse(float splat_radius,
                              const glm::vec3& impulse_position,
                              const glm::vec3& hotspot, float impulse_density,
                              float impulse_temperature, float impulse_velocity)
{
    Metrics::Instance()->OnFrameUpdateBegins();

    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::FlipParticles p;
        SetCudaParticles(&p, particles_);
        CudaMain::Instance()->EmitParticles(&p, impulse_position, hotspot,
                                            splat_radius, impulse_density,
                                            impulse_temperature,
                                            glm::vec3(impulse_velocity),
                                            density_->cuda_volume()->size());
    }
}

bool FlipFluidSolver::Initialize(GraphicsLib graphics_lib, int width,
                                 int height, int depth)
{
    velocity_ = std::make_shared<GraphicsVolume3>(graphics_lib_);
    velocity_prev_ = std::make_shared<GraphicsVolume3>(graphics_lib_);
    density_ = std::make_shared<GraphicsVolume>(graphics_lib_);
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

    result = density_->Create(width, height, depth, 1, 2, 0);
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
    result = InitParticles(particles_.get(), graphics_lib_, cell_count,
                           max_num_particles_, false);
    assert(result);
    if (!result)
        return false;

    result = InitParticles(particles_aux_.get(), graphics_lib_, cell_count,
                           max_num_particles_, true);
    assert(result);
    if (!result)
        return false;

    Reset();
    return true;
}

void FlipFluidSolver::Reset()
{
    if (density_)
        density_->Clear();

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
        CudaMain::Instance()->ResetParticles(&p, grid_size_);
    }

    frame_ = 0;
    Metrics::Instance()->Reset();
}

void FlipFluidSolver::SetDiagnosis(int diagnosis)
{
    diagnosis_ = diagnosis % NUM_DIAG_TARGETS;
}

void FlipFluidSolver::SetPressureSolver(PoissonSolver* solver)
{
    pressure_solver_ = solver;
}

void FlipFluidSolver::Solve(float delta_time)
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

    MoveParticles(delta_time);
    Metrics::Instance()->OnVelocityAvected();
    Metrics::Instance()->OnParticleNumberUpdated(num_active_particles_);

    // Apply buoyancy and gravity
    ApplyBuoyancy(delta_time);
    Metrics::Instance()->OnBuoyancyApplied();

    CudaMain::Instance()->RoundPassed(frame_++);
}

GraphicsMemPiece* FlipFluidSolver::GetActiveParticleCountMemPiece()
{
    return particles_->num_of_actives_.get();
}

GraphicsVolume* FlipFluidSolver::GetDensityVolume()
{
    return density_.get();
}

GraphicsLinearMemU16* FlipFluidSolver::GetParticleDensityField()
{
    return particles_->density_.get();
}

GraphicsLinearMemU16* FlipFluidSolver::GetParticlePosXField()
{
    return particles_->position_x_.get();
}

GraphicsLinearMemU16* FlipFluidSolver::GetParticlePosYField()
{
    return particles_->position_y_.get();
}

GraphicsLinearMemU16* FlipFluidSolver::GetParticlePosZField()
{
    return particles_->position_z_.get();
}

GraphicsLinearMemU16* FlipFluidSolver::GetParticleTemperatureField()
{
    return particles_->temperature_.get();
}

GraphicsVolume* FlipFluidSolver::GetTemperatureVolume()
{
    return temperature_.get();
}

bool FlipFluidSolver::InitParticles(FlipParticles* particles, GraphicsLib lib,
                                    int cell_count, int max_num_particles,
                                    bool aux)
{
    bool result = true;
    int n = max_num_particles;
    particles->num_of_particles_ = n;

    result &= InitParticleField(&particles->cell_index_,    lib, n);
    result &= InitParticleField(&particles->in_cell_index_, lib, n);
    result &= InitParticleField(&particles->position_x_,    lib, n);
    result &= InitParticleField(&particles->position_y_,    lib, n);
    result &= InitParticleField(&particles->position_z_,    lib, n);
    result &= InitParticleField(&particles->velocity_x_,    lib, n);
    result &= InitParticleField(&particles->velocity_y_,    lib, n);
    result &= InitParticleField(&particles->velocity_z_,    lib, n);
    result &= InitParticleField(&particles->density_,       lib, n);
    result &= InitParticleField(&particles->temperature_,   lib, n);

    if (!aux) {
        particles->num_of_actives_ = std::make_shared<GraphicsMemPiece>(lib);
        result &= particles->num_of_actives_->Create(sizeof(int));

        result &= InitParticleField(&particles->particle_index_, lib,
                                    cell_count);
        result &= InitParticleField(&particles->particle_count_, lib,
                                    cell_count);
    }

    return result;
}

void FlipFluidSolver::ApplyBuoyancy(float delta_time)
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
                                            density_->cuda_volume(), delta_time,
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

void FlipFluidSolver::ComputeResidualDiagnosis(
    std::shared_ptr<GraphicsVolume> pressure,
    std::shared_ptr<GraphicsVolume> divergence)
{
    if (diagnosis_ != DIAG_PRESSURE)
        return;

    if (!diagnosis_volume_) {
        int width = pressure->GetWidth();
        int height = pressure->GetHeight();
        int depth = pressure->GetDepth();
        std::shared_ptr<GraphicsVolume> v(new GraphicsVolume(graphics_lib_));
        bool result = v->Create(width, height, depth, 1, 4, 0);
        assert(result);
        if (!result)
            return;

        diagnosis_volume_ = v;
    }

    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ComputeResidualDiagnosis(
            diagnosis_volume_->cuda_volume(), pressure->cuda_volume(),
            divergence->cuda_volume());
    }
}

void FlipFluidSolver::MoveParticles(float delta_time)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::FlipParticles p;
        SetCudaParticles(&p, particles_);
        CudaMain::FlipParticles p_aux;
        SetCudaParticles(&p_aux, particles_aux_);
        CudaMain::Instance()->MoveParticles(
            &p, &num_active_particles_, &p_aux, velocity_->x()->cuda_volume(),
            velocity_->y()->cuda_volume(), velocity_->z()->cuda_volume(),
            velocity_prev_->x()->cuda_volume(),
            velocity_prev_->y()->cuda_volume(),
            velocity_prev_->z()->cuda_volume(), density_->cuda_volume(),
            temperature_->cuda_volume(), GetProperties().velocity_dissipation_,
            GetProperties().density_dissipation_,
            GetProperties().temperature_dissipation_, delta_time);

        SwapParticleFields(particles_.get(), particles_aux_.get());
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

    ComputeResidualDiagnosis(pressure, divergence);
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

void FlipFluidSolver::SwapParticleFields(FlipParticles* particles,
                                         FlipParticles* aux)
{
    std::swap(particles->cell_index_,    aux->cell_index_);
    std::swap(particles->in_cell_index_, aux->in_cell_index_);
    std::swap(particles->position_x_,    aux->position_x_);
    std::swap(particles->position_y_,    aux->position_y_);
    std::swap(particles->position_z_,    aux->position_z_);
    std::swap(particles->velocity_x_,    aux->velocity_x_);
    std::swap(particles->velocity_y_,    aux->velocity_y_);
    std::swap(particles->velocity_z_,    aux->velocity_z_);
    std::swap(particles->density_,       aux->density_);
    std::swap(particles->temperature_,   aux->temperature_);
}
