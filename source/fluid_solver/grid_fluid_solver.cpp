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
#include "grid_fluid_solver.h"

#include <cassert>

#include "cuda_host/cuda_main.h"
#include "cuda_host/cuda_volume.h"
#include "graphics_volume.h"
#include "graphics_volume_group.h"
#include "metrics.h"
#include "opengl/gl_volume.h"
#include "poisson_solver/poisson_solver.h"
#include "shader/fluid_shader.h"
#include "shader/multigrid_shader.h"
#include "third_party/glm/vec2.hpp"
#include "third_party/glm/vec3.hpp"
#include "third_party/opengl/glew.h"
#include "utility.h"

static struct
{
    GLuint Advect;
    GLuint Jacobi;
    GLuint DampedJacobi;
    GLuint compute_residual;
    GLuint SubtractGradient;
    GLuint ComputeDivergence;
    GLuint ApplyImpulse;
    GLuint ApplyBuoyancy;
    GLuint diagnose_;
} Programs;

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

GridFluidSolver::GridFluidSolver()
    : FluidSolver()
    , FluidBufferOwner()
    , graphics_lib_(GRAPHICS_LIB_CUDA)
    , grid_size_(128)
    , pressure_solver_(nullptr)
    , diagnosis_(DIAG_NONE)
    , velocity_()
    , velocity_prime_()
    , vorticity_()
    , aux_()
    , vort_conf_()
    , density_()
    , temperature_()
    , general1a_()
    , general1b_()
    , general1c_()
    , general1d_()
    , diagnosis_volume_()
    , frame_(0)
{
}

GridFluidSolver::~GridFluidSolver()
{
}

void GridFluidSolver::Impulse(float splat_radius,
                              const glm::vec3& impulse_position,
                              const glm::vec3& hotspot, float impulse_density,
                              float impulse_temperature, float impulse_velocity)
{

    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ApplyImpulse(velocity_->x()->cuda_volume(),
                                           velocity_->y()->cuda_volume(),
                                           velocity_->z()->cuda_volume(),
                                           density_->cuda_volume(),
                                           temperature_->cuda_volume(),
                                           velocity_->x()->cuda_volume(),
                                           velocity_->y()->cuda_volume(),
                                           velocity_->z()->cuda_volume(),
                                           density_->cuda_volume(),
                                           temperature_->cuda_volume(),
                                           impulse_position, hotspot,
                                           splat_radius, impulse_velocity,
                                           impulse_density,
                                           impulse_temperature);
    } else {
        ImpulseDensity(impulse_position, hotspot, splat_radius,
                       impulse_density);

        if (impulse_temperature > 0.0f)
            ImpulseField(temperature_, impulse_position, hotspot, splat_radius,
                         impulse_temperature);
    }
}

bool GridFluidSolver::Initialize(GraphicsLib graphics_lib, int width,
                                 int height, int depth, int poisson_byte_width)
{
    velocity_       = std::make_shared<GraphicsVolume3>(graphics_lib_);
    velocity_prime_ = std::make_shared<GraphicsVolume3>(graphics_lib_);
    vorticity_      = std::make_shared<GraphicsVolume3>(graphics_lib_);
    aux_            = std::make_shared<GraphicsVolume3>(graphics_lib_);
    vort_conf_      = std::make_shared<GraphicsVolume3>(graphics_lib_);
    density_        = std::make_shared<GraphicsVolume>(graphics_lib_);
    temperature_    = std::make_shared<GraphicsVolume>(graphics_lib_);
    general1a_      = std::make_shared<GraphicsVolume>(graphics_lib_);
    general1b_      = std::make_shared<GraphicsVolume>(graphics_lib_);
    general1c_      = std::make_shared<GraphicsVolume>(graphics_lib_);
    general1d_      = std::make_shared<GraphicsVolume>(graphics_lib_);

    // A hard lesson had told us: locality is a vital factor of the performance
    // of raycast. Even a trivial-like adjustment that packing the temperature
    // with the density field would surprisingly bring a 17% decline to the
    // performance. 
    //
    // Here is the analysis:
    // 
    // In the original design, density buffer is 128 ^ 3 * 2 byte = 4 MB,
    // where as the buffer had been increased to 128 ^ 3 * 6 byte = 12 MB in
    // our experiment(it is 6 bytes wide instead of 4 because we need to
    // swap it with the 3-byte-width buffer that shared with velocity buffer).
    // That expanded buffer size would greatly increase the possibility of
    // cache miss in GPU during raycast. So, it's a problem all about the cache
    // shortage in graphic cards.

    grid_size_ = glm::ivec3(width, height, depth);

    bool result = velocity_->Create(width, height, depth, 1, 2, 0);
    assert(result);
    if (!result)
        return false;

    result = velocity_prime_->Create(width, height, depth, 1, 2, 0);
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

    result = general1c_->Create(width, height, depth, 1, poisson_byte_width, 0);
    assert(result);
    if (!result)
        return false;

    result = general1d_->Create(width, height, depth, 1, poisson_byte_width, 0);
    assert(result);
    if (!result)
        return false;

    if (graphics_lib_ == GRAPHICS_LIB_GLSL ||
            graphics_lib_ == GRAPHICS_LIB_CUDA_DIAGNOSIS) {
        Programs.Advect = LoadProgram(FluidShader::Vertex(),
                                      FluidShader::PickLayer(),
                                      FluidShader::Advect());
        Programs.Jacobi = LoadProgram(FluidShader::Vertex(),
                                      FluidShader::PickLayer(),
                                      FluidShader::Jacobi());
        Programs.DampedJacobi = LoadProgram(
            FluidShader::Vertex(), FluidShader::PickLayer(),
            FluidShader::DampedJacobiPacked());
        Programs.compute_residual = LoadProgram(
            FluidShader::Vertex(), FluidShader::PickLayer(),
            MultigridShader::ComputeResidual());
        Programs.SubtractGradient = LoadProgram(
            FluidShader::Vertex(), FluidShader::PickLayer(),
            FluidShader::SubtractGradient());
        Programs.ComputeDivergence = LoadProgram(
            FluidShader::Vertex(), FluidShader::PickLayer(),
            FluidShader::ComputeDivergence());
        Programs.ApplyImpulse = LoadProgram(FluidShader::Vertex(),
                                            FluidShader::PickLayer(),
                                            FluidShader::Splat());
        Programs.ApplyBuoyancy = LoadProgram(FluidShader::Vertex(),
                                             FluidShader::PickLayer(),
                                             FluidShader::Buoyancy());
        Programs.diagnose_ = LoadProgram(
            FluidShader::Vertex(), FluidShader::PickLayer(),
            MultigridShader::ComputeResidualPackedDiagnosis());
    }

    Reset();
    return true;
}

void GridFluidSolver::Reset()
{
    if (density_)
        density_->Clear();

    if (temperature_)
        temperature_->Clear();

    if (general1a_)
        general1a_->Clear();

    if (general1b_)
        general1b_->Clear();

    if (general1c_)
        general1c_->Clear();

    if (general1d_)
        general1d_->Clear();

    if (velocity_ && *velocity_) {
        velocity_->x()->Clear();
        velocity_->y()->Clear();
        velocity_->z()->Clear();
    }

    if (velocity_prime_ && *velocity_prime_) {
        velocity_prime_->x()->Clear();
        velocity_prime_->y()->Clear();
        velocity_prime_->z()->Clear();
    }

    if (vorticity_ && *vorticity_) {
        vorticity_->x()->Clear();
        vorticity_->y()->Clear();
        vorticity_->z()->Clear();
    }

    if (aux_ && *aux_) {
        aux_->x()->Clear();
        aux_->y()->Clear();
        aux_->z()->Clear();
    }

    diagnosis_volume_.reset();
    frame_ = 0;

    Metrics::Instance()->Reset();
}

void GridFluidSolver::SetDiagnosis(int diagnosis)
{
    diagnosis_ = diagnosis % NUM_DIAG_TARGETS;
}

void GridFluidSolver::SetPressureSolver(PoissonSolver* solver)
{
    pressure_solver_ = solver;
}

void GridFluidSolver::Solve(float delta_time)
{
    Metrics::Instance()->OnFrameUpdateBegins();

    // Calculate divergence.
    ComputeDivergence(general1c_);
    Metrics::Instance()->OnDivergenceComputed();

    // Solve pressure-velocity Poisson equation
    SolvePressure(general1d_, general1c_);
    Metrics::Instance()->OnPressureSolved();

    // Rectify velocity via the gradient of pressure
    SubtractGradient(general1d_);
    Metrics::Instance()->OnVelocityRectified();

    // Advect density and temperature
    AdvectTemperature(delta_time);
    Metrics::Instance()->OnTemperatureAvected();

    AdvectDensity(delta_time);
    Metrics::Instance()->OnDensityAvected();

    // Advect velocity
    AdvectVelocity(delta_time);
    Metrics::Instance()->OnVelocityAvected();

    // Restore vorticity
    RestoreVorticity(delta_time);
    Metrics::Instance()->OnVorticityRestored();

    // Apply buoyancy and gravity
    ApplyBuoyancy(delta_time);
    Metrics::Instance()->OnBuoyancyApplied();

    ReviseDensity();

    // Recently in my experiments I examined the data generated by the passes
    // of simulation(for CUDA porting), and I found that in different times of
    // execution, the results always fluctuate a bit, even through I turned off
    // the random hotspot, this fluctuation remains.
    //
    // This system should have no any undetermined factor and random number
    // introduced, and the exactly same result should be produced every time
    // the simulation ran. The most suspicious part is that the in-place
    // modification pattern accessing the texture in the pressure solver, 
    // which may produce different results due to the undetermined order of
    // shader/kernel execution.
    // I may find some time to explore into it.

    CudaMain::Instance()->RoundPassed(frame_++);
}

GraphicsMemPiece* GridFluidSolver::GetActiveParticleCountMemPiece()
{
    return nullptr;
}

GraphicsVolume* GridFluidSolver::GetDensityVolume()
{
    return density_.get();
}

GraphicsLinearMemU16* GridFluidSolver::GetParticleDensityField()
{
    return nullptr;
}

GraphicsLinearMemU16* GridFluidSolver::GetParticlePosXField()
{
    return nullptr;
}

GraphicsLinearMemU16* GridFluidSolver::GetParticlePosYField()
{
    return nullptr;
}

GraphicsLinearMemU16* GridFluidSolver::GetParticlePosZField()
{
    return nullptr;
}

GraphicsLinearMemU16* GridFluidSolver::GetParticleTemperatureField()
{
    return nullptr;
}

GraphicsVolume* GridFluidSolver::GetTemperatureVolume()
{
    return temperature_.get();
}

void GridFluidSolver::AdvectDensity(float delta_time)
{
    float density_dissipation = GetProperties().density_dissipation_;
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->AdvectField(general1a_->cuda_volume(),
                                          density_->cuda_volume(),
                                          velocity_->x()->cuda_volume(),
                                          velocity_->y()->cuda_volume(),
                                          velocity_->z()->cuda_volume(),
                                          general1b_->cuda_volume(),
                                          delta_time, density_dissipation);
    } else {
        AdvectImpl(*density_, delta_time, density_dissipation);
    }
    density_->Swap(*general1a_);
}

void GridFluidSolver::AdvectImpl(const GraphicsVolume& source,
                                 float delta_time, float dissipation)
{
    glUseProgram(Programs.Advect);

    SetUniform("InverseSize", CalculateInverseSize(*source.gl_volume()));
    SetUniform("TimeStep", delta_time);
    SetUniform("Dissipation", dissipation);
    SetUniform("SourceTexture", 1);
    SetUniform("Obstacles", 2);

    glBindFramebuffer(GL_FRAMEBUFFER,
                      general1a_->gl_volume()->frame_buffer());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity_->x()->gl_volume()->texture_handle());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, source.gl_volume()->texture_handle());
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                          general1a_->gl_volume()->depth());
    ResetState();
}

void GridFluidSolver::AdvectTemperature(float delta_time)
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
    } else {
        AdvectImpl(*temperature_, delta_time, temperature_dissipation);
    }

    std::swap(temperature_, general1a_);
}

void GridFluidSolver::AdvectVelocity(float delta_time)
{
    float velocity_dissipation = GetProperties().velocity_dissipation_;
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->AdvectVelocity(
            velocity_prime_->x()->cuda_volume(),
            velocity_prime_->y()->cuda_volume(),
            velocity_prime_->z()->cuda_volume(), velocity_->x()->cuda_volume(),
            velocity_->y()->cuda_volume(), velocity_->z()->cuda_volume(),
            general1a_->cuda_volume(), delta_time, velocity_dissipation);
        velocity_->Swap(*velocity_prime_);

        if (diagnosis_ == DIAG_VELOCITY) {
            CudaMain::Instance()->PrintVolume(velocity_->x()->cuda_volume(),
                                              "VelocityX");
            CudaMain::Instance()->PrintVolume(velocity_->y()->cuda_volume(),
                                              "VelocityY");
            CudaMain::Instance()->PrintVolume(velocity_->z()->cuda_volume(),
                                              "VelocityZ");
        }
    } else {
        glUseProgram(Programs.Advect);

        SetUniform("InverseSize",
                   CalculateInverseSize(*velocity_->x()->gl_volume()));
        SetUniform("TimeStep", delta_time);
        SetUniform("Dissipation", velocity_dissipation);
        SetUniform("SourceTexture", 1);
        SetUniform("Obstacles", 2);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          general1a_->gl_volume()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D,
                      velocity_->x()->gl_volume()->texture_handle());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D,
                      velocity_->x()->gl_volume()->texture_handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              general1a_->gl_volume()->depth());
        ResetState();

        //std::swap(velocity2_.x(), general4a_);
    }

}

void GridFluidSolver::ApplyBuoyancy(float delta_time)
{
    float smoke_weight = GetProperties().weight_;
    float ambient_temperature = GetProperties().ambient_temperature_;
    float buoyancy_coef = GetProperties().buoyancy_coef_;
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ApplyBuoyancy(velocity_->x()->cuda_volume(),
                                            velocity_->y()->cuda_volume(),
                                            velocity_->z()->cuda_volume(),
                                            velocity_->x()->cuda_volume(),
                                            velocity_->y()->cuda_volume(),
                                            velocity_->z()->cuda_volume(),
                                            temperature_->cuda_volume(),
                                            density_->cuda_volume(), delta_time,
                                            ambient_temperature,
                                            buoyancy_coef, smoke_weight);
    } else {
        glUseProgram(Programs.ApplyBuoyancy);

        SetUniform("Velocity", 0);
        SetUniform("Temperature", 1);
        SetUniform("AmbientTemperature", ambient_temperature);
        SetUniform("TimeStep", delta_time);
        SetUniform("Sigma", buoyancy_coef);
        SetUniform("Kappa", smoke_weight);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          general1a_->gl_volume()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D,
                      velocity_->x()->gl_volume()->texture_handle());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, temperature_->gl_volume()->texture_handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              general1a_->gl_volume()->depth());
        ResetState();

        //std::swap(velocity_, general4a_);
    }
}

void GridFluidSolver::ComputeDivergence(
    std::shared_ptr<GraphicsVolume> divergence)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ComputeDivergence(divergence->cuda_volume(),
                                                velocity_->x()->cuda_volume(),
                                                velocity_->y()->cuda_volume(),
                                                velocity_->z()->cuda_volume());
    } else {
        float cell_size = 0.15f;
        float half_inverse_cell_size = 0.5f / cell_size;

        glUseProgram(Programs.ComputeDivergence);

        SetUniform("HalfInverseCellSize", half_inverse_cell_size);
        SetUniform("Obstacles", 1);
        SetUniform("velocity", 0);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          divergence->gl_volume()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D,
                      velocity_->x()->gl_volume()->texture_handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              divergence->gl_volume()->depth());
        ResetState();
    }
}

void GridFluidSolver::ComputeResidualDiagnosis(
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
    } else if (graphics_lib_ == GRAPHICS_LIB_GLSL) {
        float cell_size = 0.15f;
        float inverse_h_square = 1.0f / (cell_size * cell_size);

        glUseProgram(Programs.diagnose_);

        SetUniform("packed_tex", 0);
        SetUniform("inverse_h_square", inverse_h_square);

        diagnosis_volume_->gl_volume()->BindFrameBuffer();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, general1b_->gl_volume()->texture_handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              diagnosis_volume_->gl_volume()->depth());
        ResetState();

        // =====================================================================
        glFinish();
        GraphicsVolume* p = diagnosis_volume_.get();

        int w = p->GetWidth();
        int h = p->GetHeight();
        int d = p->GetDepth();
        int n = 1;
        int element_size = sizeof(float);
        GLenum format = GL_RED;

        static char* v = nullptr;
        if (!v)
            v = new char[w * h * d * element_size * n];

        memset(v, 0, w * h * d * element_size * n);
        p->gl_volume()->GetTexImage(v);

        float* f = (float*)v;
        double sum = 0.0;
        double q = 0.0;
        double m = 0.0;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < h; j++) {
                for (int k = 0; k < w; k++) {
                    for (int l = 0; l < n; l++) {
                        q = f[i * w * h * n + j * w * n + k * n + l];
                        //if (i == 30 && j == 0 && k == 56)
                            //if (q > 1)
                            sum += q;
                        m = std::max(q, m);
                    }
                }
            }
        }

        double avg = sum / (w * h * d);
        PrintDebugString("(GLSL) avg ||r||: %.8f,    max ||r||: %.8f\n", avg, m);
    }
}

void GridFluidSolver::DampedJacobi(std::shared_ptr<GraphicsVolume> pressure,
                                   std::shared_ptr<GraphicsVolume> divergence,
                                   float cell_size, int num_of_iterations)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->Relax(pressure->cuda_volume(),
                                    pressure->cuda_volume(),
                                    divergence->cuda_volume(),
                                    num_of_iterations);
    } else {
        float one_minus_omega = 0.33333333f;
        float minus_square_cell_size = -(cell_size * cell_size);
        float omega_over_beta = 0.11111111f;

        for (int i = 0; i < num_of_iterations; ++i) {
            glUseProgram(Programs.DampedJacobi);

            SetUniform("Alpha", minus_square_cell_size);
            SetUniform("InverseBeta", omega_over_beta);
            SetUniform("one_minus_omega", one_minus_omega);
            SetUniform("packed_tex", 0);

            glBindFramebuffer(GL_FRAMEBUFFER,
                              general1b_->gl_volume()->frame_buffer());
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_3D, general1b_->gl_volume()->texture_handle());
            glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                                  general1b_->gl_volume()->depth());
            ResetState();
        }
    }
}

void GridFluidSolver::ImpulseField(std::shared_ptr<GraphicsVolume> dest,
                                   const glm::vec3& position,
                                   const glm::vec3& hotspot, float splat_radius,
                                   float value)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
    } else {
        glUseProgram(Programs.ApplyImpulse);

        SetUniform("center_point", position);
        SetUniform("hotspot", hotspot);
        SetUniform("radius", splat_radius);
        SetUniform("fill_color_r", value);
        SetUniform("fill_color_g", value);

        glBindFramebuffer(GL_FRAMEBUFFER, dest->gl_volume()->frame_buffer());
        glEnable(GL_BLEND);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              dest->gl_volume()->depth());
        ResetState();
    }
}

void GridFluidSolver::ImpulseDensity(const glm::vec3& position,
                                     const glm::vec3& hotspot,
                                     float splat_radius, float value)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
    } else {
        glUseProgram(Programs.ApplyImpulse);

        SetUniform("center_point", position);
        SetUniform("hotspot", hotspot);
        SetUniform("radius", splat_radius);
        SetUniform("fill_color_r", value);
        SetUniform("fill_color_g", value);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          density_->gl_volume()->frame_buffer());
        glEnable(GL_BLEND);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              density_->gl_volume()->depth());
        ResetState();
    }
}

void GridFluidSolver::ReviseDensity()
{
    return;
    glm::vec3 pos(0.0f);
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ReviseDensity(
            density_->cuda_volume(), pos, grid_size_.x * 0.5f, 0.1f);
    }
}

void GridFluidSolver::SolvePressure(std::shared_ptr<GraphicsVolume> pressure,
                                    std::shared_ptr<GraphicsVolume> divergence)
{
    if (pressure_solver_) {
        pressure_solver_->SetDiagnosis(diagnosis_ == DIAG_PRESSURE);
        pressure_solver_->Solve(pressure, divergence);
    }

    ComputeResidualDiagnosis(pressure, divergence);
}

void GridFluidSolver::SubtractGradient(std::shared_ptr<GraphicsVolume> pressure)
{
    // In the original implementation, this coefficient was set to 1.125, which
    // I guess is a trick to compensate the inaccuracy of the solution of
    // Poisson equation. As the solution now becomes more and more precise,
    // I changed the number to 1.0 to keep the system stable.
    //
    // 2016/5/23 update:
    // During the process of verifying the staggered grid discretization, I
    // found this coefficient should be the same as that in divergence
    // calculation. This mistake was introduced at the first day the project
    // was created.
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->SubtractGradient(velocity_->x()->cuda_volume(),
                                               velocity_->y()->cuda_volume(),
                                               velocity_->z()->cuda_volume(),
                                               pressure->cuda_volume());
    } else {
        float cell_size = 0.15f;
        const float half_inverse_cell_size = 0.5f / cell_size;

        glUseProgram(Programs.SubtractGradient);

        SetUniform("GradientScale", half_inverse_cell_size);
        SetUniform("velocity", 0);
        SetUniform("packed_tex", 1);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          velocity_->x()->gl_volume()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D,
                      velocity_->x()->gl_volume()->texture_handle());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, pressure->gl_volume()->texture_handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              velocity_->x()->gl_volume()->depth());
        ResetState();
    }
}

void GridFluidSolver::AddCurlPsi(const GraphicsVolume3& psi)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->AddCurlPsi(velocity_->x()->cuda_volume(),
                                         velocity_->y()->cuda_volume(),
                                         velocity_->z()->cuda_volume(),
                                         psi.x()->cuda_volume(),
                                         psi.y()->cuda_volume(),
                                         psi.z()->cuda_volume());
    }
}

void GridFluidSolver::AdvectVortices(const GraphicsVolume3& vorticity,
                                     const GraphicsVolume3& temp,
                                     std::shared_ptr<GraphicsVolume> aux,
                                     float delta_time)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->AdvectVorticity(
            vorticity.x()->cuda_volume(), vorticity.y()->cuda_volume(),
            vorticity.z()->cuda_volume(), temp.x()->cuda_volume(),
            temp.y()->cuda_volume(), temp.z()->cuda_volume(),
            velocity_prime_->x()->cuda_volume(),
            velocity_prime_->y()->cuda_volume(),
            velocity_prime_->z()->cuda_volume(), aux->cuda_volume(), delta_time,
            0.0f);
    }
}

void GridFluidSolver::ApplyVorticityConfinemnet()
{
    const GraphicsVolume3& vort_conf = GetVorticityConfinementField();
    if (!vort_conf)
        return;

    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ApplyVorticityConfinement(
            velocity_->x()->cuda_volume(), velocity_->y()->cuda_volume(),
            velocity_->z()->cuda_volume(), vort_conf_->x()->cuda_volume(),
            vort_conf_->y()->cuda_volume(), vort_conf_->z()->cuda_volume());
    }

    //std::swap(velocity_, general4a_);
}

void GridFluidSolver::BuildVorticityConfinemnet(float delta_time)
{
    const GraphicsVolume3& vorticity = GetVorticityField();
    if (!vorticity)
        return;

    const GraphicsVolume3& vort_conf = GetVorticityConfinementField();
    if (!vort_conf)
        return;

    float vort_conf_coef = GetProperties().vorticity_confinement_;
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->BuildVorticityConfinement(
            vort_conf.x()->cuda_volume(), vort_conf.y()->cuda_volume(),
            vort_conf.z()->cuda_volume(), vorticity.x()->cuda_volume(),
            vorticity.y()->cuda_volume(), vorticity.z()->cuda_volume(),
            vort_conf_coef * delta_time);
    }
}

void GridFluidSolver::ComputeCurl(const GraphicsVolume3& vorticity,
                                  const GraphicsVolume3& velocity)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ComputeCurl(vorticity.x()->cuda_volume(),
                                          vorticity.y()->cuda_volume(),
                                          vorticity.z()->cuda_volume(),
                                          velocity.x()->cuda_volume(),
                                          velocity.y()->cuda_volume(),
                                          velocity.z()->cuda_volume());
        if (diagnosis_ == DIAG_CURL) {
            CudaMain::Instance()->PrintVolume(vorticity.x()->cuda_volume(),
                                              "CurlX");
            CudaMain::Instance()->PrintVolume(vorticity.y()->cuda_volume(),
                                              "CurlY");
            CudaMain::Instance()->PrintVolume(vorticity.z()->cuda_volume(),
                                              "CurlZ");
        }
    }
}

void GridFluidSolver::ComputeDeltaVorticity(const GraphicsVolume3& aux,
                                            const GraphicsVolume3& vorticity)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ComputeDeltaVorticity(
            aux.x()->cuda_volume(), aux.y()->cuda_volume(),
            aux.z()->cuda_volume(), vorticity.x()->cuda_volume(),
            vorticity.y()->cuda_volume(), vorticity.z()->cuda_volume());

        if (diagnosis_ == DIAG_DELTA_VORT) {
            CudaMain::Instance()->PrintVolume(aux.x()->cuda_volume(),
                                              "DeltaVortX");
            CudaMain::Instance()->PrintVolume(aux.y()->cuda_volume(),
                                              "DeltaVortY");
            CudaMain::Instance()->PrintVolume(aux.z()->cuda_volume(),
                                              "DeltaVortZ");
        }
    }
}

void GridFluidSolver::DecayVortices(const GraphicsVolume3& vorticity,
                                    std::shared_ptr<GraphicsVolume> aux,
                                    float delta_time)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ComputeDivergence(
            aux->cuda_volume(), velocity_prime_->x()->cuda_volume(),
            velocity_prime_->y()->cuda_volume(),
            velocity_prime_->z()->cuda_volume());
        CudaMain::Instance()->DecayVortices(
            vorticity.x()->cuda_volume(), vorticity.y()->cuda_volume(),
            vorticity.z()->cuda_volume(), aux->cuda_volume(), delta_time);
    }
}

void GridFluidSolver::RestoreVorticity(float delta_time)
{
    if (GetProperties().vorticity_confinement_ > 0.0f) {
        const GraphicsVolume3& vorticity = GetVorticityField();
        if (!vorticity)
            return;

        ComputeCurl(vorticity, *velocity_prime_);
        BuildVorticityConfinemnet(delta_time);

        if (false) {
            GraphicsVolume3 temp(general1a_, general1b_, general1c_);
            StretchVortices(temp, vorticity, delta_time);
            DecayVortices(temp, general1d_, delta_time);

            //////////////////////////
            general1d_->Clear();
            //////////////////////////

            AdvectVortices(vorticity, temp, general1d_, delta_time);

            //////////////////////////
            general1a_->Clear();
            general1b_->Clear();
            general1c_->Clear();
            //////////////////////////

            ComputeCurl(temp, *velocity_);
            ComputeDeltaVorticity(temp, vorticity);
            SolvePsi(vorticity, temp, 1);
            AddCurlPsi(vorticity);
        }

        ApplyVorticityConfinemnet();
    }
}

void GridFluidSolver::SolvePsi(const GraphicsVolume3& psi,
                               const GraphicsVolume3& delta_vort,
                               int num_iterations)
{
    if (!pressure_solver_) {
        for (int i = 0; i < psi.num_of_volumes(); i++) {
            psi[i]->Clear();
            for (int j = 0; j < num_iterations; j++)
                pressure_solver_->Solve(psi[i], delta_vort[i]);
        }

        if (diagnosis_ == DIAG_PSI) {
            CudaMain::Instance()->PrintVolume(psi.x()->cuda_volume(), "PsiX");
            CudaMain::Instance()->PrintVolume(psi.y()->cuda_volume(), "PsiY");
            CudaMain::Instance()->PrintVolume(psi.z()->cuda_volume(), "PsiZ");
        }
    }
}

void GridFluidSolver::StretchVortices(const GraphicsVolume3& vort_np1,
                                      const GraphicsVolume3& vorticity,
                                      float delta_time)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->StretchVortices(
            vort_np1.x()->cuda_volume(), vort_np1.y()->cuda_volume(),
            vort_np1.z()->cuda_volume(), velocity_prime_->x()->cuda_volume(),
            velocity_prime_->y()->cuda_volume(),
            velocity_prime_->z()->cuda_volume(),
            vorticity.x()->cuda_volume(), vorticity.y()->cuda_volume(),
            vorticity.z()->cuda_volume(), delta_time);
    }
}

const GraphicsVolume3& GridFluidSolver::GetVorticityField()
{
    if (!*vorticity_) {
        int width = static_cast<int>(grid_size_.x);
        int height = static_cast<int>(grid_size_.y);
        int depth = static_cast<int>(grid_size_.z);
        bool r = vorticity_->Create(width, height, depth, 1, 2, 0);
        assert(r);
    }

    return *vorticity_;
}

const GraphicsVolume3& GridFluidSolver::GetAuxField()
{
    if (!*aux_) {
        int width = static_cast<int>(grid_size_.x);
        int height = static_cast<int>(grid_size_.y);
        int depth = static_cast<int>(grid_size_.z);
        bool r = aux_->Create(width, height, depth, 1, 2, 0);
        assert(r);
    }

    return *aux_;
}

const GraphicsVolume3& GridFluidSolver::GetVorticityConfinementField()
{
    if (!*vort_conf_) {
        int width = static_cast<int>(grid_size_.x);
        int height = static_cast<int>(grid_size_.y);
        int depth = static_cast<int>(grid_size_.z);
        bool r = vort_conf_->Create(width, height, depth, 1, 2, 0);
        assert(r);
    }

    return *vort_conf_;
}
