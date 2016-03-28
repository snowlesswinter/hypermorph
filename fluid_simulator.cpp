#include "stdafx.h"
#include "fluid_simulator.h"

#include <cassert>

#include "cuda_host/cuda_volume.h"
#include "full_multigrid_poisson_solver.h"
#include "graphics_volume.h"
#include "metrics.h"
#include "multigrid_core_cuda.h"
#include "multigrid_core_glsl.h"
#include "multigrid_poisson_solver.h"
#include "opengl/gl_texture.h"
#include "shader/fluid_shader.h"
#include "shader/multigrid_shader.h"
#include "third_party/opengl/glew.h"
#include "utility.h"
#include "vmath.hpp"

//TODO
#include "multigrid_core.h"
#include "cuda_host/cuda_main.h"

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
} Programs;

FluidSimulator::FluidSimulator()
    : solver_choice_(POISSON_SOLVER_DAMPED_JACOBI)
    , num_multigrid_iterations_(5)
    , num_full_multigrid_iterations_(2)
    , velocity_()
    , density_()
    , density2_()
    , temperature_()
    , general1_()
    , general4_()
    , graphics_lib_(GRAPHICS_LIB_GLSL)
{
}

FluidSimulator::~FluidSimulator()
{
}

bool FluidSimulator::Init()
{
    velocity_.reset(new GraphicsVolume(graphics_lib_));
    density_.reset(
        new GraphicsVolume(
            graphics_lib_ == GRAPHICS_LIB_CUDA ?
                GRAPHICS_LIB_CUDA_DIAGNOSIS : GRAPHICS_LIB_GLSL));
    density2_.reset(
        new GraphicsVolume(
            graphics_lib_ == GRAPHICS_LIB_CUDA ?
                GRAPHICS_LIB_CUDA_DIAGNOSIS : GRAPHICS_LIB_GLSL));
    temperature_.reset(new GraphicsVolume(graphics_lib_));
    general1_.reset(new GraphicsVolume(graphics_lib_));
    general4_.reset(new GraphicsVolume(graphics_lib_));

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

    bool result = density_->Create(GridWidth, GridHeight, GridDepth, 1, 2);
    assert(result);
    if (!result)
        return false;

    result = density2_->Create(GridWidth, GridHeight, GridDepth, 1, 2);
    assert(result);
    if (!result)
        return false;

    result = velocity_->Create(GridWidth, GridHeight, GridDepth, 4, 2);
    assert(result);
    if (!result)
        return false;

    result = temperature_->Create(GridWidth, GridHeight, GridDepth, 1, 2);
    assert(result);
    if (!result)
        return false;

    result = general1_->Create(GridWidth, GridHeight, GridDepth, 1, 2);
    assert(result);
    if (!result)
        return false;

    result = general4_->Create(GridWidth, GridHeight, GridDepth, 4, 2);
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
    }

    return true;
}

void FluidSimulator::Reset()
{
    velocity_->Clear();
    density_->Clear();
    temperature_->Clear();

    Metrics::Instance()->Reset();
}

void FluidSimulator::Update(float delta_time, double seconds_elapsed,
                            int frame_count)
{
    float sin_factor = static_cast<float>(sin(seconds_elapsed / 4 * 3.1415926f));
    float cos_factor = static_cast<float>(cos(seconds_elapsed / 4 * 3.1415926f));
    float hotspot_x =
        cos_factor * SplatRadius * 0.8f + kImpulsePosition.getX();
    float hotspot_z =
        sin_factor * SplatRadius * 0.8f + kImpulsePosition.getZ();
    vmath::Vector3 hotspot(hotspot_x, 0.0f, hotspot_z);

    Metrics::Instance()->OnFrameUpdateBegins();

    // Advect velocity
    AdvectVelocity(delta_time);
    Metrics::Instance()->OnVelocityAvected();

    // Advect density and temperature
    AdvectTemperature(delta_time);
    Metrics::Instance()->OnTemperatureAvected();

    AdvectDensity(delta_time);
    Metrics::Instance()->OnDensityAvected();

    // Apply buoyancy and gravity
    ApplyBuoyancy(delta_time);
    Metrics::Instance()->OnBuoyancyApplied();

    // Splat new smoke
    ApplyImpulseDensity(kImpulsePosition, hotspot, ImpulseDensity);

    // Something wrong with the temperature impulsing.
    ApplyImpulse(&temperature_, kImpulsePosition, hotspot, ImpulseTemperature);
    Metrics::Instance()->OnImpulseApplied();

    // TODO: Try to slightly optimize the calculation by pre-multiplying 1/h^2.
    ComputeDivergence();
    Metrics::Instance()->OnDivergenceComputed();

    // Solve pressure-velocity Poisson equation
    SolvePressure();
    Metrics::Instance()->OnPressureSolved();

    // Rectify velocity via the gradient of pressure
    SubtractGradient();
    Metrics::Instance()->OnVelocityRectified();

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

    CudaMain::Instance()->RoundPassed(frame_count);
}

void FluidSimulator::AdvectDensity(float delta_time)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        density2_->Clear();
        CudaMain::Instance()->AdvectDensityPure(density2_->gl_texture(),
                                                velocity_->cuda_volume(),
                                                density_->gl_texture(),
                                                delta_time, DensityDissipation);
        std::swap(density_, density2_);
    } else {
        AdvectImpl(density_, delta_time, DensityDissipation);
        std::swap(density_, general1_);
    }
}

void FluidSimulator::AdvectImpl(std::shared_ptr<GraphicsVolume> source,
                                float delta_time, float dissipation)
{
    general1_->Clear();
    if (graphics_lib_ == GRAPHICS_LIB_CUDA_DIAGNOSIS) {
        CudaMain::Instance()->Advect(velocity_->gl_texture(),
                                     source->gl_texture(),
                                     general1_->gl_texture(), delta_time,
                                     dissipation);
    } else {
        GLuint p = Programs.Advect;
        glUseProgram(p);

        SetUniform("InverseSize", CalculateInverseSize(*source->gl_texture()));
        SetUniform("TimeStep", delta_time);
        SetUniform("Dissipation", dissipation);
        SetUniform("SourceTexture", 1);
        SetUniform("Obstacles", 2);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          general1_->gl_texture()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, velocity_->gl_texture()->handle());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, source->gl_texture()->handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              general1_->gl_texture()->depth());
        ResetState();
    }
}

void FluidSimulator::AdvectTemperature(float delta_time)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        general1_->Clear(); // TODO: Clearing CUDA texture could be slow.
        CudaMain::Instance()->AdvectPure(general1_->cuda_volume(),
                                         velocity_->cuda_volume(),
                                         temperature_->cuda_volume(),
                                         delta_time, TemperatureDissipation);
    } else {
        AdvectImpl(temperature_, delta_time, TemperatureDissipation);
    }

    std::swap(temperature_, general1_);
}

void FluidSimulator::AdvectVelocity(float delta_time)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->AdvectVelocityPure(general4_->cuda_volume(),
                                                 velocity_->cuda_volume(),
                                                 delta_time,
                                                 VelocityDissipation);
    } else if (graphics_lib_ == GRAPHICS_LIB_CUDA_DIAGNOSIS) {
        CudaMain::Instance()->AdvectVelocity(velocity_->gl_texture(),
                                             general4_->gl_texture(),
                                             delta_time, VelocityDissipation);
    } else {
        GLuint p = Programs.Advect;
        glUseProgram(p);

        SetUniform("InverseSize",
                   CalculateInverseSize(*velocity_->gl_texture()));
        SetUniform("TimeStep", delta_time);
        SetUniform("Dissipation", VelocityDissipation);
        SetUniform("SourceTexture", 1);
        SetUniform("Obstacles", 2);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          general4_->gl_texture()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, velocity_->gl_texture()->handle());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, velocity_->gl_texture()->handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              general4_->gl_texture()->depth());
        ResetState();
    }

    std::swap(velocity_, general4_);
}

void FluidSimulator::ApplyBuoyancy(float delta_time)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ApplyBuoyancyPure(general4_->cuda_volume(),
                                                velocity_->cuda_volume(),
                                                temperature_->cuda_volume(),
                                                delta_time, AmbientTemperature,
                                                kBuoyancyCoef, SmokeWeight);
    } else if (graphics_lib_ == GRAPHICS_LIB_CUDA_DIAGNOSIS) {
        CudaMain::Instance()->ApplyBuoyancy(velocity_->gl_texture(),
                                            temperature_->gl_texture(),
                                            general4_->gl_texture(),
                                            delta_time, AmbientTemperature,
                                            kBuoyancyCoef, SmokeWeight);
    } else {
        GLuint p = Programs.ApplyBuoyancy;
        glUseProgram(p);

        SetUniform("Velocity", 0);
        SetUniform("Temperature", 1);
        SetUniform("AmbientTemperature", AmbientTemperature);
        SetUniform("TimeStep", delta_time);
        SetUniform("Sigma", kBuoyancyCoef);
        SetUniform("Kappa", SmokeWeight);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          general4_->gl_texture()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, velocity_->gl_texture()->handle());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, temperature_->gl_texture()->handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              general4_->gl_texture()->depth());
        ResetState();
    }

    std::swap(velocity_, general4_);
}

void FluidSimulator::ApplyImpulse(std::shared_ptr<GraphicsVolume>* dest,
                                  Vectormath::Aos::Vector3 position,
                                  Vectormath::Aos::Vector3 hotspot, float value)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ApplyImpulsePure(general1_->cuda_volume(),
                                               (*dest)->cuda_volume(),
                                               position, hotspot, SplatRadius,
                                               value);
        std::swap(*dest, general1_);
    } else if (graphics_lib_ == GRAPHICS_LIB_CUDA_DIAGNOSIS) {
        CudaMain::Instance()->ApplyImpulse((*dest)->gl_texture(),
                                           kImpulsePosition, hotspot,
                                           SplatRadius, value);
    } else {
        GLuint p = Programs.ApplyImpulse;
        glUseProgram(p);

        SetUniform("center_point", position);
        SetUniform("hotspot", hotspot);
        SetUniform("radius", SplatRadius);
        SetUniform("fill_color_r", value);
        SetUniform("fill_color_g", value);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          (*dest)->gl_texture()->frame_buffer());
        glEnable(GL_BLEND);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              (*dest)->gl_texture()->depth());
        ResetState();
    }
}

void FluidSimulator::ApplyImpulseDensity(vmath::Vector3 position,
                                         vmath::Vector3 hotspot, float value)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ApplyImpulseDensityPure(density2_->gl_texture(),
                                                      density_->gl_texture(),
                                                      position, hotspot,
                                                      SplatRadius, value);
        std::swap(density_, density2_);
    } else if (graphics_lib_ == GRAPHICS_LIB_CUDA_DIAGNOSIS) {
        CudaMain::Instance()->ApplyImpulse(density_->gl_texture(),
                                           kImpulsePosition, hotspot,
                                           SplatRadius, value);
    } else {
        GLuint p = Programs.ApplyImpulse;
        glUseProgram(p);

        SetUniform("center_point", position);
        SetUniform("hotspot", hotspot);
        SetUniform("radius", SplatRadius);
        SetUniform("fill_color_r", value);
        SetUniform("fill_color_g", value);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          density_->gl_texture()->frame_buffer());
        glEnable(GL_BLEND);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              density_->gl_texture()->depth());
        ResetState();
    }
}

void FluidSimulator::ComputeDivergence()
{
    float half_inverse_cell_size = 0.5f / CellSize;
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ComputeDivergencePure(general4_->cuda_volume(),
                                                    velocity_->cuda_volume(),
                                                    half_inverse_cell_size);
    } else if (graphics_lib_ == GRAPHICS_LIB_CUDA_DIAGNOSIS) {
        CudaMain::Instance()->ComputeDivergence(velocity_->gl_texture(),
                                                general4_->gl_texture(),
                                                half_inverse_cell_size);
    } else {
        GLuint p = Programs.ComputeDivergence;
        glUseProgram(p);

        SetUniform("HalfInverseCellSize", half_inverse_cell_size);
        SetUniform("Obstacles", 1);
        SetUniform("velocity", 0);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          general4_->gl_texture()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, velocity_->gl_texture()->handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              general4_->gl_texture()->depth());
        ResetState();
    }
}

void FluidSimulator::DampedJacobi(float cell_size)
{
    float one_minus_omega = 0.33333333f;
    float minus_square_cell_size = -(cell_size * cell_size);
    float omega_over_beta = 0.11111111f;

    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->DampedJacobiPure(general4_->cuda_volume(),
                                               general4_->cuda_volume(),
                                               one_minus_omega,
                                               minus_square_cell_size,
                                               omega_over_beta);
    } else if (graphics_lib_ == GRAPHICS_LIB_CUDA_DIAGNOSIS) {
        CudaMain::Instance()->DampedJacobi(general4_->gl_texture(),
                                           general4_->gl_texture(),
                                           one_minus_omega,
                                           minus_square_cell_size,
                                           omega_over_beta);
    } else {
        GLuint p = Programs.DampedJacobi;
        glUseProgram(p);

        SetUniform("Alpha", minus_square_cell_size);
        SetUniform("InverseBeta", omega_over_beta);
        SetUniform("one_minus_omega", one_minus_omega);
        SetUniform("packed_tex", 0);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          general4_->gl_texture()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, general4_->gl_texture()->handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              general4_->gl_texture()->depth());
        ResetState();
    }
}

void FluidSimulator::Jacobi(float cell_size)
{
    GLuint p = Programs.Jacobi;
    glUseProgram(p);

    SetUniform("Alpha", -CellSize * CellSize);
    SetUniform("InverseBeta", 0.1666f);
    SetUniform("Divergence", 1);
    SetUniform("Obstacles", 2);

    glBindFramebuffer(GL_FRAMEBUFFER, general4_->gl_texture()->frame_buffer());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, general4_->gl_texture()->handle());
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                          general4_->gl_texture()->depth());
    ResetState();
}

void FluidSimulator::SolvePressure()
{
    static MultigridCore* m_core = nullptr;
    if (!m_core) {
        if (graphics_lib_ == GRAPHICS_LIB_CUDA)
            m_core = new MultigridCoreCuda();
        else
            m_core = new MultigridCoreGlsl();
    }

    switch (solver_choice_) {
        case POISSON_SOLVER_JACOBI:
        case POISSON_SOLVER_GAUSS_SEIDEL: { // Bad in parallelism. Hard to be
                                            // implemented by shader.
            for (int i = 0; i < kNumJacobiIterations; ++i)
                Jacobi(CellSize);

            break;
        }
        case POISSON_SOLVER_DAMPED_JACOBI: {
            // NOTE: If we don't clear the buffer, a lot more details are gonna
            //       be rendered. Preconditioned?
            //
            // Our experiments reveals that increasing the iteration times to
            // 80 of Jacobi will NOT lead to higher accuracy.
            for (int i = 0; i < kNumJacobiIterations; ++i)
                DampedJacobi(CellSize);

            break;
        }
        case POISSON_SOLVER_MULTI_GRID: {
            static PoissonSolver* p_solver = nullptr;
            if (!p_solver)
            {
                p_solver = new MultigridPoissonSolver(m_core);
                p_solver->Initialize(general4_->GetWidth(),
                                     general4_->GetHeight(),
                                     general4_->GetDepth());
            }

            // An iteration times lower than 4 will introduce significant
            // unnatural visual effect caused by the half-convergent state of
            // pressure. 
            //
            // If I change the value to 6, then the average |r| could be
            // reduced to around 0.004.
            //
            // And please also note that the average |r| of Damped Jacobi
            // (using a constant iteration time of 40) is stabilized at 0.025.
            // That's a pretty good score!

            for (int i = 0; i < num_multigrid_iterations_; i++)
                p_solver->Solve(general4_, CellSize, !i);

            break;
        }
        case POISSON_SOLVER_FULL_MULTI_GRID: {
            static PoissonSolver* p_solver = nullptr;
            if (!p_solver)
            {
                p_solver = new FullMultigridPoissonSolver(m_core);
                p_solver->Initialize(general4_->GetWidth(),
                                     general4_->GetHeight(),
                                     general4_->GetDepth());
            }

            // Chaos occurs if the iteration times is set to a value above 2.
            for (int i = 0; i < num_full_multigrid_iterations_; i++)
                p_solver->Solve(general4_, CellSize, !i);

            break;
        }
        default: {
            break;
        }
    }
}

void FluidSimulator::SubtractGradient()
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->SubstractGradientPure(velocity_->cuda_volume(),
                                                    general4_->cuda_volume(),
                                                    GradientScale);
    } else if (graphics_lib_ == GRAPHICS_LIB_CUDA_DIAGNOSIS) {
        CudaMain::Instance()->SubstractGradient(velocity_->gl_texture(),
                                                general4_->gl_texture(),
                                                velocity_->gl_texture(),
                                                GradientScale);
    } else {
        GLuint p = Programs.SubtractGradient;
        glUseProgram(p);

        SetUniform("GradientScale", GradientScale);
        SetUniform("HalfInverseCellSize", 0.5f / CellSize);
        SetUniform("velocity", 0);
        SetUniform("packed_tex", 1);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          velocity_->gl_texture()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, velocity_->gl_texture()->handle());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, general4_->gl_texture()->handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              velocity_->gl_texture()->depth());
        ResetState();
    }
}

const GraphicsVolume& FluidSimulator::GetDensityTexture() const
{
    return *density_;
}
