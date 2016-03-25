#include "stdafx.h"
#include "fluid_simulator.h"

#include "full_multigrid_poisson_solver.h"
#include "metrics.h"
#include "multigrid_poisson_solver.h"
#include "opengl/gl_texture.h"
#include "opengl/glew.h"
#include "shader/fluid_shader.h"
#include "shader/multigrid_shader.h"
#include "utility.h"
#include "vmath.hpp"

//TODO
#include "multigrid_core.h"

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
    : solver_choice_(POISSON_SOLVER_FULL_MULTI_GRID)
    , num_multigrid_iterations_(5)
    , num_full_multigrid_iterations_(2)
    , velocity_()
    , density_()
    , temperature_()
    , general1_()
    , general3_()
{

}

FluidSimulator::~FluidSimulator()
{

}

bool FluidSimulator::Init()
{
    Programs.Advect = LoadProgram(FluidShader::Vertex(),
                                  FluidShader::PickLayer(),
                                  FluidShader::Advect());
    Programs.Jacobi = LoadProgram(FluidShader::Vertex(),
                                  FluidShader::PickLayer(),
                                  FluidShader::Jacobi());
    Programs.DampedJacobi = LoadProgram(FluidShader::Vertex(),
                                        FluidShader::PickLayer(),
                                        FluidShader::DampedJacobi());
    Programs.compute_residual = LoadProgram(FluidShader::Vertex(),
                                            FluidShader::PickLayer(),
                                            MultigridShader::ComputeResidual());
    Programs.SubtractGradient = LoadProgram(FluidShader::Vertex(),
                                            FluidShader::PickLayer(),
                                            FluidShader::SubtractGradient());
    Programs.ComputeDivergence = LoadProgram(FluidShader::Vertex(),
                                             FluidShader::PickLayer(),
                                             FluidShader::ComputeDivergence());
    Programs.ApplyImpulse = LoadProgram(FluidShader::Vertex(),
                                        FluidShader::PickLayer(),
                                        FluidShader::Splat());
    Programs.ApplyBuoyancy = LoadProgram(FluidShader::Vertex(),
                                         FluidShader::PickLayer(),
                                         FluidShader::Buoyancy());

    MultigridCore core; // TODO
    velocity_ = core.CreateTexture(GridWidth, GridHeight, GridDepth, GL_RGBA32F, GL_RGBA);

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

    density_ = core.CreateTexture(GridWidth, GridHeight, GridDepth, GL_R32F,
                                  GL_RED);
    temperature_ = core.CreateTexture(GridWidth, GridHeight, GridDepth, GL_R32F,
                                      GL_RED);
    general1_ = core.CreateTexture(GridWidth, GridHeight, GridDepth, GL_R32F,
                                   GL_RED);
    general3_ = core.CreateTexture(GridWidth, GridHeight, GridDepth, GL_RGBA32F,
                                   GL_RGBA);

    return true;
}

void FluidSimulator::Reset()
{
    ClearSurface(velocity_.get(), 0.0f);
    ClearSurface(density_.get(), 0.0f);
    ClearSurface(temperature_.get(), 0.0f);
    Metrics::Instance()->Reset();
}

void FluidSimulator::Update(float delta_time, double seconds_elapsed,
                            int frame_count)
{

    float sin_factor = static_cast<float>(sin(0 / 4 * 3.1415926f));
    float cos_factor = static_cast<float>(cos(0 / 4 * 3.1415926f));
    float hotspot_x =
        cos_factor * SplatRadius * 0.8f + kImpulsePosition.getX();
    float hotspot_z =
        sin_factor * SplatRadius * 0.8f + kImpulsePosition.getZ();
    vmath::Vector3 hotspot(hotspot_x, 0.0f, hotspot_z);

    Metrics::Instance()->OnFrameUpdateBegins();

    // Advect velocity
    //CudaMain::Instance()->AdvectVelocity(*Surfaces.tex_velocity, *gb3, delta_time, VelocityDissipation);
    AdvectVelocity(velocity_, general3_, delta_time, VelocityDissipation);
    std::swap(velocity_, general3_);

    Metrics::Instance()->OnVelocityAvected();

    // Advect density and temperature
    ClearSurface(general1_.get(), 0.0f);
    //CudaMain::Instance()->Advect(*Surfaces.tex_velocity, *Surfaces.tex_temperature, *gb1, delta_time, TemperatureDissipation);
    Advect(velocity_, temperature_, general1_, delta_time, TemperatureDissipation);
    std::swap(temperature_, general1_);
    Metrics::Instance()->OnTemperatureAvected();

    ClearSurface(general1_.get(), 0.0f);
    //CudaMain::Instance()->Advect(*Surfaces.tex_velocity, *Surfaces.tex_density, *gb1, delta_time, DensityDissipation);
    Advect(velocity_, density_, general1_, delta_time, DensityDissipation);
    std::swap(density_, general1_);
    Metrics::Instance()->OnDensityAvected();

    // Apply buoyancy and gravity
    //CudaMain::Instance()->ApplyBuoyancy(*Surfaces.tex_velocity, *Surfaces.tex_temperature, *gb3, delta_time, AmbientTemperature, kBuoyancyCoef, SmokeWeight);
    ApplyBuoyancy(velocity_, temperature_, general3_, delta_time);
    std::swap(velocity_, general3_);
    Metrics::Instance()->OnBuoyancyApplied();

    // Splat new smoke
    //CudaMain::Instance()->ApplyImpulse(*Surfaces.tex_density, kImpulsePosition, hotspot, SplatRadius, ImpulseDensity);
    ApplyImpulse(density_, kImpulsePosition, hotspot, ImpulseDensity);

    // Something wrong with the temperature impulsing.
    //CudaMain::Instance()->ApplyImpulse(*Surfaces.tex_temperature, kImpulsePosition, hotspot, SplatRadius, ImpulseTemperature);
    ApplyImpulse(temperature_, kImpulsePosition, hotspot, ImpulseTemperature);
    Metrics::Instance()->OnImpulseApplied();

    // TODO: Try to slightly optimize the calculation by pre-multiplying 1/h^2.
    //CudaMain::Instance()->ComputeDivergence(*Surfaces.tex_velocity, *gb3, 0.5f / CellSize);
    ComputeDivergence(velocity_, general3_);
    Metrics::Instance()->OnDivergenceComputed();

    // Solve pressure-velocity Poisson equation
    SolvePressure(general3_);
    Metrics::Instance()->OnPressureSolved();

    // Rectify velocity via the gradient of pressure
    //CudaMain::Instance()->SubstractGradient(*Surfaces.tex_velocity, *gb3, *Surfaces.tex_velocity, GradientScale);
    SubtractGradient(velocity_, general3_);
    Metrics::Instance()->OnVelocityRectified();

    //CudaMain::Instance()->RoundPassed(frame_count);
}

void FluidSimulator::Advect(std::shared_ptr<GLTexture> velocity,
                            std::shared_ptr<GLTexture> source,
                            std::shared_ptr<GLTexture> dest, float delta_time,
                            float dissipation)
{
    GLuint p = Programs.Advect;
    glUseProgram(p);

    SetUniform(
        "InverseSize",
        vmath::recipPerElem(
            vmath::Vector3(float(GridWidth), float(GridHeight),
                           float(GridDepth))));
    SetUniform("TimeStep", delta_time);
    SetUniform("Dissipation", dissipation);
    SetUniform("SourceTexture", 1);
    SetUniform("Obstacles", 2);

    glBindFramebuffer(GL_FRAMEBUFFER, dest->frame_buffer());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity->handle());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, source->handle());
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest->depth());
    ResetState();
}

void FluidSimulator::AdvectVelocity(std::shared_ptr<GLTexture> velocity,
                                    std::shared_ptr<GLTexture> dest,
                                    float delta_time, float dissipation)
{
    GLuint p = Programs.Advect;
    glUseProgram(p);

    SetUniform(
        "InverseSize",
        vmath::recipPerElem(
        vmath::Vector3(float(GridWidth), float(GridHeight),
        float(GridDepth))));
    SetUniform("TimeStep", delta_time);
    SetUniform("Dissipation", dissipation);
    SetUniform("SourceTexture", 1);
    SetUniform("Obstacles", 2);

    glBindFramebuffer(GL_FRAMEBUFFER, dest->frame_buffer());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity->handle());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, velocity->handle());
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest->depth());
    ResetState();
}

void FluidSimulator::Jacobi(std::shared_ptr<GLTexture> pressure,
                            std::shared_ptr<GLTexture> divergence)
{
    GLuint p = Programs.Jacobi;
    glUseProgram(p);

    SetUniform("Alpha", -CellSize * CellSize);
    SetUniform("InverseBeta", 0.1666f);
    SetUniform("Divergence", 1);
    SetUniform("Obstacles", 2);

    glBindFramebuffer(GL_FRAMEBUFFER, pressure->frame_buffer());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, pressure->handle());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, divergence->handle());
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, pressure->depth());
    ResetState();
}

void FluidSimulator::DampedJacobi(std::shared_ptr<GLTexture> pressure,
                                  std::shared_ptr<GLTexture> divergence,
                                  float cell_size)
{
    GLuint p = Programs.DampedJacobi;
    glUseProgram(p);

    SetUniform("Alpha", -(cell_size * cell_size));
    SetUniform("InverseBeta", 0.11111111f);
    SetUniform("one_minus_omega", 0.33333333f);
    SetUniform("Pressure", 0);
    SetUniform("Divergence", 1);
    SetUniform("Obstacles", 2);

    glBindFramebuffer(GL_FRAMEBUFFER, pressure->frame_buffer());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, pressure->handle());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, divergence->handle());
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, pressure->depth());
    ResetState();
}

void FluidSimulator::SolvePressure(std::shared_ptr<GLTexture> packed)
{
    switch (solver_choice_)
    {
        case POISSON_SOLVER_JACOBI:
        case POISSON_SOLVER_GAUSS_SEIDEL: { // Bad in parallelism. Hard to be
            // implemented by shader.
            //             ClearSurface(pressure, 0.0f);
            //             for (int i = 0; i < NumJacobiIterations; ++i) {
            //                 Jacobi(pressure, divergence, obstacles);
            //             }
            break;
        }
        case POISSON_SOLVER_DAMPED_JACOBI: {
            // NOTE: If we don't clear the buffer, a lot more details are gonna
            //       be rendered. Preconditioned?
            //
            // Our experiments reveals that increasing the iteration times to
            // 80 of Jacobi will NOT lead to higher accuracy.

            //             ClearSurface(pressure, 0.0f);
            //             for (int i = 0; i < NumJacobiIterations; ++i) {
            //                 DampedJacobi(pressure, divergence, obstacles, CellSize);
            //             }
            break;
        }
        case POISSON_SOLVER_MULTI_GRID: {
            static PoissonSolver* p_solver = nullptr;
            if (!p_solver)
            {
                p_solver = new MultigridPoissonSolver();
                p_solver->Initialize(GridWidth, GridWidth, GridWidth);
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
                p_solver->Solve(packed, CellSize, !i);

            break;
        }
        case POISSON_SOLVER_FULL_MULTI_GRID: {
            static PoissonSolver* p_solver = nullptr;
            if (!p_solver)
            {
                p_solver = new FullMultigridPoissonSolver();
                p_solver->Initialize(GridWidth, GridWidth, GridWidth);
            }

            // Chaos occurs if the iteration times is set to a value above 2.
            for (int i = 0; i < num_full_multigrid_iterations_; i++)
                p_solver->Solve(packed, CellSize, !i);

            break;
        }
        default: {
            break;
        }
    }
}

void FluidSimulator::SubtractGradient(std::shared_ptr<GLTexture> velocity,
                                      std::shared_ptr<GLTexture> packed)
{
    GLuint p = Programs.SubtractGradient;
    glUseProgram(p);

    SetUniform("GradientScale", GradientScale);
    SetUniform("HalfInverseCellSize", 0.5f / CellSize);
    SetUniform("velocity", 0);
    SetUniform("packed_tex", 1);

    glBindFramebuffer(GL_FRAMEBUFFER, velocity->frame_buffer());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity->handle());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, packed->handle());
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, velocity->depth());
    ResetState();
}

void FluidSimulator::ComputeDivergence(std::shared_ptr<GLTexture> velocity,
                                       std::shared_ptr<GLTexture> dest)
{
    GLuint p = Programs.ComputeDivergence;
    glUseProgram(p);

    SetUniform("HalfInverseCellSize", 0.5f / CellSize);
    SetUniform("Obstacles", 1);
    SetUniform("velocity", 0);

    glBindFramebuffer(GL_FRAMEBUFFER, dest->frame_buffer());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity->handle());
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest->depth());
    ResetState();
}

void FluidSimulator::ApplyImpulse(std::shared_ptr<GLTexture> dest,
                                  Vectormath::Aos::Vector3 position,
                                  Vectormath::Aos::Vector3 hotspot, float value)
{
    GLuint p = Programs.ApplyImpulse;
    glUseProgram(p);

    SetUniform("center_point", position);
    SetUniform("hotspot", hotspot);
    SetUniform("radius", SplatRadius);
    SetUniform("fill_color_r", value);
    SetUniform("fill_color_g", value);

    glBindFramebuffer(GL_FRAMEBUFFER, dest->frame_buffer());
    glEnable(GL_BLEND);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest->depth());
    ResetState();
}

void FluidSimulator::ApplyBuoyancy(std::shared_ptr<GLTexture> velocity,
                                   std::shared_ptr<GLTexture> temperature,
                                   std::shared_ptr<GLTexture> dest,
                                   float delta_time)
{
    GLuint p = Programs.ApplyBuoyancy;
    glUseProgram(p);

    SetUniform("Velocity", 0);
    SetUniform("Temperature", 1);
    SetUniform("AmbientTemperature", AmbientTemperature);
    SetUniform("TimeStep", delta_time);
    SetUniform("Sigma", kBuoyancyCoef);
    SetUniform("Kappa", SmokeWeight);

    glBindFramebuffer(GL_FRAMEBUFFER, dest->frame_buffer());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity->handle());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, temperature->handle());
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest->depth());
    ResetState();
}

const GLTexture& FluidSimulator::GetDensityTexture() const
{
    return *density_;
}
