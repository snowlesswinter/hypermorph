#include "stdafx.h"
#include "fluid_simulator.h"

#include <cassert>

#include "cuda_host/cuda_main.h"
#include "cuda_host/cuda_volume.h"
#include "fluid_config.h"
#include "graphics_volume.h"
#include "metrics.h"
#include "opengl/gl_volume.h"
#include "poisson_solver/full_multigrid_poisson_solver.h"
#include "poisson_solver/multigrid_core_cuda.h"
#include "poisson_solver/multigrid_core_glsl.h"
#include "poisson_solver/multigrid_poisson_solver.h"
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

FluidSimulator::FluidSimulator()
    : graphics_lib_(GRAPHICS_LIB_GLSL)
    , solver_choice_(POISSON_SOLVER_FULL_MULTI_GRID)
    , multigrid_core_()
    , solver_()
    , num_multigrid_iterations_(5)
    , num_full_multigrid_iterations_(2)
    , volume_byte_width_(2)
    , diagnosis_(false)
    , velocity_()
    , density_()
    , density2_()
    , temperature_()
    , packed_()
    , general1_()
    , general4_()
    , diagnosis_volume_()
    , manual_impulse_()
{
}

FluidSimulator::~FluidSimulator()
{
}

bool FluidSimulator::Init()
{
    velocity_.reset(new GraphicsVolume(graphics_lib_));
    density_.reset(new GraphicsVolume(graphics_lib_));
    density2_.reset(new GraphicsVolume(graphics_lib_));
    temperature_.reset(new GraphicsVolume(graphics_lib_));
    packed_.reset(new GraphicsVolume(graphics_lib_));
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

    result = packed_->Create(GridWidth, GridHeight, GridDepth, 2, volume_byte_width_);
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
        Programs.diagnose_ = LoadProgram(
            FluidShader::Vertex(), FluidShader::PickLayer(),
            MultigridShader::ComputeResidualPackedDiagnosis());
    }

    return true;
}

void FluidSimulator::Reset()
{
    velocity_->Clear();
    density_->Clear();
    temperature_->Clear();

    diagnosis_volume_.reset();

    Metrics::Instance()->Reset();
}

std::shared_ptr<GraphicsVolume> FluidSimulator::GetDensityField() const
{
    return density_;
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
    Metrics::Instance()->OnFrameUpdateBegins();

    float fixed_time_step = FluidConfig::Instance()->fixed_time_step();
    float proper_delta_time = fixed_time_step > 0.0f ?
        fixed_time_step : std::min(delta_time, kMaxTimeStep);

    // Splat new smoke
    ApplyImpulse(seconds_elapsed, proper_delta_time);
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

    // Advect velocity
    AdvectVelocity(proper_delta_time);
    Metrics::Instance()->OnVelocityAvected();

    // Advect density and temperature
    AdvectTemperature(proper_delta_time);
    Metrics::Instance()->OnTemperatureAvected();

    AdvectDensity(proper_delta_time);
    Metrics::Instance()->OnDensityAvected();

    // Apply buoyancy and gravity
    ApplyBuoyancy(proper_delta_time);
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

    CudaMain::Instance()->RoundPassed(frame_count);
}

void FluidSimulator::UpdateImpulsing(float x, float y)
{
    if (manual_impulse_) {
        *manual_impulse_ = glm::vec2(x, y);
    }
}

void FluidSimulator::AdvectDensity(float delta_time)
{
    float density_dissipation = FluidConfig::Instance()->density_dissipation();
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->AdvectDensity(density2_->cuda_volume(),
                                            velocity_->cuda_volume(),
                                            density_->cuda_volume(),
                                            delta_time, density_dissipation);
        std::swap(density_, density2_);
    } else {
        AdvectImpl(density_, delta_time, density_dissipation);
        std::swap(density_, general1_);
    }
}

void FluidSimulator::AdvectImpl(std::shared_ptr<GraphicsVolume> source,
                                float delta_time, float dissipation)
{
    glUseProgram(Programs.Advect);

    SetUniform("InverseSize", CalculateInverseSize(*source->gl_volume()));
    SetUniform("TimeStep", delta_time);
    SetUniform("Dissipation", dissipation);
    SetUniform("SourceTexture", 1);
    SetUniform("Obstacles", 2);

    glBindFramebuffer(GL_FRAMEBUFFER,
                      general1_->gl_volume()->frame_buffer());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity_->gl_volume()->texture_handle());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, source->gl_volume()->texture_handle());
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                          general1_->gl_volume()->depth());
    ResetState();
}

void FluidSimulator::AdvectTemperature(float delta_time)
{
    float temperature_dissipation =
        FluidConfig::Instance()->temperature_dissipation();
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->Advect(general1_->cuda_volume(),
                                     velocity_->cuda_volume(),
                                     temperature_->cuda_volume(),
                                     delta_time, temperature_dissipation);
    } else {
        AdvectImpl(temperature_, delta_time, temperature_dissipation);
    }

    std::swap(temperature_, general1_);
}

void FluidSimulator::AdvectVelocity(float delta_time)
{
    float velocity_dissipation =
        FluidConfig::Instance()->velocity_dissipation();
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->AdvectVelocity(general4_->cuda_volume(),
                                             velocity_->cuda_volume(),
                                             delta_time, velocity_dissipation);
    } else {
        glUseProgram(Programs.Advect);

        SetUniform("InverseSize",
                   CalculateInverseSize(*velocity_->gl_volume()));
        SetUniform("TimeStep", delta_time);
        SetUniform("Dissipation", velocity_dissipation);
        SetUniform("SourceTexture", 1);
        SetUniform("Obstacles", 2);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          general4_->gl_volume()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, velocity_->gl_volume()->texture_handle());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, velocity_->gl_volume()->texture_handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              general4_->gl_volume()->depth());
        ResetState();
    }

    std::swap(velocity_, general4_);
}

void FluidSimulator::ApplyBuoyancy(float delta_time)
{
    float smoke_weight = FluidConfig::Instance()->smoke_weight();
    float ambient_temperature = FluidConfig::Instance()->ambient_temperature();
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ApplyBuoyancy(general4_->cuda_volume(),
                                            velocity_->cuda_volume(),
                                            temperature_->cuda_volume(),
                                            delta_time, ambient_temperature,
                                            kBuoyancyCoef, smoke_weight);
    } else {
        glUseProgram(Programs.ApplyBuoyancy);

        SetUniform("Velocity", 0);
        SetUniform("Temperature", 1);
        SetUniform("AmbientTemperature", ambient_temperature);
        SetUniform("TimeStep", delta_time);
        SetUniform("Sigma", kBuoyancyCoef);
        SetUniform("Kappa", smoke_weight);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          general4_->gl_volume()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, velocity_->gl_volume()->texture_handle());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, temperature_->gl_volume()->texture_handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              general4_->gl_volume()->depth());
        ResetState();
    }

    std::swap(velocity_, general4_);
}

void FluidSimulator::ApplyImpulse(double seconds_elapsed, float delta_time)
{
    double ¦Ð = 3.1415926;
    float splat_radius =
        GridWidth * FluidConfig::Instance()->splat_radius_factor();
    float sin_factor = static_cast<float>(sin(seconds_elapsed / 4.0 * 2.0 * ¦Ð));
    float cos_factor = static_cast<float>(cos(seconds_elapsed / 4.0 * 2.0 * ¦Ð));
    float hotspot_x = cos_factor * splat_radius * 0.5f + kImpulsePosition.x;
    float hotspot_z = sin_factor * splat_radius * 0.5f + kImpulsePosition.z;
    glm::vec3 hotspot(hotspot_x, 0.0f, hotspot_z);

    if (manual_impulse_)
        hotspot = glm::vec3(0.5f * GridWidth * (manual_impulse_->x + 1.0f),
                            0.0f,
                            0.5f * GridDepth * (manual_impulse_->y + 1.0f));
    else if (!FluidConfig::Instance()->auto_impulse())
        return;

    ImpulseDensity(kImpulsePosition, hotspot, splat_radius,
                   FluidConfig::Instance()->impulse_density());

    glm::vec3 temperature(FluidConfig::Instance()->impulse_temperature(), 0.0f,
                          0.0f);
    Impulse(temperature_, kImpulsePosition, hotspot, splat_radius, temperature,
            1);

    return; // Not necessary for this scene.
    float v_coef = static_cast<float>(sin(seconds_elapsed * 3.0 * 2.0 * ¦Ð));
    glm::vec3 initial_velocity(
        0.0f,
        (1.0f + v_coef) * FluidConfig::Instance()->impulse_velocity(),
        0.0f
    );
    Impulse(velocity_, kImpulsePosition, hotspot, splat_radius,
            initial_velocity, 7);
}

void FluidSimulator::ComputeDivergence()
{
    float half_inverse_cell_size = 0.5f / CellSize;
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ComputeDivergence(packed_->cuda_volume(),
                                                velocity_->cuda_volume(),
                                                half_inverse_cell_size);
    } else {
        glUseProgram(Programs.ComputeDivergence);

        SetUniform("HalfInverseCellSize", half_inverse_cell_size);
        SetUniform("Obstacles", 1);
        SetUniform("velocity", 0);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          packed_->gl_volume()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, velocity_->gl_volume()->texture_handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              packed_->gl_volume()->depth());
        ResetState();
    }
}

void FluidSimulator::ComputeResidualDiagnosis(float cell_size)
{
    if (!diagnosis_)
        return;

    if (!diagnosis_volume_) {
        std::shared_ptr<GraphicsVolume> v(new GraphicsVolume(graphics_lib_));
        bool result = v->Create(GridWidth, GridHeight, GridDepth, 1, 4);
        assert(result);
        if (!result)
            return;

        diagnosis_volume_ = v;
    }

    float inverse_h_square = 1.0f / (cell_size * cell_size);
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ComputeResidualPackedDiagnosis(
            diagnosis_volume_->cuda_volume(), packed_->cuda_volume(),
            inverse_h_square);
    } else if (graphics_lib_ == GRAPHICS_LIB_GLSL) {
        glUseProgram(Programs.diagnose_);

        SetUniform("packed_tex", 0);
        SetUniform("inverse_h_square", inverse_h_square);

        diagnosis_volume_->gl_volume()->BindFrameBuffer();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, packed_->gl_volume()->texture_handle());
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

void FluidSimulator::DampedJacobi(float cell_size, int num_of_iterations)
{
    float one_minus_omega = 0.33333333f;
    float minus_square_cell_size = -(cell_size * cell_size);
    float omega_over_beta = 0.11111111f;

    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->DampedJacobi(packed_->cuda_volume(),
                                               packed_->cuda_volume(),
                                               minus_square_cell_size,
                                               omega_over_beta,
                                               num_of_iterations);
    } else {
        for (int i = 0; i < num_of_iterations; ++i) {
            glUseProgram(Programs.DampedJacobi);

            SetUniform("Alpha", minus_square_cell_size);
            SetUniform("InverseBeta", omega_over_beta);
            SetUniform("one_minus_omega", one_minus_omega);
            SetUniform("packed_tex", 0);

            glBindFramebuffer(GL_FRAMEBUFFER,
                              packed_->gl_volume()->frame_buffer());
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_3D, packed_->gl_volume()->texture_handle());
            glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                                  packed_->gl_volume()->depth());
            ResetState();
        }
    }
}

void FluidSimulator::Impulse(std::shared_ptr<GraphicsVolume> dest,
                             const glm::vec3& position,
                             const glm::vec3& hotspot, float splat_radius,
                             const glm::vec3& value, uint32_t mask)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ApplyImpulse(dest->cuda_volume(),
                                           dest->cuda_volume(),
                                           position, hotspot, splat_radius,
                                           value, mask);
    } else {
        glUseProgram(Programs.ApplyImpulse);

        SetUniform("center_point", position);
        SetUniform("hotspot", hotspot);
        SetUniform("radius", splat_radius);
        SetUniform("fill_color_r", value[0]);
        SetUniform("fill_color_g", value[1]);

        glBindFramebuffer(GL_FRAMEBUFFER, dest->gl_volume()->frame_buffer());
        glEnable(GL_BLEND);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              dest->gl_volume()->depth());
        ResetState();
    }
}

void FluidSimulator::ImpulseDensity(const glm::vec3& position,
                                    const glm::vec3& hotspot,
                                    float splat_radius, float value)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ApplyImpulseDensity(density_->cuda_volume(),
                                                  position, hotspot,
                                                  splat_radius, value);
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

void FluidSimulator::Jacobi(float cell_size)
{
    glUseProgram(Programs.Jacobi);

    SetUniform("Alpha", -CellSize * CellSize);
    SetUniform("InverseBeta", 0.1666f);
    SetUniform("Divergence", 1);
    SetUniform("Obstacles", 2);

    glBindFramebuffer(GL_FRAMEBUFFER, packed_->gl_volume()->frame_buffer());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, packed_->gl_volume()->texture_handle());
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                          packed_->gl_volume()->depth());
    ResetState();
}

void FluidSimulator::ReviseDensity()
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        float splat_radius =
            GridWidth * FluidConfig::Instance()->splat_radius_factor();
        CudaMain::Instance()->ReviseDensity(
            density_->cuda_volume(), kImpulsePosition, splat_radius,
            FluidConfig::Instance()->impulse_density() * 0.2f);
    }
}

void FluidSimulator::SolvePressure()
{
    if (!multigrid_core_) {
        if (graphics_lib_ == GRAPHICS_LIB_CUDA)
            multigrid_core_.reset(new MultigridCoreCuda());
        else
            multigrid_core_.reset(new MultigridCoreGlsl());
    }

    int num_jacobi_iterations =
        FluidConfig::Instance()->num_jacobi_iterations();
    switch (solver_choice_) {
        case POISSON_SOLVER_JACOBI:
        case POISSON_SOLVER_GAUSS_SEIDEL: { // Bad in parallelism. Hard to be
                                            // implemented by shader.
            for (int i = 0; i < num_jacobi_iterations; ++i)
                Jacobi(CellSize);

            break;
        }
        case POISSON_SOLVER_DAMPED_JACOBI: {
            // NOTE: If we don't clear the buffer, a lot more details are gonna
            //       be rendered. Preconditioned?
            //
            // Our experiments reveals that increasing the iteration times to
            // 80 of Jacobi will NOT lead to higher accuracy.
            
            DampedJacobi(CellSize, num_jacobi_iterations);
            break;
        }
        case POISSON_SOLVER_MULTI_GRID: {
            if (!solver_) {
                solver_.reset(
                    new MultigridPoissonSolver(multigrid_core_.get()));
                solver_->Initialize(packed_->GetWidth(), packed_->GetHeight(),
                                    packed_->GetDepth(), volume_byte_width_);
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
                solver_->Solve(packed_, CellSize, !i);

            break;
        }
        case POISSON_SOLVER_FULL_MULTI_GRID: {
            if (!solver_) {
                solver_.reset(
                    new FullMultigridPoissonSolver(multigrid_core_.get()));
                solver_->Initialize(packed_->GetWidth(), packed_->GetHeight(),
                                    packed_->GetDepth(), volume_byte_width_);
            }

            // Chaos occurs if the iteration times is set to a value above 2.
            for (int i = 0; i < num_full_multigrid_iterations_; i++)
                solver_->Solve(packed_, CellSize, !i);

            break;
        }
        default: {
            break;
        }
    }

    ComputeResidualDiagnosis(CellSize);
}

void FluidSimulator::SubtractGradient()
{
    // In the original implementation, this coefficient was set to 1.125, which
    // I guess is a trick to compensate the inaccuracy of the solution of
    // Poisson equation. As the solution now becomes more and more precise,
    // I changed the number to 1.0 to keep the system stable.
    const float gradient_scale = 1.0f / CellSize;
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->SubtractGradient(velocity_->cuda_volume(),
                                               packed_->cuda_volume(),
                                               gradient_scale);
    } else {
        glUseProgram(Programs.SubtractGradient);

        SetUniform("GradientScale", gradient_scale);
        SetUniform("HalfInverseCellSize", 0.5f / CellSize);
        SetUniform("velocity", 0);
        SetUniform("packed_tex", 1);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          velocity_->gl_volume()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, velocity_->gl_volume()->texture_handle());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, packed_->gl_volume()->texture_handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              velocity_->gl_volume()->depth());
        ResetState();
    }
}
