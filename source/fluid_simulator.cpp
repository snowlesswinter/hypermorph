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
#include "poisson_solver/open_boundary_multigrid_poisson_solver.h"
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

int size_tweak = 1;
const int kVelGridWidth = GridWidth + size_tweak;
const int kVelGridHeight = GridHeight + size_tweak;
const int kVelGridDepth = GridDepth + size_tweak;

int sphere = 0;

FluidSimulator::FluidSimulator()
    : graphics_lib_(GRAPHICS_LIB_CUDA)
    , solver_choice_(POISSON_SOLVER_FULL_MULTI_GRID)
    , multigrid_core_()
    , pressure_solver_()
    , psi_solver_()
    , num_multigrid_iterations_(5)
    , num_full_multigrid_iterations_(2)
    , volume_byte_width_(2)
    , diagnosis_(false)
    , velocity_()
    , velocity2_(GRAPHICS_LIB_CUDA)
    , vorticity_(GRAPHICS_LIB_CUDA)
    , aux_(GRAPHICS_LIB_CUDA)
    , vort_conf_(GRAPHICS_LIB_CUDA)
    , density_()
    , temperature_()
    , pressure_()
    , general1a_()
    , general1b_()
    , general1c_()
    , general1d_()
    , general4a_()
    , general4b_()
    , diagnosis_volume_()
    , manual_impulse_()
{
}

FluidSimulator::~FluidSimulator()
{
}

bool FluidSimulator::Init()
{
    velocity_ = std::make_shared<GraphicsVolume>(graphics_lib_);
    density_ = std::make_shared<GraphicsVolume>(graphics_lib_);
    temperature_ = std::make_shared<GraphicsVolume>(graphics_lib_);
    pressure_ = std::make_shared<GraphicsVolume>(graphics_lib_);
    general1a_ = std::make_shared<GraphicsVolume>(graphics_lib_);
    general1b_ = std::make_shared<GraphicsVolume>(graphics_lib_);
    general1c_ = std::make_shared<GraphicsVolume>(graphics_lib_);
    general1d_ = std::make_shared<GraphicsVolume>(graphics_lib_);
    general4a_ = std::make_shared<GraphicsVolume>(graphics_lib_);
    general4b_ = std::make_shared<GraphicsVolume>(graphics_lib_);

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

    result = velocity_->Create(kVelGridWidth, kVelGridHeight, kVelGridDepth, 4,
                               2);
    assert(result);
    if (!result)
        return false;

    result = velocity2_.Create(kVelGridWidth, kVelGridHeight, kVelGridDepth, 1,
                               2);
    assert(result);
    if (!result)
        return false;

    result = temperature_->Create(GridWidth, GridHeight, GridDepth, 1, 2);
    assert(result);
    if (!result)
        return false;

    result = pressure_->Create(GridWidth, GridHeight, GridDepth, 1,
                               volume_byte_width_);
    assert(result);
    if (!result)
        return false;

    result = general1a_->Create(GridWidth, GridHeight, GridDepth, 1, 2);
    assert(result);
    if (!result)
        return false;

    result = general1b_->Create(GridWidth, GridHeight, GridDepth, 1, 2);
    assert(result);
    if (!result)
        return false;

    result = general1c_->Create(GridWidth, GridHeight, GridDepth, 1, 2);
    assert(result);
    if (!result)
        return false;

    result = general1d_->Create(GridWidth, GridHeight, GridDepth, 1, 2);
    assert(result);
    if (!result)
        return false;

    result = general4a_->Create(kVelGridWidth, kVelGridHeight, kVelGridDepth, 4,
                                2);
    assert(result);
    if (!result)
        return false;

    result = general4b_->Create(kVelGridWidth, kVelGridHeight, kVelGridDepth, 4,
                                2);
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

void FluidSimulator::Reset()
{
    velocity_->Clear();
    density_->Clear();
    temperature_->Clear();
    general1a_->Clear();
    general1b_->Clear();
    general1c_->Clear();
    general1d_->Clear();
    general4a_->Clear();
    general4b_->Clear();

    if (velocity2_) {
        velocity2_.x()->Clear();
        velocity2_.y()->Clear();
        velocity2_.z()->Clear();
    }

    if (vorticity_) {
        vorticity_.x()->Clear();
        vorticity_.y()->Clear();
        vorticity_.z()->Clear();
    }

    if (aux_) {
        aux_.x()->Clear();
        aux_.y()->Clear();
        aux_.z()->Clear();
    }

    diagnosis_volume_.reset();

    Metrics::Instance()->Reset();
}

std::shared_ptr<GraphicsVolume> FluidSimulator::GetDensityField() const
{
    return density_;
}

void FluidSimulator::SetStaggered(bool staggered)
{
    CudaMain::Instance()->SetStaggered(staggered);
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

    // Advect density and temperature
    AdvectTemperature(proper_delta_time);
    Metrics::Instance()->OnTemperatureAvected();

    AdvectDensity(proper_delta_time);
    Metrics::Instance()->OnDensityAvected();

    // Advect velocity
    AdvectVelocity(proper_delta_time);
    Metrics::Instance()->OnVelocityAvected();

    // Restore vorticity
    RestoreVorticity(proper_delta_time);
    Metrics::Instance()->OnVorticityRestored();

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
        CudaMain::Instance()->AdvectField(general1a_->cuda_volume(),
                                          density_->cuda_volume(),
                                          velocity2_.x()->cuda_volume(),
                                          velocity2_.y()->cuda_volume(),
                                          velocity2_.z()->cuda_volume(),
                                          general1b_->cuda_volume(), delta_time,
                                          density_dissipation);
    } else {
        AdvectImpl(density_, delta_time, density_dissipation);
    }
    std::swap(density_, general1a_);
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
                      general1a_->gl_volume()->frame_buffer());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, velocity_->gl_volume()->texture_handle());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, source->gl_volume()->texture_handle());
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                          general1a_->gl_volume()->depth());
    ResetState();
}

void FluidSimulator::AdvectTemperature(float delta_time)
{
    float temperature_dissipation =
        FluidConfig::Instance()->temperature_dissipation();
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->AdvectField(general1a_->cuda_volume(),
                                          temperature_->cuda_volume(),
                                          velocity2_.x()->cuda_volume(),
                                          velocity2_.y()->cuda_volume(),
                                          velocity2_.z()->cuda_volume(),
                                          general1b_->cuda_volume(), delta_time,
                                          temperature_dissipation);
    } else {
        AdvectImpl(temperature_, delta_time, temperature_dissipation);
    }

    std::swap(temperature_, general1a_);
}

void FluidSimulator::AdvectVelocity(float delta_time)
{
    float velocity_dissipation =
        FluidConfig::Instance()->velocity_dissipation();
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::AdvectionMethod method =
            FluidConfig::Instance()->advection_method();
        CudaMain::Instance()->AdvectVelocity(general1a_->cuda_volume(),
                                             general1b_->cuda_volume(),
                                             general1c_->cuda_volume(),
                                             velocity2_.x()->cuda_volume(),
                                             velocity2_.y()->cuda_volume(),
                                             velocity2_.z()->cuda_volume(),
                                             general4b_->cuda_volume(),
                                             delta_time, velocity_dissipation,
                                             method);
        velocity2_.Swap(GraphicsVolume3(general1a_, general1b_, general1c_));
    } else {
        glUseProgram(Programs.Advect);

        SetUniform("InverseSize",
                   CalculateInverseSize(*velocity_->gl_volume()));
        SetUniform("TimeStep", delta_time);
        SetUniform("Dissipation", velocity_dissipation);
        SetUniform("SourceTexture", 1);
        SetUniform("Obstacles", 2);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          general4a_->gl_volume()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, velocity_->gl_volume()->texture_handle());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, velocity_->gl_volume()->texture_handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              general4a_->gl_volume()->depth());
        ResetState();

        std::swap(velocity_, general4a_);
    }

}

void FluidSimulator::ApplyBuoyancy(float delta_time)
{
    float smoke_weight = FluidConfig::Instance()->smoke_weight();
    float ambient_temperature = FluidConfig::Instance()->ambient_temperature();
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ApplyBuoyancy(velocity2_.x()->cuda_volume(),
                                            velocity2_.y()->cuda_volume(),
                                            velocity2_.z()->cuda_volume(),
                                            temperature_->cuda_volume(),
                                            density_->cuda_volume(), delta_time,
                                            ambient_temperature,
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
                          general4a_->gl_volume()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, velocity_->gl_volume()->texture_handle());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, temperature_->gl_volume()->texture_handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              general4a_->gl_volume()->depth());
        ResetState();

        std::swap(velocity_, general4a_);
    }

}

void FluidSimulator::ApplyImpulse(double seconds_elapsed, float delta_time)
{
    double ¦Ð = 3.1415926;
    float splat_radius =
        GridWidth * FluidConfig::Instance()->splat_radius_factor();
    float time_stretch = FluidConfig::Instance()->time_stretch() + 0.00001f;
    float sin_factor = static_cast<float>(sin(seconds_elapsed / time_stretch * 2.0 * ¦Ð));
    float cos_factor = static_cast<float>(cos(seconds_elapsed / time_stretch * 2.0 * ¦Ð));
    float hotspot_x = cos_factor * splat_radius * 0.8f + kImpulsePosition.x;
    float hotspot_z = sin_factor * splat_radius * 0.8f + kImpulsePosition.z;
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

void FluidSimulator::ComputeCurl(const GraphicsVolume3* vorticity,
                                 std::shared_ptr<GraphicsVolume> velocity)
{
    float inverse_cell_size = 1.0f / CellSize;
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ComputeCurl(vorticity->x()->cuda_volume(),
                                          vorticity->y()->cuda_volume(),
                                          vorticity->z()->cuda_volume(),
                                          velocity->cuda_volume(),
                                          inverse_cell_size);
    }
}

void FluidSimulator::ComputeDivergence()
{
    float half_inverse_cell_size = 0.5f / CellSize;
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ComputeDivergence(general1a_->cuda_volume(),
                                                velocity2_.x()->cuda_volume(),
                                                velocity2_.y()->cuda_volume(),
                                                velocity2_.z()->cuda_volume(),
                                                half_inverse_cell_size);
    } else {
        glUseProgram(Programs.ComputeDivergence);

        SetUniform("HalfInverseCellSize", half_inverse_cell_size);
        SetUniform("Obstacles", 1);
        SetUniform("velocity", 0);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          general1a_->gl_volume()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, velocity_->gl_volume()->texture_handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              general1a_->gl_volume()->depth());
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
        CudaMain::Instance()->ComputeResidualDiagnosis(
            diagnosis_volume_->cuda_volume(), pressure_->cuda_volume(),
            general1a_->cuda_volume(), inverse_h_square);
    } else if (graphics_lib_ == GRAPHICS_LIB_GLSL) {
        glUseProgram(Programs.diagnose_);

        SetUniform("packed_tex", 0);
        SetUniform("inverse_h_square", inverse_h_square);

        diagnosis_volume_->gl_volume()->BindFrameBuffer();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, pressure_->gl_volume()->texture_handle());
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
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->Relax(pressure_->cuda_volume(),
                                    pressure_->cuda_volume(),
                                    general1a_->cuda_volume(), cell_size,
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
                              pressure_->gl_volume()->frame_buffer());
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_3D, pressure_->gl_volume()->texture_handle());
            glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                                  pressure_->gl_volume()->depth());
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

void FluidSimulator::ReviseDensity()
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        if (!sphere)
            CudaMain::Instance()->ReviseDensity(
                density_->cuda_volume(), kImpulsePosition, GridWidth * 0.5f,
                0.1f);
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
        case POISSON_SOLVER_GAUSS_SEIDEL:
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
            if (!pressure_solver_) {
                pressure_solver_.reset(
                    new MultigridPoissonSolver(multigrid_core_.get()));
                pressure_solver_->Initialize(pressure_->GetWidth(),
                                             pressure_->GetHeight(),
                                             pressure_->GetDepth(),
                                             volume_byte_width_);
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
                pressure_solver_->Solve(pressure_, general1a_, CellSize, !i);

            break;
        }
        case POISSON_SOLVER_FULL_MULTI_GRID: {
            if (!pressure_solver_) {
                pressure_solver_.reset(
                    new FullMultigridPoissonSolver(multigrid_core_.get()));
                pressure_solver_->Initialize(pressure_->GetWidth(),
                                             pressure_->GetHeight(),
                                             pressure_->GetDepth(),
                                             volume_byte_width_);
            }

            // The reason why the prolongation in FMG taking the last result 
            // into account would produce a better solution remains a mystery.
            // Recently I found that in a new time step the pressure must be
            // reinitialized to 0 before iteration, or the FMG is going to
            // blow the velocity. This would kind of reveal that the current
            // prolongation scheme is not providing an accurate answer(go to
            // chaos when the iteration times is set to above 3).
            //
            // Also note that, the result of the first iteration in the V-cycle
            // is the error of the 0 guess.

            pressure_->Clear();

            // Chaos occurs if the iteration times is set to a value above 2.
            for (int i = 0; i < num_full_multigrid_iterations_; i++)
                pressure_solver_->Solve(pressure_, general1a_, CellSize, !i);

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
    //
    // 2016/5/23 update:
    // During the process of verifying the staggered grid discretization, I
    // found this coefficient should be the same as that in divergence
    // calculation. This mistake was introduced at the first day the project
    // was created.

    const float half_inverse_cell_size = 0.5f / CellSize;
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->SubtractGradient(velocity2_.x()->cuda_volume(),
                                               velocity2_.y()->cuda_volume(),
                                               velocity2_.z()->cuda_volume(),
                                               pressure_->cuda_volume(),
                                               half_inverse_cell_size);
    } else {
        glUseProgram(Programs.SubtractGradient);

        SetUniform("GradientScale", half_inverse_cell_size);
        SetUniform("velocity", 0);
        SetUniform("packed_tex", 1);

        glBindFramebuffer(GL_FRAMEBUFFER,
                          velocity_->gl_volume()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, velocity_->gl_volume()->texture_handle());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, pressure_->gl_volume()->texture_handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              velocity_->gl_volume()->depth());
        ResetState();
    }
}

void FluidSimulator::AddCurlPsi()
{
    const GraphicsVolume3& psi = GetVorticityField();
    if (!psi)
        return;

    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->AddCurlPsi(velocity_->cuda_volume(),
                                         psi.x()->cuda_volume(),
                                         psi.y()->cuda_volume(),
                                         psi.z()->cuda_volume(), CellSize);
    }
}

void FluidSimulator::AdvectVortices(float delta_time)
{
    const GraphicsVolume3& vorticity = GetVorticityField();
    if (!vorticity)
        return;

    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->AdvectVorticityFields(
            vorticity.x()->cuda_volume(), vorticity.y()->cuda_volume(),
            vorticity.z()->cuda_volume(), general1a_->cuda_volume(),
            general1b_->cuda_volume(), general1c_->cuda_volume(),
            general1d_->cuda_volume(), general4a_->cuda_volume(), delta_time,
            0.0f);
    }
}

void FluidSimulator::ApplyVorticityConfinemnet()
{
    const GraphicsVolume3& vort_conf = GetVorticityConfinementField();
    if (!vort_conf)
        return;

    // Please note that the nth velocity in still within |general4a_|.
    float inverse_cell_size = 1.0f / CellSize;
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ApplyVorticityConfinement(
            general4a_->cuda_volume(), velocity_->cuda_volume(),
            vort_conf.x()->cuda_volume(), vort_conf.y()->cuda_volume(),
            vort_conf.z()->cuda_volume());
    }

    std::swap(velocity_, general4a_);
}

void FluidSimulator::BuildVorticityConfinemnet(float delta_time)
{
    const GraphicsVolume3& vorticity = GetVorticityField();
    if (!vorticity)
        return;

    const GraphicsVolume3& vort_conf = GetVorticityConfinementField();
    if (!vort_conf)
        return;

    // Please note that the nth velocity in still within |general4a_|.
    float inverse_cell_size = 1.0f / CellSize;
    float vort_conf_coef = FluidConfig::Instance()->vorticity_confinement();
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->BuildVorticityConfinement(
            vort_conf.x()->cuda_volume(), vort_conf.y()->cuda_volume(),
            vort_conf.z()->cuda_volume(), vorticity.x()->cuda_volume(),
            vorticity.y()->cuda_volume(), vorticity.z()->cuda_volume(),
            vort_conf_coef * delta_time, CellSize);
    }
}

void FluidSimulator::ComputeDeltaVorticity()
{
    const GraphicsVolume3& vorticity = GetVorticityField();
    if (!vorticity)
        return;

    const GraphicsVolume3& aux = GetAuxField();
    if (!aux)
        return;

    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ComputeDeltaVorticity(
            aux.x()->cuda_volume(), aux.y()->cuda_volume(),
            aux.z()->cuda_volume(), vorticity.x()->cuda_volume(),
            vorticity.y()->cuda_volume(), vorticity.z()->cuda_volume());
    }
}

void FluidSimulator::DecayVortices(float delta_time, float cell_size)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->ComputeDivergenceForVort(
            general1d_->cuda_volume(), general4a_->cuda_volume(), cell_size);
        CudaMain::Instance()->DecayVortices(
            general1a_->cuda_volume(), general1b_->cuda_volume(),
            general1c_->cuda_volume(), general1d_->cuda_volume(), delta_time);
    }
}

void FluidSimulator::SolvePsi()
{
    if (!multigrid_core_) {
        if (graphics_lib_ == GRAPHICS_LIB_CUDA)
            multigrid_core_.reset(new MultigridCoreCuda());
        else
            multigrid_core_.reset(new MultigridCoreGlsl());
    }

    const GraphicsVolume3& psi = GetVorticityField();
    if (!psi)
        return;

    const GraphicsVolume3& delta_vort = GetAuxField();
    if (!delta_vort)
        return;

    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        if (!psi_solver_) {
            psi_solver_.reset(
                new OpenBoundaryMultigridPoissonSolver(multigrid_core_.get()));
            psi_solver_->Initialize(psi.x()->GetWidth(), psi.x()->GetHeight(),
                                    psi.x()->GetDepth(), volume_byte_width_);
        }

        for (int i = 0; i < psi.num_of_volumes(); i++) {
            for (int j = 0; j < num_multigrid_iterations_; j++)
                psi_solver_->Solve(psi[i], delta_vort[i], CellSize);
        }
    }
}

void FluidSimulator::StretchVortices(float delta_time, float cell_size)
{
    const GraphicsVolume3& vorticity = GetVorticityField();
    if (!vorticity)
        return;

    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->StretchVortices(
            general1a_->cuda_volume(), general1b_->cuda_volume(),
            general1c_->cuda_volume(), general4a_->cuda_volume(),
            vorticity.x()->cuda_volume(), vorticity.y()->cuda_volume(),
            vorticity.z()->cuda_volume(), CellSize, delta_time);
    }
}

const GraphicsVolume3& FluidSimulator::GetVorticityField()
{
    if (!vorticity_) {
        bool r = vorticity_.Create(GridWidth, GridHeight, GridDepth, 1, 2);
        assert(r);
    }

    return vorticity_;
}

const GraphicsVolume3& FluidSimulator::GetAuxField()
{
    if (!aux_) {
        bool r = aux_.Create(GridWidth, GridHeight, GridDepth, 1, 2);
        assert(r);
    }

    return aux_;
}

const GraphicsVolume3& FluidSimulator::GetVorticityConfinementField()
{
    if (!vort_conf_) {
        bool r = vort_conf_.Create(GridWidth, GridHeight, GridDepth, 1, 2);
        assert(r);
    }

    return vort_conf_;
}

void FluidSimulator::RestoreVorticity(float delta_time)
{
    if (FluidConfig::Instance()->vorticity_confinement() > 0.0f) {
        const GraphicsVolume3& vorticity = GetVorticityField();
        if (!vorticity)
            return;

        // Please note that the nth velocity in still within |general4a_|.
        ComputeCurl(&vorticity, general4a_);
        BuildVorticityConfinemnet(delta_time);

        StretchVortices(delta_time, CellSize);
        DecayVortices(delta_time, CellSize);
        AdvectVortices(delta_time);

        const GraphicsVolume3& aux = GetAuxField();
        if (!aux)
            return;

        ComputeCurl(&aux, velocity_);
        ComputeDeltaVorticity();
        SolvePsi();
        AddCurlPsi();

        ApplyVorticityConfinemnet();
    }
}
