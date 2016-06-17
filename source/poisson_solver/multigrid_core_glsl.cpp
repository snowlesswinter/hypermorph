#include "stdafx.h"
#include "multigrid_core_glsl.h"

#include <cassert>

#include "cuda_host/cuda_main.h"
#include "graphics_volume.h"
#include "graphics_volume_group.h"
#include "opengl/gl_program.h"
#include "opengl/gl_volume.h"
#include "shader/fluid_shader.h"
#include "shader/multigrid_shader.h"
#include "third_party/glm/vec3.hpp"
#include "utility.h"

MultigridCoreGlsl::MultigridCoreGlsl()
    : MultigridCore()
    , prolongate_and_relax_program_()
    , prolongate_packed_program_()
    , relax_packed_program_()
    , relax_zero_guess_packed_program_()
    , residual_packed_program_()
    , restrict_packed_program_()
    , restrict_residual_packed_program_()
{

}

MultigridCoreGlsl::~MultigridCoreGlsl()
{

}

std::shared_ptr<GraphicsMemPiece> MultigridCoreGlsl::CreateMemPiece(int size)
{
    return std::shared_ptr<GraphicsMemPiece>();
}

std::shared_ptr<GraphicsVolume> MultigridCoreGlsl::CreateVolume(
    int width, int height, int depth, int num_of_components, int byte_width)
{
    std::shared_ptr<GraphicsVolume> r(new GraphicsVolume(GRAPHICS_LIB_GLSL));
    bool succeeded = r->Create(width, height, depth, num_of_components,
                               byte_width, 0);

    return succeeded ? r : std::shared_ptr<GraphicsVolume>();
}

std::shared_ptr<GraphicsVolume3> MultigridCoreGlsl::CreateVolumeGroup(
    int width, int height, int depth, int num_of_components, int byte_width)
{
    std::shared_ptr<GraphicsVolume3> r(new GraphicsVolume3(GRAPHICS_LIB_GLSL));
    bool succeeded = r->Create(width, height, depth, num_of_components,
                               byte_width, 0);

    return succeeded ? r : std::shared_ptr<GraphicsVolume3>();
}

void MultigridCoreGlsl::ComputeResidual(const GraphicsVolume& r,
                                        const GraphicsVolume& u,
                                        const GraphicsVolume& b,
                                        float cell_size)
{
    GetResidualPackedProgram()->Use();

    SetUniform("packed_tex", 0);
    SetUniform("inverse_h_square", 1.0f / (cell_size * cell_size));

    glBindFramebuffer(GL_FRAMEBUFFER, r.gl_volume()->frame_buffer());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, u.gl_volume()->texture_handle());
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                          u.gl_volume()->depth());
    ResetState();
}

void MultigridCoreGlsl::Prolongate(const GraphicsVolume& fine,
                                   const GraphicsVolume& coarse)
{
    GetProlongatePackedProgram()->Use();

    SetUniform("fine", 0);
    SetUniform("s", 1);
    SetUniform("inverse_size_f", CalculateInverseSize(*fine.gl_volume()));
    SetUniform("inverse_size_c", CalculateInverseSize(*coarse.gl_volume()));
    SetUniform("overlay", 1.0f);

    glBindFramebuffer(GL_FRAMEBUFFER, fine.gl_volume()->frame_buffer());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, fine.gl_volume()->texture_handle());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, coarse.gl_volume()->texture_handle());
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                          fine.gl_volume()->depth());
    ResetState();
}

void MultigridCoreGlsl::ProlongateError(const GraphicsVolume& fine,
                                        const GraphicsVolume& coarse)
{
    GetProlongatePackedProgram()->Use();

    SetUniform("fine", 0);
    SetUniform("s", 1);
    SetUniform("inverse_size_f", CalculateInverseSize(*fine.gl_volume()));
    SetUniform("inverse_size_c", CalculateInverseSize(*coarse.gl_volume()));
    SetUniform("overlay", 1.0f);

    glBindFramebuffer(GL_FRAMEBUFFER, fine.gl_volume()->frame_buffer());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, fine.gl_volume()->texture_handle());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, coarse.gl_volume()->texture_handle());
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                          fine.gl_volume()->depth());
    ResetState();
}

void MultigridCoreGlsl::Relax(const GraphicsVolume& u, const GraphicsVolume& b,
                              float cell_size, int num_of_iterations)
{
    for (int i = 0; i < num_of_iterations; i++) {
        GetRelaxPackedProgram()->Use();

        SetUniform("packed_tex", 0);
        SetUniform("one_minus_omega", 0.33333333f);
        SetUniform("minus_h_square", -(cell_size * cell_size));
        SetUniform("omega_over_beta", 0.11111111f);

        glBindFramebuffer(GL_FRAMEBUFFER, u.gl_volume()->frame_buffer());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, u.gl_volume()->texture_handle());
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                              u.gl_volume()->depth());
        ResetState();
    }
}

void MultigridCoreGlsl::RelaxWithZeroGuess(const GraphicsVolume& u,
                                           const GraphicsVolume& b,
                                           float cell_size)
{
    GetRelaxZeroGuessPackedProgram()->Use();

    SetUniform("packed_tex", 0);
    SetUniform("alpha_omega_over_beta",
               -(cell_size * cell_size) * 0.11111111f);
    SetUniform("one_minus_omega", 0.33333333f);
    SetUniform("minus_h_square", -(cell_size * cell_size));
    SetUniform("omega_times_inverse_beta", 0.11111111f);

    glBindFramebuffer(GL_FRAMEBUFFER, u.gl_volume()->frame_buffer());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, u.gl_volume()->texture_handle());
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                          u.gl_volume()->depth());
    ResetState();
}

void MultigridCoreGlsl::Restrict(const GraphicsVolume& coarse,
                                 const GraphicsVolume& fine)
{
    GetRestrictPackedProgram()->Use();

    SetUniform("s", 0);
    SetUniform("inverse_size", CalculateInverseSize(*fine.gl_volume()));

    glBindFramebuffer(GL_FRAMEBUFFER, coarse.gl_volume()->frame_buffer());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, fine.gl_volume()->texture_handle());
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
                          coarse.gl_volume()->depth());
    ResetState();
}

void MultigridCoreGlsl::ApplyStencil(const GraphicsVolume& aux,
                                     const GraphicsVolume& search,
                                     float cell_size)
{

}

void MultigridCoreGlsl::ComputeRho(const GraphicsMemPiece& rho,
                                   const GraphicsVolume& aux,
                                   const GraphicsVolume& r)
{

}

GLProgram* MultigridCoreGlsl::GetProlongatePackedProgram()
{
    if (!prolongate_packed_program_)
    {
        prolongate_packed_program_.reset(new GLProgram());
        prolongate_packed_program_->Load(FluidShader::Vertex(),
                                         FluidShader::PickLayer(),
                                         MultigridShader::ProlongatePacked());
    }

    return prolongate_packed_program_.get();
}

GLProgram* MultigridCoreGlsl::GetRelaxPackedProgram()
{
    if (!relax_packed_program_)
    {
        relax_packed_program_.reset(new GLProgram());
        relax_packed_program_->Load(FluidShader::Vertex(),
                                    FluidShader::PickLayer(),
                                    MultigridShader::RelaxPacked());
    }

    return relax_packed_program_.get();
}

GLProgram* MultigridCoreGlsl::GetRelaxZeroGuessPackedProgram()
{
    if (!relax_zero_guess_packed_program_)
    {
        relax_zero_guess_packed_program_.reset(new GLProgram());
        relax_zero_guess_packed_program_->Load(
            FluidShader::Vertex(), FluidShader::PickLayer(),
            MultigridShader::RelaxWithZeroGuessPacked());
    }

    return relax_zero_guess_packed_program_.get();
}

GLProgram* MultigridCoreGlsl::GetResidualPackedProgram()
{
    if (!residual_packed_program_)
    {
        residual_packed_program_.reset(new GLProgram());
        residual_packed_program_->Load(FluidShader::Vertex(),
                                       FluidShader::PickLayer(),
                                       MultigridShader::ComputeResidual());
    }

    return residual_packed_program_.get();
}

GLProgram* MultigridCoreGlsl::GetRestrictPackedProgram()
{
    if (!restrict_packed_program_)
    {
        restrict_packed_program_.reset(new GLProgram());
        restrict_packed_program_->Load(FluidShader::Vertex(),
                                       FluidShader::PickLayer(),
                                       MultigridShader::RestrictPacked());
    }

    return restrict_packed_program_.get();
}

GLProgram* MultigridCoreGlsl::GetRestrictResidualPackedProgram()
{
    if (!restrict_residual_packed_program_)
    {
        restrict_residual_packed_program_.reset(new GLProgram());
        restrict_residual_packed_program_->Load(
            FluidShader::Vertex(), FluidShader::PickLayer(),
            MultigridShader::RestrictResidualPacked());
    }

    return restrict_residual_packed_program_.get();
}
