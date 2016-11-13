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
#include "poisson_core_glsl.h"

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

PoissonCoreGlsl::PoissonCoreGlsl()
    : PoissonCore()
    , prolongate_and_relax_program_()
    , prolongate_packed_program_()
    , relax_packed_program_()
    , relax_zero_guess_packed_program_()
    , residual_packed_program_()
    , restrict_packed_program_()
    , restrict_residual_packed_program_()
{

}

PoissonCoreGlsl::~PoissonCoreGlsl()
{

}

std::shared_ptr<GraphicsMemPiece> PoissonCoreGlsl::CreateMemPiece(int size)
{
    return std::shared_ptr<GraphicsMemPiece>();
}

std::shared_ptr<GraphicsVolume> PoissonCoreGlsl::CreateVolume(
    int width, int height, int depth, int num_of_components, int byte_width)
{
    std::shared_ptr<GraphicsVolume> r(new GraphicsVolume(GRAPHICS_LIB_GLSL));
    bool succeeded = r->Create(width, height, depth, num_of_components,
                               byte_width, 0);

    return succeeded ? r : std::shared_ptr<GraphicsVolume>();
}

std::shared_ptr<GraphicsVolume3> PoissonCoreGlsl::CreateVolumeGroup(
    int width, int height, int depth, int num_of_components, int byte_width)
{
    std::shared_ptr<GraphicsVolume3> r(new GraphicsVolume3(GRAPHICS_LIB_GLSL));
    bool succeeded = r->Create(width, height, depth, num_of_components,
                               byte_width, 0);

    return succeeded ? r : std::shared_ptr<GraphicsVolume3>();
}

void PoissonCoreGlsl::ComputeResidual(const GraphicsVolume& r,
                                      const GraphicsVolume& u,
                                      const GraphicsVolume& b)
{
    float cell_size = 0.15f;
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

void PoissonCoreGlsl::Prolongate(const GraphicsVolume& fine,
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

void PoissonCoreGlsl::ProlongateError(const GraphicsVolume& fine,
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

void PoissonCoreGlsl::Relax(const GraphicsVolume& u, const GraphicsVolume& b,
                            int num_of_iterations)
{
    float cell_size = 0.15f;
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

void PoissonCoreGlsl::RelaxWithZeroGuess(const GraphicsVolume& u,
                                         const GraphicsVolume& b)
{
    float cell_size = 0.15f;
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

void PoissonCoreGlsl::Restrict(const GraphicsVolume& coarse,
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

void PoissonCoreGlsl::ApplyStencil(const GraphicsVolume& aux,
                                   const GraphicsVolume& search)
{

}

void PoissonCoreGlsl::ComputeAlpha(const GraphicsMemPiece& alpha,
                                   const GraphicsMemPiece& rho,
                                   const GraphicsVolume& aux,
                                   const GraphicsVolume& search)
{

}

void PoissonCoreGlsl::ComputeRho(const GraphicsMemPiece& rho,
                                 const GraphicsVolume& search,
                                 const GraphicsVolume& residual)
{

}

void PoissonCoreGlsl::ComputeRhoAndBeta(const GraphicsMemPiece& beta,
                                        const GraphicsMemPiece& rho_new,
                                        const GraphicsMemPiece& rho,
                                        const GraphicsVolume& aux,
                                        const GraphicsVolume& residual)
{

}

void PoissonCoreGlsl::ScaledAdd(const GraphicsVolume& dest,
                                const GraphicsVolume& v0,
                                const GraphicsVolume& v1,
                                const GraphicsMemPiece& coef, float sign)
{

}

void PoissonCoreGlsl::ScaleVector(const GraphicsVolume& dest,
                                  const GraphicsVolume& v,
                                  const GraphicsMemPiece& coef, float sign)
{

}

GLProgram* PoissonCoreGlsl::GetProlongatePackedProgram()
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

GLProgram* PoissonCoreGlsl::GetRelaxPackedProgram()
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

GLProgram* PoissonCoreGlsl::GetRelaxZeroGuessPackedProgram()
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

GLProgram* PoissonCoreGlsl::GetResidualPackedProgram()
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

GLProgram* PoissonCoreGlsl::GetRestrictPackedProgram()
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

GLProgram* PoissonCoreGlsl::GetRestrictResidualPackedProgram()
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
