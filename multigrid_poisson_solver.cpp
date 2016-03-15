#include "stdafx.h"
#include "multigrid_poisson_solver.h"

#include <cassert>
#include <tuple>

#include "gl_program.h"
#include "utility.h"
#include "fluid_shader.h"
#include "multigrid_shader.h"

MultigridPoissonSolver::MultigridPoissonSolver()
    : multi_grid_surfaces_()
    , restricted_residual_()
    , temp_surface_()
    , residual_program_()
    , restrict_program_()
    , prolongate_program_()
{
}

MultigridPoissonSolver::~MultigridPoissonSolver()
{

}

void MultigridPoissonSolver::Initialize(int grid_width)
{
    assert(!multi_grid_surfaces_);
    multi_grid_surfaces_.reset(new MultiGridSurfaces());

    // Placeholder for the solution buffer.
    multi_grid_surfaces_->push_back(
        std::make_tuple(SurfacePod(), SurfacePod(), SurfacePod()));

    int width = grid_width >> 1;
    while (width > 16) {
        multi_grid_surfaces_->push_back(
            std::make_tuple(
                CreateVolume(width, width, width, 1),
                CreateVolume(width, width, width, 1),
                CreateVolume(width, width, width, 1)));

        width >>= 1;
    }

//     restricted_residual_.reset(
//         new SurfacePod(
//             CreateVolume(grid_width >> 1, grid_width >> 1, grid_width >> 1,
//                          1)));

    temp_surface_.reset(
        new SurfacePod(
        CreateVolume(grid_width, grid_width, grid_width, 1)));
    diagnosis_.reset(
        new SurfacePod(
        CreateVolume(grid_width, grid_width, grid_width, 1)));

    residual_program_.reset(new GLProgram());
    residual_program_->Load(FluidShader::GetVertexShaderCode(),
                            FluidShader::GetPickLayerShaderCode(),
                            MultigridShader::GetComputeResidualShaderCode());
    restrict_program_.reset(new GLProgram());
    restrict_program_->Load(FluidShader::GetVertexShaderCode(),
                            FluidShader::GetPickLayerShaderCode(),
                            MultigridShader::GetRestrictShaderCode());
    prolongate_program_.reset(new GLProgram());
    prolongate_program_->Load(FluidShader::GetVertexShaderCode(),
                              FluidShader::GetPickLayerShaderCode(),
                              MultigridShader::GetProlongateShaderCode());
}

void MultigridPoissonSolver::Solve(const SurfacePod& pressure,
                                   const SurfacePod& divergence,
                                   bool as_precondition)
{
    assert(multi_grid_surfaces_);
    assert(multi_grid_surfaces_->size() > 1);
//     assert(restricted_residual_);
//     if (!multi_grid_surfaces_ || multi_grid_surfaces_->empty() ||
//             !restricted_residual_)
//         return;
    if (!multi_grid_surfaces_ || multi_grid_surfaces_->empty())
        return;

    int times_to_iterate = 2;
    (*multi_grid_surfaces_)[0] = std::make_tuple(pressure, divergence,
                                                 *temp_surface_);

    const int num_of_levels = static_cast<int>(multi_grid_surfaces_->size());
    for (int i = 0; i < num_of_levels - 1; i++)
    {
        Surface& fine_surf = (*multi_grid_surfaces_)[i];
        Surface& coarse_surf = (*multi_grid_surfaces_)[i + 1];

        if (!as_precondition)
            ClearSurface(std::get<0>(fine_surf), 0.0f);

        Relax(std::get<0>(fine_surf), std::get<1>(fine_surf), CellSize,
              times_to_iterate);
        ComputeResidual(std::get<0>(fine_surf), std::get<1>(fine_surf),
                        std::get<2>(fine_surf), CellSize);
        Restrict(std::get<2>(fine_surf), std::get<1>(coarse_surf));

        times_to_iterate += 2;
    }

    Surface coarsest = (*multi_grid_surfaces_)[num_of_levels - 1];
    if (!as_precondition)
        ClearSurface(std::get<0>(coarsest), 0.0f);

    Relax(std::get<0>(coarsest), std::get<1>(coarsest), CellSize,
          times_to_iterate);

    for (int j = num_of_levels - 2; j >= 0; j--)
    {
        Surface& coarse_surf = (*multi_grid_surfaces_)[j + 1];
        Surface& fine_surf = (*multi_grid_surfaces_)[j];
        times_to_iterate -= 2;

        Prolongate(std::get<0>(coarse_surf), std::get<0>(fine_surf));
        Relax(std::get<0>(fine_surf), std::get<1>(fine_surf), CellSize,
              times_to_iterate);
    }

    // For diagnosis.
    ComputeResidual(pressure, divergence, *diagnosis_, CellSize);
    static int diagnosis = 0;
    if (diagnosis)
    {
        glFinish();
        SurfacePod* p = diagnosis_.get();

        int w = p->Width;
        int h = p->Height;
        int d = p->Depth;

        static char* v = nullptr;
        if (!v)
            v = new char[w * h * d * 4];

        memset(v, 0, w * h * d * 4);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, p->FboHandle);
        glReadPixels(0, 0, w, h, GL_RED, GL_FLOAT, v);
        float* f = (float*)v;
        double sum = 0.0;
        for (int i = 0; i < w * h * d; i++)
            sum += abs(f[i]);

        double avg = sum / (w * h * d);
        PezDebugString("avg ||r||: %.4f\n", (float)avg);
    }
    
}

void MultigridPoissonSolver::ComputeResidual(const SurfacePod& u,
                                             const SurfacePod& b,
                                             const SurfacePod& residual,
                                             float cell_size)
{
    assert(residual_program_);
    if (!residual_program_)
        return;

    residual_program_->Use();

    SetUniform("residual", 0);
    SetUniform("u", 1);
    SetUniform("b", 2);
    SetUniform("inverse_h_square", 1.0f / (cell_size * cell_size));

    glBindFramebuffer(GL_FRAMEBUFFER, residual.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, residual.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, u.ColorTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_3D, b.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, residual.Depth);
    ResetState();
}

void MultigridPoissonSolver::Prolongate(const SurfacePod& coarse_solution,
                                        const SurfacePod& fine_solution)
{
    assert(prolongate_program_);
    if (!prolongate_program_)
        return;

    prolongate_program_->Use();

    SetUniform("fine", 0);
    SetUniform("c", 1);

    glBindFramebuffer(GL_FRAMEBUFFER, fine_solution.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, fine_solution.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, coarse_solution.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, fine_solution.Depth);
    ResetState();
}

void MultigridPoissonSolver::Relax(const SurfacePod& u, const SurfacePod& b,
                                   float cell_size, int times)
{
    for (int i = 0; i < times; i++)
        DampedJacobi(u, b, SurfacePod(), cell_size);
}

void MultigridPoissonSolver::Restrict(const SurfacePod& source,
                                      const SurfacePod& dest)
{
    assert(restrict_program_);
    if (!restrict_program_)
        return;

    restrict_program_->Use();

    SetUniform("s", 0);

    glBindFramebuffer(GL_FRAMEBUFFER, dest.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, source.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, dest.Depth);
    ResetState();
}
