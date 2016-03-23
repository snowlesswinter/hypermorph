#include "stdafx.h"
#include "full_multigrid_poisson_solver.h"

#include <cassert>
#include <algorithm>

#include "multigrid_core.h"
#include "multigrid_poisson_solver.h"
#include "opengl/gl_program.h"
#include "opengl/gl_texture.h"
#include "shader/fluid_shader.h"
#include "shader/multigrid_shader.h"
#include "shader/multigrid_staggered_shader.h"
#include "utility.h"

const int kWidthOfCoarsestLevel = 32;

FullMultigridPoissonSolver::FullMultigridPoissonSolver()
    : solver_(new MultigridPoissonSolver())
    , packed_surfaces_()
    , packed_textures_()
    , restrict_packed_program_()
{

}

FullMultigridPoissonSolver::~FullMultigridPoissonSolver()
{

}

void FullMultigridPoissonSolver::Initialize(int width, int height, int depth)
{
    solver_->Initialize(width, height, depth);

    // Placeholder for the solution buffer.
    packed_surfaces_.push_back(SurfacePod());
    packed_textures_.push_back(std::shared_ptr<GLTexture>());

    int min_width = std::min(std::min(width, height), depth);

    int scale = 2;
    while (min_width / scale > 32) { // TODO: CUDA has some problem with 32^3
                                     //       volumes.
        int w = width / scale;
        int h = height / scale;
        int d = depth / scale;

        std::shared_ptr<GLTexture> t = 
            solver_->core()->CreateTexture(w, h, d, GL_RGBA32F, GL_RGBA);
        packed_textures_.push_back(t);

        SurfacePod kk;
        kk.FboHandle = t->frame_buffer();
        kk.ColorTexture = t->handle();
        kk.Width = t->width();
        kk.Height = t->height();
        kk.Depth = t->depth();
        packed_surfaces_.push_back(kk);

        scale <<= 1;
    }

    restrict_packed_program_.reset(new GLProgram());
    restrict_packed_program_->Load(
        FluidShader::Vertex(), FluidShader::PickLayer(),
        MultigridStaggeredShader::RestrictPacked());
}

void FullMultigridPoissonSolver::Solve(const SurfacePod& u_and_b,
                                       float cell_size,
                                       bool as_precondition,
                                       std::shared_ptr<GLTexture> t)
{
    if (u_and_b.Width < 32) {
        solver_->Solve(u_and_b, cell_size, true, t);
        return;
    }

    assert(packed_surfaces_.size() > 1);
    if (packed_surfaces_.size() <= 1)
        return;

    // With less iterations in each level but more iterating in every V-Cycle
    // will out perform the case visa versa(less time cost, lower avg/max |r|),
    // especially in high divergence cases.
    solver_->SetBaseRelaxationTimes(5);
    packed_surfaces_[0] = u_and_b;
    packed_textures_[0] = t;

    const int num_of_levels = static_cast<int>(packed_surfaces_.size());
    float level_cell_size = cell_size;
    for (int i = 0; i < num_of_levels - 1; i++) {
        SurfacePod fine_volume = packed_surfaces_[i];
        SurfacePod coarse_volume = packed_surfaces_[i + 1];

        Restrict(fine_volume, coarse_volume);

        level_cell_size *= 1.0f;
    }

    SurfacePod coarsest = packed_surfaces_[num_of_levels - 1];
    if (as_precondition)
        solver_->RelaxWithZeroGuessPacked(coarsest, level_cell_size);

    solver_->RelaxPacked(coarsest, level_cell_size, 15);

    int times_to_iterate = 1;
    for (int j = num_of_levels - 2; j >= 0; j--) {
        SurfacePod coarse_volume = packed_surfaces_[j + 1];
        SurfacePod fine_volume = packed_surfaces_[j];

        //solver_->ProlongatePacked(coarse_volume, fine_volume);

        solver_->ProlongatePacked2(packed_textures_[j + 1], packed_textures_[j]);

        for (int k = 0; k < times_to_iterate; k++)
            solver_->Solve(fine_volume, level_cell_size, false, t);

        // For comparison.
        // 
        // Damped Jacobi is still faster than Multigrid, no much though. With
        // a base relaxation times 5, Multigrid had achieved a notable
        // lower avg/max |r| compared to Jacobi in our experiments.

        //solver_->RelaxPacked(fine_volume, level_cell_size, 15);

        // Experiments revealed that iterations in different levels almost
        // equally contribute to the final result, thus we are not going to
        // reduce the iteration times in coarsen level.
        times_to_iterate += 0;
        level_cell_size *= 1.0f;
    }

    if (!as_precondition)
        solver_->Diagnose(t.get());
}

void FullMultigridPoissonSolver::Restrict(const SurfacePod& fine,
                                          const SurfacePod& coarse)
{
    assert(restrict_packed_program_);
    if (!restrict_packed_program_)
        return;

    restrict_packed_program_->Use();

    SetUniform("s", 0);
    SetUniform("inverse_size",  CalculateInverseSize(fine));

    glBindFramebuffer(GL_FRAMEBUFFER, coarse.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, fine.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, coarse.Depth);
    ResetState();
}
