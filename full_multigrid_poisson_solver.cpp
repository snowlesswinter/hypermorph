#include "stdafx.h"
#include "full_multigrid_poisson_solver.h"

#include <cassert>
#include <algorithm>

#include "gl_program.h"
#include "multigrid_poisson_solver.h"
#include "shader/fluid_shader.h"
#include "shader/multigrid_shader.h"
#include "utility.h"

const int kWidthOfCoarsestLevel = 32;

FullMultigridPoissonSolver::FullMultigridPoissonSolver()
    : solver_(new MultigridPoissonSolver())
    , packed_surfaces_()
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

    int min_width = std::min(std::min(width, height), depth);

    int scale = 2;
    while (min_width / scale > 16) {
        int w = width / scale;
        int h = height / scale;
        int d = depth / scale;

        packed_surfaces_.push_back(CreateVolume(w, h, d, 3));

        scale <<= 1;
    }

    restrict_packed_program_.reset(new GLProgram());
    restrict_packed_program_->Load(
        FluidShader::Vertex(), FluidShader::PickLayer(),
        MultigridShader::RestrictUBPacked());
}

void FullMultigridPoissonSolver::Solve(const SurfacePod& u_and_b,
                                       bool as_precondition)
{
    if (u_and_b.Width < 32) {
        solver_->Solve(u_and_b, true);
        return;
    }

    assert(packed_surfaces_.size() > 1);
    if (packed_surfaces_.size() <= 1)
        return;

    packed_surfaces_[0] = u_and_b;

    const int num_of_levels = static_cast<int>(packed_surfaces_.size());
    for (int i = 0; i < num_of_levels - 1; i++) {
        SurfacePod fine_volume = packed_surfaces_[i];
        SurfacePod coarse_volume = packed_surfaces_[i + 1];

        Restrict(fine_volume, coarse_volume);
    }

    SurfacePod coarsest = packed_surfaces_[num_of_levels - 1];
    solver_->RelaxWithZeroGuessPacked(coarsest, CellSize);
    solver_->RelaxPacked(coarsest, CellSize, 15);

    for (int j = num_of_levels - 2; j >= 0; j--) {
        SurfacePod coarse_volume = packed_surfaces_[j + 1];
        SurfacePod fine_volume = packed_surfaces_[j];

        solver_->ProlongatePacked(coarse_volume, fine_volume);
//         solver_->Solve(fine_volume, as_precondition);
//         solver_->Solve(fine_volume, false);
//         solver_->Solve(fine_volume, false);
        solver_->RelaxPacked(fine_volume, CellSize, 15);
    }

    //solver_->RelaxPacked(u_and_b, CellSize, 40);
    solver_->Diagnose(u_and_b);
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
