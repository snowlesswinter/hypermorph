#include "stdafx.h"
#include "multigrid_poisson_solver.h"

#include <cassert>
#include <tuple>

#include "gl_program.h"
#include "utility.h"
#include "metrics.h"
#include "shader/fluid_shader.h"
#include "shader/multigrid_shader.h"

// A summary for lately experiments:
//
// Conclusion:
//
// * The relaxation to the finest level domains the time cost and visual
//   result of the algorithm, though the average |r| will eventually stabilize
//   to around 0.09 in spite of the setting of parameters. I think this is
//   bottleneck for the current algorithm.
// * Increasing the iteration times of coarsen level will not affect the time
//   cost much, neither the visual effect.
// * As said that the first 2 times Jacobi are the most efficient, reducing
//   this number will probably introduce significant artifact to the result.
//   So this is also the number of iterations we choose for the finest level
//   smoothing.

MultigridPoissonSolver::MultigridPoissonSolver()
    : multi_grid_surfaces_()
    , packed_surfaces_()
    , temp_surface_()
    , residual_program_()
    , restrict_program_()
    , prolongate_program_()
    , relax_zero_guess_program_()
    , diagnosis_(false)
    , prolongate_and_relax_program_()
    , prolongate_packed_program_()
    , relax_packed_program_()
    , relax_zero_guess_packed_program_()
    , residual_packed_program_()
    , restrict_packed_program_()
    , absolute_program_()
    , residual_diagnosis_program_()
    , diagnosis_volume_()
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
    packed_surfaces_.push_back(SurfacePod());

    int width = grid_width >> 1;
    while (width > 16) {
        if (diagnosis_)
            multi_grid_surfaces_->push_back(
            std::make_tuple(
            CreateVolume(width, width, width, 1),
            CreateVolume(width, width, width, 1),
            CreateVolume(width, width, width, 1)));
        else
            packed_surfaces_.push_back(CreateVolume(width, width, width, 3));

        width >>= 1;
    }

    diagnosis_volume_.reset(
        new SurfacePod(
        CreateVolume(grid_width, grid_width, grid_width, 1)));

    if (diagnosis_) {
        temp_surface_.reset(
            new SurfacePod(
                CreateVolume(grid_width, grid_width, grid_width, 1)));

        residual_program_.reset(new GLProgram());
        residual_program_->Load(FluidShader::Vertex(), FluidShader::PickLayer(),
                                MultigridShader::ComputeResidual());
        restrict_program_.reset(new GLProgram());
        restrict_program_->Load(FluidShader::Vertex(), FluidShader::PickLayer(),
                                MultigridShader::RestrictShader());
        prolongate_program_.reset(new GLProgram());
        prolongate_program_->Load(FluidShader::Vertex(),
                                  FluidShader::PickLayer(),
                                  MultigridShader::Prolongate());
        relax_zero_guess_program_.reset(new GLProgram());
        relax_zero_guess_program_->Load(FluidShader::Vertex(),
                                        FluidShader::PickLayer(),
                                        MultigridShader::RelaxWithZeroGuess());
        absolute_program_.reset(new GLProgram());
        absolute_program_->Load(FluidShader::Vertex(), FluidShader::PickLayer(),
                                MultigridShader::Absolute());
    } else {
        residual_packed_program_.reset(new GLProgram());
        residual_packed_program_->Load(
            FluidShader::Vertex(), FluidShader::PickLayer(),
            MultigridShader::ComputeResidualPacked());
        restrict_packed_program_.reset(new GLProgram());
        restrict_packed_program_->Load(
            FluidShader::Vertex(), FluidShader::PickLayer(),
            MultigridShader::RestrictPacked());
        prolongate_and_relax_program_.reset(new GLProgram());
        prolongate_and_relax_program_->Load(
            FluidShader::Vertex(), FluidShader::PickLayer(),
            MultigridShader::ProlongateAndRelax());
        prolongate_packed_program_.reset(new GLProgram());
        prolongate_packed_program_->Load(FluidShader::Vertex(),
                                         FluidShader::PickLayer(),
                                         MultigridShader::ProlongatePacked());
        relax_packed_program_.reset(new GLProgram());
        relax_packed_program_->Load(FluidShader::Vertex(),
                                    FluidShader::PickLayer(),
                                    MultigridShader::RelaxPacked());
        relax_zero_guess_packed_program_.reset(new GLProgram());
        relax_zero_guess_packed_program_->Load(
            FluidShader::Vertex(), FluidShader::PickLayer(),
            MultigridShader::RelaxWithZeroGuessPacked());
        absolute_program_.reset(new GLProgram());
        absolute_program_->Load(FluidShader::Vertex(),
                                FluidShader::PickLayer(),
                                MultigridShader::Absolute());
    }
    residual_diagnosis_program_.reset(new GLProgram());
    residual_diagnosis_program_->Load(
        FluidShader::Vertex(), FluidShader::PickLayer(),
        MultigridShader::ComputeResidualPackedDiagnosis());
}

void MultigridPoissonSolver::Solve(const SurfacePod& packed,
                                   bool as_precondition)
{
    if (diagnosis_)
        SolvePlain(packed, as_precondition);
    else
        SolveOpt(packed, as_precondition);

    // For diagnosis.
    ComputeResidualPackedDiagnosis(packed, *diagnosis_volume_, CellSize);
    static int diagnosis = 0;
    if (diagnosis)
    {
        glFinish();
        const SurfacePod* p = diagnosis_volume_.get();

        int w = p->Width;
        int h = p->Height;
        int d = p->Depth;
        int n = 1;
        int element_size = sizeof(float);
        GLenum format = GL_RED;

        static char* v = nullptr;
        if (!v)
            v = new char[w * h * d * element_size * n];

        memset(v, 0, w * h * d * element_size * n);
        glBindTexture(GL_TEXTURE_3D, p->ColorTexture);
        glGetTexImage(GL_TEXTURE_3D, 0, format, GL_FLOAT, v);
        float* f = (float*)v;
        double sum = 0.0;
        float q = 0.0f;
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < h; j++)
            {
                for (int k = 0; k < w; k++)
                {
                    for (int l = 0; l < n; l++)
                    {
                        q = abs(f[i * w * h * n + j * w * n + k * n + l]);
                        //if (l % n == 2)
                            sum += q;
                    }
                }
            }
        }

        double avg = sum / (w * h * d);
        PezDebugString("avg ||r||: %.8f\n", avg);
    }
}

void MultigridPoissonSolver::ComputeResidual(const SurfacePod& u,
                                             const SurfacePod& b,
                                             const SurfacePod& residual,
                                             float cell_size, bool diagnosis)
{
    assert(residual_program_);
    if (!residual_program_)
        return;

    residual_program_->Use();

    SetUniform("packed_tex", 0);
    SetUniform("b", 1);
    SetUniform("inverse_h_square", 1.0f / (cell_size * cell_size));

    glBindFramebuffer(GL_FRAMEBUFFER, residual.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, u.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, b.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, residual.Depth);
    ResetState();

    // For diagnosis
    if (!diagnosis || !absolute_program_)
        return;

    absolute_program_->Use();

    SetUniform("t", 0);

    glBindFramebuffer(GL_FRAMEBUFFER, residual.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, residual.ColorTexture);
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

void MultigridPoissonSolver::RelaxWithZeroGuess(const SurfacePod& u,
                                                const SurfacePod& b,
                                                float cell_size)
{
    assert(relax_zero_guess_program_);
    if (!relax_zero_guess_program_)
        return;

    relax_zero_guess_program_->Use();

    SetUniform("b", 0);
    SetUniform("alpha_omega_over_beta",
               -(cell_size * cell_size) * 0.11111111f);

    glBindFramebuffer(GL_FRAMEBUFFER, u.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, b.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, u.Depth);
    ResetState();
}

void MultigridPoissonSolver::Restrict(const SurfacePod& fine,
                                      const SurfacePod& coarse)
{
    assert(restrict_program_);
    if (!restrict_program_)
        return;

    restrict_program_->Use();

    SetUniform("s", 0);

    glBindFramebuffer(GL_FRAMEBUFFER, fine.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, coarse.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, fine.Depth);
    ResetState();
}

void MultigridPoissonSolver::SolvePlain(const SurfacePod& packed,
                                        bool as_precondition)
{
    assert(multi_grid_surfaces_);
    assert(multi_grid_surfaces_->size() > 1);
    if (!multi_grid_surfaces_ || multi_grid_surfaces_->size() <= 1)
        return;

    int times_to_iterate = 2;
    (*multi_grid_surfaces_)[0] = std::make_tuple(packed, packed,
                                                 *temp_surface_);

    const int num_of_levels = static_cast<int>(multi_grid_surfaces_->size());
    for (int i = 0; i < num_of_levels - 1; i++) {
        Surface& fine_surf = (*multi_grid_surfaces_)[i];
        Surface& coarse_surf = (*multi_grid_surfaces_)[i + 1];

        if (i || as_precondition)
            RelaxWithZeroGuess(std::get<0>(fine_surf), std::get<1>(fine_surf),
                               CellSize);
        else
            Relax(std::get<0>(fine_surf), std::get<1>(fine_surf), CellSize, 1);

        Relax(std::get<0>(fine_surf), std::get<1>(fine_surf), CellSize,
              times_to_iterate - 1);
        ComputeResidual(std::get<0>(fine_surf), std::get<1>(fine_surf),
                        std::get<2>(fine_surf), CellSize, false);
        Restrict(std::get<2>(fine_surf), std::get<1>(coarse_surf));

        times_to_iterate *= 2;
    }

    Surface coarsest = (*multi_grid_surfaces_)[num_of_levels - 1];
    RelaxWithZeroGuess(std::get<0>(coarsest), std::get<1>(coarsest), CellSize);
    Relax(std::get<0>(coarsest), std::get<1>(coarsest), CellSize,
          times_to_iterate);

    for (int j = num_of_levels - 2; j >= 0; j--) {
        Surface& coarse_surf = (*multi_grid_surfaces_)[j + 1];
        Surface& fine_surf = (*multi_grid_surfaces_)[j];
        times_to_iterate /= 2;

        Prolongate(std::get<0>(coarse_surf), std::get<0>(fine_surf));
        Relax(std::get<0>(fine_surf), std::get<1>(fine_surf), CellSize,
              times_to_iterate);
    }
}

void MultigridPoissonSolver::ComputeResidualPacked(const SurfacePod& packed,
                                                   float cell_size)
{
    assert(residual_packed_program_);
    if (!residual_packed_program_)
        return;

    residual_packed_program_->Use();

    SetUniform("packed_tex", 0);
    SetUniform("inverse_h_square", 1.0f / (cell_size * cell_size));

    glBindFramebuffer(GL_FRAMEBUFFER, packed.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, packed.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, packed.Depth);
    ResetState();
}

void MultigridPoissonSolver::ProlongateAndRelax(const SurfacePod& coarse,
                                                const SurfacePod& fine)
{
    assert(prolongate_and_relax_program_);
    if (!prolongate_and_relax_program_)
        return;

    prolongate_and_relax_program_->Use();

    SetUniform("fine", 0);
    SetUniform("c", 1);

    glBindFramebuffer(GL_FRAMEBUFFER, fine.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, fine.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, coarse.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, fine.Depth);
    ResetState();
}

void MultigridPoissonSolver::ProlongatePacked(const SurfacePod& coarse,
                                              const SurfacePod& fine)
{
    assert(prolongate_packed_program_);
    if (!prolongate_packed_program_)
        return;

    prolongate_packed_program_->Use();

    SetUniform("fine", 0);
    SetUniform("c", 1);

    glBindFramebuffer(GL_FRAMEBUFFER, fine.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, fine.ColorTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, coarse.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, fine.Depth);
    ResetState();
}

void MultigridPoissonSolver::RelaxPacked(const SurfacePod& u_and_b,
                                         float cell_size, int times)
{
    for (int i = 0; i < times; i++)
        RelaxPackedImpl(u_and_b, cell_size);
}

void MultigridPoissonSolver::RelaxPackedImpl(const SurfacePod& u_and_b, float cell_size)
{
    assert(relax_packed_program_);
    if (!relax_packed_program_)
        return;

    relax_packed_program_->Use();

    SetUniform("packed_tex", 0);
    SetUniform("one_minus_omega", 0.33333333f);
    SetUniform("minus_h_square", -(cell_size * cell_size));
    SetUniform("omega_over_beta", 0.11111111f);

    glBindFramebuffer(GL_FRAMEBUFFER, u_and_b.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, u_and_b.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, u_and_b.Depth);
    ResetState();
}

void MultigridPoissonSolver::RelaxWithZeroGuessAndComputeResidual(
    const SurfacePod& packed_volumes, float cell_size, int times)
{
}

void MultigridPoissonSolver::RelaxWithZeroGuessPacked(
    const SurfacePod& packed_volumes, float cell_size)
{
    assert(relax_zero_guess_packed_program_);
    if (!relax_zero_guess_packed_program_)
        return;

    relax_zero_guess_packed_program_->Use();

    SetUniform("packed_tex", 0);
    SetUniform("alpha_omega_over_beta",
               -(cell_size * cell_size) * 0.11111111f);
    SetUniform("one_minus_omega", 0.33333333f);
    SetUniform("minus_h_square", -(cell_size * cell_size));
    SetUniform("omega_times_inverse_beta", 0.11111111f);

    glBindFramebuffer(GL_FRAMEBUFFER, packed_volumes.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, packed_volumes.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, packed_volumes.Depth);
    ResetState();
}

void MultigridPoissonSolver::RestrictPacked(const SurfacePod& fine,
                                            const SurfacePod& coarse)
{
    assert(restrict_packed_program_);
    if (!restrict_packed_program_)
        return;

    restrict_packed_program_->Use();

    SetUniform("s", 0);

    glBindFramebuffer(GL_FRAMEBUFFER, coarse.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, fine.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, coarse.Depth);
    ResetState();
}

void MultigridPoissonSolver::SolveOpt(const SurfacePod& u_and_b,
                                      bool as_precondition)
{
    assert(packed_surfaces_.size() > 1);
    if (packed_surfaces_.size() <= 1)
        return;

    int times_to_iterate = 2;
    packed_surfaces_[0] = u_and_b;

    const int num_of_levels = static_cast<int>(packed_surfaces_.size());
    for (int i = 0; i < num_of_levels - 1; i++) {
        SurfacePod fine_volume = packed_surfaces_[i];
        SurfacePod coarse_volume = packed_surfaces_[i + 1];

        if (i || as_precondition)
            RelaxWithZeroGuessPacked(fine_volume, CellSize);
        else
            RelaxPacked(fine_volume, CellSize, 2);

        RelaxPacked(fine_volume, CellSize, times_to_iterate - 2);
        ComputeResidualPacked(fine_volume, CellSize);
        RestrictPacked(fine_volume, coarse_volume);

        times_to_iterate *= 2;
    }

    SurfacePod coarsest = packed_surfaces_[num_of_levels - 1];
    RelaxWithZeroGuessPacked(coarsest, CellSize);
    RelaxPacked(coarsest, CellSize, times_to_iterate - 2);

    for (int j = num_of_levels - 2; j >= 0; j--) {
        SurfacePod coarse_volume = packed_surfaces_[j + 1];
        SurfacePod fine_volume = packed_surfaces_[j];
        times_to_iterate /= 2;

        ProlongatePacked(coarse_volume, fine_volume);
        RelaxPacked(fine_volume, CellSize, times_to_iterate/* - 1*/);
    }
}

void MultigridPoissonSolver::ComputeResidualPackedDiagnosis(
    const SurfacePod& packed, const SurfacePod& diagnosis, float cell_size)
{
    assert(residual_diagnosis_program_);
    if (!residual_diagnosis_program_)
        return;

    residual_diagnosis_program_->Use();

    SetUniform("packed_tex", 0);
    SetUniform("inverse_h_square", 1.0f / (cell_size * cell_size));

    glBindFramebuffer(GL_FRAMEBUFFER, diagnosis.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, packed.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, diagnosis.Depth);
    ResetState();
}
