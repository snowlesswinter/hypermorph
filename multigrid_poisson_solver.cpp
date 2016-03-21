#include "stdafx.h"
#include "multigrid_poisson_solver.h"

#include <cassert>
#include <tuple>

#include "cuda/cuda_main.h"
#include "metrics.h"
#include "opengl/gl_program.h"
#include "opengl/gl_texture.h"
#include "shader/fluid_shader.h"
#include "shader/multigrid_shader.h"
#include "shader/multigrid_staggered_shader.h"
#include "utility.h"


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
    , surf_resource()
    , temp_surface_()
    , residual_program_()
    , restrict_program_()
    , prolongate_program_()
    , relax_zero_guess_program_()
    , times_to_iterate_(2)
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

void MultigridPoissonSolver::Initialize(int width, int height, int depth)
{
    assert(!multi_grid_surfaces_);
    multi_grid_surfaces_.reset(new MultiGridSurfaces());

    // Placeholder for the solution buffer.
    multi_grid_surfaces_->push_back(
        std::make_tuple(SurfacePod(), SurfacePod(), SurfacePod()));

    int min_width = std::min(std::min(width, height), depth);

    int scale = 2;
    while (min_width / scale > 16) {
        int w = width / scale;
        int h = height / scale;
        int d = depth / scale;

        if (diagnosis_)
            multi_grid_surfaces_->push_back(
                std::make_tuple(
                    CreateVolume(w, h, d, 1), CreateVolume(w, h, d, 1),
                    CreateVolume(w, h, d, 1)));
        else
            surf_resource.push_back(CreateVolume(w, h, d, 3));

        scale <<= 1;
    }

    if (diagnosis_) {
        temp_surface_.reset(
            new SurfacePod(CreateVolume(width, height, depth, 1)));

        residual_program_.reset(new GLProgram());
        residual_program_->Load(FluidShader::Vertex(), FluidShader::PickLayer(),
                                MultigridShader::ComputeResidual());
        restrict_program_.reset(new GLProgram());
        restrict_program_->Load(FluidShader::Vertex(), FluidShader::PickLayer(),
                                MultigridShader::Restrict());
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
            MultigridStaggeredShader::RestrictResidualPacked());
        prolongate_and_relax_program_.reset(new GLProgram());
        prolongate_and_relax_program_->Load(
            FluidShader::Vertex(), FluidShader::PickLayer(),
            MultigridShader::ProlongateAndRelax());
        prolongate_packed_program_.reset(new GLProgram());
        prolongate_packed_program_->Load(
            FluidShader::Vertex(), FluidShader::PickLayer(),
            MultigridStaggeredShader::ProlongatePacked());
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

void MultigridPoissonSolver::Solve(const SurfacePod& u_and_b, float cell_size,
                                   bool as_precondition)
{
    if (!ValidateVolume(u_and_b))
        return;

    if (diagnosis_)
        SolvePlain(u_and_b, cell_size, as_precondition);
    else
        SolveOpt(u_and_b, cell_size, as_precondition);

    //Diagnose(u_and_b);
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
    SetUniform("inverse_size", CalculateInverseSize(fine));

    glBindFramebuffer(GL_FRAMEBUFFER, fine.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, coarse.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, fine.Depth);
    ResetState();
}

void MultigridPoissonSolver::SetBaseRelaxationTimes(int base_times)
{
    times_to_iterate_ = base_times;
}

void MultigridPoissonSolver::SolvePlain(const SurfacePod& u_and_b,
                                        float cell_size, bool as_precondition)
{
    assert(multi_grid_surfaces_);
    assert(multi_grid_surfaces_->size() > 1);
    if (!multi_grid_surfaces_ || multi_grid_surfaces_->size() <= 1)
        return;

    int times_to_iterate = times_to_iterate_;
    (*multi_grid_surfaces_)[0] = std::make_tuple(u_and_b, u_and_b,
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

bool MultigridPoissonSolver::ValidateVolume(const SurfacePod& u_and_b)
{
    if (surf_resource.empty())
        return false;

    if (u_and_b.Width > surf_resource[0].Width * 2 ||
            u_and_b.Height > surf_resource[0].Height * 2 ||
            u_and_b.Depth > surf_resource[0].Depth * 2)
        return false;

    return true;
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
    SetUniform("s", 1);
    SetUniform("inverse_size_f", CalculateInverseSize(fine));
    SetUniform("inverse_size_c", CalculateInverseSize(coarse));

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
    SetUniform("inverse_size", CalculateInverseSize(fine));

    glBindFramebuffer(GL_FRAMEBUFFER, coarse.FboHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, fine.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, coarse.Depth);
    ResetState();
}

void MultigridPoissonSolver::SolveOpt(const SurfacePod& u_and_b,
                                      float cell_size, bool as_precondition)
{
    std::vector<SurfacePod> surfs(1, u_and_b);
    auto i = surf_resource.begin();
    for (; i != surf_resource.end(); ++i) {
        if (i->Width * 2 == u_and_b.Width)
            break;
    }

    assert(i != surf_resource.end());
    if (i == surf_resource.end())
        return;

    surfs.insert(surfs.end(), i, surf_resource.end());

    int times_to_iterate = times_to_iterate_;

    const int num_of_levels = static_cast<int>(surfs.size());
    float level_cell_size = cell_size;
    for (int i = 0; i < num_of_levels - 1; i++) {
        SurfacePod fine_volume = surfs[i];
        SurfacePod coarse_volume = surfs[i + 1];

        if (i || as_precondition)
            RelaxWithZeroGuessPacked(fine_volume, level_cell_size);
        else
            RelaxPacked(fine_volume, level_cell_size, 2);

        RelaxPacked(fine_volume, level_cell_size, times_to_iterate - 2);
        ComputeResidualPacked(fine_volume, level_cell_size);
        RestrictPacked(fine_volume, coarse_volume);

        times_to_iterate *= 2;
        level_cell_size /= 1.0f; // Reducing the h every level will give us
                                 // worse result of |r|. Need digging.
    }

    SurfacePod coarsest = surfs[num_of_levels - 1];
    RelaxWithZeroGuessPacked(coarsest, level_cell_size);
    RelaxPacked(coarsest, level_cell_size, times_to_iterate - 2);

    for (int j = num_of_levels - 2; j >= 0; j--) {
        SurfacePod coarse_volume = surfs[j + 1];
        SurfacePod fine_volume = surfs[j];

        times_to_iterate /= 2;
        level_cell_size *= 1.0f;

        ProlongatePacked(coarse_volume, fine_volume);
        RelaxPacked(fine_volume, level_cell_size, times_to_iterate/* - 1*/);
    }
}

void MultigridPoissonSolver::ComputeResidualPackedDiagnosis(
    const SurfacePod& packed, const GLTexture& diagnosis, float cell_size)
{
    assert(residual_diagnosis_program_);
    if (!residual_diagnosis_program_)
        return;

    residual_diagnosis_program_->Use();

    SetUniform("packed_tex", 0);
    SetUniform("inverse_h_square", 1.0f / (cell_size * cell_size));

    diagnosis.BindFrameBuffer();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, packed.ColorTexture);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, diagnosis.depth());
    ResetState();

//     assert(absolute_program_);
//     if (!absolute_program_)
//         return;
// 
//     absolute_program_->Use();
// 
//     SetUniform("t", 0);
// 
//     diagnosis.BindFrameBuffer();
//     glActiveTexture(GL_TEXTURE0);
//     diagnosis.Bind();
//     glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, diagnosis.depth());
//     ResetState();
}

void MultigridPoissonSolver::Diagnose(const SurfacePod& packed)
{
    extern int g_diagnosis;
    if (g_diagnosis) {
        if (!diagnosis_volume_ || diagnosis_volume_->width() != packed.Width ||
                diagnosis_volume_->height() != packed.Height ||
                diagnosis_volume_->depth() != packed.Depth) {
            diagnosis_volume_.reset(new GLTexture());
            bool r = diagnosis_volume_->Create(packed.Width, packed.Height,
                                               packed.Depth, GL_R16F, GL_RED);
            if (r)
                CudaMain::Instance()->RegisterGLImage(diagnosis_volume_);
        }

        ComputeResidualPackedDiagnosis(packed, *diagnosis_volume_, CellSize);
        glFinish();
        GLTexture* p = diagnosis_volume_.get();

        int w = p->width();
        int h = p->height();
        int d = p->depth();
        int n = 1;
        int element_size = sizeof(float);
        GLenum format = GL_RED;

        static char* v = nullptr;
        if (!v)
            v = new char[w * h * d * element_size * n];

        memset(v, 0, w * h * d * element_size * n);
        p->GetTexImage(format, GL_FLOAT, v);
        
        float* f = (float*)v;
        double sum = 0.0;
        double q = 0.0f;
        double m = 0.0f;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < h; j++) {
                for (int k = 0; k < w; k++) {
                    for (int l = 0; l < n; l++) {
                        q = f[i * w * h * n + j * w * n + k * n + l];
                        //if (l % n == 2)
                        sum += q;
                        m = std::max(q, m);
                    }
                }
            }
        }

        // =========================================================================
        CudaMain::Instance()->Absolute(diagnosis_volume_);
        // =========================================================================

        double avg = sum / (w * h * d);
        PezDebugString("avg ||r||: %.8f,    max ||r||: %.8f\n", avg, m);
    }
}