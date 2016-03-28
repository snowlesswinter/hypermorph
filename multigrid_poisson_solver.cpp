#include "stdafx.h"
#include "multigrid_poisson_solver.h"

#include <cassert>
#include <tuple>

#include "cuda_host/cuda_main.h"
#include "graphics_volume.h"
#include "metrics.h"
#include "multigrid_core.h"
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

MultigridPoissonSolver::MultigridPoissonSolver(MultigridCore* core)
    : core_(core)
    , volume_resource()
    , times_to_iterate_(2)
    , diagnosis_(false)
    , diagnosis_volume_()
{
}

MultigridPoissonSolver::~MultigridPoissonSolver()
{

}

bool MultigridPoissonSolver::Initialize(int width, int height, int depth)
{
    volume_resource.clear();

    int min_width = std::min(std::min(width, height), depth);
    int scale = 2;
    while (min_width / scale > 16) {
        int w = width / scale;
        int h = height / scale;
        int d = depth / scale;
        std::shared_ptr<GraphicsVolume> v = core_->CreateVolume(w, h, d, 4, 2);
        if (!v)
            return false;

        volume_resource.push_back(v);

        scale <<= 1;
    }

    return true;
}

void MultigridPoissonSolver::Solve(std::shared_ptr<GraphicsVolume> u_and_b,
                                   float cell_size,
                                   bool as_precondition)
{
    if (!ValidateVolume(u_and_b))
        return;

    SolveOpt(u_and_b, cell_size, as_precondition);

    //Diagnose(u_and_b);
}

void MultigridPoissonSolver::SetBaseRelaxationTimes(int base_times)
{
    times_to_iterate_ = base_times;
}

bool MultigridPoissonSolver::ValidateVolume(
    std::shared_ptr<GraphicsVolume> u_and_b)
{
    if (volume_resource.empty())
        return false;

    if (u_and_b->GetWidth() > volume_resource[0]->GetWidth() * 2 ||
            u_and_b->GetHeight() > volume_resource[0]->GetHeight() * 2 ||
            u_and_b->GetDepth() > volume_resource[0]->GetDepth() * 2)
        return false;

    return true;
}

void MultigridPoissonSolver::RelaxPacked(
    std::shared_ptr<GraphicsVolume> u_and_b, float cell_size, int times)
{
    for (int i = 0; i < times; i++)
        core_->RelaxPacked(*u_and_b, cell_size);
}

void MultigridPoissonSolver::SolveOpt(std::shared_ptr<GraphicsVolume> u_and_b,
                                      float cell_size, bool as_precondition)
{
    std::vector<std::shared_ptr<GraphicsVolume>> surfs(1, u_and_b);
    auto i = volume_resource.begin();
    for (; i != volume_resource.end(); ++i) {
        if ((*i)->GetWidth() * 2 == u_and_b->GetWidth())
            break;
    }

    assert(i != volume_resource.end());
    if (i == volume_resource.end())
        return;

    surfs.insert(surfs.end(), i, volume_resource.end());

    int times_to_iterate = times_to_iterate_;

    const int num_of_levels = static_cast<int>(surfs.size());
    float level_cell_size = cell_size;
    for (int i = 0; i < num_of_levels - 1; i++) {
        std::shared_ptr<GraphicsVolume> fine_volume = surfs[i];
        std::shared_ptr<GraphicsVolume> coarse_volume = surfs[i + 1];

        if (i || as_precondition)
            core_->RelaxWithZeroGuessPacked(*fine_volume, level_cell_size);
        else
            RelaxPacked(fine_volume, level_cell_size, 2);

        RelaxPacked(fine_volume, level_cell_size, times_to_iterate - 2);
        core_->ComputeResidualPacked(*fine_volume, level_cell_size);
        core_->RestrictResidualPacked(*fine_volume, *coarse_volume);

        times_to_iterate *= 2;
        level_cell_size /= 1.0f; // Reducing the h every level will give us
                                 // worse result of |r|. Need digging.
    }

    std::shared_ptr<GraphicsVolume> coarsest = surfs[num_of_levels - 1];
    core_->RelaxWithZeroGuessPacked(*coarsest, level_cell_size);
    RelaxPacked(coarsest, level_cell_size, times_to_iterate - 2);

    for (int j = num_of_levels - 2; j >= 0; j--) {
        std::shared_ptr<GraphicsVolume> coarse_volume = surfs[j + 1];
        std::shared_ptr<GraphicsVolume> fine_volume = surfs[j];

        times_to_iterate /= 2;
        level_cell_size *= 1.0f;

        core_->ProlongatePacked(*coarse_volume, *fine_volume);
        RelaxPacked(fine_volume, level_cell_size, times_to_iterate/* - 1*/);
    }
}

void MultigridPoissonSolver::Diagnose(GraphicsVolume* packed)
{
    extern int g_diagnosis;
    if (g_diagnosis) {
        if (!diagnosis_volume_ ||
                diagnosis_volume_->GetWidth() != packed->GetWidth() ||
                diagnosis_volume_->GetHeight() != packed->GetHeight() ||
                diagnosis_volume_->GetDepth() != packed->GetDepth()) {
            
            diagnosis_volume_ = core_->CreateVolume(packed->GetWidth(),
                                                    packed->GetHeight(),
                                                    packed->GetDepth(), 4, 4);
        }

        //ComputeResidualPackedDiagnosis(*packed, *diagnosis_volume_, CellSize);
        glFinish();
        GraphicsVolume* p = packed;// diagnosis_volume_.get();

        int w = p->GetWidth();
        int h = p->GetHeight();
        int d = p->GetDepth();
        int n = 4;
        int element_size = sizeof(float);
        GLenum format = GL_RGBA;

        static char* v = nullptr;
        if (!v)
            v = new char[w * h * d * element_size * n];

        memset(v, 0, w * h * d * element_size * n);
        p->gl_texture()->GetTexImage(format, GL_FLOAT, v);
        
        float* f = (float*)v;
        double sum = 0.0;
        double q = 0.0f;
        double m = 0.0f;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < h; j++) {
                for (int k = 0; k < w; k++) {
                    for (int l = 0; l < n; l++) {
                        q = f[i * w * h * n + j * w * n + k * n + l];
                        if (i == 30 && j == 0 && k == 56)
                        //if (q > 1)
                        sum += q;
                        m = std::max(q, m);
                    }
                }
            }
        }

        // =========================================================================
        //CudaMain::Instance()->Absolute(diagnosis_volume_);
        // =========================================================================

        double avg = sum / (w * h * d);
        PrintDebugString("sum: %.8f\n", sum);
        //PrintDebugString("avg ||r||: %.8f,    max ||r||: %.8f\n", avg, m);
    }
}