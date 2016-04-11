#include "stdafx.h"
#include "multigrid_poisson_solver.h"

#include <cassert>
#include <tuple>

#include "graphics_volume.h"
#include "metrics.h"
#include "multigrid_core.h"
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
    , residual_volume_()
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
    residual_volume_ = core_->CreateVolume(width, height, depth, 1, 2);

    int min_width = std::min(std::min(width, height), depth);
    int scale = 2;
    while (min_width / scale > 16) {
        int w = width / scale;
        int h = height / scale;
        int d = depth / scale;
        std::shared_ptr<GraphicsVolume> v0 = core_->CreateVolume(w, h, d, 2, 2);
        if (!v0)
            return false;

        std::shared_ptr<GraphicsVolume> v1 = core_->CreateVolume(w, h, d, 1, 2);
        if (!v1)
            return false;

        volume_resource.push_back(std::make_pair(v0, v1));

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

    if (u_and_b->GetWidth() > volume_resource[0].first->GetWidth() * 2 ||
            u_and_b->GetHeight() > volume_resource[0].first->GetHeight() * 2 ||
            u_and_b->GetDepth() > volume_resource[0].first->GetDepth() * 2)
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
    auto i = volume_resource.begin();
    auto prev = i;
    for (; i != volume_resource.end(); ++i) {
        if (i->first->GetWidth() * 2 == u_and_b->GetWidth())
            break;

        prev = i;
    }

    assert(i != volume_resource.end());
    if (i == volume_resource.end())
        return;

    std::shared_ptr<GraphicsVolume> residual_volume =
        i == prev ? residual_volume_ : prev->second;
    std::vector<VolumePair> surfs(1, std::make_pair(u_and_b, residual_volume));
    surfs.insert(surfs.end(), i, volume_resource.end());

    int times_to_iterate = times_to_iterate_;

    const int num_of_levels = static_cast<int>(surfs.size());
    float level_cell_size = cell_size;
    for (int i = 0; i < num_of_levels - 1; i++) {
        VolumePair fine_volumes = surfs[i];
        std::shared_ptr<GraphicsVolume> coarse_volume = surfs[i + 1].first;

        if (i || as_precondition)
            core_->RelaxWithZeroGuessPacked(*fine_volumes.first,
                                            level_cell_size);
        else
            RelaxPacked(fine_volumes.first, level_cell_size, 2);

        RelaxPacked(fine_volumes.first, level_cell_size, times_to_iterate - 2);
        core_->ComputeResidual(*fine_volumes.first, *fine_volumes.second,
                               level_cell_size);
        core_->RestrictResidualPacked(*fine_volumes.second, *coarse_volume);

        times_to_iterate *= 2;
        level_cell_size /= 1.0f; // Reducing the h every level will give us
                                 // worse result of |r|. Need digging.
    }

    std::shared_ptr<GraphicsVolume> coarsest = surfs[num_of_levels - 1].first;
    core_->RelaxWithZeroGuessPacked(*coarsest, level_cell_size);
    RelaxPacked(coarsest, level_cell_size, times_to_iterate - 2);

    for (int j = num_of_levels - 2; j >= 0; j--) {
        std::shared_ptr<GraphicsVolume> coarse_volume = surfs[j + 1].first;
        std::shared_ptr<GraphicsVolume> fine_volume = surfs[j].first;

        times_to_iterate /= 2;
        level_cell_size *= 1.0f;

        core_->ProlongateResidualPacked(*coarse_volume, *fine_volume);
        RelaxPacked(fine_volume, level_cell_size, times_to_iterate/* - 1*/);
    }
}

void MultigridPoissonSolver::Diagnose(GraphicsVolume* packed)
{
//     extern int g_diagnosis;
//     if (g_diagnosis) {
//         if (!diagnosis_volume_ ||
//                 diagnosis_volume_->GetWidth() != packed->GetWidth() ||
//                 diagnosis_volume_->GetHeight() != packed->GetHeight() ||
//                 diagnosis_volume_->GetDepth() != packed->GetDepth()) {
//             
//             diagnosis_volume_ = core_->CreateVolume(packed->GetWidth(),
//                                                     packed->GetHeight(),
//                                                     packed->GetDepth(), 4, 4);
//         }
// 
//         //ComputeResidualPackedDiagnosis(*packed, *diagnosis_volume_, CellSize);
//         glFinish();
//         GraphicsVolume* p = packed;// diagnosis_volume_.get();
// 
//         int w = p->GetWidth();
//         int h = p->GetHeight();
//         int d = p->GetDepth();
//         int n = 4;
//         int element_size = sizeof(float);
//         GLenum format = GL_RGBA;
// 
//         static char* v = nullptr;
//         if (!v)
//             v = new char[w * h * d * element_size * n];
// 
//         memset(v, 0, w * h * d * element_size * n);
//         p->gl_texture()->GetTexImage(format, GL_FLOAT, v);
//         
//         float* f = (float*)v;
//         double sum = 0.0;
//         double q = 0.0f;
//         double m = 0.0f;
//         for (int i = 0; i < d; i++) {
//             for (int j = 0; j < h; j++) {
//                 for (int k = 0; k < w; k++) {
//                     for (int l = 0; l < n; l++) {
//                         q = f[i * w * h * n + j * w * n + k * n + l];
//                         if (i == 30 && j == 0 && k == 56)
//                         //if (q > 1)
//                         sum += q;
//                         m = std::max(q, m);
//                     }
//                 }
//             }
//         }
// 
//         double avg = sum / (w * h * d);
//         PrintDebugString("sum: %.8f\n", sum);
//         //PrintDebugString("avg ||r||: %.8f,    max ||r||: %.8f\n", avg, m);
//     }
}