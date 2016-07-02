//
// Fluid3d - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Fluid3d license (* see part 1 below)
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
#include "multigrid_poisson_solver.h"

#include <cassert>
#include <tuple>

#include "graphics_volume.h"
#include "graphics_volume_group.h"
#include "metrics.h"
#include "poisson_core.h"
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

MultigridPoissonSolver::MultigridPoissonSolver(PoissonCore* core)
    : core_(core)
    , volume_resource_()
    , residual_volume_()
    , num_finest_level_iteration_per_pass_(2)
    , diagnosis_(false)
    , diagnosis_volume_()
{
}

MultigridPoissonSolver::~MultigridPoissonSolver()
{

}

bool MultigridPoissonSolver::Initialize(int width, int height, int depth,
                                        int byte_width, int minimum_grid_width)
{
    volume_resource_.clear();
    residual_volume_ = core_->CreateVolume(width, height, depth, 1, byte_width);

    int min_width = std::min(std::min(width, height), depth);
    int scale = 2;
    while (min_width / scale > minimum_grid_width - 1) {
        int w = width / scale;
        int h = height / scale;
        int d = depth / scale;

        std::shared_ptr<GraphicsVolume3> g = core_->CreateVolumeGroup(
            w, h, d, 1, byte_width);
        if (!g)
            return false;

        volume_resource_.push_back(g);

        scale <<= 1;
    }

    return true;
}

void MultigridPoissonSolver::SetAuxiliaryVolumes(
    const std::vector<std::shared_ptr<GraphicsVolume>>& volumes)
{

}

void MultigridPoissonSolver::SetDiagnosis(bool diagnosis)
{

}

void MultigridPoissonSolver::SetNestedSolverIterations(int num_iterations)
{

}

void MultigridPoissonSolver::Solve(std::shared_ptr<GraphicsVolume> u,
                                   std::shared_ptr<GraphicsVolume> b,
                                   int iteration_times)
{
    if (!ValidateVolume(u) || !ValidateVolume(b))
        return;

    for (int i = 0; i < iteration_times; i++)
        Iterate(u, b, !i);
}

bool MultigridPoissonSolver::ValidateVolume(
    std::shared_ptr<GraphicsVolume> v)
{
    if (volume_resource_.empty())
        return false;

    if (v->GetWidth() > volume_resource_[0]->x()->GetWidth() * 2 ||
            v->GetHeight() > volume_resource_[0]->x()->GetHeight() * 2 ||
            v->GetDepth() > volume_resource_[0]->x()->GetDepth() * 2)
        return false;

    return true;
}

void MultigridPoissonSolver::Iterate(std::shared_ptr<GraphicsVolume> u,
                                     std::shared_ptr<GraphicsVolume> b,
                                     bool apply_initial_guess)
{
    auto i = volume_resource_.begin();
    auto prev = i;
    for (; i != volume_resource_.end(); ++i) {
        if ((*i)->x()->GetWidth() * 2 == u->GetWidth())
            break;

        prev = i;
    }

    assert(i != volume_resource_.end());
    if (i == volume_resource_.end())
        return;

    std::shared_ptr<GraphicsVolume> residual_volume =
        i == prev ? residual_volume_ : (*prev)->z();
    std::vector<std::shared_ptr<GraphicsVolume3>> volumes(
        1, std::make_shared<GraphicsVolume3>(u, b, residual_volume));

    volumes.insert(volumes.end(), i, volume_resource_.end());

    int times_to_iterate = num_finest_level_iteration_per_pass_;

    const int num_of_levels = static_cast<int>(volumes.size());
    for (int i = 0; i < num_of_levels - 1; i++) {
        std::shared_ptr<GraphicsVolume3> fine_volumes = volumes[i];
        std::shared_ptr<GraphicsVolume> coarse_volume = volumes[i + 1]->y();

        if (i || apply_initial_guess)
            core_->RelaxWithZeroGuess(*fine_volumes->x(), *fine_volumes->y());
        else
            core_->Relax(*fine_volumes->x(), *fine_volumes->y(), 2);

        core_->Relax(*fine_volumes->x(), *fine_volumes->y(),
                     times_to_iterate - 2);
        core_->ComputeResidual(*fine_volumes->z(), *fine_volumes->x(),
                               *fine_volumes->y());
        core_->Restrict(*coarse_volume, *fine_volumes->z());

        times_to_iterate *= 2;
    }

    std::shared_ptr<GraphicsVolume3> coarsest = volumes[num_of_levels - 1];
    core_->RelaxWithZeroGuess(*coarsest->x(), *coarsest->y());
    core_->Relax(*coarsest->x(), *coarsest->y(), times_to_iterate - 2 + 30);

    for (int j = num_of_levels - 2; j >= 0; j--) {
        std::shared_ptr<GraphicsVolume> coarse_volume = volumes[j + 1]->x();
        std::shared_ptr<GraphicsVolume3> fine_volume = volumes[j];

        times_to_iterate /= 2;

        core_->ProlongateError(*fine_volume->x(), *coarse_volume);
        core_->Relax(*fine_volume->x(), *fine_volume->y(), times_to_iterate);
    }
}
