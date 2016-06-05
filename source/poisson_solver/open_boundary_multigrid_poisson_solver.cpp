#include "stdafx.h"
#include "open_boundary_multigrid_poisson_solver.h"

#include <cassert>
#include <tuple>

#include "graphics_volume.h"
#include "multigrid_core.h"

OpenBoundaryMultigridPoissonSolver::OpenBoundaryMultigridPoissonSolver(
        MultigridCore* core)
    : core_(core)
    , volume_resource_()
    , residual_volume_()
    , num_finest_level_iteration_per_pass_(2)
{
}

OpenBoundaryMultigridPoissonSolver::~OpenBoundaryMultigridPoissonSolver()
{

}

bool OpenBoundaryMultigridPoissonSolver::Initialize(int width, int height,
                                                    int depth,
                                                    int byte_width)
{
    volume_resource_.clear();
    residual_volume_ = core_->CreateVolume(width, height, depth, 1, byte_width);

    int min_width = std::min(std::min(width, height), depth);
    int scale = 2;
    while (min_width / scale > 8) {
        int w = width / scale;
        int h = height / scale;
        int d = depth / scale;

        std::shared_ptr<GraphicsVolume3> v = core_->CreateVolumeGroup(
            w, h, d, 1, byte_width);
        if (!v)
            return false;

        volume_resource_.push_back(v);

        scale <<= 1;
    }

    return true;
}

void OpenBoundaryMultigridPoissonSolver::Solve(
    std::shared_ptr<GraphicsVolume> u, std::shared_ptr<GraphicsVolume> b,
    float cell_size)
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

        if (i)
            core_->RelaxWithZeroGuess(*fine_volumes->x(), *fine_volumes->y(),
                                      cell_size);
        else
            Relax(fine_volumes->x(), fine_volumes->y(), cell_size, 2);

        Relax(fine_volumes->x(), fine_volumes->y(), cell_size,
              times_to_iterate - 2);
        core_->ComputeResidual(*fine_volumes->z(), *fine_volumes->x(),
                               *fine_volumes->y(), cell_size);
        core_->Restrict(*coarse_volume, *fine_volumes->z());

        times_to_iterate *= 2;
    }

    std::shared_ptr<GraphicsVolume3> coarsest = volumes[num_of_levels - 1];
    core_->RelaxWithZeroGuess(*coarsest->x(), *coarsest->y(), cell_size);
    Relax(coarsest->x(), coarsest->y(), cell_size, times_to_iterate - 2);

    for (int j = num_of_levels - 2; j >= 0; j--) {
        std::shared_ptr<GraphicsVolume> coarse_volume = volumes[j + 1]->x();
        std::shared_ptr<GraphicsVolume3> fine_volume = volumes[j];

        times_to_iterate /= 2;

        core_->ProlongateResidual(*fine_volume->x(), *coarse_volume);
        Relax(fine_volume->x(), fine_volume->y(), cell_size, times_to_iterate);
    }
}

void OpenBoundaryMultigridPoissonSolver::Relax(
    std::shared_ptr<GraphicsVolume> u, std::shared_ptr<GraphicsVolume> b,
    float cell_size, int times)
{
    core_->Relax(*u, *b, cell_size, times);
}