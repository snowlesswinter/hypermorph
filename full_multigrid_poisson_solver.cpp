#include "stdafx.h"
#include "full_multigrid_poisson_solver.h"

#include <cassert>
#include <algorithm>

#include "cuda_host/cuda_main.h"
#include "graphics_volume.h"
#include "multigrid_core.h"
#include "multigrid_poisson_solver.h"
#include "opengl/gl_program.h"
#include "opengl/gl_texture.h"
#include "shader/fluid_shader.h"
#include "shader/multigrid_shader.h"
#include "shader/multigrid_staggered_shader.h"
#include "utility.h"

const int kWidthOfCoarsestLevel = 32;

FullMultigridPoissonSolver::FullMultigridPoissonSolver(MultigridCore* core)
    : core_(core)
    , solver_(new MultigridPoissonSolver(core))
    , packed_textures_()
{

}

FullMultigridPoissonSolver::~FullMultigridPoissonSolver()
{

}

bool FullMultigridPoissonSolver::Initialize(int width, int height, int depth)
{
    if (!solver_->Initialize(width, height, depth))
        return false;

    // Placeholder for the solution buffer.
    packed_textures_.clear();
    packed_textures_.push_back(std::shared_ptr<GraphicsVolume>());

    int min_width = std::min(std::min(width, height), depth);

    int scale = 2;
    while (min_width / scale > 32) { // TODO: CUDA has some problem with 32^3
                                     //       volumes.
        int w = width / scale;
        int h = height / scale;
        int d = depth / scale;

        std::shared_ptr<GraphicsVolume> v = core_->CreateVolume(w, h, d, 4, 2);
        if (!v)
            return false;

        packed_textures_.push_back(v);

        scale <<= 1;
    }

    return true;
}

void FullMultigridPoissonSolver::Solve(std::shared_ptr<GraphicsVolume> u_and_b,
                                       float cell_size,
                                       bool as_precondition)
{
    if (u_and_b->GetWidth() < 32) {
        solver_->Solve(u_and_b, cell_size, true);
        return;
    }

    assert(packed_textures_.size() > 1);
    if (packed_textures_.size() <= 1)
        return;

    // With less iterations in each level but more iterating in every V-Cycle
    // will out perform the case visa versa(less time cost, lower avg/max |r|),
    // especially in high divergence cases.
    solver_->SetBaseRelaxationTimes(5);
    packed_textures_[0] = u_and_b;

    const int num_of_levels = static_cast<int>(packed_textures_.size());
    float level_cell_size = cell_size;
    for (int i = 0; i < num_of_levels - 1; i++) {
        std::shared_ptr<GraphicsVolume> fine_volume = packed_textures_[i];
        std::shared_ptr<GraphicsVolume> coarse_volume = packed_textures_[i + 1];

        core_->RestrictPacked(*fine_volume, *coarse_volume);

        level_cell_size *= 1.0f;
    }

    std::shared_ptr<GraphicsVolume> coarsest =
        packed_textures_[num_of_levels - 1];
    if (as_precondition)
        core_->RelaxWithZeroGuessPacked(*coarsest, level_cell_size);

    RelaxPacked(coarsest, level_cell_size, 15);

    int times_to_iterate = 1;
    for (int j = num_of_levels - 2; j >= 0; j--) {
        std::shared_ptr<GraphicsVolume> coarse_volume = packed_textures_[j + 1];
        std::shared_ptr<GraphicsVolume> fine_volume = packed_textures_[j];

        core_->ProlongatePacked(*coarse_volume, *fine_volume);

        for (int k = 0; k < times_to_iterate; k++)
            solver_->Solve(fine_volume, level_cell_size, false);

        // For comparison.
        // 
        // Damped Jacobi is still faster than Multigrid, no much though. With
        // a base relaxation times 5, Multigrid had achieved a notable
        // lower avg/max |r| compared to Jacobi in our experiments.

        //RelaxPacked(fine_volume, level_cell_size, 15);

        // Experiments revealed that iterations in different levels almost
        // equally contribute to the final result, thus we are not going to
        // reduce the iteration times in coarsen level.
        times_to_iterate += 0;
        level_cell_size *= 1.0f;
    }

//     if (!as_precondition)
//         core_->Diagnose(t.get());
}

void FullMultigridPoissonSolver::RelaxPacked(
    std::shared_ptr<GraphicsVolume> u_and_b, float cell_size, int times)
{
    for (int i = 0; i < times; i++)
        core_->RelaxPacked(*u_and_b, cell_size);
}
