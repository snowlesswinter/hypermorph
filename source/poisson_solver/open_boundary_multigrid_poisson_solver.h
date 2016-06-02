#ifndef _OPEN_BOUNDARY_MULTIGRID_POISSON_SOLVER_H_
#define _OPEN_BOUNDARY_MULTIGRID_POISSON_SOLVER_H_

#include <memory>
#include <vector>

#include "graphics_volume_group.h"

class MultigridCore;
class OpenBoundaryMultigridPoissonSolver
{
public:
    explicit OpenBoundaryMultigridPoissonSolver(MultigridCore* core);
    virtual ~OpenBoundaryMultigridPoissonSolver();

    bool Initialize(int width, int height, int depth,
                    int byte_width);
    void Solve(std::shared_ptr<GraphicsVolume> u,
               std::shared_ptr<GraphicsVolume> b, float cell_size);

    void set_num_finest_level_iteration_per_pass(int n) {
        num_finest_level_iteration_per_pass_ = n;
    }

private:
    void Relax(std::shared_ptr<GraphicsVolume> u,
               std::shared_ptr<GraphicsVolume> b, float cell_size, int times);

    MultigridCore* core_;
    std::vector<std::shared_ptr<GraphicsVolume3>> volume_resource;
    std::shared_ptr<GraphicsVolume> residual_volume_;
    int num_finest_level_iteration_per_pass_;
};

#endif // _OPEN_BOUNDARY_MULTIGRID_POISSON_SOLVER_H_