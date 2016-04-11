#ifndef _FULL_MULTIGRID_POISSON_SOLVER_H_
#define _FULL_MULTIGRID_POISSON_SOLVER_H_

#include <memory>
#include <vector>

#include "poisson_solver.h"

class MultigridCore;
class MultigridPoissonSolver;
class FullMultigridPoissonSolver : public PoissonSolver
{
public:
    explicit FullMultigridPoissonSolver(MultigridCore* core);
    virtual ~FullMultigridPoissonSolver();

    virtual bool Initialize(int width, int height, int depth,
                            int byte_width) override;
    virtual void Solve(std::shared_ptr<GraphicsVolume> u_and_b, float cell_size,
                       bool as_precondition) override;

private:
    void RelaxPacked(std::shared_ptr<GraphicsVolume> u_and_b, float cell_size,
                     int times);

    MultigridCore* core_;
    std::unique_ptr<MultigridPoissonSolver> solver_;
    std::vector<std::shared_ptr<GraphicsVolume>> packed_textures_;
};

#endif // _FULL_MULTIGRID_POISSON_SOLVER_H_