#ifndef _FULL_MULTIGRID_POISSON_SOLVER_H_
#define _FULL_MULTIGRID_POISSON_SOLVER_H_

#include <memory>
#include <vector>

#include "poisson_solver.h"

class MultigridPoissonSolver;
class PoissonCore;
class FullMultigridPoissonSolver : public PoissonSolver
{
public:
    explicit FullMultigridPoissonSolver(PoissonCore* core);
    virtual ~FullMultigridPoissonSolver();

    virtual bool Initialize(int width, int height, int depth,
                            int byte_width, int minimum_grid_width) override;
    virtual void SetAuxiliaryVolumes(
        const std::vector<std::shared_ptr<GraphicsVolume>>& volumes) override;
    virtual void SetDiagnosis(bool diagnosis) override;
    virtual void SetNestedSolverIterations(int num_iterations) override;
    virtual void Solve(std::shared_ptr<GraphicsVolume> u,
                       std::shared_ptr<GraphicsVolume> b, float cell_size,
                       int iteration_times) override;

private:
    typedef std::pair<std::shared_ptr<GraphicsVolume>,
        std::shared_ptr<GraphicsVolume>> VolumePair;

    void Iterate(std::shared_ptr<GraphicsVolume> u,
                 std::shared_ptr<GraphicsVolume> b, float cell_size,
                 bool apply_initial_guess);

    PoissonCore* core_;
    std::unique_ptr<MultigridPoissonSolver> solver_;
    std::vector<VolumePair> volume_resource_;
    int num_nested_iterations_;
};

#endif // _FULL_MULTIGRID_POISSON_SOLVER_H_