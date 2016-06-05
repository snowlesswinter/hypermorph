#ifndef _MULTIGRID_POISSON_SOLVER_H_
#define _MULTIGRID_POISSON_SOLVER_H_

#include <memory>
#include <vector>

#include "poisson_solver.h"

class GraphicsVolume;
class GraphicsVolume3;
class MultigridCore;
class MultigridPoissonSolver : public PoissonSolver
{
public:
    explicit MultigridPoissonSolver(MultigridCore* core);
    virtual ~MultigridPoissonSolver();

    virtual bool Initialize(int width, int height, int depth,
                            int byte_width) override;
    virtual void Solve(std::shared_ptr<GraphicsVolume> u,
                       std::shared_ptr<GraphicsVolume> b, float cell_size,
                       bool as_precondition) override;

    void set_num_finest_level_iteration_per_pass(int n) {
        num_finest_level_iteration_per_pass_ = n;
    }

    // TODO
    void Diagnose(GraphicsVolume* packed);

private:
    void SolveOpt(std::shared_ptr<GraphicsVolume> u,
                  std::shared_ptr<GraphicsVolume> b, float cell_size,
                  bool as_precondition);
    bool ValidateVolume(std::shared_ptr<GraphicsVolume> v);

    MultigridCore* core_;
    std::vector<std::shared_ptr<GraphicsVolume3>> volume_resource_;
    std::shared_ptr<GraphicsVolume> residual_volume_;
    int num_finest_level_iteration_per_pass_;
    bool diagnosis_;

    // For diagnosis.
    std::shared_ptr<GraphicsVolume> diagnosis_volume_;
};

#endif // _MULTIGRID_POISSON_SOLVER_H_