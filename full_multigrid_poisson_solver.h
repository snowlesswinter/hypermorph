#ifndef _FULL_MULTIGRID_POISSON_SOLVER_H_
#define _FULL_MULTIGRID_POISSON_SOLVER_H_

#include <memory>
#include <vector>

#include "poisson_solver.h"

class GLProgram;
class MultigridPoissonSolver;
class FullMultigridPoissonSolver : public PoissonSolver
{
public:
    FullMultigridPoissonSolver();
    virtual ~FullMultigridPoissonSolver();

    virtual void Initialize(int width, int height, int depth) override;
    virtual void Solve(const SurfacePod& u_and_b,
                       bool as_precondition) override;

private:
    void Restrict(const SurfacePod& fine, const SurfacePod& coarse);

    std::unique_ptr<MultigridPoissonSolver> solver_;
    std::vector<SurfacePod> packed_surfaces_;
    std::unique_ptr<GLProgram> restrict_packed_program_;
};

#endif // _FULL_MULTIGRID_POISSON_SOLVER_H_