#ifndef _FULL_MULTIGRID_POISSON_SOLVER_H_
#define _FULL_MULTIGRID_POISSON_SOLVER_H_

#include <memory>
#include <vector>

#include "poisson_solver.h"

class GLProgram;
class GLTexture;
class MultigridPoissonSolver;
class FullMultigridPoissonSolver : public PoissonSolver
{
public:
    FullMultigridPoissonSolver();
    virtual ~FullMultigridPoissonSolver();

    virtual void Initialize(int width, int height, int depth) override;
    virtual void Solve(const SurfacePod& u_and_b, float cell_size,
                       bool as_precondition,
                       std::shared_ptr<GLTexture> t) override;

private:
    void Restrict(const SurfacePod& fine, const SurfacePod& coarse);

    std::unique_ptr<MultigridPoissonSolver> solver_;
    std::vector<SurfacePod> packed_surfaces_;
    std::vector<std::shared_ptr<GLTexture>> packed_textures_;
    std::unique_ptr<GLProgram> restrict_packed_program_;
};

#endif // _FULL_MULTIGRID_POISSON_SOLVER_H_