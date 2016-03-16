#ifndef _POISSON_SOLVER_H_
#define _POISSON_SOLVER_H_

struct SurfacePod;

class PoissonSolver
{
public:
    PoissonSolver();
    virtual ~PoissonSolver();

    virtual void Initialize(int grid_width) = 0;
    virtual void Solve(const SurfacePod& pressure,
                       const SurfacePod& divergence, bool as_precondition) = 0;
};

#endif // _POISSON_SOLVER_H_