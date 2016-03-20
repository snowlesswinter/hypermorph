#ifndef _POISSON_SOLVER_H_
#define _POISSON_SOLVER_H_

struct SurfacePod;

class PoissonSolver
{
public:
    PoissonSolver();
    virtual ~PoissonSolver();

    virtual void Initialize(int width, int height, int depth) = 0;
    virtual void Solve(const SurfacePod& u_and_b, float cell_size,
                       bool as_precondition) = 0;
};

#endif // _POISSON_SOLVER_H_