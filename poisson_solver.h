#ifndef _POISSON_SOLVER_H_
#define _POISSON_SOLVER_H_

#include <memory>

class GLTexture;
class PoissonSolver
{
public:
    PoissonSolver();
    virtual ~PoissonSolver();

    virtual void Initialize(int width, int height, int depth) = 0;
    virtual void Solve(std::shared_ptr<GLTexture> u_and_b, float cell_size,
                       bool as_precondition) = 0;
};

#endif // _POISSON_SOLVER_H_