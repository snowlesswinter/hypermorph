#ifndef _POISSON_SOLVER_H_
#define _POISSON_SOLVER_H_

#include <memory>

class GraphicsVolume;
class PoissonSolver
{
public:
    PoissonSolver();
    virtual ~PoissonSolver();

    virtual bool Initialize(int width, int height, int depth,
                            int byte_width) = 0;
    virtual void Solve(std::shared_ptr<GraphicsVolume> u_and_b, float cell_size,
                       bool as_precondition) = 0;
};

#endif // _POISSON_SOLVER_H_