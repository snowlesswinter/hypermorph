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
                            int byte_width, int minimum_grid_width) = 0;
    virtual void Solve(std::shared_ptr<GraphicsVolume> u,
                       std::shared_ptr<GraphicsVolume> b, float cell_size,
                       int iteration_times) = 0;
};

#endif // _POISSON_SOLVER_H_