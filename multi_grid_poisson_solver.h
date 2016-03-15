#ifndef _MULTI_GRID_POISSON_SOLVER_H_
#define _MULTI_GRID_POISSON_SOLVER_H_

#include <memory>
#include <vector>

#include "poisson_solver.h"

class GLProgram;
class MultiGridPoissonSolver : public PoissonSolver
{
public:
    MultiGridPoissonSolver();
    virtual ~MultiGridPoissonSolver();

    virtual void Initialize(int grid_width) override;
    virtual void Solve(const SurfacePod& pressure,
                       const SurfacePod& divergence,
                       bool as_precondition) override;

private:
    typedef std::vector<std::tuple<SurfacePod, SurfacePod, SurfacePod>>
        MultiGridSurfaces;
    typedef MultiGridSurfaces::value_type Surface;

    void ComputeResidual(const SurfacePod& pressure,
                         const SurfacePod& divergence,
                         const SurfacePod& residual, float cell_size);
    void Prolongate(const SurfacePod& coarse_solution,
                    const SurfacePod& fine_solution);
    void Relax(const SurfacePod& u, const SurfacePod& b, float cell_size,
               int times);
    void Restrict(const SurfacePod& source, const SurfacePod& dest);

    std::unique_ptr<MultiGridSurfaces> multi_grid_surfaces_;
    std::unique_ptr<SurfacePod> restricted_residual_;
    std::unique_ptr<SurfacePod> temp_surface_; // TODO
    std::unique_ptr<GLProgram> residual_program_;
    std::unique_ptr<GLProgram> restrict_program_;
    std::unique_ptr<GLProgram> prolongate_program_;

    std::unique_ptr<SurfacePod> diagnosis_; // TODO
};

#endif // _MULTI_GRID_POISSON_SOLVER_H_