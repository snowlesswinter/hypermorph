#ifndef _MULTIGRID_POISSON_SOLVER_H_
#define _MULTIGRID_POISSON_SOLVER_H_

#include <memory>
#include <vector>

#include "poisson_solver.h"

class GraphicsVolume;
class MultigridCore;
class MultigridPoissonSolver : public PoissonSolver
{
public:
    explicit MultigridPoissonSolver(MultigridCore* core);
    virtual ~MultigridPoissonSolver();

    virtual bool Initialize(int width, int height, int depth) override;
    virtual void Solve(std::shared_ptr<GraphicsVolume> u_and_b, float cell_size,
                       bool as_precondition) override;

    void SetBaseRelaxationTimes(int base_times);

    // TODO
    void Diagnose(GraphicsVolume* packed);

private:
    typedef std::pair<std::shared_ptr<GraphicsVolume>,
        std::shared_ptr<GraphicsVolume>> VolumePair;

    void RelaxPacked(std::shared_ptr<GraphicsVolume> u_and_b, float cell_size,
                     int times);
    void SolveOpt(std::shared_ptr<GraphicsVolume> u_and_b, float cell_size,
                  bool as_precondition);
    bool ValidateVolume(std::shared_ptr<GraphicsVolume> u_and_b);

    MultigridCore* core_;
    std::vector<VolumePair> volume_resource;
    std::shared_ptr<GraphicsVolume> residual_volume_;
    int times_to_iterate_;
    bool diagnosis_;

    // For diagnosis.
    std::shared_ptr<GraphicsVolume> diagnosis_volume_;
};

#endif // _MULTIGRID_POISSON_SOLVER_H_