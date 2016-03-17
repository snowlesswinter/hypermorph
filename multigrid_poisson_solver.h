#ifndef _MULTIGRID_POISSON_SOLVER_H_
#define _MULTIGRID_POISSON_SOLVER_H_

#include <memory>
#include <vector>

#include "poisson_solver.h"

class GLProgram;
class MultigridPoissonSolver : public PoissonSolver
{
public:
    MultigridPoissonSolver();
    virtual ~MultigridPoissonSolver();

    virtual void Initialize(int grid_width) override;
    virtual void Solve(const SurfacePod& packed,
                       bool as_precondition) override;

private:
    typedef std::vector<std::tuple<SurfacePod, SurfacePod, SurfacePod>>
        MultiGridSurfaces;
    typedef MultiGridSurfaces::value_type Surface;

    void ComputeResidual(const SurfacePod& u, const SurfacePod& b,
                         const SurfacePod& residual, float cell_size,
                         bool diagnosis);
    void Prolongate(const SurfacePod& coarse_solution,
                    const SurfacePod& fine_solution);
    void Relax(const SurfacePod& u, const SurfacePod& b, float cell_size,
               int times);
    void RelaxWithZeroGuess(const SurfacePod& u, const SurfacePod& b,
                            float cell_size);
    void Restrict(const SurfacePod& fine, const SurfacePod& coarse);
    void SolvePlain(const SurfacePod& packed, bool as_precondition);

    // Optimization.
    void ComputeResidualPacked(const SurfacePod& packed, float cell_size);
    void ProlongateAndRelax(const SurfacePod& coarse, const SurfacePod& fine);
    void ProlongatePacked(const SurfacePod& coarse, const SurfacePod& fine);
    void RelaxPacked(const SurfacePod& u_and_b, float cell_size, int times);
    void RelaxPackedImpl(const SurfacePod& u_and_b, float cell_size);
    void RelaxWithZeroGuessAndComputeResidual(const SurfacePod& packed_volumes,
                                              float cell_size, int times);
    void RelaxWithZeroGuessPacked(const SurfacePod& packed_volumes,
                                  float cell_size);
    void RestrictPacked(const SurfacePod& fine, const SurfacePod& coarse);
    void SolveOpt(const SurfacePod& u_and_b, bool as_precondition);

    // For diagnosis.
    void ComputeResidualPackedDiagnosis(const SurfacePod& packed,
                                        const SurfacePod& diagnosis,
                                        float cell_size);

    std::unique_ptr<MultiGridSurfaces> multi_grid_surfaces_;
    std::vector<SurfacePod> packed_surfaces_;
    std::unique_ptr<SurfacePod> temp_surface_; // TODO
    std::unique_ptr<GLProgram> residual_program_;
    std::unique_ptr<GLProgram> restrict_program_;
    std::unique_ptr<GLProgram> prolongate_program_;
    std::unique_ptr<GLProgram> relax_zero_guess_program_;
    bool diagnosis_;

    // Optimization.
    std::unique_ptr<GLProgram> prolongate_and_relax_program_;
    std::unique_ptr<GLProgram> prolongate_packed_program_;
    std::unique_ptr<GLProgram> relax_packed_program_;
    std::unique_ptr<GLProgram> relax_zero_guess_packed_program_;
    std::unique_ptr<GLProgram> residual_packed_program_;
    std::unique_ptr<GLProgram> restrict_packed_program_;

    // For diagnosis.
    std::unique_ptr<GLProgram> absolute_program_; 
    std::unique_ptr<GLProgram> residual_diagnosis_program_;
    std::unique_ptr<SurfacePod> diagnosis_volume_;
};

#endif // _MULTIGRID_POISSON_SOLVER_H_