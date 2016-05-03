#ifndef _MULTIGRID_CORE_GLSL_H_
#define _MULTIGRID_CORE_GLSL_H_

#include <memory>

#include "multigrid_core.h"

class GLProgram;
class MultigridCoreGlsl : public MultigridCore
{
public:
    MultigridCoreGlsl();
    virtual ~MultigridCoreGlsl();

    virtual std::shared_ptr<GraphicsVolume> CreateVolume(
        int width, int height, int depth, int num_of_components,
        int byte_width) override;

    virtual void ComputeResidual(const GraphicsVolume& packed,
                                 const GraphicsVolume& residual,
                                 float cell_size) override;

    // ProlongateAndRelax() is deprecated. Though it seems to be a bit useful
    // that saving one time texture fetch, it still need to read the texture in
    // prolong-and-add operation. So the advantage of fetch-saving would be
    // trivial if we sum all things up.
    virtual void ProlongatePacked(const GraphicsVolume& coarse,
                                  const GraphicsVolume& fine) override;
    virtual void ProlongateResidualPacked(const GraphicsVolume& coarse,
                                          const GraphicsVolume& fine) override;
    virtual void RelaxPacked(const GraphicsVolume& u_and_b, float cell_size,
                             int num_of_iterations) override;
    virtual void RelaxWithZeroGuessAndComputeResidual(
        const GraphicsVolume& packed_volumes, float cell_size,
        int times) override;
    virtual void RelaxWithZeroGuessPacked(const GraphicsVolume& packed,
                                          float cell_size) override;
    virtual void RestrictPacked(const GraphicsVolume& fine,
                                const GraphicsVolume& coarse) override;
    virtual void RestrictResidualPacked(const GraphicsVolume& fine,
                                        const GraphicsVolume& coarse) override;

private:
    GLProgram* GetProlongatePackedProgram();
    GLProgram* GetRelaxPackedProgram();
    GLProgram* GetRelaxZeroGuessPackedProgram();
    GLProgram* GetResidualPackedProgram();
    GLProgram* GetRestrictPackedProgram();
    GLProgram* GetRestrictResidualPackedProgram();

    // Optimization.
    std::unique_ptr<GLProgram> prolongate_and_relax_program_;
    std::unique_ptr<GLProgram> prolongate_packed_program_;
    std::unique_ptr<GLProgram> relax_packed_program_;
    std::unique_ptr<GLProgram> relax_zero_guess_packed_program_;
    std::unique_ptr<GLProgram> residual_packed_program_;
    std::unique_ptr<GLProgram> restrict_packed_program_;
    std::unique_ptr<GLProgram> restrict_residual_packed_program_;
};

#endif // _MULTIGRID_CORE_GLSL_H_