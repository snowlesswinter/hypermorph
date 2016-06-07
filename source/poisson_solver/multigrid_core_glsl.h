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
    virtual std::shared_ptr<GraphicsVolume3> CreateVolumeGroup(
        int width, int height, int depth, int num_of_components,
        int byte_width) override;

    virtual void ComputeResidual(const GraphicsVolume& r,
                                 const GraphicsVolume& u,
                                 const GraphicsVolume& b,
                                 float cell_size) override;
    virtual void Prolongate(const GraphicsVolume& fine,
                            const GraphicsVolume& coarse) override;

    virtual void ProlongateError(const GraphicsVolume& fine,
                                 const GraphicsVolume& coarse) override;
    virtual void Relax(const GraphicsVolume& u, const GraphicsVolume& b,
                       float cell_size, int num_of_iterations) override;
    virtual void RelaxWithZeroGuess(const GraphicsVolume& u,
                                    const GraphicsVolume& b,
                                    float cell_size) override;
    virtual void Restrict(const GraphicsVolume& coarse,
                          const GraphicsVolume& fine) override;

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