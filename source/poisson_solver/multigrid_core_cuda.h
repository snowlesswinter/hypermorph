#ifndef _MULTIGRID_CORE_CUDA_H_
#define _MULTIGRID_CORE_CUDA_H_

#include <memory>

#include "multigrid_core.h"

class GLProgram;
class MultigridCoreCuda : public MultigridCore
{
public:
    MultigridCoreCuda();
    virtual ~MultigridCoreCuda();

    virtual std::shared_ptr<GraphicsVolume> CreateVolume(
        int width, int height, int depth, int num_of_components,
        int byte_width) override;

    virtual void ComputeResidual(const GraphicsVolume& packed,
                                 const GraphicsVolume& residual,
                                 float cell_size) override;
    virtual void ProlongatePacked(const GraphicsVolume& coarse,
                                  const GraphicsVolume& fine) override;
    virtual void RelaxPacked(const GraphicsVolume& u_and_b, float cell_size);
    virtual void RelaxWithZeroGuessAndComputeResidual(
        const GraphicsVolume& packed_volumes, float cell_size,
        int times) override;
    virtual void RelaxWithZeroGuessPacked(const GraphicsVolume& packed,
                                          float cell_size) override;
    virtual void RestrictPacked(const GraphicsVolume& fine,
                                const GraphicsVolume& coarse) override;
    virtual void RestrictResidualPacked(const GraphicsVolume& fine,
                                        const GraphicsVolume& coarse) override;
};

#endif // _MULTIGRID_CORE_CUDA_H_