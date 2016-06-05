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
    virtual std::shared_ptr<GraphicsVolume3> CreateVolumeGroup(
        int width, int height, int depth, int num_of_components,
        int byte_width) override;

    virtual void ComputeResidual(const GraphicsVolume& packed,
                                 const GraphicsVolume& residual,
                                 float cell_size) override;
    virtual void ComputeResidual(const GraphicsVolume& r,
                                 const GraphicsVolume& u,
                                 const GraphicsVolume& b,
                                 float cell_size) override;
    virtual void Prolongate(const GraphicsVolume& fine,
                            const GraphicsVolume& coarse) override;
    virtual void ProlongatePacked(const GraphicsVolume& coarse,
                                  const GraphicsVolume& fine) override;
    virtual void ProlongateResidual(const GraphicsVolume& fine,
                                    const GraphicsVolume& coarse) override;
    virtual void ProlongateResidualPacked(const GraphicsVolume& coarse,
                                          const GraphicsVolume& fine) override;
    virtual void Relax(const GraphicsVolume& u, const GraphicsVolume& b,
                       float cell_size, int num_of_iterations) override;
    virtual void RelaxWithZeroGuessAndComputeResidual(
        const GraphicsVolume& packed_volumes, float cell_size,
        int times) override;
    virtual void RelaxWithZeroGuess(const GraphicsVolume& u,
                                    const GraphicsVolume& b,
                                    float cell_size) override;
    virtual void RelaxWithZeroGuessPacked(const GraphicsVolume& packed,
                                          float cell_size) override;
    virtual void Restrict(const GraphicsVolume& coarse,
                          const GraphicsVolume& fine) override;
    virtual void RestrictPacked(const GraphicsVolume& fine,
                                const GraphicsVolume& coarse) override;
    virtual void RestrictResidualPacked(const GraphicsVolume& fine,
                                        const GraphicsVolume& coarse) override;
};

#endif // _MULTIGRID_CORE_CUDA_H_