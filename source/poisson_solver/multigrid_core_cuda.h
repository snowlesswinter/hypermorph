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

    virtual std::shared_ptr<GraphicsMemPiece> CreateMemPiece(int size) override;
    virtual std::shared_ptr<GraphicsVolume> CreateVolume(
        int width, int height, int depth, int num_of_components,
        int byte_width) override;
    virtual std::shared_ptr<GraphicsVolume3> CreateVolumeGroup(
        int width, int height, int depth, int num_of_components,
        int byte_width) override;

    // Multigrid.
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

    // Conjugate gradient.
    virtual void ApplyStencil(const GraphicsVolume& aux,
                              const GraphicsVolume& search,
                              float cell_size) override;
    virtual void ComputeAlpha(const GraphicsMemPiece& alpha,
                              const GraphicsMemPiece& rho,
                              const GraphicsVolume& aux,
                              const GraphicsVolume& search) override;
    virtual void ComputeRhoAndBeta(const GraphicsMemPiece& beta,
                                   const GraphicsMemPiece& rho_new,
                                   const GraphicsMemPiece& rho,
                                   const GraphicsVolume& aux,
                                   const GraphicsVolume& residual) override;
    virtual void ComputeRho(const GraphicsMemPiece& rho,
                            const GraphicsVolume& search,
                            const GraphicsVolume& residual) override;
    virtual void UpdateVector(const GraphicsVolume& dest,
                              const GraphicsVolume& v0,
                              const GraphicsVolume& v1,
                              const GraphicsMemPiece& coef,
                              float sign) override;
};

#endif // _MULTIGRID_CORE_CUDA_H_