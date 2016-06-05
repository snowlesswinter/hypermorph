#ifndef _MULTIGRID_CORE_H_
#define _MULTIGRID_CORE_H_

#include <memory>

class GraphicsVolume;
class GraphicsVolume3;
class MultigridCore
{
public:
    MultigridCore();
    virtual ~MultigridCore();

    virtual std::shared_ptr<GraphicsVolume> CreateVolume(int width, int height,
                                                         int depth,
                                                         int num_of_components,
                                                         int byte_width) = 0;
    virtual std::shared_ptr<GraphicsVolume3> CreateVolumeGroup(
        int width, int height, int depth, int num_of_components,
        int byte_width) = 0;

    virtual void ComputeResidual(const GraphicsVolume& packed,
                                 const GraphicsVolume& residual,
                                 float cell_size) = 0;
    virtual void ComputeResidual(const GraphicsVolume& r,
                                 const GraphicsVolume& u,
                                 const GraphicsVolume& b, float cell_size) = 0;
    virtual void Prolongate(const GraphicsVolume& fine,
                            const GraphicsVolume& coarse) = 0;
    virtual void ProlongatePacked(const GraphicsVolume& coarse,
                                  const GraphicsVolume& fine) = 0;
    virtual void ProlongateResidual(const GraphicsVolume& fine,
                                    const GraphicsVolume& coarse) = 0;
    virtual void ProlongateResidualPacked(const GraphicsVolume& coarse,
                                          const GraphicsVolume& fine) = 0;
    virtual void Relax(const GraphicsVolume& u, const GraphicsVolume& b,
                       float cell_size, int num_of_iterations) = 0;
    virtual void RelaxPacked(const GraphicsVolume& u_and_b, float cell_size,
                             int num_of_iterations) = 0;
    virtual void RelaxWithZeroGuessAndComputeResidual(
        const GraphicsVolume& packed_volumes, float cell_size, int times) = 0;
    virtual void RelaxWithZeroGuess(const GraphicsVolume& u,
                                    const GraphicsVolume& b,
                                    float cell_size) = 0;
    virtual void RelaxWithZeroGuessPacked(const GraphicsVolume& packed,
                                          float cell_size) = 0;
    virtual void Restrict(const GraphicsVolume& coarse,
                          const GraphicsVolume& fine) = 0;
    virtual void RestrictPacked(const GraphicsVolume& fine,
                                const GraphicsVolume& coarse) = 0;
    virtual void RestrictResidualPacked(const GraphicsVolume& fine,
                                        const GraphicsVolume& coarse) = 0;
};

#endif // _MULTIGRID_CORE_H_