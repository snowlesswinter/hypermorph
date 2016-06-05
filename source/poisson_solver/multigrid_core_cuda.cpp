#include "stdafx.h"
#include "multigrid_core_cuda.h"

#include <cassert>

#include "cuda_host/cuda_main.h"
#include "graphics_volume.h"
#include "graphics_volume_group.h"
#include "utility.h"

MultigridCoreCuda::MultigridCoreCuda()
    : MultigridCore()
{

}

MultigridCoreCuda::~MultigridCoreCuda()
{

}

std::shared_ptr<GraphicsVolume> MultigridCoreCuda::CreateVolume(
    int width, int height, int depth, int num_of_components, int byte_width)
{
    std::shared_ptr<GraphicsVolume> r(new GraphicsVolume(GRAPHICS_LIB_CUDA));
    bool succeeded = r->Create(width, height, depth, num_of_components,
                               byte_width);

    return succeeded ? r : std::shared_ptr<GraphicsVolume>();
}

std::shared_ptr<GraphicsVolume3> MultigridCoreCuda::CreateVolumeGroup(
    int width, int height, int depth, int num_of_components, int byte_width)
{
    std::shared_ptr<GraphicsVolume3> r(new GraphicsVolume3(GRAPHICS_LIB_CUDA));
    bool succeeded = r->Create(width, height, depth, num_of_components,
                               byte_width);
    return succeeded ? r : std::shared_ptr<GraphicsVolume3>();
}

void MultigridCoreCuda::ComputeResidual(const GraphicsVolume& packed,
                                        const GraphicsVolume& residual,
                                        float cell_size)
{
    float inverse_h_square = 1.0f / (cell_size * cell_size);
    CudaMain::Instance()->ComputeResidualPacked(residual.cuda_volume(),
                                                packed.cuda_volume(),
                                                inverse_h_square);
    
}

void MultigridCoreCuda::ComputeResidual(const GraphicsVolume& r,
                                        const GraphicsVolume& u,
                                        const GraphicsVolume& b,
                                        float cell_size)
{
    CudaMain::Instance()->ComputeResidual(r.cuda_volume(), u.cuda_volume(),
                                          b.cuda_volume(), cell_size);
}

void MultigridCoreCuda::Prolongate(const GraphicsVolume& fine,
                                   const GraphicsVolume& coarse)
{
    CudaMain::Instance()->Prolongate(fine.cuda_volume(), coarse.cuda_volume());
}

void MultigridCoreCuda::ProlongatePacked(const GraphicsVolume& coarse,
                                         const GraphicsVolume& fine)
{
    CudaMain::Instance()->ProlongatePacked(coarse.cuda_volume(),
                                           fine.cuda_volume(), 1.0f);
}

void MultigridCoreCuda::ProlongateResidual(const GraphicsVolume& fine,
                                           const GraphicsVolume& coarse)
{
    CudaMain::Instance()->Prolongate(fine.cuda_volume(), coarse.cuda_volume());
}

void MultigridCoreCuda::ProlongateResidualPacked(const GraphicsVolume& coarse,
                                                 const GraphicsVolume& fine)
{
    CudaMain::Instance()->ProlongatePacked(coarse.cuda_volume(),
                                           fine.cuda_volume(), 1.0f);
}

void MultigridCoreCuda::Relax(const GraphicsVolume& u, const GraphicsVolume& b,
                              float cell_size, int num_of_iterations)
{
    CudaMain::Instance()->Relax(u.cuda_volume(), u.cuda_volume(),
                                b.cuda_volume(), cell_size, num_of_iterations);
}

void MultigridCoreCuda::RelaxWithZeroGuessAndComputeResidual(
    const GraphicsVolume& packed_volumes, float cell_size, int times)
{
    // Just wait and see how the profiler tells us.
}

void MultigridCoreCuda::RelaxWithZeroGuess(const GraphicsVolume& u,
                                           const GraphicsVolume& b,
                                           float cell_size)
{
    CudaMain::Instance()->RelaxWithZeroGuess(u.cuda_volume(), b.cuda_volume(),
                                             cell_size);
}

void MultigridCoreCuda::RelaxWithZeroGuessPacked(const GraphicsVolume& packed,
                                                 float cell_size)
{
    float alpha_omega_over_beta = -(cell_size * cell_size) * 0.11111111f;
    float one_minus_omega = 0.33333333f;
    float minus_h_square = -(cell_size * cell_size);
    float omega_times_inverse_beta = 0.11111111f;
    CudaMain::Instance()->RelaxWithZeroGuessPacked(packed.cuda_volume(),
                                                   packed.cuda_volume(),
                                                   alpha_omega_over_beta,
                                                   one_minus_omega,
                                                   minus_h_square,
                                                   omega_times_inverse_beta);
}

void MultigridCoreCuda::Restrict(const GraphicsVolume& coarse,
                                 const GraphicsVolume& fine)
{
    CudaMain::Instance()->Restrict(coarse.cuda_volume(), fine.cuda_volume());
}

void MultigridCoreCuda::RestrictPacked(const GraphicsVolume& fine,
                                       const GraphicsVolume& coarse)
{
    CudaMain::Instance()->RestrictPacked(coarse.cuda_volume(),
                                         fine.cuda_volume());
}

void MultigridCoreCuda::RestrictResidualPacked(const GraphicsVolume& fine,
                                               const GraphicsVolume& coarse)
{
    CudaMain::Instance()->RestrictResidualPacked(coarse.cuda_volume(),
                                                 fine.cuda_volume());
}
