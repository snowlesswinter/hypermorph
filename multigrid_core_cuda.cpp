#include "stdafx.h"
#include "multigrid_core_cuda.h"

#include <cassert>

#include "cuda_host/cuda_main.h"
#include "graphics_volume.h"
#include "shader/fluid_shader.h"
#include "shader/multigrid_shader.h"
#include "shader/multigrid_staggered_shader.h"
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

void MultigridCoreCuda::ComputeResidualPacked(const GraphicsVolume& packed,
                                              float cell_size)
{
    
}

void MultigridCoreCuda::ProlongatePacked(const GraphicsVolume& coarse,
                                         const GraphicsVolume& fine)
{
    CudaMain::Instance()->ProlongatePackedPure(coarse.cuda_volume(),
                                               fine.cuda_volume());
}

void MultigridCoreCuda::RelaxPacked(const GraphicsVolume& u_and_b,
                                    float cell_size)
{
}

void MultigridCoreCuda::RelaxWithZeroGuessAndComputeResidual(
    const GraphicsVolume& packed_volumes, float cell_size, int times)
{
    // Just wait and see how the profiler tells us.
}

void MultigridCoreCuda::RelaxWithZeroGuessPacked(const GraphicsVolume& packed,
                                                 float cell_size)
{
    float alpha_omega_over_beta = -(cell_size * cell_size) * 0.11111111f;
    float one_minus_omega = 0.33333333f;
    float minus_h_square = -(cell_size * cell_size);
    float omega_times_inverse_beta = 0.11111111f;
    CudaMain::Instance()->RelaxWithZeroGuessPackedPure(
        packed.cuda_volume(), packed.cuda_volume(), alpha_omega_over_beta,
        one_minus_omega, minus_h_square, omega_times_inverse_beta);
}

void MultigridCoreCuda::RestrictPacked(const GraphicsVolume& fine,
                                       const GraphicsVolume& coarse)
{
    
}

void MultigridCoreCuda::RestrictResidualPacked(const GraphicsVolume& fine,
                                               const GraphicsVolume& coarse)
{
    
}

void MultigridCoreCuda::ComputeResidualPackedDiagnosis(
    const GraphicsVolume& packed, const GraphicsVolume& diagnosis,
    float cell_size)
{
    
}