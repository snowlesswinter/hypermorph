#include "stdafx.h"
#include "multigrid_core_cuda.h"

#include <cassert>

#include "cuda_host/cuda_main.h"
#include "graphics_mem_piece.h"
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

std::shared_ptr<GraphicsMemPiece> MultigridCoreCuda::CreateMemPiece(int size)
{
    std::shared_ptr<GraphicsMemPiece> r =
        std::make_shared<GraphicsMemPiece>(GRAPHICS_LIB_CUDA);
    bool succeeded = r->Create(size);
    return succeeded ? r : std::shared_ptr<GraphicsMemPiece>();
}

std::shared_ptr<GraphicsVolume> MultigridCoreCuda::CreateVolume(
    int width, int height, int depth, int num_of_components, int byte_width)
{
    std::shared_ptr<GraphicsVolume> r =
        std::make_shared<GraphicsVolume>(GRAPHICS_LIB_CUDA);
    bool succeeded = r->Create(width, height, depth, num_of_components,
                               byte_width, 0);

    return succeeded ? r : std::shared_ptr<GraphicsVolume>();
}

std::shared_ptr<GraphicsVolume3> MultigridCoreCuda::CreateVolumeGroup(
    int width, int height, int depth, int num_of_components, int byte_width)
{
    std::shared_ptr<GraphicsVolume3> r(new GraphicsVolume3(GRAPHICS_LIB_CUDA));
    bool succeeded = r->Create(width, height, depth, num_of_components,
                               byte_width, 0);
    return succeeded ? r : std::shared_ptr<GraphicsVolume3>();
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

void MultigridCoreCuda::ProlongateError(const GraphicsVolume& fine,
                                        const GraphicsVolume& coarse)
{
    CudaMain::Instance()->ProlongateError(fine.cuda_volume(),
                                          coarse.cuda_volume());
}

void MultigridCoreCuda::Relax(const GraphicsVolume& u, const GraphicsVolume& b,
                              float cell_size, int num_of_iterations)
{
    CudaMain::Instance()->Relax(u.cuda_volume(), u.cuda_volume(),
                                b.cuda_volume(), cell_size, num_of_iterations);
}

void MultigridCoreCuda::RelaxWithZeroGuess(const GraphicsVolume& u,
                                           const GraphicsVolume& b,
                                           float cell_size)
{
    CudaMain::Instance()->RelaxWithZeroGuess(u.cuda_volume(), b.cuda_volume(),
                                             cell_size);
}

void MultigridCoreCuda::Restrict(const GraphicsVolume& coarse,
                                 const GraphicsVolume& fine)
{
    CudaMain::Instance()->Restrict(coarse.cuda_volume(), fine.cuda_volume());
}

void MultigridCoreCuda::ComputeRho(const GraphicsMemPiece& rho,
                                   const GraphicsVolume& z,
                                   const GraphicsVolume& r)
{
    CudaMain::Instance()->ComputeRho(rho.cuda_mem_piece(), z.cuda_volume(),
                                     r.cuda_volume());
}
