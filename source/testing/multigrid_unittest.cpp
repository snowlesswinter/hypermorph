#include "stdafx.h"
#include "multigrid_unittest.h"

#include "cuda_host/cuda_volume.h"
#include "graphics_volume.h"
#include "multigrid_core_cuda.h"
#include "multigrid_core_glsl.h"
#include "opengl/gl_texture.h"
#include "unittest_common.h"
#include "utility.h"

void MultigridUnittest::TestResidualCalculation(int random_seed)
{
    srand(random_seed);

    GraphicsVolume cuda_volume(GRAPHICS_LIB_CUDA);
    GraphicsVolume glsl_volume(GRAPHICS_LIB_GLSL);

    cuda_volume.Create(128, 128, 128, 4, 2);
    glsl_volume.Create(128, 128, 128, 4, 2);

    int width = cuda_volume.GetWidth();
    int height = cuda_volume.GetHeight();
    int depth = cuda_volume.GetDepth();
    int n = 4;
    int pitch = width * sizeof(uint16_t) * n;
    int size = pitch * height * depth;
    UnittestCommon::InitializeVolume4(&cuda_volume, &glsl_volume, width, height,
                                      depth, n, pitch, size,
                                      std::make_pair(-2.0f, 2.0f));

    MultigridCoreCuda cuda_core;
    cuda_core.ComputeResidualPacked(cuda_volume, CellSize);

    MultigridCoreGlsl glsl_core;
    glsl_core.ComputeResidualPacked(glsl_volume, CellSize);

    UnittestCommon::CollectAndVerifyResult(width, height, depth, size, pitch, n,
                                           4, &cuda_volume, &glsl_volume,
                                           __FUNCTION__);
}

void MultigridUnittest::TestZeroGuessRelaxation(int random_seed)
{
    srand(random_seed);

    GraphicsVolume cuda_volume(GRAPHICS_LIB_CUDA);
    GraphicsVolume glsl_volume(GRAPHICS_LIB_GLSL);

    cuda_volume.Create(128, 128, 128, 4, 2);
    glsl_volume.Create(128, 128, 128, 4, 2);

    int width = cuda_volume.GetWidth();
    int height = cuda_volume.GetHeight();
    int depth = cuda_volume.GetDepth();
    int n = 4;
    int pitch = width * sizeof(uint16_t) * n;
    int size = pitch * height * depth;
    UnittestCommon::InitializeVolume4(&cuda_volume, &glsl_volume, width, height,
                                      depth, n, pitch, size,
                                      std::make_pair(-4.0f, 4.0f));

    MultigridCoreCuda cuda_core;
    cuda_core.RelaxWithZeroGuessPacked(cuda_volume, CellSize);

    MultigridCoreGlsl glsl_core;
    glsl_core.RelaxWithZeroGuessPacked(glsl_volume, CellSize);

    UnittestCommon::CollectAndVerifyResult(width, height, depth, size, pitch, n,
                                           1, &cuda_volume, &glsl_volume,
                                           __FUNCTION__);
}
