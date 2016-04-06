#include "stdafx.h"
#include "multigrid_unittest.h"

#include "cuda_host/cuda_volume.h"
#include "graphics_volume.h"
#include "opengl/gl_texture.h"
#include "poisson_solver/multigrid_core_cuda.h"
#include "poisson_solver/multigrid_core_glsl.h"
#include "unittest_common.h"
#include "utility.h"

void MultigridUnittest::TestProlongation(int random_seed)
{
    srand(random_seed);

    GraphicsVolume cuda_fine(GRAPHICS_LIB_CUDA);
    GraphicsVolume glsl_fine(GRAPHICS_LIB_GLSL);
    GraphicsVolume cuda_coarse(GRAPHICS_LIB_CUDA);
    GraphicsVolume glsl_coarse(GRAPHICS_LIB_GLSL);

    cuda_fine.Create(128, 128, 128, 4, 2);
    glsl_fine.Create(128, 128, 128, 4, 2);
    cuda_coarse.Create(64, 64, 64, 4, 2);
    glsl_coarse.Create(64, 64, 64, 4, 2);

    int width = cuda_coarse.GetWidth();
    int height = cuda_coarse.GetHeight();
    int depth = cuda_coarse.GetDepth();
    int n = 4;
    int pitch = width * sizeof(uint16_t) * n;
    int size = pitch * height * depth;
    UnittestCommon::InitializeVolume4(&cuda_coarse, &glsl_coarse, width, height,
                                      depth, n, pitch, size,
                                      std::make_pair(-4.0f, 4.0f));

    MultigridCoreCuda cuda_core;
    cuda_core.ProlongatePacked(cuda_coarse, cuda_fine);

    MultigridCoreGlsl glsl_core;
    glsl_core.ProlongatePacked(glsl_coarse, glsl_fine);

    int pitch_fine = cuda_fine.GetWidth() * sizeof(uint16_t) * n;
    int size_fine = pitch_fine * cuda_fine.GetHeight() *
        cuda_fine.GetDepth();
    UnittestCommon::CollectAndVerifyResult(cuda_fine.GetWidth(),
                                           cuda_fine.GetHeight(),
                                           cuda_fine.GetDepth(), size_fine,
                                           pitch_fine, n, 3, &cuda_fine,
                                           &glsl_fine, __FUNCTION__);
}

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

void MultigridUnittest::TestResidualRestriction(int random_seed)
{
    srand(random_seed);

    GraphicsVolume cuda_fine(GRAPHICS_LIB_CUDA);
    GraphicsVolume glsl_fine(GRAPHICS_LIB_GLSL);
    GraphicsVolume cuda_coarse(GRAPHICS_LIB_CUDA);
    GraphicsVolume glsl_coarse(GRAPHICS_LIB_GLSL);

    cuda_fine.Create(128, 128, 128, 4, 2);
    glsl_fine.Create(128, 128, 128, 4, 2);
    cuda_coarse.Create(64, 64, 64, 4, 2);
    glsl_coarse.Create(64, 64, 64, 4, 2);

    int width = cuda_fine.GetWidth();
    int height = cuda_fine.GetHeight();
    int depth = cuda_fine.GetDepth();
    int n = 4;
    int pitch = width * sizeof(uint16_t) * n;
    int size = pitch * height * depth;
    UnittestCommon::InitializeVolume4(&cuda_fine, &glsl_fine, width, height,
                                      depth, n, pitch, size,
                                      std::make_pair(-4.0f, 4.0f));

    MultigridCoreCuda cuda_core;
    cuda_core.RestrictResidualPacked(cuda_fine, cuda_coarse);

    MultigridCoreGlsl glsl_core;
    glsl_core.RestrictResidualPacked(glsl_fine, glsl_coarse);

    int pitch_coarse = cuda_coarse.GetWidth() * sizeof(uint16_t) * n;
    int size_coarse = pitch_coarse * cuda_coarse.GetHeight() *
        cuda_coarse.GetDepth();
    UnittestCommon::CollectAndVerifyResult(cuda_coarse.GetWidth(),
                                           cuda_coarse.GetHeight(),
                                           cuda_coarse.GetDepth(), size_coarse,
                                           pitch_coarse, n, 2, &cuda_coarse,
                                           &glsl_coarse, __FUNCTION__);
}

void MultigridUnittest::TestRestriction(int random_seed)
{
    srand(random_seed);

    GraphicsVolume cuda_fine(GRAPHICS_LIB_CUDA);
    GraphicsVolume glsl_fine(GRAPHICS_LIB_GLSL);
    GraphicsVolume cuda_coarse(GRAPHICS_LIB_CUDA);
    GraphicsVolume glsl_coarse(GRAPHICS_LIB_GLSL);

    cuda_fine.Create(128, 128, 128, 4, 2);
    glsl_fine.Create(128, 128, 128, 4, 2);
    cuda_coarse.Create(64, 64, 64, 4, 2);
    glsl_coarse.Create(64, 64, 64, 4, 2);

    int width = cuda_fine.GetWidth();
    int height = cuda_fine.GetHeight();
    int depth = cuda_fine.GetDepth();
    int n = 4;
    int pitch = width * sizeof(uint16_t) * n;
    int size = pitch * height * depth;
    UnittestCommon::InitializeVolume4(&cuda_fine, &glsl_fine, width, height,
                                      depth, n, pitch, size,
                                      std::make_pair(-4.0f, 4.0f));

    MultigridCoreCuda cuda_core;
    cuda_core.RestrictPacked(cuda_fine, cuda_coarse);

    MultigridCoreGlsl glsl_core;
    glsl_core.RestrictPacked(glsl_fine, glsl_coarse);

    int pitch_coarse = cuda_coarse.GetWidth() * sizeof(uint16_t) * n;
    int size_coarse = pitch_coarse * cuda_coarse.GetHeight() *
        cuda_coarse.GetDepth();
    UnittestCommon::CollectAndVerifyResult(cuda_coarse.GetWidth(),
                                           cuda_coarse.GetHeight(),
                                           cuda_coarse.GetDepth(), size_coarse,
                                           pitch_coarse, n, 3, &cuda_coarse,
                                           &glsl_coarse, __FUNCTION__);
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
