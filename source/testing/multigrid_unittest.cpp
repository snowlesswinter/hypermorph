//
// Fluid3d - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Fluid3d license (* see part 1 below)
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. Acknowledgement of the
//    original author is required if you publish this in a paper, or use it
//    in a product.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

#include "stdafx.h"
#include "multigrid_unittest.h"

#include "cuda_host/cuda_volume.h"
#include "graphics_volume.h"
#include "opengl/gl_texture.h"
#include "poisson_solver/poisson_core_cuda.h"
#include "poisson_solver/poisson_core_glsl.h"
#include "unittest_common.h"
#include "utility.h"

const float kCellSize = 0.15f;

void MultigridUnittest::TestProlongation(int random_seed)
{
    srand(random_seed);

    GraphicsVolume cuda_fine(GRAPHICS_LIB_CUDA);
    GraphicsVolume glsl_fine(GRAPHICS_LIB_GLSL);
    GraphicsVolume cuda_coarse(GRAPHICS_LIB_CUDA);
    GraphicsVolume glsl_coarse(GRAPHICS_LIB_GLSL);

    cuda_fine.Create(128, 128, 128, 2, 2, 0);
    glsl_fine.Create(128, 128, 128, 2, 2, 0);
    cuda_coarse.Create(64, 64, 64, 2, 2, 0);
    glsl_coarse.Create(64, 64, 64, 2, 2, 0);

    int width = cuda_coarse.GetWidth();
    int height = cuda_coarse.GetHeight();
    int depth = cuda_coarse.GetDepth();
    int n = 2;
    int pitch = width * sizeof(uint16_t) * n;
    int size = pitch * height * depth;
    UnittestCommon::InitializeVolume4(&cuda_coarse, &glsl_coarse, width, height,
                                      depth, n, pitch, size,
                                      std::make_pair(-4.0f, 4.0f));

    PoissonCoreCuda cuda_core;
    cuda_core.Prolongate(cuda_coarse, cuda_fine);

    PoissonCoreGlsl glsl_core;
    glsl_core.Prolongate(glsl_coarse, glsl_fine);

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
    GraphicsVolume cuda_residual(GRAPHICS_LIB_CUDA);
    GraphicsVolume glsl_volume(GRAPHICS_LIB_GLSL);
    GraphicsVolume glsl_residual(GRAPHICS_LIB_GLSL);

    cuda_volume.Create(128, 128, 128, 2, 2, 0);
    cuda_residual.Create(128, 128, 128, 1, 2, 0);
    glsl_volume.Create(128, 128, 128, 2, 2, 0);
    glsl_residual.Create(128, 128, 128, 1, 2, 0);

    int width = cuda_volume.GetWidth();
    int height = cuda_volume.GetHeight();
    int depth = cuda_volume.GetDepth();
    int n_2 = 2;
    int n_1 = 1;
    int pitch_2 = width * sizeof(uint16_t) * n_2;
    int pitch_1 = width * sizeof(uint16_t) * n_1;
    int size_2 = pitch_2 * height * depth;
    int size_1 = pitch_1 * height * depth;
    UnittestCommon::InitializeVolume2(&cuda_volume, &glsl_volume, width, height,
                                      depth, n_2, pitch_2, size_2,
                                      std::make_pair(-2.0f, 2.0f));

    PoissonCoreCuda cuda_core;
    cuda_core.ComputeResidual(cuda_volume, cuda_volume, cuda_residual,
                              kCellSize);

    PoissonCoreGlsl glsl_core;
    glsl_core.ComputeResidual(glsl_volume, glsl_volume, glsl_residual,
                              kCellSize);

    UnittestCommon::CollectAndVerifyResult(width, height, depth, size_1,
                                           pitch_1, n_1, 1, &cuda_residual,
                                           &glsl_residual, __FUNCTION__);
}

void MultigridUnittest::TestResidualRestriction(int random_seed)
{
    srand(random_seed);

    GraphicsVolume cuda_fine(GRAPHICS_LIB_CUDA);
    GraphicsVolume glsl_fine(GRAPHICS_LIB_GLSL);
    GraphicsVolume cuda_coarse(GRAPHICS_LIB_CUDA);
    GraphicsVolume glsl_coarse(GRAPHICS_LIB_GLSL);

    cuda_fine.Create(128, 128, 128, 1, 2, 0);
    glsl_fine.Create(128, 128, 128, 1, 2, 0);
    cuda_coarse.Create(64, 64, 64, 2, 2, 0);
    glsl_coarse.Create(64, 64, 64, 2, 2, 0);

    int width = cuda_fine.GetWidth();
    int height = cuda_fine.GetHeight();
    int depth = cuda_fine.GetDepth();
    int n_2 = 2;
    int n_1 = 1;
    int pitch_2 = width * sizeof(uint16_t) * n_2;
    int pitch_1 = width * sizeof(uint16_t) * n_1;
    int size_2 = pitch_2 * height * depth;
    int size_1 = pitch_1 * height * depth;
    UnittestCommon::InitializeVolume1(&cuda_fine, &glsl_fine, width, height,
                                      depth, n_1, pitch_1, size_1,
                                      std::make_pair(-4.0f, 4.0f));

    PoissonCoreCuda cuda_core;
    cuda_core.Restrict(cuda_fine, cuda_coarse);

    PoissonCoreGlsl glsl_core;
    glsl_core.Restrict(glsl_fine, glsl_coarse);

    int pitch_coarse = cuda_coarse.GetWidth() * sizeof(uint16_t) * n_2;
    int size_coarse = pitch_coarse * cuda_coarse.GetHeight() *
        cuda_coarse.GetDepth();
    UnittestCommon::CollectAndVerifyResult(cuda_coarse.GetWidth(),
                                           cuda_coarse.GetHeight(),
                                           cuda_coarse.GetDepth(), size_coarse,
                                           pitch_coarse, n_2, 2, &cuda_coarse,
                                           &glsl_coarse, __FUNCTION__);
}

void MultigridUnittest::TestRestriction(int random_seed)
{
    srand(random_seed);

    GraphicsVolume cuda_fine(GRAPHICS_LIB_CUDA);
    GraphicsVolume glsl_fine(GRAPHICS_LIB_GLSL);
    GraphicsVolume cuda_coarse(GRAPHICS_LIB_CUDA);
    GraphicsVolume glsl_coarse(GRAPHICS_LIB_GLSL);

    cuda_fine.Create(128, 128, 128, 2, 2, 0);
    glsl_fine.Create(128, 128, 128, 2, 2, 0);
    cuda_coarse.Create(64, 64, 64, 2, 2, 0);
    glsl_coarse.Create(64, 64, 64, 2, 2, 0);

    int width = cuda_fine.GetWidth();
    int height = cuda_fine.GetHeight();
    int depth = cuda_fine.GetDepth();
    int n = 2;
    int pitch = width * sizeof(uint16_t) * n;
    int size = pitch * height * depth;
    UnittestCommon::InitializeVolume2(&cuda_fine, &glsl_fine, width, height,
                                      depth, n, pitch, size,
                                      std::make_pair(-4.0f, 4.0f));

    PoissonCoreCuda cuda_core;
    cuda_core.Restrict(cuda_fine, cuda_coarse);

    PoissonCoreGlsl glsl_core;
    glsl_core.Restrict(glsl_fine, glsl_coarse);

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

    cuda_volume.Create(128, 128, 128, 2, 2, 0);
    glsl_volume.Create(128, 128, 128, 2, 2, 0);

    int width = cuda_volume.GetWidth();
    int height = cuda_volume.GetHeight();
    int depth = cuda_volume.GetDepth();
    int n = 2;
    int pitch = width * sizeof(uint16_t) * n;
    int size = pitch * height * depth;
    UnittestCommon::InitializeVolume4(&cuda_volume, &glsl_volume, width, height,
                                      depth, n, pitch, size,
                                      std::make_pair(-4.0f, 4.0f));

    PoissonCoreCuda cuda_core;
    cuda_core.RelaxWithZeroGuess(cuda_volume, cuda_volume, kCellSize);

    PoissonCoreGlsl glsl_core;
    glsl_core.RelaxWithZeroGuess(glsl_volume, glsl_volume, kCellSize);

    UnittestCommon::CollectAndVerifyResult(width, height, depth, size, pitch, n,
                                           1, &cuda_volume, &glsl_volume,
                                           __FUNCTION__);
}
