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

#ifndef _UNITTEST_COMMON_H_
#define _UNITTEST_COMMON_H_

#include <utility>
#include <vector>

#include <stdint.h>

class GraphicsVolume;
class GridFluidSolver;
class UnittestCommon
{
public:
    static void CollectAndVerifyResult(int width, int height, int depth,
                                       int size, int pitch, int n,
                                       uint32_t channel_mask,
                                       GraphicsVolume* cuda_volume,
                                       GraphicsVolume* glsl_volume,
                                       char* function_name);
    static bool InitializeSimulators(GridFluidSolver* sim_cuda,
                                     GridFluidSolver* sim_glsl);
    static void InitializeVolume1(GraphicsVolume* cuda_volume,
                                  GraphicsVolume* glsl_volume, int width,
                                  int height, int depth, int n, int pitch,
                                  int size,
                                  const std::pair<float, float>& scope);
    static void InitializeVolume2(GraphicsVolume* cuda_volume,
                                  GraphicsVolume* glsl_volume, int width,
                                  int height, int depth, int n, int pitch,
                                  int size,
                                  const std::pair<float, float>& scope);
    static void InitializeVolume4(GraphicsVolume* cuda_volume,
                                  GraphicsVolume* glsl_volume, int width,
                                  int height, int depth, int n, int pitch,
                                  int size,
                                  const std::pair<float, float>& scope);
    static float RandomFloat(const std::pair<float, float>& scope);
    static void VerifyResult1(const std::vector<uint16_t>& result_cuda,
                              const std::vector<uint16_t>& result_glsl,
                              int width, int height, int depth, int n,
                              char* function_name);

private:
    UnittestCommon();
    ~UnittestCommon();
};

#endif // _UNITTEST_COMMON_H_