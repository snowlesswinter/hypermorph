//
// Hypermorph - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Hypermorph license (* see part 1 below)
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

#include "cuda_common_host.h"

#include <cassert>

bool IsPow2(uint x)
{
    return ((x & (x - 1)) == 0);
}

bool CopyVolumeAsync(cudaArray* dest, cudaArray* source,
                     const uint3& volume_size)
{
    cudaMemcpy3DParms cpy_parms = {};
    cpy_parms.dstArray = dest;
    cpy_parms.srcArray = source;
    cpy_parms.extent.width = volume_size.x;
    cpy_parms.extent.height = volume_size.y;
    cpy_parms.extent.depth = volume_size.z;
    cpy_parms.kind = cudaMemcpyDeviceToDevice;
    cudaError_t e = cudaMemcpy3DAsync(&cpy_parms);
    assert(e == cudaSuccess);
    return e == cudaSuccess;
}