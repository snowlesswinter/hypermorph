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

#include "block_arrangement.h"

#include <algorithm>
#include <cassert>

#include <cuda_runtime.h>

BlockArrangement::BlockArrangement()
    : dev_prop_(new cudaDeviceProp())
{
    memset(dev_prop_.get(), 0, sizeof(*dev_prop_));
}

BlockArrangement::~BlockArrangement()
{

}

void BlockArrangement::Init(int dev_id)
{
    cudaGetDeviceProperties(dev_prop_.get(), dev_id);
}

void BlockArrangement::ArrangeGrid(dim3* grid, const dim3& block,
                                   const uint3& volume_size)
{
    if (!grid || !block.x || !block.y || !block.z)
        return;

    int bw = block.x;
    int bh = block.y;
    int bd = block.z;
    *grid = dim3((volume_size.x + bw - 1) / bw, (volume_size.y + bh - 1) / bh,
                 (volume_size.z + bd - 1) / bd);
}

void BlockArrangement::ArrangeRowScan(dim3* block, dim3* grid,
                                      const uint3& volume_size)
{
    if (!block || !grid || !volume_size.x)
        return;

    int max_threads = dev_prop_->maxThreadsPerBlock >> 1;
    int bw = volume_size.x;
    int bh = std::min(max_threads / bw, static_cast<int>(volume_size.y));
    int bd = std::min(max_threads / bw / bh, static_cast<int>(volume_size.z));
    if (!bh || !bd)
        return;

    *block = dim3(bw, bh, bd);
    *grid = dim3((volume_size.x + bw - 1) / bw, (volume_size.y + bh - 1) / bh,
                 (volume_size.z + bd - 1) / bd);
}

void BlockArrangement::ArrangePrefer3dLocality(dim3* block, dim3* grid,
                                               const uint3& volume_size)
{
    if (!block || !grid)
        return;

    int bw = 8;
    int bh = 8;
    int bd = 8;
    *block = dim3(bw, bh, bd);
    *grid = dim3((volume_size.x + bw - 1) / bw, (volume_size.y + bh - 1) / bh,
                 (volume_size.z + bd - 1) / bd);
}

void BlockArrangement::ArrangeSequential(dim3* block, dim3* grid,
                                         const uint3& volume_size)
{
    if (!block || !grid || !volume_size.x)
        return;

    int max_threads = dev_prop_->maxThreadsPerBlock;
    int bw = max_threads;
    int bh = 1;
    int bd = 1;

    int elements = volume_size.x * volume_size.y * volume_size.z;
    int blocks = std::max(1, elements / (bw * 2));

    const int kMaxBlocks = 64;
    blocks = std::min(kMaxBlocks, blocks);

    *block = dim3(bw, bh, bd);
    *grid = dim3(blocks, 1, 1);
}
