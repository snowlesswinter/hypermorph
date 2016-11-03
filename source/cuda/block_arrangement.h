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

#ifndef _BLOCK_ARRANGEMENT_H_
#define _BLOCK_ARRANGEMENT_H_

#include <memory>

struct cudaDeviceProp;
struct dim3;
struct uint3;
class BlockArrangement
{
public:
    BlockArrangement();
    ~BlockArrangement();

    void Init(int dev_id);

    void ArrangeGrid(dim3* grid, const dim3& block, const uint3& volume_size);
    void ArrangeLinear(dim3* grid, dim3* block, int num_of_elements);
    void ArrangeLinearReduction(dim3* grid, dim3* block, int* num_of_blocks,
                                int* np2_last_block, int* elements_last_block,
                                int* threads_last_block, int num_of_elements);
    void ArrangePrefer3dLocality(dim3* block, dim3* grid,
                                 const uint3& volume_size);
    void ArrangeRowScan(dim3* block, dim3* grid, const uint3& volume_size);
    void ArrangeSequential(dim3* block, dim3* grid, const uint3& volume_size);

    // TODO: Kernel strategy?
    int GetSharedMemPerSMInKB() const;

private:
    std::unique_ptr<cudaDeviceProp> dev_prop_;
};

#endif // _BLOCK_ARRANGEMENT_H_