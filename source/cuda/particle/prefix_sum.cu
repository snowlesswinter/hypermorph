/*
FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
Copyright (C) 2012. Rama Hoetzlein, http://fluids3.com

Fluids-ZLib license (* see part 1 below)
This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. Acknowledgement of the
original author is required if you publish this in a paper, or use it
in a product. (See fluids3.com for details)
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

// =============================================================================

// MODIFIED by JIANWEN TAN(jianwen.tan@gmail.com).
//
// 1. Reoganized the structure of code.
// 2. Rewrote as c++ style.
// 

#include <cassert>
#include <functional>
#include <memory>
#include <vector>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "cuda/aux_buffer_manager.h"
#include "cuda/block_arrangement.h"
#include "cuda/cuda_common_host.h"
#include "cuda/cuda_common_kern.h"

namespace
{
typedef std::vector<std::unique_ptr<uint, std::function<void(void*)>>>
    BlockSums;
}

template <bool StoreBlockSum>
__device__ void ParallelPrefixSum(uint* block_sums, uint* smem, uint tid,
                                  uint block_index)
{
    uint stride = 1;
    uint base_offset;
    int offset0;
    int offset1;
    for (int i = blockDim.x; i > 0; i >>= 1) {
        __syncthreads();
        if (tid < i) {
            base_offset = __umul24(__umul24(2, stride), tid) + stride - 1;
            offset0 = base_offset;
            offset1 = base_offset + stride;

            smem[offset1] += smem[offset0];
        }

        stride <<= 1;
    }

    if (tid == 0) {
        if (StoreBlockSum)
            block_sums[block_index ? block_index : blockIdx.x] = smem[offset1];

        smem[offset1] = 0;
    }

    for (int i = 1; i <= blockDim.x; i <<= 1) {
        stride >>= 1;
        __syncthreads();

        if (tid < i) {
            base_offset = __umul24(__umul24(2, stride), tid) + stride - 1;
            offset0 = base_offset;
            offset1 = base_offset + stride;

            uint t = smem[offset0];
            smem[offset0] = smem[offset1];
            smem[offset1] += t;
        }
    }
}

template <bool LastBlock>
__global__ void ApplyBlockResultsKernel(uint* prefix_sum,
                                        const uint* block_sums,
                                        uint num_of_elements, uint block_offset,
                                        uint base_index)
{
    __shared__ int sum;
    if (threadIdx.x == 0)
        sum = block_sums[blockIdx.x + block_offset];

    uint index = __umul24(blockIdx.x, (blockDim.x << 1)) + base_index +
        threadIdx.x;

    __syncthreads();

    prefix_sum[index] += sum;
    if (!LastBlock || threadIdx.x + blockDim.x < num_of_elements)
        prefix_sum[index + blockDim.x] += sum;
}

template <bool StoreBlockSum, bool NP2>
__global__ void BuildPrefixSumKernel(uint* prefix_sum, uint* block_sums,
                                     const uint* cell_particles_counts,
                                     uint num_of_elements, uint block_index,
                                     uint element_index)
{
    extern __shared__ uint smem[];

    uint tid = threadIdx.x;
    uint base_index = element_index ?
        element_index : __umul24(blockIdx.x, (blockDim.x << 1));
    uint i0 = base_index + threadIdx.x;
    uint i1 = base_index + threadIdx.x + blockDim.x;

    // Load bin count into shared memory.
    uint smem_index0 = tid;
    uint smem_index1 = tid + blockDim.x;

    smem[smem_index0] = cell_particles_counts[i0];
    if (NP2) {
        smem[smem_index1] = (smem_index1 < num_of_elements) ?
            cell_particles_counts[i1] : 0;
    } else {
        smem[smem_index1] = cell_particles_counts[i1];
    }

    ParallelPrefixSum<StoreBlockSum>(block_sums, smem, tid, block_index);

    __syncthreads();

    // Save prefix sum into global memory.
    prefix_sum[i0] = smem[smem_index0];
    if (NP2) {
        if (smem_index1 < num_of_elements)
            prefix_sum[i1] = smem[smem_index1];
    } else {
        prefix_sum[i1] = smem[smem_index1];
    }
}

// =============================================================================

void PrefixSumRecursive(uint* particle_offsets,
                        const uint* cell_particles_counts, int num_of_elements,
                        const BlockSums& block_results, int level,
                        BlockArrangement* ba)
{
    dim3 block;
    dim3 grid;
    int num_of_blocks = 0;
    int np2_last_block = 0;
    int elements_last_block = 0;
    int threads_last_block = 0;
    ba->ArrangeLinearReduction(&grid, &block, &num_of_blocks, &np2_last_block,
                               &elements_last_block, &threads_last_block,
                               num_of_elements);

    int shared_size = sizeof(int) * block.x * 2;
    int shared_size_last_block = sizeof(int) * threads_last_block * 2;
    if (num_of_blocks > 1) {
        BuildPrefixSumKernel<true, false><<<grid, block, shared_size>>>(
            particle_offsets, block_results[level].get(), cell_particles_counts,
            block.x * 2, 0, 0);
        if (np2_last_block)
            BuildPrefixSumKernel<true, true>
                <<<1, threads_last_block, shared_size_last_block>>>(
                    particle_offsets, block_results[level].get(),
                    cell_particles_counts, elements_last_block,
                    num_of_blocks - 1, num_of_elements - elements_last_block);

        PrefixSumRecursive(block_results[level].get(),
                           block_results[level].get(), num_of_blocks,
                           block_results, level + 1, ba);

        ApplyBlockResultsKernel<false><<<grid, block>>>(
            particle_offsets, block_results[level].get(),
            num_of_elements - (np2_last_block ? elements_last_block : 0), 0, 0);
        if (np2_last_block)
            ApplyBlockResultsKernel<true><<<1, threads_last_block>>>(
                particle_offsets, block_results[level].get(),
                elements_last_block, num_of_blocks - 1,
                num_of_elements - elements_last_block);
    } else {
        if (IsPow2(num_of_elements))
            BuildPrefixSumKernel<false, false><<<grid, block, shared_size>>>(
                particle_offsets, 0, cell_particles_counts, block.x * 2, 0, 0);
        else
            BuildPrefixSumKernel<false, true><<<grid, block, shared_size>>>(
                particle_offsets, 0, cell_particles_counts, num_of_elements, 0,
                0);
    }
}

namespace kern_launcher
{
void BuildCellOffsets(uint* particle_offsets, const uint* cell_particles_counts,
                      int num_of_cells, BlockArrangement* ba,
                      AuxBufferManager* bm)
{
    int elements = num_of_cells;
    BlockSums block_sums;
    do {
        dim3 block;
        dim3 grid;
        int np2_last_block = 0;
        int num_of_blocks = 0;
        ba->ArrangeLinearReduction(&grid, &block, &num_of_blocks,
                                   &np2_last_block, nullptr, nullptr, elements);
        if (num_of_blocks > 1)
            block_sums.push_back(
                BlockSums::value_type(
                    reinterpret_cast<uint*>(
                        bm->Allocate(num_of_blocks * sizeof(uint))),
                    [&bm](void* p) { bm->Free(p); }));

        elements = num_of_blocks;
    } while (elements > 1);

    PrefixSumRecursive(particle_offsets, cell_particles_counts, num_of_cells,
                       block_sums, 0, ba);
}
}
