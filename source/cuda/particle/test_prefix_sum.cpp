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

#include <cassert>
#include <fstream>
#include <functional>
#include <memory>
#include <vector>

#include "third_party/opengl/glew.h"

#include <cuda_runtime.h>
#include <helper_math.h>

#include "cuda/aux_buffer_manager.h"
#include "cuda/block_arrangement.h"
#include "cuda/kernel_launcher.h"

std::vector<uint> PreparePrefixSumTest(uint32_t* cell_particle_counts,
                                       uint num_of_cells)
{
    std::vector<uint> cell_particle_counts_cpu(num_of_cells);
    cudaError_t e = cudaMemcpy(&cell_particle_counts_cpu[0],
                               cell_particle_counts,
                               num_of_cells * sizeof(uint),
                               cudaMemcpyDeviceToHost);
    assert(e == cudaSuccess);
    if (e != cudaSuccess)
        return std::vector<uint>();

    return std::move(cell_particle_counts_cpu);
}

void DumpResult(const std::vector<uint>& particle_index, int line_separator)
{
    std::ofstream f("e:/prefix_sum_result.txt");
    int count = 0;
    for (auto i = particle_index.begin(); i != particle_index.end(); i++) {
        f << *i << ", ";
        
        if (!(++count % line_separator))
            f << std::endl;
    }
    f.flush();
}

bool DoPrefixSumTest(const std::vector<uint>& cell_particle_counts,
                     uint32_t* particle_index,
                     BlockArrangement* ba, AuxBufferManager* bm)
{
    const int num_of_cells = cell_particle_counts.size();
    std::vector<uint> particle_index_cpu(num_of_cells);
    std::vector<uint> particle_index_gpu(num_of_cells);

    uint sum = 0;
    for (int i = 0; i < num_of_cells; i++) {
        particle_index_cpu[i] = sum;
        sum += cell_particle_counts[i];
    }

    cudaThreadSynchronize();
    cudaError_t e = cudaMemcpy(&particle_index_gpu[0], particle_index,
                               num_of_cells * sizeof(uint),
                               cudaMemcpyDeviceToHost);
    assert(e == cudaSuccess);
    if (e != cudaSuccess)
        return false;

    int bad = 0;
    for (int i = 0; i < num_of_cells; i++) {
        int a = particle_index_cpu[i];
        int b = particle_index_gpu[i];
        if (a != b)
            bad++;
    }

    //DumpResult(particle_index_gpu, 96);
    return !bad;
}

bool TestPrefixSum(BlockArrangement* ba, AuxBufferManager* bm)
{
    srand(0x18923074);

    const int num_of_cells = 10000000;

    std::unique_ptr<uint[]> cell_particle_counts(new uint[num_of_cells]);
    std::unique_ptr<uint[]> cell_offsets_cpu(new uint[num_of_cells]);
    std::unique_ptr<uint[]> cell_offsets_gpu(new uint[num_of_cells]);

    std::unique_ptr<uint, std::function<void(void*)>> idata(
        reinterpret_cast<uint*>(bm->Allocate(num_of_cells * sizeof(uint))),
        [&bm](void* p) { bm->Free(p); });
    std::unique_ptr<uint, std::function<void(void*)>> odata(
        reinterpret_cast<uint*>(bm->Allocate(num_of_cells * sizeof(uint))),
        [&bm](void* p) { bm->Free(p); });

    for (int i = 0; i < num_of_cells; i++)
        cell_particle_counts[i] = rand() % 8;

    uint sum = 0;
    for (int i = 0; i < num_of_cells; i++) {
        cell_offsets_cpu[i] = sum;
        sum += cell_particle_counts[i];
    }

    cudaMemcpy(idata.get(), cell_particle_counts.get(),
               num_of_cells * sizeof(uint), cudaMemcpyHostToDevice);
    kern_launcher::BuildCellOffsets(odata.get(), idata.get(), num_of_cells, ba,
                                    bm);

    cudaThreadSynchronize();
    cudaMemcpy(cell_offsets_gpu.get(), odata.get(), num_of_cells * sizeof(uint),
               cudaMemcpyDeviceToHost);

    int bad = 0;
    for (int i = 0; i < num_of_cells; i++) {
        int a = cell_offsets_cpu[i];
        int b = cell_offsets_gpu[i];
        if (a != b)
            bad++;
    }

    return !bad;
}