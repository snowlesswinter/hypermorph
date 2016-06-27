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
#include <functional>
#include <memory>

#include "third_party/opengl/glew.h"

#include <cuda_runtime.h>
#include <helper_math.h>

#include "cuda/aux_buffer_manager.h"
#include "cuda/block_arrangement.h"
#include "cuda/kernel_launcher.h"

bool TestPrefixSum(BlockArrangement* ba, AuxBufferManager* bm)
{
    srand(0x18923074);

    const int num_of_cells = 1000000;

    std::unique_ptr<uint[]> cell_particles_counts(new uint[num_of_cells]);
    std::unique_ptr<uint[]> cell_offsets_cpu(new uint[num_of_cells]);
    std::unique_ptr<uint[]> cell_offsets_gpu(new uint[num_of_cells]);

    std::unique_ptr<uint, std::function<void(void*)>> idata(
        reinterpret_cast<uint*>(bm->Allocate(num_of_cells * sizeof(uint))),
        [&bm](void* p) { bm->Free(p); });
    std::unique_ptr<uint, std::function<void(void*)>> odata(
        reinterpret_cast<uint*>(bm->Allocate(num_of_cells * sizeof(uint))),
        [&bm](void* p) { bm->Free(p); });

    for (int i = 0; i < num_of_cells; i++)
        cell_particles_counts[i] = rand() % 8;

    uint sum = 0;
    for (int i = 0; i < num_of_cells; i++) {
        cell_offsets_cpu[i] = sum;
        sum += cell_particles_counts[i];
    }

    cudaMemcpy(idata.get(), cell_particles_counts.get(),
               num_of_cells * sizeof(uint), cudaMemcpyHostToDevice);
    LaunchBuildCellOffsets(odata.get(), idata.get(), num_of_cells, ba, bm);

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