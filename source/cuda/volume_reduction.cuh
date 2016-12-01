/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
Parallel reduction kernels
*/

// 
// A customized version for volume reduction by JINWEN TAN(jianwen.tan@gmail.com).
//

#ifndef _VOLUME_REDUCTION_H_
#define _VOLUME_REDUCTION_H_

template <uint BlockSize, typename FPType>
__device__ void ReduceBlock(volatile uint* sdata_raw, FPType my_sum,
                            const uint tid)
{
    volatile FPType* sdata = reinterpret_cast<volatile FPType*>(sdata_raw);
    sdata[tid] = my_sum;
    __syncthreads();

    if (BlockSize >= 1024) {
        if (tid < 512) {
            my_sum += sdata[tid + 512];
            sdata[tid] = my_sum;
        }

        __syncthreads();
    }

    if (BlockSize >= 512) {
        if (tid < 256) {
            my_sum += sdata[tid + 256];
            sdata[tid] = my_sum;
        }

        __syncthreads();
    }

    if (BlockSize >= 256) {
        if (tid < 128) {
            my_sum += sdata[tid + 128];
            sdata[tid] = my_sum;
        }

        __syncthreads();
    }

    if (BlockSize >= 128) {
        if (tid < 64) {
            my_sum += sdata[tid + 64];
            sdata[tid] = my_sum;
        }

        __syncthreads();
    }

    if (tid < 32) {
        if (BlockSize >= 64) {
            my_sum += sdata[tid + 32];
            sdata[tid] = my_sum;
        }

        if (BlockSize >= 32) {
            my_sum += sdata[tid + 16];
            sdata[tid] = my_sum;
        }

        if (BlockSize >= 16) {
            my_sum += sdata[tid + 8];
            sdata[tid] = my_sum;
        }

        if (BlockSize >= 8) {
            my_sum += sdata[tid + 4];
            sdata[tid] = my_sum;
        }

        if (BlockSize >= 4) {
            my_sum += sdata[tid + 2];
            sdata[tid] = my_sum;
        }

        if (BlockSize >= 2) {
            my_sum += sdata[tid + 1];
            sdata[tid] = my_sum;
        }
    }
}

template <uint BlockSize, bool IsPow2, typename FPType, typename DataScheme>
__device__ void ReduceBlocks(FPType* block_results, uint total_elements,
                             uint row_stride, uint slice_stride,
                             DataScheme scheme)
{
    extern __shared__ uint sdata_raw[];
    FPType* sdata = reinterpret_cast<FPType*>(sdata_raw);

    uint tid = threadIdx.x;
    uint i = blockIdx.x * (BlockSize * 2) + threadIdx.x;
    uint grid_size = BlockSize * 2 * gridDim.x;
    FPType my_sum = 0.0f;

    while (i < total_elements) {
        my_sum += scheme.Load(i, row_stride, slice_stride);
        if (IsPow2 || i + BlockSize < total_elements)
            my_sum += scheme.Load(i + BlockSize, row_stride, slice_stride);

        i += grid_size;
    }

    ReduceBlock<BlockSize>(sdata_raw, my_sum, tid);

    if (tid == 0)
        block_results[blockIdx.x] = sdata[0];
}

__device__ uint retirement_count = 0;

template <uint BlockSize, bool IsPow2, typename FPType, typename DataScheme>
__global__ void ReduceVolumeKernel(FPType* dest, FPType* block_results,
                                   uint total_elements, uint row_stride,
                                   uint slice_stride, DataScheme scheme)
{
    ReduceBlocks<BlockSize, IsPow2>(block_results, total_elements, row_stride,
                                    slice_stride, scheme);

    const uint tid = threadIdx.x;
    __shared__ bool last_block;
    extern uint __shared__ smem[];

    __threadfence();

    if (tid == 0) {
        uint ticket = atomicInc(&retirement_count, gridDim.x);
        last_block = (ticket == gridDim.x - 1);
    }

    __syncthreads();

    if (last_block) {
        int i = tid;
        FPType my_sum = 0.0f;

        while (i < gridDim.x) {
            my_sum += block_results[i];
            i += BlockSize;
        }

        ReduceBlock<BlockSize>(smem, my_sum, tid);

        if (tid == 0) {
            scheme.Save(dest, reinterpret_cast<FPType*>(smem)[0]);
            retirement_count = 0;
        }
    }
}

// =============================================================================

template <typename FPType, typename DataScheme>
void ReduceVolume(FPType* dest, const DataScheme& scheme, uint3 volume_size,
                  BlockArrangement* ba, AuxBufferManager* bm)
{
    uint total_elements = volume_size.x * volume_size.y * volume_size.z;

    dim3 grid;
    dim3 block;
    ba->ArrangeSequential(&grid, &block, volume_size);

    std::unique_ptr<FPType, std::function<void(void*)>> block_results(
        reinterpret_cast<FPType*>(bm->Allocate(grid.x * sizeof(FPType))),
        [&bm](void* p) { bm->Free(p); });

    uint num_of_threads = block.x;
    uint smem_size = num_of_threads * sizeof(FPType);
    uint row_stride = volume_size.x;
    uint slice_stride = volume_size.x * volume_size.y;

    if (IsPow2(total_elements)) {
        switch (num_of_threads) {
            case 1024:
                ReduceVolumeKernel<1024, true><<<grid, block, smem_size>>>(
                    dest, block_results.get(), total_elements, row_stride,
                    slice_stride, scheme);
                break;
            case 512:
                ReduceVolumeKernel< 512, true><<<grid, block, smem_size>>>(
                    dest, block_results.get(), total_elements, row_stride,
                    slice_stride, scheme);
                break;
            case 256:
                ReduceVolumeKernel< 256, true><<<grid, block, smem_size>>>(
                    dest, block_results.get(), total_elements, row_stride,
                    slice_stride, scheme);
                break;
            case 128:
                ReduceVolumeKernel< 128, true><<<grid, block, smem_size>>>(
                    dest, block_results.get(), total_elements, row_stride,
                    slice_stride, scheme);
                break;
            case 64:
                ReduceVolumeKernel<  64, true><<<grid, block, smem_size>>>(
                    dest, block_results.get(), total_elements, row_stride,
                    slice_stride, scheme);
                break;
        }
    } else {
        switch (num_of_threads) {
            case 1024:
                ReduceVolumeKernel<1024, false><<<grid, block, smem_size>>>(
                    dest, block_results.get(), total_elements, row_stride,
                    slice_stride, scheme);
                break;
            case 512:
                ReduceVolumeKernel< 512, false><<<grid, block, smem_size>>>(
                    dest, block_results.get(), total_elements, row_stride,
                    slice_stride, scheme);
                break;
            case 256:
                ReduceVolumeKernel< 256, false><<<grid, block, smem_size>>>(
                    dest, block_results.get(), total_elements, row_stride,
                    slice_stride, scheme);
                break;
            case 128:
                ReduceVolumeKernel< 128, false><<<grid, block, smem_size>>>(
                    dest, block_results.get(), total_elements, row_stride,
                    slice_stride, scheme);
                break;
            case 64:
                ReduceVolumeKernel<  64, false><<<grid, block, smem_size>>>(
                    dest, block_results.get(), total_elements, row_stride,
                    slice_stride, scheme);
                break;
        }
    }
}

#endif  // _VOLUME_REDUCTION_H_