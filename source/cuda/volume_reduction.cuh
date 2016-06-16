#ifndef _VOLUME_REDUCTION_H_
#define _VOLUME_REDUCTION_H_

template <uint BlockSize>
__device__ void ReduceBlock(volatile float *sdata, float my_sum, const uint tid)
{
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

template <uint BlockSize, bool IsPow2>
__device__ void ReduceBlocks(float* block_results, uint total_elements,
                             uint row_stride, uint slice_stride)
{
    extern __shared__ float sdata[];

    uint tid = threadIdx.x;
    uint i = blockIdx.x * (BlockSize * 2) + threadIdx.x;
    uint grid_size = BlockSize * 2 * gridDim.x;
    float my_sum = 0.0f;

    while (i < total_elements) {
        my_sum += ReadFromTexture(i, row_stride, slice_stride);
        if (IsPow2 || i + BlockSize < total_elements)
            my_sum += ReadFromTexture(i + BlockSize, row_stride, slice_stride);

        i += grid_size;
    }

    ReduceBlock<BlockSize>(sdata, my_sum, tid);

    if (tid == 0)
        block_results[blockIdx.x] = sdata[0];
}

__device__ uint retirement_count = 0;

template <uint BlockSize, bool IsPow2, typename SaveScheme>
__global__ void ReduceVolumeKernel(float* dest, float* block_results,
                                   uint total_elements, uint row_stride,
                                   uint slice_stride)
{
    ReduceBlocks<BlockSize, IsPow2>(block_results, total_elements, row_stride,
                                    slice_stride);

    const uint tid = threadIdx.x;
    __shared__ bool last_block;
    extern float __shared__ smem[];

    __threadfence();

    if (tid == 0) {
        uint ticket = atomicInc(&retirement_count, gridDim.x);
        last_block = (ticket == gridDim.x - 1);
    }

    __syncthreads();

    if (last_block) {
        int i = tid;
        float my_sum = 0.0f;

        while (i < gridDim.x) {
            my_sum += block_results[i];
            i += BlockSize;
        }

        ReduceBlock<BlockSize>(smem, my_sum, tid);

        if (tid == 0) {
            SaveResult<SaveScheme>(dest, smem[0]);
            retirement_count = 0;
        }
    }
}

// =============================================================================

bool IsPow2(uint x)
{
    return ((x & (x - 1)) == 0);
}

template <typename SaveScheme>
void ReduceVolume(float* dest, float* block_results, uint3 volume_size,
                  BlockArrangement* ba)
{
    uint total_elements = volume_size.x * volume_size.y * volume_size.z;

    dim3 block;
    dim3 grid;
    ba->ArrangeSequential(&block, &grid, volume_size);

    uint num_of_threads = block.x;
    uint smem_size = num_of_threads * sizeof(float);
    uint row_stride = volume_size.x;
    uint slice_stride = volume_size.x * volume_size.y;

    if (IsPow2(total_elements)) {
        switch (num_of_threads) {
            case 1024:
                ReduceVolumeKernel<1024, true, SaveScheme><<<grid, block, smem_size>>>(
                    dest, block_results, total_elements, row_stride,
                    slice_stride);
                break;
            case 512:
                ReduceVolumeKernel< 512, true, SaveScheme><<<grid, block, smem_size>>>(
                    dest, block_results, total_elements, row_stride,
                    slice_stride);
                break;
            case 256:
                ReduceVolumeKernel< 256, true, SaveScheme><<<grid, block, smem_size>>>(
                    dest, block_results, total_elements, row_stride,
                    slice_stride);
                break;
            case 128:
                ReduceVolumeKernel< 128, true, SaveScheme><<<grid, block, smem_size>>>(
                    dest, block_results, total_elements, row_stride,
                    slice_stride);
                break;
            case 64:
                ReduceVolumeKernel<  64, true, SaveScheme><<<grid, block, smem_size>>>(
                    dest, block_results, total_elements, row_stride,
                    slice_stride);
                break;
        }
    } else {
        switch (num_of_threads) {
            case 1024:
                ReduceVolumeKernel<1024, false, SaveScheme><<<grid, block, smem_size>>>(
                    dest, block_results, total_elements, row_stride,
                    slice_stride);
                break;
            case 512:
                ReduceVolumeKernel< 512, false, SaveScheme><<<grid, block, smem_size>>>(
                    dest, block_results, total_elements, row_stride,
                    slice_stride);
                break;
            case 256:
                ReduceVolumeKernel< 256, false, SaveScheme><<<grid, block, smem_size>>>(
                    dest, block_results, total_elements, row_stride,
                    slice_stride);
                break;
            case 128:
                ReduceVolumeKernel< 128, false, SaveScheme><<<grid, block, smem_size>>>(
                    dest, block_results, total_elements, row_stride,
                    slice_stride);
                break;
            case 64:
                ReduceVolumeKernel<  64, false, SaveScheme><<<grid, block, smem_size>>>(
                    dest, block_results, total_elements, row_stride,
                    slice_stride);
                break;
        }
    }
}

#endif  // _VOLUME_REDUCTION_H_