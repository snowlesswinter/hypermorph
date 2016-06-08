#include "block_arrangement.h"

#include <algorithm>

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

    int threads = dev_prop_->maxThreadsPerBlock >> 1;
    int bw = volume_size.x;
    int bh = std::min(threads / bw, static_cast<int>(volume_size.y));
    int bd = std::min(threads / bw / bh, static_cast<int>(volume_size.z));
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
