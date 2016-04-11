#include "block_arrangement.h"

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

void BlockArrangement::Arrange(dim3* block, dim3* grid, const int3& volume_size)
{
    if (!block || !grid || !volume_size.x)
        return;

    int threads = dev_prop_->maxThreadsPerBlock >> 1;
    int bw = volume_size.x;
    int bh = threads / volume_size.x;
    int bd = threads / bw / bh;
    if (!bh || !bd)
        return;

    *block = dim3(bw, bh, bd);
    *grid = dim3((volume_size.x + bw - 1) / bw, (volume_size.x + bh - 1) / bh,
                 (volume_size.x + bd - 1) / bd);
}

void BlockArrangement::ArrangePrefer3dLocality(dim3* block, dim3* grid,
                                               const int3& volume_size)
{
    if (!block || !grid)
        return;

    int bw = 8;
    int bh = 8;
    int bd = 8;
    *block = dim3(bw, bh, bd);
    *grid = dim3((volume_size.x + bw - 1) / bw, (volume_size.x + bh - 1) / bh,
                 (volume_size.x + bd - 1) / bd);
}
