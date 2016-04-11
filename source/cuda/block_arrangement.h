#ifndef _BLOCK_ARRANGEMENT_H_
#define _BLOCK_ARRANGEMENT_H_

#include <memory>

struct cudaDeviceProp;
struct dim3;
struct int3;
class BlockArrangement
{
public:
    BlockArrangement();
    ~BlockArrangement();

    void Init(int dev_id);

    void Arrange(dim3* block, dim3* grid, const int3& volume_size);
    void ArrangePrefer3dLocality(dim3* block, dim3* grid,
                                 const int3& volume_size);

private:
    std::unique_ptr<cudaDeviceProp> dev_prop_;
};

#endif // _BLOCK_ARRANGEMENT_H_