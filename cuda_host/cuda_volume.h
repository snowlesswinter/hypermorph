#ifndef _CUDA_VOLUME_H_
#define _CUDA_VOLUME_H_

#include <map>
#include <memory>

struct cudaPitchedPtr;
class CudaVolume
{
public:
    CudaVolume();
    ~CudaVolume();

    bool Create(int width, int height, int depth, int num_of_components,
                int byte_width);

    cudaPitchedPtr* dev_mem() const { return dev_mem_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int depth() const { return depth_; }

private:
    cudaPitchedPtr* dev_mem_;
    int width_;
    int height_;
    int depth_;
};

#endif // _CUDA_VOLUME_H_