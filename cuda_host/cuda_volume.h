#ifndef _CUDA_VOLUME_H_
#define _CUDA_VOLUME_H_

#include <map>
#include <memory>

struct cudaArray;
struct cudaPitchedPtr;
class CudaVolume
{
public:
    CudaVolume();
    ~CudaVolume();

    bool Create(int width, int height, int depth, int num_of_components,
                int byte_width);
    bool CreateInPlace(int width, int height, int depth, int num_of_components,
                       int byte_width);

    cudaArray* dev_array() const { return dev_array_; }
    cudaPitchedPtr* dev_mem() const { return dev_mem_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int depth() const { return depth_; }

private:
    cudaArray* dev_array_;
    cudaPitchedPtr* dev_mem_;
    int width_;
    int height_;
    int depth_;
};

#endif // _CUDA_VOLUME_H_