#ifndef _CUDA_ARRAY_GROUP_H_
#define _CUDA_ARRAY_GROUP_H_

#include <array>

class CudaArrayGroup
{
public:
    CudaArrayGroup(cudaArray_t x, cudaArray_t y, cudaArray_t z)
        : x_(x)
        , y_(y)
        , z_(z)
        , v_({x_, y_, z_})
        , offset_x_(make_float3(0.0f, 0.0f, 0.0f))
        , offset_y_(make_float3(0.0f, 0.0f, 0.0f))
        , offset_z_(make_float3(0.0f, 0.0f, 0.0f))
        , ov_({offset_x_, offset_y_, offset_z_})
    {
    }
    CudaArrayGroup(cudaArray_t x, cudaArray_t y, cudaArray_t z, float3 offset_x,
                   float3 offset_y, float3 offset_z)
        : x_(x)
        , y_(y)
        , z_(z)
        , v_({x_, y_, z_})
        , offset_x_(offset_x)
        , offset_y_(offset_y)
        , offset_z_(offset_z)
        , ov_({offset_x_, offset_y_, offset_z_})
    {
    }

    int num_of_components() const { return num_of_components_; }

    cudaArray_t x() const { return x_; }
    cudaArray_t y() const { return y_; }
    cudaArray_t z() const { return z_; }

    float3 offset_x() const { return offset_x_; }
    float3 offset_y() const { return offset_y_; }
    float3 offset_z() const { return offset_z_; }

    cudaArray_t operator[](int n) const
    {
        return v_[n];
    }
    float3 offset(int n) const
    {
        return ov_[n];
    }


private:
    static const int num_of_components_ = 3;

    cudaArray_t x_;
    cudaArray_t y_;
    cudaArray_t z_;
    std::array<cudaArray_t, num_of_components_> v_;

    float3 offset_x_;
    float3 offset_y_;
    float3 offset_z_;
    std::array<float3, num_of_components_> ov_;
};

#endif // _CUDA_ARRAY_GROUP_H_