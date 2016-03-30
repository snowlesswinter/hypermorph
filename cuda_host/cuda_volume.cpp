#include "stdafx.h"
#include "cuda_volume.h"

#include <cassert>

#include "cuda/cuda_core.h"
#include "vmath.hpp"

namespace
{
vmath::Vector3 FromIntValues(int x, int y, int z)
{
    return vmath::Vector3(static_cast<float>(x), static_cast<float>(y),
                          static_cast<float>(z));
}
} // Anonymous namespace.

CudaVolume::CudaVolume()
    : dev_array_(nullptr)
    , dev_mem_(nullptr)
    , width_(0)
    , height_(0)
    , depth_(0)
    , num_of_components_(0)
    , byte_width_(0)
{

}

CudaVolume::~CudaVolume()
{
    if (dev_array_) {
        CudaCore::FreeVolumeMemory(dev_array_);
        dev_array_ = nullptr;
    }

    if (dev_mem_) {
        CudaCore::FreeVolumeInPlaceMemory(dev_mem_);
        dev_mem_ = nullptr;
    }
}

void CudaVolume::Clear()
{
    if (dev_array_) {
        CudaCore::ClearVolume(
            dev_array_, vmath::Vector4(0.0f),
            vmath::Vector3(static_cast<float>(width_),
                           static_cast<float>(height_),
                           static_cast<float>(depth_)));
    }
}

bool CudaVolume::Create(int width, int height, int depth, int num_of_components,
                        int byte_width)
{
    assert(!dev_array_);
    if (dev_array_)
        return false;

    bool result = CudaCore::AllocVolumeMemory(
        &dev_array_, FromIntValues(width, height, depth), num_of_components,
        byte_width);
    if (result) {
        width_ = width;
        height_ = height;
        depth_ = depth;
        num_of_components_ = num_of_components;
        byte_width_ = byte_width;
    }
    return result;
}

bool CudaVolume::CreateInPlace(int width, int height, int depth,
                               int num_of_components, int byte_width)
{
    assert(!dev_mem_);
    if (dev_mem_)
        return false;

    bool result = CudaCore::AllocVolumeInPlaceMemory(
        &dev_mem_, FromIntValues(width, height, depth), num_of_components,
        byte_width);
    if (result) {
        width_ = width;
        height_ = height;
        depth_ = depth;
        num_of_components_ = num_of_components;
        byte_width_ = byte_width;
    }
    return result;
}
