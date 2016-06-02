#include "stdafx.h"
#include "cuda_volume.h"

#include <cassert>

#include "cuda/cuda_core.h"
#include "cuda_host/cuda_main.h"
#include "third_party/glm/vec3.hpp"
#include "third_party/glm/vec4.hpp"

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
        CudaMain::Instance()->ClearVolume(this, glm::vec4(0.0f),
                                          glm::ivec3(width_, height_, depth_));
    }
}

bool CudaVolume::Create(int width, int height, int depth, int num_of_components,
                        int byte_width)
{
    assert(!dev_array_);
    if (dev_array_)
        return false;

    bool result = CudaCore::AllocVolumeMemory(
        &dev_array_, glm::ivec3(width, height, depth), num_of_components,
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
        &dev_mem_, glm::ivec3(width, height, depth), num_of_components,
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

bool CudaVolume::HasSameProperties(const CudaVolume& other) const
{
    return width_ == other.width_ && height_ == other.height_ &&
        depth_ == other.depth_ &&
        num_of_components_ == other.num_of_components_ &&
        byte_width_ == other.byte_width_;
}

glm::ivec3 CudaVolume::size() const
{
    return glm::ivec3(width_, height_, depth_);
}
