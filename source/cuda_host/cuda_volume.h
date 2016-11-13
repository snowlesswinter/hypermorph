//
// Hypermorph - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Hypermorph license (* see part 1 below)
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. Acknowledgement of the
//    original author is required if you publish this in a paper, or use it
//    in a product.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

#ifndef _CUDA_VOLUME_H_
#define _CUDA_VOLUME_H_

#include <memory>

#include "third_party/glm/fwd.hpp"

struct cudaArray;
struct cudaPitchedPtr;
class CudaVolume
{
public:
    CudaVolume();
    ~CudaVolume();

    void Clear();
    bool Create(int width, int height, int depth, int num_of_components,
                int byte_width);
    bool CreateInPlace(int width, int height, int depth, int num_of_components,
                       int byte_width);
    bool HasSameProperties(const CudaVolume& other) const;

    cudaArray* dev_array() const { return dev_array_; }
    cudaPitchedPtr* dev_mem() const { return dev_mem_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int depth() const { return depth_; }
    int num_of_components() const { return num_of_components_; }
    int byte_width() const { return byte_width_; }
    glm::ivec3 size() const;

private:
    cudaArray* dev_array_;
    cudaPitchedPtr* dev_mem_;
    int width_;
    int height_;
    int depth_;
    int num_of_components_;
    int byte_width_;
};

#endif // _CUDA_VOLUME_H_