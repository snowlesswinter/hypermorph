//
// Fluid3d - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Fluid3d license (* see part 1 below)
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

#ifndef _GRAPHICS_LINEAR_MEM_H_
#define _GRAPHICS_LINEAR_MEM_H_

#include <cassert>
#include <memory>

#include <stdint.h>

#include "cuda_host/cuda_linear_mem.h"
#include "graphics_lib_enum.h"

template <typename T>
class GraphicsLinearMem
{
public:
    explicit GraphicsLinearMem(GraphicsLib lib)
        : graphics_lib_(lib)
        , cuda_linear_mem_()
    {
    }
    ~GraphicsLinearMem() {}

    bool Create(int num_of_elements)
    {
        if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
            std::shared_ptr<CudaLinearMem<T>> r =
                std::make_shared<CudaLinearMem<T>>();
            bool result = r->Create(num_of_elements);
            if (result) {
                cuda_linear_mem_ = r;
            }

            return result;
        }

        return false;
    }

    GraphicsLib graphics_lib() const { return graphics_lib_; }
    std::shared_ptr<CudaLinearMem<T>> cuda_linear_mem() const
    {
        assert(cuda_linear_mem_);
        return cuda_linear_mem_;
    }

private:
    GraphicsLib graphics_lib_;
    std::shared_ptr<CudaLinearMem<T>> cuda_linear_mem_;
};

typedef GraphicsLinearMem<uint8_t> GraphicsLinearMemU8;
typedef GraphicsLinearMem<uint16_t> GraphicsLinearMemU16;
typedef GraphicsLinearMem<uint32_t> GraphicsLinearMemU32;

#endif // _GRAPHICS_LINEAR_MEM_H_