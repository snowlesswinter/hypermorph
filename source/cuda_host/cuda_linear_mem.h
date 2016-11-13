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

#ifndef _CUDA_LINEAR_MEM_H_
#define _CUDA_LINEAR_MEM_H_

#include <memory>

#include <stdint.h>

namespace detail
{
class CudaLinearMemBase
{
public:
    CudaLinearMemBase();
    virtual ~CudaLinearMemBase();

    void* Create(int num_of_elements, int byte_width);
    void Destroy(void* mem);
};
}

// =============================================================================

template <typename T>
class CudaLinearMem : public detail::CudaLinearMemBase
{
public:
    CudaLinearMem()
        : mem_(nullptr)
        , num_of_elements_(0)
    {
    }

    virtual ~CudaLinearMem()
    {
        Destroy(mem_);
    }

    bool Create(int num_of_elements)
    {
        mem_ = reinterpret_cast<T*>(
            CudaLinearMemBase::Create(num_of_elements, sizeof(T)));
        if (mem_) {
            num_of_elements_ = num_of_elements;
            return true;
        }

        return false;
    }

    T* mem() const { return mem_; }

private:
    CudaLinearMem(const CudaLinearMem&);
    void operator=(const CudaLinearMem&);

    T* mem_;
    int num_of_elements_;
};

typedef CudaLinearMem<uint8_t> CudaLinearMemU8;
typedef CudaLinearMem<uint16_t> CudaLinearMemU16;
typedef CudaLinearMem<uint32_t> CudaLinearMemU32;

#endif // _CUDA_LINEAR_MEM_H_