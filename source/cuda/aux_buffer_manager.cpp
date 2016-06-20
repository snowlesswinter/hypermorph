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

#include "aux_buffer_manager.h"

#include <cassert>

#include "cuda_runtime.h"

struct AuxBufferManager::DevMemDeleter
{
    void operator()(void* p) {
        if (p)
            cudaFree(p);
    }
};

AuxBufferManager::AuxBufferManager()
    : free_()
    , in_use_()
{
}

AuxBufferManager::~AuxBufferManager()
{
}

void* AuxBufferManager::Allocate(int size)
{
    std::list<DevMemPtr>& ptr_list = free_[size];
    if (ptr_list.empty()) {
        void* r = nullptr;
        cudaError_t e = cudaMalloc(&r, size);
        if (e == cudaSuccess) {
            in_use_[r] = size;
            return r;
        }
    } else {
        DevMemPtr p = std::move(ptr_list.front());
        ptr_list.pop_front();
        in_use_[p.get()] = size;
        return p.release();
    }

    return nullptr;
}

void AuxBufferManager::Free(void* p)
{
    auto& i = in_use_.find(p);
    assert(i != in_use_.end());
    if (i == in_use_.end())
        return;

    int size = i->second;
    in_use_.erase(i);

    std::list<DevMemPtr>& ptr_list = free_[size];

    DevMemPtr w(p, DevMemDeleter());
    ptr_list.push_back(std::move(w));
}
