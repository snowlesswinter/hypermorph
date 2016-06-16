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
