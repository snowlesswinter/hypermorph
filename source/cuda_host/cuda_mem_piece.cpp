#include "stdafx.h"
#include "cuda_mem_piece.h"

#include <cassert>

#include "cuda/cuda_core.h"

CudaMemPiece::CudaMemPiece()
    : mem_(nullptr)
    , size_(0)
{
}

CudaMemPiece::~CudaMemPiece()
{
    if (mem_) {
        CudaCore::FreeMemPiece(mem_);
        mem_ = nullptr;
    }
}

bool CudaMemPiece::Create(int size)
{
    void* r = nullptr;
    if (CudaCore::AllocMemPiece(&r, size)) {
        mem_ = r;
        size_ = size;
        return true;
    }

    return false;
}
