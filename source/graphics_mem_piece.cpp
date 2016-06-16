#include "stdafx.h"
#include "graphics_mem_piece.h"

#include <cassert>

#include "cuda_host/cuda_main.h"
#include "cuda_host/cuda_mem_piece.h"

GraphicsMemPiece::GraphicsMemPiece(GraphicsLib lib)
    : graphics_lib_(lib)
    , cuda_mem_piece_()
{
}

GraphicsMemPiece::~GraphicsMemPiece()
{
}

bool GraphicsMemPiece::Create(int size)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        std::shared_ptr<CudaMemPiece> r = std::make_shared<CudaMemPiece>();
        bool result = r->Create(size);
        if (result) {
            cuda_mem_piece_ = r;
        }

        return result;
    }

    return false;
}

std::shared_ptr<CudaMemPiece> GraphicsMemPiece::cuda_mem_piece() const
{
    assert(cuda_mem_piece_);
    return cuda_mem_piece_;
}
