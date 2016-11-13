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
