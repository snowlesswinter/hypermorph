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

#ifndef _MEM_PIECE_H_
#define _MEM_PIECE_H_

namespace internal
{
class MemPieceBase
{
public:
    virtual ~MemPieceBase();

    int byte_width() const { return byte_width_; }

protected:
    MemPieceBase(void* mem, int byte_width);

    void CheckType(int byte_width) const;
    void* mem() const { return mem_; }

private:
    void* mem_;
    int byte_width_;
};
} // namespace internal

// A simple wrapping class holds type information. Not responsible for
// ownership management.
class MemPiece : public internal::MemPieceBase
{
public:
    MemPiece(void* mem, int byte_width);
    virtual ~MemPiece();

    template <typename T>
    T* AsType() const
    {
        CheckType(sizeof(T));
        return reinterpret_cast<T*>(mem());
    }
};

#endif // _MEM_PIECE_H_