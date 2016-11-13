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

#ifndef _GRAPHICS_VOLUME_H_
#define _GRAPHICS_VOLUME_H_

#include <memory>

#include "graphics_lib_enum.h"

class CudaVolume;
class GLVolume;
class GraphicsVolume
{
public:
    explicit GraphicsVolume(GraphicsLib lib);
    ~GraphicsVolume();

    void Clear();
    bool Create(int width, int height, int depth, int num_of_components,
                int byte_width, int border);
    bool HasSameProperties(const GraphicsVolume& other) const;
    void Swap(GraphicsVolume& other);

    GraphicsLib graphics_lib() const { return graphics_lib_; }
    int GetWidth() const;
    int GetHeight() const;
    int GetDepth() const;
    int GetByteWidth() const;

    std::shared_ptr<GLVolume> gl_volume() const;
    std::shared_ptr<CudaVolume> cuda_volume() const;

private:
    GraphicsLib graphics_lib_;
    std::shared_ptr<GLVolume> gl_volume_;
    std::shared_ptr<CudaVolume> cuda_volume_;
    int border_;
};

#endif // _GRAPHICS_VOLUME_H_