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

#ifndef _RENDERER_H_
#define _RENDERER_H_

#include "graphics_lib_enum.h"
#include "third_party/glm/mat4x4.hpp"
#include "third_party/glm/vec2.hpp"
#include "third_party/glm/vec3.hpp"

class FluidBufferOwner;
class Renderer
{
public:
    virtual ~Renderer();

    virtual void OnViewportSized(const glm::ivec2& viewport_size);
    virtual void Render(FluidBufferOwner* buf_owner) = 0;
    virtual void Update(float zoom, const glm::mat4& rotation) = 0;

    void set_graphics_lib(GraphicsLib lib) { graphics_lib_ = lib; }
    void set_grid_size(const glm::vec3& grid_size) { grid_size_ = grid_size; }
    void set_fov(float fov) { fov_ = fov; }

protected:
    explicit Renderer();

    GraphicsLib graphics_lib() const { return graphics_lib_; }
    const glm::ivec2& viewport_size() const { return viewport_size_; }
    const glm::vec3& grid_size() const { return grid_size_; }
    float fov() const { return fov_; }

    void set_viewport_size(const glm::ivec2& viewport_size)
    {
        viewport_size_ = viewport_size;
    }

private:
    GraphicsLib graphics_lib_;
    glm::ivec2 viewport_size_;
    glm::vec3 grid_size_;
    float fov_;
};

#endif // _RENDERER_H_