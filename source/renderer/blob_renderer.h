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

#ifndef _BLOB_RENDERER_H_
#define _BLOB_RENDERER_H_

#include <memory>

#include "renderer/renderer.h"

class GLProgram;
class BlobRenderer : public Renderer
{
public:
    BlobRenderer();
    virtual ~BlobRenderer();

    // Overridden from Renderer:
    virtual void Render(FluidBufferOwner* buf_owner) override;
    virtual void Update(float zoom, const glm::mat4& rotation) override;

    bool Init(int particle_count, const glm::ivec2& viewport_size);

    void set_crit_density(float crit_density) { crit_density_ = crit_density; }

private:
    void CopyToVbo(FluidBufferOwner* buf_owner);
    GLProgram* GetRenderProgram();

    int particle_count_;
    glm::mat4 model_view_proj_;
    glm::mat4 perspective_proj_;
    float point_scale_;

    std::shared_ptr<GLProgram> prog_;
    uint32_t point_vbo_;
    float crit_density_;
};

#endif // _BLOB_RENDERER_H_