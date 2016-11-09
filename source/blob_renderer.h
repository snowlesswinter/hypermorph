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

#include "graphics_lib_enum.h"
#include "third_party/glm/mat4x4.hpp"
#include "third_party/glm/vec2.hpp"
#include "third_party/glm/vec3.hpp"

class GLProgram;
class GLSurface;
class GraphicsVolume;
class BlobRenderer
{
public:
    BlobRenderer();
    ~BlobRenderer();

    bool Init(int particle_count, const glm::ivec2& viewport_size);
    void OnViewportSized(const glm::ivec2& viewport_size);
    void Render();
    void Update(const glm::vec3& eye_position, const glm::mat4& rotation,
                const glm::mat4& perspective);

    void set_graphics_lib(GraphicsLib lib) { graphics_lib_ = lib; }
    void set_fov(float fov) { fov_ = fov; }

    // FIXME:
    uint32_t point_vbo() const { return point_vbo_; }

private:
    GLProgram* GetRenderProgram();

    GraphicsLib graphics_lib_;
    glm::ivec2 viewport_size_;

    int particle_count_;
    float fov_;
    glm::mat4 model_view_;
    glm::mat4 perspective_projection_;
    float point_scale_;

    std::shared_ptr<GLProgram> prog_;
    uint32_t point_vbo_;
};

#endif // _BLOB_RENDERER_H_