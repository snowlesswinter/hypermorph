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

#ifndef _VOLUME_RENDERER_H_
#define _VOLUME_RENDERER_H_

#include <memory>

#include "renderer/renderer.h"

class FluidBufferOwner;
class GLProgram;
class GLSurface;
class GraphicsVolume;
struct MeshPod;
class VolumeRenderer : public Renderer
{
public:
    VolumeRenderer();
    virtual ~VolumeRenderer();

    // Overridden from Renderer:
    virtual void OnViewportSized(const glm::ivec2& viewport_size) override;
    virtual void Render(FluidBufferOwner* buf_owner) override;
    virtual void Update(float zoom, const glm::mat4& rotation) override;

    bool Init(const glm::ivec2& viewport_size);

private:
    uint32_t GetCubeCenterVbo();
    MeshPod* GetQuadMesh();
    GLProgram* GetRaycastProgram();

    void RenderImplCuda();
    void RenderImplGlsl(GraphicsVolume* density_volume, float focal_length);

    glm::mat4 model_view_proj_;
    glm::mat4 view_proj_;
    glm::mat4 perspective_proj_;
    glm::mat4 mvp_proj_;
    glm::vec3 eye_position_;
    float focal_length_;

    std::shared_ptr<GLSurface> surf_;
    std::shared_ptr<GLProgram> render_texture_;
    std::shared_ptr<GLProgram> raycast_;
    MeshPod* quad_mesh_;
    uint32_t cube_center_vbo_;
};

#endif // _VOLUME_RENDERER_H_