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

#include "stdafx.h"
#include "volume_renderer.h"

#include "cuda_host/cuda_main.h"
#include "fluid_config.h"
#include "graphics_volume.h"
#include "opengl/gl_program.h"
#include "opengl/gl_surface.h"
#include "opengl/gl_volume.h"
#include "shader/raycast_shader.h"
#include "utility.h"
#include "third_party/glm/gtc/matrix_transform.hpp"

VolumeRenderer::VolumeRenderer()
    : graphics_lib_(GRAPHICS_LIB_CUDA)
    , viewport_size_(0)
    , model_view_()
    , view_()
    , projection_()
    , model_view_projection_()
    , eye_position_()
    , surf_()
    , render_texture_(new GLProgram())
    , raycast_()
    , quad_mesh_(nullptr)
    , cube_center_vbo_(0)
{
}

VolumeRenderer::~VolumeRenderer()
{
    if (cube_center_vbo_) {
        glDeleteBuffers(1, &cube_center_vbo_);
        cube_center_vbo_ = 0;
    }

    if (quad_mesh_) {
        delete quad_mesh_;
        quad_mesh_ = nullptr;
    }
}

bool VolumeRenderer::Init(const glm::ivec2& viewport_size)
{
    viewport_size_ = viewport_size;

    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        std::shared_ptr<GLSurface> s(new GLSurface());
        bool result = s->Create(viewport_size, GL_RGBA16F,
                                GL_RGBA, 2);
        if (!result)
            return false;

        std::shared_ptr<GLProgram> p(new GLProgram());
        if (!p->Load(RaycastShader::ApplyTextureVert(), "",
            RaycastShader::ApplyTextureFrag()))
            return false;

        CudaMain::Instance()->RegisterGLImage(s);
        surf_ = s;
        render_texture_ = p;
        return true;
    }

    return true;
}

void VolumeRenderer::OnViewportSized(const glm::ivec2& viewport_size)
{
    if (surf_ && viewport_size_ != viewport_size) {

        // A shitty bug in CUDA 7.5 that make the program go crash if I
        // unregister any opengl resources during debug/profiling.
        //
        // Be sure to check all the places that opengl-CUDA inter-operate.

        CudaMain::Instance()->UnregisterGLImage(surf_);
        surf_.reset();
    }

    viewport_size_ = viewport_size;
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        std::shared_ptr<GLSurface> s(new GLSurface());
        bool result = s->Create(viewport_size, GL_RGBA16F, GL_RGBA, 2);
        if (!result)
            return;

        CudaMain::Instance()->RegisterGLImage(s);
        surf_ = s;
    }
}

void VolumeRenderer::Render(std::shared_ptr<GraphicsVolume> density_volume,
                            float focal_length)
{
    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->Raycast(
            surf_, density_volume->cuda_volume(), model_view_, eye_position_,
            FluidConfig::Instance()->light_color(),
            FluidConfig::Instance()->light_position(),
            FluidConfig::Instance()->light_intensity(), focal_length,
            FluidConfig::Instance()->num_raycast_samples(),
            FluidConfig::Instance()->num_raycast_light_samples(),
            FluidConfig::Instance()->light_absorption(),
            FluidConfig::Instance()->raycast_density_factor(),
            FluidConfig::Instance()->raycast_occlusion_factor());
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, viewport_size_.x, viewport_size_.y);
    glEnable(GL_BLEND);

    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        RenderImplCuda();
    } else if (graphics_lib_ == GRAPHICS_LIB_GLSL) {
        RenderImplGlsl(density_volume, focal_length);
    }
}

void VolumeRenderer::Update(glm::vec3 eye_position, const glm::mat4& look_at,
                            const glm::mat4& rotation,
                            const glm::mat4& perspective)
{
    view_ = look_at;
    model_view_ = view_ * rotation;
    projection_ = perspective;
    model_view_projection_ = projection_ * model_view_;
    eye_position_ =
        (glm::transpose(model_view_) * glm::vec4(eye_position, 1.0f)).xyz();
}

uint32_t VolumeRenderer::GetCubeCenterVbo()
{
    if (!cube_center_vbo_) {
        cube_center_vbo_ = CreatePointVbo(0, 0, 0);
    }
    return cube_center_vbo_;
}

MeshPod* VolumeRenderer::GetQuadMesh()
{
    if (!quad_mesh_) {
        quad_mesh_ = new MeshPod(CreateQuadMesh(-1.0f, 1.0f, 1.0f, -1.0f));
    }

    return quad_mesh_;
}

GLProgram* VolumeRenderer::GetRaycastProgram()
{
    if (!raycast_) {
        raycast_.reset(new GLProgram());
        raycast_->Load(RaycastShader::Vertex(), RaycastShader::Geometry(),
                       RaycastShader::Fragment());
    }
    return raycast_.get();
}

void VolumeRenderer::RenderImplCuda()
{
    if (!surf_)
        return;

    glEnable(GL_BLEND);

    render_texture_->Use();
    render_texture_->SetUniform("depth", 1.0f);
    render_texture_->SetUniform("sampler", 0);
    render_texture_->SetUniform("viewport_size",
                                static_cast<float>(surf_->width()),
                                static_cast<float>(surf_->height()));

    glBindTexture(GL_TEXTURE_2D, surf_->texture_handle());
    RenderMesh(*GetQuadMesh());

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_BLEND);

    render_texture_->Unuse();
}

void VolumeRenderer::RenderImplGlsl(
    std::shared_ptr<GraphicsVolume> density_volume, float focal_length)
{
    GLProgram* raycast = GetRaycastProgram();
    raycast->Use();
    raycast->SetUniform("ModelviewProjection", model_view_projection_);
    raycast->SetUniform("Modelview", model_view_);
    raycast->SetUniform("ViewMatrix", view_);
    raycast->SetUniform("ProjectionMatrix", projection_);
    raycast->SetUniform("RayOrigin", eye_position_);
    raycast->SetUniform("FocalLength", focal_length);
    raycast->SetUniform("WindowSize", static_cast<float>(viewport_size_.x),
                        static_cast<float>(viewport_size_.y));

    glBindBuffer(GL_ARRAY_BUFFER, GetCubeCenterVbo());
    glVertexAttribPointer(SlotPosition, 3, GL_FLOAT, GL_FALSE,
                          3 * sizeof(float), 0);
    glBindTexture(GL_TEXTURE_3D, density_volume->gl_volume()->texture_handle());

    glDrawArrays(GL_POINTS, 0, 1);

    glBindTexture(GL_TEXTURE_3D, 0);
    raycast->Unuse();
}
