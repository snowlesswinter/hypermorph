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
#include "fluid_solver/fluid_buffer_owner.h"
#include "graphics_volume.h"
#include "opengl/gl_program.h"
#include "opengl/gl_surface.h"
#include "opengl/gl_volume.h"
#include "shader/raycast_shader.h"
#include "utility.h"
#include "third_party/glm/gtc/matrix_transform.hpp"

VolumeRenderer::VolumeRenderer()
    : Renderer()
    , inverse_rotation_proj_()
    , view_proj_()
    , perspective_proj_()
    , mvp_proj_()
    , eye_position_()
    , screen_size_()
    , focal_length_(1.0f)
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

void VolumeRenderer::OnViewportSized(const glm::ivec2& viewport_size)
{
    if (surf_ && Renderer::viewport_size() != viewport_size) {

        // A shitty bug in CUDA 7.5 that make the program go crash if I
        // unregister any opengl resources during debug/profiling.
        //
        // Be sure to check all the places that opengl-CUDA inter-operate.

        CudaMain::Instance()->UnregisterGLImage(surf_);
        surf_.reset();
    }

    Renderer::OnViewportSized(viewport_size);

    if (graphics_lib() == GRAPHICS_LIB_CUDA) {
        std::shared_ptr<GLSurface> s(new GLSurface());
        bool result = s->Create(viewport_size, GL_RGBA16F, GL_RGBA, 2);
        if (!result)
            return;

        CudaMain::Instance()->RegisterGLImage(s);
        surf_ = s;
    }
}

void VolumeRenderer::Render(FluidBufferOwner* buf_owner)
{
    if (graphics_lib() == GRAPHICS_LIB_CUDA) {
        CudaMain::Instance()->Raycast(
            surf_, buf_owner->GetDensityVolume()->cuda_volume(),
            inverse_rotation_proj_, eye_position_,
            FluidConfig::Instance()->light_color(),
            FluidConfig::Instance()->light_position(),
            FluidConfig::Instance()->light_intensity(), focal_length_,
            screen_size_, FluidConfig::Instance()->num_raycast_samples(),
            FluidConfig::Instance()->num_raycast_light_samples(),
            FluidConfig::Instance()->light_absorption(),
            FluidConfig::Instance()->raycast_density_factor(),
            FluidConfig::Instance()->raycast_occlusion_factor());
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, viewport_size().x, viewport_size().y);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

    if (graphics_lib() == GRAPHICS_LIB_CUDA) {
        RenderImplCuda();
    } else if (graphics_lib() == GRAPHICS_LIB_GLSL) {
        RenderImplGlsl(buf_owner->GetDensityVolume(), focal_length_);
    }

    glDisable(GL_BLEND);
}

void VolumeRenderer::Update(float zoom, const glm::mat4& rotation)
{
    // Volume rendering uses normalized coordinates.
    float max_length =
        std::max(std::max(grid_size().x, grid_size().y), grid_size().z);
    glm::vec3 half_size = grid_size() / max_length;

    float half_diag = glm::length(half_size);

    // Make sure the camera is able to capture every corner however the
    // rotation goes.
    float eye_dist     = half_diag / std::sin(fov() / 2.0f);
    float near_pos     = eye_dist - half_diag;
    float far_pos      = eye_dist + half_diag;
    float aspect_ratio =
        static_cast<float>(viewport_size().x) / viewport_size().y;
    if (aspect_ratio > 1.0f) {
        float ¦È = std::atan(std::tan(fov() / 2.0f) * aspect_ratio);
        eye_dist = half_diag / std::sin(¦È);
        near_pos = eye_dist - half_diag;
        far_pos = eye_dist + half_diag;
    }

    glm::vec3 eye(0.0f, 0.0f, eye_dist * (1.0f + zoom));
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::vec3 target(0.0f);
    view_proj_ = glm::lookAt(eye, target, up);
    perspective_proj_ = glm::perspective(fov(), aspect_ratio, near_pos,
                                         far_pos);
    mvp_proj_ = perspective_proj_ * view_proj_ * rotation;
    eye_position_ = (glm::inverse(rotation) * glm::vec4(eye, 1.0f)).xyz();
    inverse_rotation_proj_ = glm::inverse(rotation);

    focal_length_ = near_pos;
    screen_size_.y = focal_length_ * tan(fov() / 2.0f);
    screen_size_.x = screen_size_.y * aspect_ratio;
}

bool VolumeRenderer::Init(const glm::ivec2& viewport_size)
{
    set_viewport_size(viewport_size);

    if (graphics_lib() == GRAPHICS_LIB_CUDA) {
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

    render_texture_->Use();
    render_texture_->SetUniform("depth", 1.0f);
    render_texture_->SetUniform("sampler", 0);
    render_texture_->SetUniform("viewport_size",
                                static_cast<float>(surf_->width()),
                                static_cast<float>(surf_->height()));

    glBindTexture(GL_TEXTURE_2D, surf_->texture_handle());
    RenderMesh(*GetQuadMesh());

    glBindTexture(GL_TEXTURE_2D, 0);
    render_texture_->Unuse();
}

void VolumeRenderer::RenderImplGlsl(GraphicsVolume* density_volume,
                                    float focal_length)
{
    GLProgram* raycast = GetRaycastProgram();
    raycast->Use();
    raycast->SetUniform("ModelviewProjection", mvp_proj_);
    raycast->SetUniform("Modelview", inverse_rotation_proj_);
    raycast->SetUniform("ViewMatrix", view_proj_);
    raycast->SetUniform("ProjectionMatrix", perspective_proj_);
    raycast->SetUniform("RayOrigin", eye_position_);
    raycast->SetUniform("FocalLength", focal_length);
    raycast->SetUniform("WindowSize", static_cast<float>(viewport_size().x),
                        static_cast<float>(viewport_size().y));

    glBindBuffer(GL_ARRAY_BUFFER, GetCubeCenterVbo());
    glVertexAttribPointer(SlotPosition, 3, GL_FLOAT, GL_FALSE,
                          3 * sizeof(float), 0);
    glBindTexture(GL_TEXTURE_3D, density_volume->gl_volume()->texture_handle());

    glDrawArrays(GL_POINTS, 0, 1);

    glBindTexture(GL_TEXTURE_3D, 0);
    raycast->Unuse();
}
