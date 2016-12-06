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
#include "blob_renderer.h"

#include <cassert>

#include "cuda_host/cuda_main.h"
#include "particle_buffer_owner.h"
#include "graphics_linear_mem.h"
#include "graphics_mem_piece.h"
#include "graphics_volume.h"
#include "opengl/gl_program.h"
#include "utility.h"
#include "third_party/glm/gtc/matrix_transform.hpp"

namespace
{
#define STRINGIFY(A) #A

const char* blob_vs = STRINGIFY(
    
uniform mat4 u_mv_matrix;
uniform float u_point_scale;
uniform float u_v_max;
uniform float u_render_temp;

in vec4 in_position;
in float in_extra;
out float vs_blob_size;
out float vs_v;

void main()
{
    vec3 pos_in_eye_coord = vec3(u_mv_matrix * in_position);
    float dist = length(pos_in_eye_coord);
    vs_blob_size = u_point_scale / dist;

    gl_Position = in_position;
    vs_v = u_v_max - in_position.z;
    if (u_render_temp > 0.0f)
        vs_v = in_extra;
}
);

const char* blob_gs = STRINGIFY(

layout(points)           in;
layout(triangle_strip)   out;
layout(max_vertices = 4) out;

uniform mat4 u_mvp_matrix;
uniform float u_inv_aspect_ratio;

in float vs_blob_size[];
in float vs_v[];
out vec2 gs_tex_coord;
out float gs_v;

void main()
{
    gs_v = vs_v[0];
    vec4 pos = u_mvp_matrix * gl_in[0].gl_Position;
    float blob_size = vs_blob_size[0];

    gl_Position = pos + vec4(blob_size * u_inv_aspect_ratio, -blob_size, 0, 0);
    gs_tex_coord = vec2(1.0f, 0.0f);
    EmitVertex();

    gl_Position = pos + vec4(blob_size * u_inv_aspect_ratio, blob_size, 0, 0);
    gs_tex_coord = vec2(1.0f, 1.0f);
    EmitVertex();

    gl_Position = pos + vec4(-blob_size * u_inv_aspect_ratio, -blob_size, 0, 0);
    gs_tex_coord = vec2(0.0f, 0.0f);
    EmitVertex();

    gl_Position = pos + vec4(-blob_size * u_inv_aspect_ratio, blob_size, 0, 0);
    gs_tex_coord = vec2(0.0f, 1.0f);
    EmitVertex();

    EndPrimitive();
}
);

const char *blob_fs = STRINGIFY(

uniform mat4 u_mv_matrix;
uniform vec3 u_light_dir;
uniform float u_v_max;

in vec2 gs_tex_coord;
in float gs_v;
out vec4 out_color;

// HSV <-> RGB conversion from http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = c.g < c.b ? vec4(c.bg, K.wz) : vec4(c.gb, K.xy);
    vec4 q = c.r < p.x ? vec4(p.xyw, c.r) : vec4(c.r, p.yzx);

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main()
{
    const vec3 light_dir = vec3(0.3, 0.75, 1.0);

    vec3 blob_normal;
    blob_normal.xy = gs_tex_coord.xy * 2.0f - 1.0f;
    float mag = dot(blob_normal.xy, blob_normal.xy);
    if (mag > 1.0f)
        discard;

    blob_normal.z = sqrt(1.0f - mag);

    float diffuse = max(0.0f, dot(light_dir, blob_normal));

    // RGB: 74.0f, 222.0f, 247.0f
    vec3 hsv_color = vec3(189.0f / 360.0f, 0.7f, 0.97f);

    hsv_color.x += (1.0f - gs_tex_coord.y) * (13.0f / 360.0f);
    hsv_color.y += (1.0f - diffuse) * 0.09f;
    hsv_color.z -= (1.0f - diffuse) * 0.4f;

    hsv_color.x += gs_v / u_v_max * (130.0f / 360.0f);
    out_color = vec4(hsv2rgb(hsv_color), 1.0f);
}
);

const int kPointFieldCount = 3;
const int kExtraFieldCount = 1;
const float kRenderTemperature = -1.0f;
} // Anonymous namespace.

BlobRenderer::BlobRenderer()
    : Renderer()
    , particle_count_(0)
    , model_view_proj_()
    , perspective_proj_()
    , point_scale_(1.0f)
    , prog_()
    , point_vbo_(0)
    , extra_vbo_(0)
    , crit_density_(0.1f)
    , impulse_temperature_(1.0f)
{
}

BlobRenderer::~BlobRenderer()
{
    if (point_vbo_) {
        CudaMain::Instance()->UnregisterGBuffer(point_vbo_);
        glDeleteBuffers(1, &point_vbo_);
        point_vbo_ = 0;
    }

    if (extra_vbo_) {
        CudaMain::Instance()->UnregisterGBuffer(extra_vbo_);
        glDeleteBuffers(1, &extra_vbo_);
        extra_vbo_ = 0;
    }
}

void BlobRenderer::Render(FluidFieldOwner* field_owner,
                          ParticleBufferOwner* buf_owner)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, viewport_size().x, viewport_size().y);
    glClearColor(0.7f, 0.7f, 0.7f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!buf_owner || !CopyToVbo(buf_owner))
        return;

    glDisable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_CULL_FACE);
    glDisable(GL_ALPHA_TEST);

    glDepthMask(GL_TRUE);
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);

    GLProgram* blob_program = GetRenderProgram();
    blob_program->Use();
    blob_program->SetUniform("u_mv_matrix", model_view_proj_);
    blob_program->SetUniform("u_mvp_matrix",
                             perspective_proj_ * model_view_proj_);
    blob_program->SetUniform("u_point_scale", point_scale_);
    blob_program->SetUniform(
        "u_inv_aspect_ratio",
        static_cast<float>(viewport_size().y) / viewport_size().x);
    blob_program->SetUniform(
        "u_v_max", kRenderTemperature > 0.0f ?
            impulse_temperature_ : grid_size().z);
    blob_program->SetUniform("u_render_temp", kRenderTemperature);

    glBindBuffer(GL_ARRAY_BUFFER, point_vbo_);
    glVertexAttribPointer(SlotPosition, kPointFieldCount, GL_HALF_FLOAT,
                          GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(SlotPosition);

    glBindBuffer(GL_ARRAY_BUFFER, extra_vbo_);
    glVertexAttribPointer(SlotTexCoord, kExtraFieldCount, GL_HALF_FLOAT,
                          GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(SlotTexCoord);

    glEnableClientState(GL_VERTEX_ARRAY);

    glDrawArrays(GL_POINTS, 0, particle_count_);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);

    blob_program->Unuse();
    glDisable(GL_DEPTH_TEST);
}

void BlobRenderer::Update(float zoom, const glm::mat4& rotation)
{
    glm::vec3 half_size = grid_size() * 0.5f;
    glm::mat4 translate = glm::translate(
        glm::mat4(), glm::vec3(-half_size.x, -half_size.y, -half_size.z));

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
        far_pos  = eye_dist + half_diag;
    }

    glm::vec3 eye(0.0f, 0.0f, eye_dist * (1.0f + zoom));
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::vec3 target(0.0f);
    glm::mat4 look_at = glm::lookAt(eye, target, up);

    model_view_proj_ = look_at * rotation * translate;
    perspective_proj_ = glm::perspective(fov(), aspect_ratio, near_pos,
                                         far_pos + half_diag * 2.0f);

    float inv_tan_fov = std::tan(fov() / 2.0f);
    point_scale_ = grid_size().y / inv_tan_fov / inv_tan_fov * 0.25f;
}

bool BlobRenderer::Init(int particle_count, const glm::ivec2& viewport_size)
{
    particle_count_ = particle_count;
    OnViewportSized(viewport_size);

    if (graphics_lib() == GRAPHICS_LIB_CUDA) {
        assert(!point_vbo_);

        GLuint vbo = CreateDynamicVbo(particle_count, kPointFieldCount);
        assert(vbo);
        if (!vbo)
            return false;

        if (CudaMain::Instance()->RegisterGLBuffer(vbo)) {
            glDeleteBuffers(1, &vbo);
            return false;
        }

        GLuint vbo2 = CreateDynamicVbo(particle_count, kExtraFieldCount);
        assert(vbo2);
        if (!vbo2) {
            glDeleteBuffers(1, &vbo);
            return false;
        }

        if (CudaMain::Instance()->RegisterGLBuffer(vbo2)) {
            glDeleteBuffers(1, &vbo);
            glDeleteBuffers(1, &vbo2);
            return false;
        }

        point_vbo_ = vbo;
        extra_vbo_ = vbo2;
        return true;
    }

    return true;
}

bool BlobRenderer::CopyToVbo(ParticleBufferOwner* buf_owner)
{
    if (!buf_owner->GetParticlePosXField() ||
            !buf_owner->GetParticlePosYField() ||
            !buf_owner->GetParticlePosZField() ||
            !buf_owner->GetParticleDensityField())
        return false;

    auto mem_piece = buf_owner->GetActiveParticleCountMemPiece();
    std::shared_ptr<CudaMemPiece> count =
        mem_piece ? mem_piece->cuda_mem_piece() : nullptr;
    auto mem = buf_owner->GetParticleTemperatureField();
    std::shared_ptr<CudaLinearMemU16> temp_field =
        mem ? mem->cuda_linear_mem() : nullptr;
    return CudaMain::Instance()->CopyToVbo(
        point_vbo_, extra_vbo_,
        buf_owner->GetParticlePosXField()->cuda_linear_mem(),
        buf_owner->GetParticlePosYField()->cuda_linear_mem(),
        buf_owner->GetParticlePosZField()->cuda_linear_mem(),
        buf_owner->GetParticleDensityField()->cuda_linear_mem(),
        temp_field, count, crit_density_, particle_count_);
}

GLProgram* BlobRenderer::GetRenderProgram()
{
    if (!prog_) {
        prog_.reset(new GLProgram());
        prog_->Load(blob_vs, blob_gs, blob_fs);
    }
    return prog_.get();
}
