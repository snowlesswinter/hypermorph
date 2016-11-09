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
#include "blob_renderer.h"

#include "cuda_host/cuda_main.h"
#include "fluid_config.h"
#include "graphics_volume.h"
#include "opengl/gl_program.h"
#include "utility.h"
#include "third_party/glm/gtc/matrix_transform.hpp"

namespace
{
#define STRINGIFY(A) #A

const char* blob_vs = STRINGIFY(
    
uniform mat4 u_mv_matrix;
uniform float point_scale;

in vec4 in_position;
out float vs_blob_size;

void main()
{
    vec3 pos_in_eye_coord = vec3(u_mv_matrix * in_position);
    float dist = length(pos_in_eye_coord);
    vs_blob_size = point_scale / dist;

    gl_Position = in_position;
}
);

const char* blob_gs = STRINGIFY(

layout(points)           in;
layout(triangle_strip)   out;
layout(max_vertices = 4) out;

uniform mat4 u_mvp_matrix;

in float vs_blob_size[];
out vec2 gs_tex_coord;
out float gs_z;

void main()
{
    gs_z = gl_in[0].gl_Position.z;
    vec4 pos = u_mvp_matrix * gl_in[0].gl_Position;
    float blob_size = vs_blob_size[0];

    gl_Position = pos + vec4(blob_size, -blob_size, 0, 0);
    gs_tex_coord = vec2(1.0f, 0.0f);
    EmitVertex();

    gl_Position = pos + vec4(blob_size, blob_size, 0, 0);
    gs_tex_coord = vec2(1.0f, 1.0f);
    EmitVertex();

    gl_Position = pos + vec4(-blob_size, -blob_size, 0, 0);
    gs_tex_coord = vec2(0.0f, 0.0f);
    EmitVertex();

    gl_Position = pos + vec4(-blob_size, blob_size, 0, 0);
    gs_tex_coord = vec2(0.0f, 1.0f);
    EmitVertex();

    EndPrimitive();
}
);

const char *blob_fs = STRINGIFY(

uniform mat4 u_mv_matrix;
uniform vec3 u_light_dir;

in vec2 gs_tex_coord;
in float gs_z;
out vec4 out_color;

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

    hsv_color.x += (gs_z / 96.0f) * (130.0f / 360.0f);
    out_color = vec4(hsv2rgb(hsv_color), 1.0f);
}
);

} // Anonymous namespace.

BlobRenderer::BlobRenderer()
    : graphics_lib_(GRAPHICS_LIB_CUDA)
    , viewport_size_(0)
    , particle_count_(0)
    , fov_(1.0f)
    , model_view_()
    , perspective_projection_()
    , point_scale_(1.0f)
    , prog_()
    , point_vbo_(0)
{
}

BlobRenderer::~BlobRenderer()
{
    if (point_vbo_) {
        glDeleteBuffers(1, reinterpret_cast<GLuint*>(&point_vbo_));
        point_vbo_ = 0;
    }
}

bool BlobRenderer::Init(int particle_count, const glm::ivec2& viewport_size)
{
    particle_count_ = particle_count;
    viewport_size_ = viewport_size;

    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        point_vbo_ = CreateDynamicVbo(particle_count);
        CudaMain::Instance()->RegisterGLBuffer(point_vbo_);
        return true;
    }

    return true;
}

void BlobRenderer::OnViewportSized(const glm::ivec2& viewport_size)
{
    viewport_size_ = viewport_size;
}

void BlobRenderer::Render()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, viewport_size_.x, viewport_size_.y);
    glClearColor(0.7f, 0.7f, 0.7f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glDisable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_CULL_FACE);
    glDisable(GL_ALPHA_TEST);

    glDepthMask(GL_TRUE);
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);

    GLProgram* blob_program = GetRenderProgram();
    blob_program->Use();
    blob_program->SetUniform("u_mv_matrix", model_view_);
    blob_program->SetUniform("u_mvp_matrix",
                             perspective_projection_ * model_view_);
    blob_program->SetUniform("point_scale", point_scale_);

    glBindBuffer(GL_ARRAY_BUFFER, point_vbo_);
    glVertexPointer(3, GL_HALF_FLOAT, 0, 0);
    glVertexAttribPointer(SlotPosition, 3, GL_HALF_FLOAT, GL_FALSE,
                          0, nullptr);
    glEnableVertexAttribArray(SlotPosition);
    glEnableClientState(GL_VERTEX_ARRAY);

    glDrawArrays(GL_POINTS, 0, particle_count_);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);

    blob_program->Unuse();
}

void BlobRenderer::Update(const glm::vec3& eye_position,
                          const glm::mat4& rotation,
                          const glm::mat4& perspective)
{
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::vec3 target(0.0f);
    glm::mat4 look_at = glm::lookAt(eye_position, target, up);
    model_view_ = look_at * rotation;
    perspective_projection_ = perspective;
    point_scale_ = 100.0f / tanf(fov_ / 2.0f);
}

GLProgram* BlobRenderer::GetRenderProgram()
{
    if (!prog_) {
        prog_.reset(new GLProgram());
        prog_->Load(blob_vs, blob_gs, blob_fs);
    }
    return prog_.get();
}