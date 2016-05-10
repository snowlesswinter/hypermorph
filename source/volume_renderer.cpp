#include "stdafx.h"
#include "volume_renderer.h"

#include "cuda_host/cuda_main.h"
#include "graphics_volume.h"
#include "opengl/gl_program.h"
#include "opengl/gl_surface.h"
#include "shader/raycast_shader.h"
#include "utility.h"

VolumeRenderer::VolumeRenderer()
    : surf_()
    , render_texture_(new GLProgram())
    , quad_mesh_(nullptr)
{

}

VolumeRenderer::~VolumeRenderer()
{
    if (quad_mesh_) {
        delete quad_mesh_;
        quad_mesh_ = nullptr;
    }
}

bool VolumeRenderer::Init(int viewport_width, int viewport_height)
{
    std::shared_ptr<GLSurface> s(new GLSurface());
    bool result = s->Create(viewport_width, viewport_height, GL_RGBA16F,
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

void VolumeRenderer::OnViewportSized(int viewport_width, int viewport_height)
{
    if (surf_ && surf_->width() != viewport_width &&
            surf_->height() != viewport_height) {

        // A shitty bug in CUDA 7.5 that make the program go crash if I
        // unregister any opengl resources during debug/profiling.
        //
        // Be sure to check all the places that opengl-CUDA inter-operate.

        CudaMain::Instance()->UnregisterGLImage(surf_);
        surf_.reset();
    }

    std::shared_ptr<GLSurface> s(new GLSurface());
    bool result = s->Create(viewport_width, viewport_height, GL_RGBA16F,
                            GL_RGBA, 2);
    if (!result)
        return;

    CudaMain::Instance()->RegisterGLImage(s);
    surf_ = s;
}

void VolumeRenderer::Raycast(std::shared_ptr<GraphicsVolume> density_volume,
                             const glm::mat4& model_view,
                             const glm::vec3& eye_pos, float focal_length)
{
    CudaMain::Instance()->Raycast(surf_, density_volume->cuda_volume(),
                                  model_view, eye_pos, focal_length);
}

void VolumeRenderer::Render()
{
    if (!surf_)
        return;

    glEnable(GL_BLEND);

    render_texture_->Use();
    render_texture_->SetUniform("depth", 1.0f);
    render_texture_->SetUniform("sampler", 0);
    render_texture_->SetUniform("viewport_size", static_cast<float>(surf_->width()),
                         static_cast<float>(surf_->height()));

    glBindTexture(GL_TEXTURE_2D, surf_->texture_handle());
    RenderMesh(*GetQuadMesh());

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_BLEND);

    render_texture_->Unuse();
}

MeshPod* VolumeRenderer::GetQuadMesh()
{
    if (!quad_mesh_) {
        quad_mesh_ = new MeshPod(CreateQuadMesh(-1.0f, 1.0f, 1.0f, -1.0f));
    }

    return quad_mesh_;
}
