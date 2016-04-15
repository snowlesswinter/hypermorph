#include "stdafx.h"
#include "volume_renderer.h"

#include "cuda_host/cuda_main.h"
#include "graphics_volume.h"
#include "opengl/gl_program.h"
#include "opengl/gl_surface.h"
#include "utility.h"

#include "shader/overlay_shader.h" // TODO

VolumeRenderer::VolumeRenderer()
    : surf_()
    , program_(new GLProgram())
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

    CudaMain::Instance()->RegisterGLImage(s);

    std::shared_ptr<GLProgram> p(new GLProgram());
    if (!p->Load(OverlayShader::Vertex(), "",  OverlayShader::Fragment()))
        return false;

    surf_ = s;
    program_ = p;
    return true;
}

void VolumeRenderer::Render(std::shared_ptr<GraphicsVolume> density_volume)
{
    CudaMain::Instance()->Raycast(surf_, std::shared_ptr<CudaVolume>());
    RenderSurface();
}

void VolumeRenderer::RenderSurface()
{
    glEnable(GL_BLEND);

    program_->Use();
    program_->SetUniform("depth", 1.0f);
    program_->SetUniform("sampler", 0);
    program_->SetUniform("viewport_size", static_cast<float>(ViewportWidth),
                         static_cast<float>(ViewportWidth));

    glBindTexture(GL_TEXTURE_2D, surf_->texture_handle());
    RenderMesh(*GetQuadMesh());

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_BLEND);

    program_->Unuse();
}

MeshPod* VolumeRenderer::GetQuadMesh()
{
    if (!quad_mesh_) {
        quad_mesh_ = new MeshPod(CreateQuadMesh(-1.0f, 1.0f, 1.0f, -1.0f));
    }

    return quad_mesh_;
}
