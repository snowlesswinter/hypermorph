#include "stdafx.h"
#include "cuda_main.h"

#include <cassert>

#include "opengl/gl_texture.h"
#include "cuda_core.h"
#include "graphics_resource.h"

// =============================================================================
std::pair<GLuint, GraphicsResource*> GetPBO(CudaCore* core, int n, int c)
{
    static std::pair<GLuint, GraphicsResource*> pixel_buffer[10][4] = {};
    std::pair<GLuint, GraphicsResource*>& ref = pixel_buffer[n][c];
    if (!ref.first)
    {
        int width = 128 / n;
        size_t size = width * width * width * c * 4;

        // create buffer object
        glGenBuffers(1, &(ref.first));
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, ref.first);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, size, 0, GL_DYNAMIC_DRAW);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Believe or not: using another CudaCore instance would cause
        //                 cudaGraphicsMapResources() crash or returning
        //                 unknown error!
        //                 This shit just tortured me for a whole day.
        //
        // So, don't treat .cu file as normal cpp files, CUDA must has done
        // something dirty with it. Just put as less as cpp code inside it
        // as possible.

        ref.second = new GraphicsResource(core);
        core->RegisterGLBuffer(ref.first, ref.second);
    }

    return ref;
}
// =============================================================================

namespace
{
void FlushPBO(GLuint pbo, GLuint format, GLTexture* dest)
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    glBindTexture(GL_TEXTURE_3D, dest->handle());
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0,
                    dest->width(), dest->height(), dest->depth(), format,
                    GL_FLOAT, nullptr);
    assert(glGetError() == 0);
    glBindTexture(GL_TEXTURE_3D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}
} // Anonymous namespace

CudaMain* CudaMain::Instance()
{
    static CudaMain* instance = nullptr;
    if (!instance) {
        instance = new CudaMain();
        instance->Init();
    }

    return instance;
}

CudaMain::CudaMain()
    : core_(new CudaCore())
    , registerd_textures_()
{

}

CudaMain::~CudaMain()
{
}

bool CudaMain::Init()
{
    return core_->Init();
}

int CudaMain::RegisterGLImage(std::shared_ptr<GLTexture> texture)
{
    if (registerd_textures_.find(texture) != registerd_textures_.end())
        return 0;

    std::unique_ptr<GraphicsResource> g(new GraphicsResource(core_.get()));
    int r = core_->RegisterGLImage(texture->handle(), texture->target(),
                                   g.get());
    if (r)
        return r;

    registerd_textures_.insert(std::make_pair(texture, std::move(g)));
    return 0;
}

void CudaMain::Absolute(std::shared_ptr<GLTexture> texture)
{
    auto i = registerd_textures_.find(texture);
    if (i == registerd_textures_.end())
        return;

    core_->Absolute(i->second.get(), i->first->handle());
}

void CudaMain::ProlongatePacked(std::shared_ptr<GLTexture> coarse,
                                std::shared_ptr<GLTexture> fine)
{
    auto i = registerd_textures_.find(coarse);
    auto j = registerd_textures_.find(fine);
    assert(i != registerd_textures_.end() && j != registerd_textures_.end());
    if (i == registerd_textures_.end() || j == registerd_textures_.end())
        return;

    int n = 128 / fine->width();
    auto pbo = GetPBO(core_.get(), n, 4);
    core_->ProlongatePacked(i->second.get(), j->second.get(), pbo.second,
                            coarse->width());

    FlushPBO(pbo.first, GL_RGBA, fine.get());
}

void CudaMain::AdvectVelocity(std::shared_ptr<GLTexture> velocity,
                              std::shared_ptr<GLTexture> dest, float time_step,
                              float dissipation)
{
    auto i = registerd_textures_.find(velocity);
    if (i == registerd_textures_.end())
        return;

    int n = 128 / velocity->width();
    auto pbo = GetPBO(core_.get(), n, 4);
    core_->AdvectVelocity(i->second.get(), pbo.second, time_step, dissipation,
                          velocity->width());

    FlushPBO(pbo.first, GL_RGBA, dest.get());
}

void CudaMain::Advect(std::shared_ptr<GLTexture> velocity,
                      std::shared_ptr<GLTexture> source,
                      std::shared_ptr<GLTexture> dest, float time_step,
                      float dissipation)
{
    auto i = registerd_textures_.find(velocity);
    auto j = registerd_textures_.find(source);
    assert(i != registerd_textures_.end() && j != registerd_textures_.end());
    if (i == registerd_textures_.end() || j == registerd_textures_.end())
        return;

    int n = 128 / velocity->width();
    auto pbo = GetPBO(core_.get(), n, 1);
    core_->Advect(i->second.get(), j->second.get(), pbo.second, time_step,
                  dissipation, velocity->width());

    FlushPBO(pbo.first, GL_RED, dest.get());
}

void CudaMain::RoundPassed(int round)
{
    core_->RoundPassed(round);
}
