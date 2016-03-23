#include "stdafx.h"
#include "cuda_main.h"

#include <cassert>

#include "opengl/gl_texture.h"
#include "cuda_core.h"
#include "graphics_resource.h"

// =============================================================================
std::pair<GLuint, GraphicsResource*> GetProlongPBO(CudaCore* core, int n)
{
    static std::pair<GLuint, GraphicsResource*> pixel_buffer[10] = {};
//     if (pixel_buffer[n].first) {
//         delete pixel_buffer[n].second;
//         glDeleteBuffers(1, &pixel_buffer[n].first);
//     }

    if (!pixel_buffer[n].first)
    {
        int width = 128 / n;
        size_t size = width * width * width * 4 * 4;

        // create buffer object
        glGenBuffers(1, &(pixel_buffer[n].first));
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixel_buffer[n].first);
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

        pixel_buffer[n].second = new GraphicsResource(core);
        core->RegisterGLBuffer(pixel_buffer[n].first, pixel_buffer[n].second);
    }

    return pixel_buffer[n];
}
// =============================================================================

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
    auto pbo = GetProlongPBO(core_.get(), n);
    core_->ProlongatePacked(i->second.get(), j->second.get(), pbo.second,
                            coarse->width());

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo.first);

    glBindTexture(GL_TEXTURE_3D, fine->handle());
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0,
                    fine->width(), fine->height(), fine->depth(),
                    GL_RGBA, GL_FLOAT, nullptr);
    assert(glGetError() == 0);
    glBindTexture(GL_TEXTURE_3D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}