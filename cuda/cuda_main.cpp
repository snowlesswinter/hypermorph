#include "stdafx.h"
#include "cuda_main.h"

#include "opengl/gl_texture.h"
#include "cuda_core.h"
#include "graphics_resource.h"

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

int CudaMain::RegisterGLImage(const std::shared_ptr<GLTexture>& texture)
{
    if (registerd_textures_.find(texture) != registerd_textures_.end())
        return 0;

    std::unique_ptr<GraphicsResource> g(new GraphicsResource(core_.get()));
    int r = core_->RegisterGLImage(*texture, g.get());
    if (r)
        return r;

    registerd_textures_.insert(std::make_pair(texture, std::move(g)));
    return 0;
}

void CudaMain::Absolute(const std::shared_ptr<GLTexture>& texture)
{
    auto i = registerd_textures_.find(texture);
    if (i == registerd_textures_.end())
        return;

    core_->Absolute(i->second.get(),i->first->buffer());
}
