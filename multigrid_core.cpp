#include "stdafx.h"
#include "multigrid_core.h"

#include "cuda/cuda_main.h"
#include "opengl/gl_texture.h"

MultigridCore::MultigridCore()
{

}

MultigridCore::~MultigridCore()
{

}

void MultigridCore::ProlongatePacked(std::shared_ptr<GLTexture> coarse,
                                     std::shared_ptr<GLTexture> fine)
{
    CudaMain::Instance()->ProlongatePacked(coarse, fine);
}

std::shared_ptr<GLTexture> MultigridCore::CreateTexture(int width, int height,
                                                        int depth,
                                                        GLuint internal_format,
                                                        GLenum format)
{
    std::shared_ptr<GLTexture> r(new GLTexture());
    r->Create(width, height, depth, internal_format, format);

    // Here we found something supernatural:
    //
    // If we don't register the texture immediately, we will never get a
    // chance to successfully register it. This unbelievable behavior had
    // tortured me a few hours, so I surrendered, and put the image register
    // code here.

    CudaMain::Instance()->RegisterGLImage(r);
    return r;
}
