#include "stdafx.h"
#include "graphics_volume.h"

#include <cassert>

#include "cuda_host/cuda_main.h"
#include "cuda_host/cuda_volume.h"
#include "opengl/gl_texture.h"
#include "utility.h"

GraphicsVolume::GraphicsVolume(GraphicsLib lib)
    : graphics_lib_(lib)
    , gl_texture_()
    , cuda_volume_()
{
}

GraphicsVolume::~GraphicsVolume()
{
}


void GraphicsVolume::Clear()
{
    if (gl_texture_)
        ClearSurface(gl_texture_.get(), 0.0f);

    if (cuda_volume_)
        cuda_volume_->Clear();
}

bool GraphicsVolume::Create(int width, int height, int depth,
                            int num_of_components, int byte_width)
{
    if (byte_width != 2)
        return false;   // Not supported yet.

    if (graphics_lib_ == GRAPHICS_LIB_CUDA) {
        std::shared_ptr<CudaVolume> r(new CudaVolume());
        bool result = r->Create(width, height, depth, num_of_components,
                                byte_width);
        if (result)
            cuda_volume_ = r;

        Clear(); // TODO
        return result;
    } else {
        GLuint internal_format = GL_RGBA16F;
        GLenum format = GL_RGBA;
        if (num_of_components != 4) {
            if (num_of_components != 1)
                return false;

            internal_format = GL_R16F;
            format = GL_RED;
        }

        std::shared_ptr<GLTexture> r(new GLTexture());
        bool result = r->Create(width, height, depth, internal_format, format);
        if (result) {
            if (graphics_lib_ == GRAPHICS_LIB_CUDA_DIAGNOSIS)
                CudaMain::Instance()->RegisterGLImage(r);

            gl_texture_ = r;
        }

        return result;
    }

    return false;
}

std::shared_ptr<GLTexture> GraphicsVolume::gl_texture() const
{
    assert(gl_texture_);
    return gl_texture_;
}

std::shared_ptr<CudaVolume> GraphicsVolume::cuda_volume() const
{
    assert(cuda_volume_);
    return cuda_volume_;
}

int GraphicsVolume::GetWidth() const
{
    assert(gl_texture_ || cuda_volume_);
    if (!gl_texture_ && !cuda_volume_)
        return 0;

    if (gl_texture_)
        return gl_texture_->width();
    else
        return cuda_volume_->width();
}

int GraphicsVolume::GetHeight() const
{
    assert(gl_texture_ || cuda_volume_);
    if (!gl_texture_ && !cuda_volume_)
        return 0;

    if (gl_texture_)
        return gl_texture_->height();
    else
        return cuda_volume_->height();
}

int GraphicsVolume::GetDepth() const
{
    assert(gl_texture_ || cuda_volume_);
    if (!gl_texture_ && !cuda_volume_)
        return 0;

    if (gl_texture_)
        return gl_texture_->depth();
    else
        return cuda_volume_->depth();
}