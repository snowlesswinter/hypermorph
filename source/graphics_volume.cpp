#include "stdafx.h"
#include "graphics_volume.h"

#include <cassert>

#include "cuda_host/cuda_main.h"
#include "cuda_host/cuda_volume.h"
#include "opengl/gl_volume.h"
#include "utility.h"

GraphicsVolume::GraphicsVolume(GraphicsLib lib)
    : graphics_lib_(lib)
    , gl_volume_()
    , cuda_volume_()
{
}

GraphicsVolume::~GraphicsVolume()
{
}


void GraphicsVolume::Clear()
{
    if (gl_volume_)
        ClearSurface(gl_volume_.get(), 0.0f);

    if (cuda_volume_)
        cuda_volume_->Clear();
}

bool GraphicsVolume::Create(int width, int height, int depth,
                            int num_of_components, int byte_width)
{
    if (byte_width != 2 && byte_width != 4)
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
        GLuint internal_format = byte_width == 2 ? GL_RGBA16F : GL_RGBA32F;
        GLenum format = GL_RGBA;
        switch (num_of_components) {
            case 4:
                break;
            case 2:
                internal_format = byte_width == 2 ? GL_RG16F : GL_RG32F;
                format = GL_RG;
                break;
            case 1:
                internal_format = byte_width == 2 ? GL_R16F : GL_R32F;
                format = GL_RED;
                break;
            default:
                return false;
        }

        std::shared_ptr<GLVolume> r(new GLVolume());
        bool result = r->Create(width, height, depth, internal_format, format,
                                byte_width);
        if (result) {
            if (graphics_lib_ == GRAPHICS_LIB_CUDA_DIAGNOSIS) {
                // Here we found something supernatural:
                //
                // If we don't register the texture immediately, we will never
                // get a chance to successfully register it. This unbelievable
                // behavior had tortured me a few hours, so I surrendered, and
                // put the image register code here.
                CudaMain::Instance()->RegisterGLImage(r);
            }

            gl_volume_ = r;
        }

        return result;
    }

    return false;
}

std::shared_ptr<GLVolume> GraphicsVolume::gl_volume() const
{
    assert(gl_volume_);
    return gl_volume_;
}

std::shared_ptr<CudaVolume> GraphicsVolume::cuda_volume() const
{
    assert(cuda_volume_);
    return cuda_volume_;
}

int GraphicsVolume::GetWidth() const
{
    assert(gl_volume_ || cuda_volume_);
    if (!gl_volume_ && !cuda_volume_)
        return 0;

    if (gl_volume_)
        return gl_volume_->width();
    else
        return cuda_volume_->width();
}

int GraphicsVolume::GetHeight() const
{
    assert(gl_volume_ || cuda_volume_);
    if (!gl_volume_ && !cuda_volume_)
        return 0;

    if (gl_volume_)
        return gl_volume_->height();
    else
        return cuda_volume_->height();
}

int GraphicsVolume::GetDepth() const
{
    assert(gl_volume_ || cuda_volume_);
    if (!gl_volume_ && !cuda_volume_)
        return 0;

    if (gl_volume_)
        return gl_volume_->depth();
    else
        return cuda_volume_->depth();
}