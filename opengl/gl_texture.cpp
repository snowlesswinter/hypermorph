#include "stdafx.h"
#include "gl_texture.h"

#include "utility.h"

GLTexture GLTexture::FromSurfacePod(const SurfacePod* sp)
{
    GLTexture r;
    r.frame_buffer_ = sp->FboHandle;
    r.handle_ = sp->ColorTexture;
    r.target_ = GL_TEXTURE_3D;
    r.width_ = sp->Width;
    r.height_ = sp->Height;
    r.depth_ = sp->Depth;

    return r;
}

GLTexture::GLTexture()
    : frame_buffer_(0)
    , buffer_(0)
    , handle_(0)
    , target_(0)
    , width_(0)
    , height_(0)
    , depth_(0)
{

}

GLTexture::~GLTexture()
{
    if (handle_) {
        glDeleteTextures(1, &handle_);
        handle_ = 0;
    }

    if (frame_buffer_) {
        glDeleteFramebuffers(1, &frame_buffer_);
        frame_buffer_ = 0;
    }
}

void GLTexture::Bind() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_);
    glBindRenderbuffer(GL_RENDERBUFFER, buffer_);
    glBindTexture(GL_TEXTURE_3D, handle_);
}

void GLTexture::BindFrameBuffer() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_);
}

bool GLTexture::Create(int width, int height, int depth, GLint internal_format,
                       GLenum format)
{
    GLuint frame_buffer = 0;
    GLuint color_buffer = 0;
    GLuint texture_handle = 0;
    do {
        glGenFramebuffers(1, &frame_buffer);
        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer);

        glGenTextures(1, &texture_handle);
        glBindTexture(GL_TEXTURE_3D, texture_handle);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage3D(GL_TEXTURE_3D, 0, internal_format, width, height, depth, 0,
                     format, GL_HALF_FLOAT, 0);
        if (glGetError() != GL_NO_ERROR)
            break;

        GLuint color_buffer;
        glGenRenderbuffers(1, &color_buffer);
        glBindRenderbuffer(GL_RENDERBUFFER, color_buffer);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture_handle, 0);
        if (glGetError() != GL_NO_ERROR)
            break;

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) !=
                GL_FRAMEBUFFER_COMPLETE)
            break;

        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glBindTexture(GL_TEXTURE_3D, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        frame_buffer_ = frame_buffer;
        buffer_ = color_buffer;
        handle_ = texture_handle;
        target_ = GL_TEXTURE_3D;
        width_ = width;
        height_ = height;
        depth_ = depth;
        return true;
    } while (0);

    if (texture_handle)
        glDeleteTextures(1, &texture_handle);

    if (color_buffer)
        glDeleteBuffers(1, &color_buffer);

    if (frame_buffer)
        glDeleteFramebuffers(1, &frame_buffer);

    return false;
}

void GLTexture::GetTexImage(GLenum format, GLenum type, void* buffer)
{
    Bind();
    glGetTexImage(GL_TEXTURE_3D, 0, format, type, buffer);
}

void GLTexture::Unbind() const
{
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}