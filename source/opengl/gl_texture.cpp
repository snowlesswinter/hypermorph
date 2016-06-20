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
#include "gl_texture.h"

#include <cassert>

#include "utility.h"

GLTexture::GLTexture()
    : frame_buffer_(0)
    , buffer_(0)
    , texture_handle_(0)
    , target_(0)
    , width_(0)
    , height_(0)
    , byte_width_(0)
    , internal_format_(0)
    , format_(0)
{

}

GLTexture::~GLTexture()
{
    if (texture_handle_) {
        glDeleteTextures(1, &texture_handle_);
        texture_handle_ = 0;
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
    glBindTexture(GL_TEXTURE_3D, texture_handle_);
}

void GLTexture::BindFrameBuffer() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_);
}

bool GLTexture::Create2dTexture(int width, int height, GLint internal_format,
                                GLenum format, int byte_width)
{
    assert(byte_width == 2 || byte_width == 4);
    if (byte_width != 2 && byte_width != 4)
        return false;

    GLuint texture_handle = 0;
    do {
        glGenTextures(1, &texture_handle);
        glBindTexture(GL_TEXTURE_2D, texture_handle);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0,
                     format, GL_HALF_FLOAT, 0);
        if (glGetError() != GL_NO_ERROR)
            break;

        glBindTexture(GL_TEXTURE_2D, 0);

        texture_handle_ = texture_handle;
        target_ = GL_TEXTURE_2D;
        width_ = width;
        height_ = height;
        byte_width_ = byte_width;
        internal_format_ = internal_format;
        format_ = format;
        return true;
    } while (0);

    if (texture_handle)
        glDeleteTextures(1, &texture_handle);

    return false;
}

bool GLTexture::Create3dTexture(int width, int height, int depth,
                                GLint internal_format, GLenum format,
                                int byte_width)
{
    assert(byte_width == 2 || byte_width == 4);
    if (byte_width != 2 && byte_width != 4)
        return false;

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
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             texture_handle, 0);
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
        texture_handle_ = texture_handle;
        target_ = GL_TEXTURE_3D;
        width_ = width;
        height_ = height;
        byte_width_ = byte_width;
        internal_format_ = internal_format;
        format_ = format;
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

void GLTexture::Unbind() const
{
    glBindTexture(GL_TEXTURE_3D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
