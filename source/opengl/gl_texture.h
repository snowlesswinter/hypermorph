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

#ifndef _GL_TEXTURE_H_
#define _GL_TEXTURE_H_

#include <string>

#include "third_party/opengl/glew.h"

class GLTexture
{
public:
    GLTexture();
    virtual ~GLTexture();

    virtual void GetTexImage(void* buffer) = 0;
    virtual void SetTexImage(const void* buffer) = 0;

    void Bind() const;
    void BindFrameBuffer() const;
    void Unbind() const;

    GLuint frame_buffer() const { return frame_buffer_; }
    GLuint buffer() const { return buffer_; }
    GLuint texture_handle() const { return texture_handle_; }
    GLenum target() const { return target_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int byte_width() const { return byte_width_; }

protected:
    bool Create2dTexture(int width, int height, GLint internal_format,
                         GLenum format, int byte_width);
    bool Create3dTexture(int width, int height, int depth, GLint internal_format,
                         GLenum format, int byte_width);

    GLint internal_format() const { return internal_format_; }
    GLenum format() const { return format_; }

private:
    GLuint frame_buffer_;
    GLuint buffer_;
    GLuint texture_handle_;
    GLenum target_;
    int width_;
    int height_;
    int byte_width_;
    GLint internal_format_;
    GLenum format_;
};

#endif // _GL_PROGRAM_H_