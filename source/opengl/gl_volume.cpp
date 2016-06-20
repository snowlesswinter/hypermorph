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
#include "gl_volume.h"

#include <cassert>


GLVolume::GLVolume()
    : GLTexture()
    , depth_(0)
{

}

GLVolume::~GLVolume()
{

}

bool GLVolume::Create(int width, int height, int depth, GLint internal_format,
                      GLenum format, int byte_width)
{
    bool result = Create3dTexture(width, height, depth, internal_format, format,
                                  byte_width);
    if (result)
        depth_ = depth;

    return result;
}

void GLVolume::GetTexImage(void* buffer)
{
    Bind();
    glGetTexImage(GL_TEXTURE_3D, 0, format(),
                  byte_width() == 2 ? GL_HALF_FLOAT : GL_FLOAT, buffer);
    GLenum e = glGetError();
    assert(e == GL_NO_ERROR);
    Unbind();
}

void GLVolume::SetTexImage(const void* buffer)
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    Bind();
    glTexImage3D(GL_TEXTURE_3D, 0, internal_format(), width(), height(),
                 depth(), 0, format(),
                 byte_width() == 2 ? GL_HALF_FLOAT : GL_FLOAT, buffer);
    GLenum e = glGetError();
    assert(e == GL_NO_ERROR);
    Unbind();
}

bool GLVolume::HasSameProperties(const GLVolume& other) const
{
    return width() == other.width() && height() == other.height() &&
        depth() == other.depth() &&
        internal_format() == other.internal_format() &&
        format() == other.format() && byte_width() == other.byte_width();
}
