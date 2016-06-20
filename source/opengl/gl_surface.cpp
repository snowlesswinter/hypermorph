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
#include "gl_surface.h"

#include <cassert>

#include <stdint.h>

#include "third_party/glm/vec2.hpp"

GLSurface::GLSurface()
    : GLTexture()
{

}

GLSurface::~GLSurface()
{
}

bool GLSurface::Create(const glm::ivec2& size, GLint internal_format,
                       GLenum format, int byte_width)
{
    return Create2dTexture(size.x, size.y, internal_format, format, byte_width);
}

void GLSurface::GetTexImage(void* buffer)
{

}

void GLSurface::SetTexImage(const void* buffer)
{

}

glm::ivec2 GLSurface::size() const
{
    return glm::ivec2(width(), height());
}
