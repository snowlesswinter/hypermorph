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

bool GLSurface::Create(int width, int height, GLint internal_format,
                       GLenum format, int byte_width)
{
    return Create2dTexture(width, height, internal_format, format, byte_width);
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
