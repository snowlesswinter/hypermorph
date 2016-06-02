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
