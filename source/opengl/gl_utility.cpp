#include "stdafx.h"
#include "gl_utility.h"

#include <cassert>

#include "opengl/gl_texture.h"
#include "vmath.hpp"

void GLUtility::CopyFromVolume(void* data, size_t size_in_bytes, size_t pitch,
                               const vmath::Vector3& volume_size,
                               GLTexture* texture)
{
}

void GLUtility::CopyToVolume(GLTexture* texture, void* data,
                             size_t size_in_bytes, size_t pitch, 
                             const vmath::Vector3& volume_size)
{
}
