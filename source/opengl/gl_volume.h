//
// Hypermorph - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Hypermorph license (* see part 1 below)
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

#ifndef _GL_VOLUME_H_
#define _GL_VOLUME_H_

#include "gl_texture.h"

class GLVolume : public GLTexture
{
public:
    GLVolume();
    virtual ~GLVolume();

    virtual void GetTexImage(void* buffer) override;
    virtual void SetTexImage(const void* buffer) override;

    bool Create(int width, int height, int depth, GLint internal_format,
                GLenum format, int byte_width);
    bool HasSameProperties(const GLVolume& other) const;

    int depth() const { return depth_; }

private:
    int depth_;
};

#endif // _GL_VOLUME_H_