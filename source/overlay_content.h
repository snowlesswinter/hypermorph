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

#ifndef _OVERLAY_CONTENT_H_
#define _OVERLAY_CONTENT_H_

#include <string>

#include "third_party/opengl/glew.h"

struct MeshPod;
class OverlayContent
{
public:
    OverlayContent();
    ~OverlayContent();

    void RenderText(const std::string& text, int viewport_width,
                    int viewport_height);

private:
    struct BitmapBuf
    {
        HBITMAP hbitmap_;
        HBRUSH hbrush_;
        void* bits_;

        BitmapBuf() : hbitmap_(nullptr), hbrush_(nullptr), bits_(nullptr) {}
    };
    GLuint GetTexture();
    GLuint GetProgram();
    MeshPod* GetQuadMesh();
    BitmapBuf GetBitmapBuf();

    std::string last_text_;
    GLuint texture_ = 0;
    GLsizei width_ = 0;
    GLsizei height_ = 0;
    GLuint program_ = 0;
    MeshPod* quad_mesh_;
    BitmapBuf bitmap_buf_;
};

#endif // _OVERLAY_CONTENT_H_