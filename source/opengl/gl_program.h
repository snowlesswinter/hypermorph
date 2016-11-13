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

#ifndef _GL_PROGRAM_H_
#define _GL_PROGRAM_H_

#include <string>

#include "third_party/opengl/glew.h"
#include "third_party/glm/fwd.hpp"

class GLProgram
{
public:
    GLProgram();
    ~GLProgram();

    bool Load(const std::string& vs_source, const std::string& gs_source,
              const std::string& fs_source);
    void SetUniform(const char* name, int value);
    void SetUniform(const char* name, float value);
    void SetUniform(const char* name, float value0, float value1);
    void SetUniform(const char* name, glm::mat4 value);
    void SetUniform(const char* name, glm::vec3 value);
    void Unuse();
    void Use();

    GLuint program_handle() const { return program_handle_; }

private:
    GLuint program_handle_;
};

#endif // _GL_PROGRAM_H_