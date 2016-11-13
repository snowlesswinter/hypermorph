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

#include "stdafx.h"
#include "gl_program.h"

#include "utility.h"
#include "third_party/glm/mat4x4.hpp"
#include "third_party/glm/vec3.hpp"

GLProgram::GLProgram()
    : program_handle_(0)
{

}

GLProgram::~GLProgram()
{

}

bool GLProgram::Load(const std::string& vs_source, const std::string& gs_source,
                     const std::string& fs_source)
{
    program_handle_ = LoadProgram(vs_source, gs_source, fs_source);
    return !!program_handle_;
}

void GLProgram::SetUniform(const char* name, int value)
{
    ::SetUniform(name, value);
}

void GLProgram::SetUniform(const char* name, float value)
{
    ::SetUniform(name, value);
}

void GLProgram::SetUniform(const char* name, float value0, float value1)
{
    ::SetUniform(name, value0, value1);
}

void GLProgram::SetUniform(const char* name, glm::mat4 value)
{
    ::SetUniform(name, value);
}

void GLProgram::SetUniform(const char* name, glm::vec3 value)
{
    ::SetUniform(name, value);
}

void GLProgram::Unuse()
{
    glUseProgram(0);
}

void GLProgram::Use()
{
    glUseProgram(program_handle_);
}
