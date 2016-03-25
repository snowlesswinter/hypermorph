#include "stdafx.h"
#include "gl_program.h"

#include "utility.h"

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

void GLProgram::Use()
{
    glUseProgram(program_handle_);
}

