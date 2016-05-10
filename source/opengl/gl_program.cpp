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
