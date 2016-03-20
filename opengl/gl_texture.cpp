#include "stdafx.h"
#include "gl_texture.h"

#include "utility.h"

GLTexture::GLTexture()
    : handle_(0)
    , target_(0)
{

}

GLTexture::~GLTexture()
{

}

bool GLTexture::Load(const std::string& vs_source, const std::string& gs_source,
                     const std::string& fs_source)
{
    handle_ = LoadProgram(vs_source, gs_source, fs_source);
    return !!handle_;
}

void GLTexture::Use()
{
    glUseProgram(handle_);
}

