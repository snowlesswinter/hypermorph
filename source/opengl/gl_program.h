#ifndef _GL_PROGRAM_H_
#define _GL_PROGRAM_H_

#include <string>

#include "third_party/opengl/glew.h"

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
    void Unuse();
    void Use();

    GLuint program_handle() const { return program_handle_; }

private:
    GLuint program_handle_;
};

#endif // _GL_PROGRAM_H_