#ifndef _GL_TEXTURE_H_
#define _GL_TEXTURE_H_

#include <string>

#include <glew.h>

class GLTexture
{
public:
    GLTexture();
    ~GLTexture();

    bool Load(const std::string& vs_source, const std::string& gs_source,
              const std::string& fs_source);
    void Use();

    GLuint handle() const { return handle_; }
    GLenum target() const { return target_; }

private:
    GLuint handle_;
    GLenum target_;
};

#endif // _GL_PROGRAM_H_