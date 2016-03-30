#ifndef _GL_TEXTURE_H_
#define _GL_TEXTURE_H_

#include <string>

#include "third_party/opengl/glew.h"

class GLTexture
{
public:
    GLTexture();
    ~GLTexture();

    void Bind() const;
    void BindFrameBuffer() const;
    bool Create(int width, int height, int depth, GLint internal_format,
                GLenum format, int byte_width);
    void GetTexImage(void* buffer);
    void TexImage3D(void* buffer);
    void Unbind() const;

    GLuint frame_buffer() const { return frame_buffer_; }
    GLuint buffer() const { return buffer_; }
    GLuint handle() const { return handle_; }
    GLenum target() const { return target_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int depth() const { return depth_; }
    int byte_width() const { return byte_width_; }

private:
    GLuint frame_buffer_;
    GLuint buffer_;
    GLuint handle_;
    GLenum target_;
    int width_;
    int height_;
    int depth_;
    int byte_width_;
    GLint internal_format_;
    GLenum format_;
};

#endif // _GL_PROGRAM_H_