#ifndef _GL_SURFACE_H_
#define _GL_SURFACE_H_

#include "gl_texture.h"

class GLSurface : public GLTexture
{
public:
    GLSurface();
    virtual ~GLSurface();

    virtual void GetTexImage(void* buffer) override;
    virtual void SetTexImage(const void* buffer) override;

    bool Create(int width, int height, GLint internal_format, GLenum format,
                int byte_width);
};

#endif // _GL_SURFACE_H_