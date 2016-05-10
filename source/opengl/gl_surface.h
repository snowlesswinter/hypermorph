#ifndef _GL_SURFACE_H_
#define _GL_SURFACE_H_

#include "gl_texture.h"
#include "third_party/glm/fwd.hpp"

class GLSurface : public GLTexture
{
public:
    GLSurface();
    virtual ~GLSurface();

    virtual void GetTexImage(void* buffer) override;
    virtual void SetTexImage(const void* buffer) override;

    bool Create(const glm::ivec2& size, GLint internal_format, GLenum format,
                int byte_width);

    glm::ivec2 size() const;
};

#endif // _GL_SURFACE_H_