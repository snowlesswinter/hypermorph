#ifndef _GL_VOLUME_H_
#define _GL_VOLUME_H_

#include "gl_texture.h"

class GLVolume : public GLTexture
{
public:
    GLVolume();
    virtual ~GLVolume();

    virtual void GetTexImage(void* buffer) override;
    virtual void SetTexImage(const void* buffer) override;

    bool Create(int width, int height, int depth, GLint internal_format,
                GLenum format, int byte_width);

    int depth() const { return depth_; }

private:
    int depth_;
};

#endif // _GL_VOLUME_H_