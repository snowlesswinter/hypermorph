#ifndef _MULTIGRID_CORE_H_
#define _MULTIGRID_CORE_H_

#include <memory>

#include "third_party/opengl/glew.h"

class GLTexture;
class MultigridCore
{
public:
    MultigridCore();
    ~MultigridCore();

    std::shared_ptr<GLTexture> CreateTexture(int width, int height, int depth,
                                             GLuint internal_format,
                                             GLenum format, bool enable_cuda);
    void ProlongatePacked(std::shared_ptr<GLTexture> coarse,
                          std::shared_ptr<GLTexture> fine);

private:
};

#endif // _MULTIGRID_CORE_H_