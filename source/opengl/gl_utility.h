#ifndef _GL_UTILITY_H_
#define _GL_UTILITY_H_

#include "third_party/opengl/glew.h"

namespace Vectormath
{
namespace Aos
{
class Vector3;
}
}
class GLTexture;
class GLUtility
{
public:
    static void CopyFromVolume(void* data, size_t size_in_bytes,
                               size_t pitch,
                               const Vectormath::Aos::Vector3& volume_size,
                               GLTexture* texture);
    static void CopyToVolume(GLTexture* texture, void* data,
                             size_t size_in_bytes, size_t pitch,
                             const Vectormath::Aos::Vector3& volume_size);

private:
    GLUtility();
    ~GLUtility();
};

#endif // _GL_UTILITY_H_