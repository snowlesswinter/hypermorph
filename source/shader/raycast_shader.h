#ifndef _RAYCAST_SHADER_H_
#define _RAYCAST_SHADER_H_

#include <string>

class RaycastShader
{
public:
    static std::string Vertex();
    static std::string Geometry();
    static std::string Fragment();

    static std::string ApplyTextureVert();
    static std::string ApplyTextureFrag();
};

#endif // _RAYCAST_SHADER_H_