#ifndef _RAYCAST_SHADER_H_
#define _RAYCAST_SHADER_H_

#include <string>

class RaycastShader
{
public:
    static std::string GetVertexShaderCode();
    static std::string GetGeometryShaderCode();
    static std::string GetFragmentShaderCode();
};

#endif // _RAYCAST_SHADER_H_