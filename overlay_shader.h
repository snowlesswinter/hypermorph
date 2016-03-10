#ifndef _OVERLAY_SHADER_H_
#define _OVERLAY_SHADER_H_

#include <string>

class OverlayShader
{
public:
    static std::string GetVertexShaderCode();
    static std::string GetFragmentShaderCode();
};

#endif // _OVERLAY_SHADER_H_