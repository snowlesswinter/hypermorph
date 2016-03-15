#ifndef _MULTIGRID_SHADER_H_
#define _MULTIGRID_SHADER_H_

#include <string>

class MultigridShader
{
public:
    static std::string GetComputeResidualShaderCode();
    static std::string GetRestrictShaderCode();
    static std::string GetProlongateShaderCode();
};

#endif // _MULTIGRID_SHADER_H_