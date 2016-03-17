#ifndef _MULTIGRID_SHADER_H_
#define _MULTIGRID_SHADER_H_

#include <string>

class MultigridShader
{
public:
    static std::string GetComputeResidualShaderCode();
    static std::string GetRestrictShaderCode();
    static std::string GetProlongateShaderCode();
    static std::string GetRelaxWithZeroGuessShaderCode();
    static std::string GetAbsoluteShaderCode(); // For diagnosis.

    // Optimization.
    static std::string GetComputeResidualPackedShaderCode();
    static std::string GetProlongateAndRelaxShaderCode();
    static std::string GetProlongatePackedShaderCode();
    static std::string GetRelaxAndComputeResidualShaderCode();
    static std::string GetRelaxPackedShaderCode();
    static std::string GetRelaxWithZeroGuessPackedShaderCode();
    static std::string GetRestrictPackedShaderCode();
};

#endif // _MULTIGRID_SHADER_H_