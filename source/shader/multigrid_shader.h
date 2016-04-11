#ifndef _MULTIGRID_SHADER_H_
#define _MULTIGRID_SHADER_H_

#include <string>

class MultigridShader
{
public:
    static std::string ComputeResidual();
    static std::string ComputeResidualPackedDiagnosis();
    static std::string ProlongatePacked();
    static std::string RelaxPacked();
    static std::string RelaxWithZeroGuessPacked();
    static std::string RestrictPacked();
    static std::string RestrictResidualPacked();
};

#endif // _MULTIGRID_SHADER_H_