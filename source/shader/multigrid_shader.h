#ifndef _MULTIGRID_SHADER_H_
#define _MULTIGRID_SHADER_H_

#include <string>

class MultigridShader
{
public:
    static std::string ComputeResidual();
    static std::string Restrict();
    static std::string Prolongate();

    static std::string RelaxAndComputeResidual();
    static std::string RelaxPacked();
    static std::string RelaxWithZeroGuessPacked();

    static std::string ComputeResidualPackedDiagnosis();
};

#endif // _MULTIGRID_SHADER_H_