#ifndef _MULTIGRID_SHADER_H_
#define _MULTIGRID_SHADER_H_

#include <string>

class MultigridShader
{
public:
    static std::string ComputeResidual();
    static std::string Restrict();
    static std::string Prolongate();
    static std::string RelaxWithZeroGuess();

    // Optimization.
    static std::string ComputeResidualPacked();
    static std::string ProlongateAndRelax();
    static std::string RelaxAndComputeResidual();
    static std::string RelaxPacked();
    static std::string RelaxWithZeroGuessPacked();

    // For diagnosis.
    static std::string Absolute();
    static std::string ComputeResidualPackedDiagnosis();
};

#endif // _MULTIGRID_SHADER_H_