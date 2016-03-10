#ifndef _FLUID_SHADER_H_
#define _FLUID_SHADER_H_

#include <string>

class FluidShader
{
public:
    static std::string GetVertexShaderCode();
    static std::string GetPickLayerShaderCode();
    static std::string GetFillShaderCode();
    static std::string GetAvectShaderCode();
    static std::string GetJacobiShaderCode();
    static std::string GetDampedJacobiShaderCode();
    static std::string GetComputeResidualShaderCode();
    static std::string GetComputeDivergenceShaderCode();
    static std::string GetSubtractGradientShaderCode();
    static std::string GetSplatShaderCode();
    static std::string GetBuoyancyShaderCode();
};

#endif // _FLUID_SHADER_H_