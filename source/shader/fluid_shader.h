#ifndef _FLUID_SHADER_H_
#define _FLUID_SHADER_H_

#include <string>

class FluidShader
{
public:
    static std::string Vertex();
    static std::string PickLayer();
    static std::string Fill();
    static std::string Advect();
    static std::string Jacobi();
    static std::string DampedJacobi();
    static std::string DampedJacobiPacked();
    static std::string ComputeDivergence();
    static std::string SubtractGradient();
    static std::string Splat();
    static std::string Buoyancy();
};

#endif // _FLUID_SHADER_H_