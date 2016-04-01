#ifndef _FLUID_UNITTEST_H_
#define _FLUID_UNITTEST_H_

class FluidUnittest
{
public:
    FluidUnittest();
    ~FluidUnittest();

    static void TestBuoyancyApplication(int random_seed);
    static void TestDampedJacobi(int random_seed);
    static void TestDensityAdvection(int random_seed);
    static void TestDivergenceCalculation(int random_seed);
    static void TestGradientSubtraction(int random_seed);
    static void TestTemperatureAdvection(int random_seed);
    static void TestVelocityAdvection(int random_seed);
};

#endif // _FLUID_UNITTEST_H_