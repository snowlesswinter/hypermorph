#ifndef _MULTIGRID_UNITTEST_H_
#define _MULTIGRID_UNITTEST_H_

class MultigridUnittest
{
public:
    static void TestProlongation(int random_seed);

    // Residual calculation test doesn't pass unless we turn to use
    // single-precision floating point numbers. Maybe I should choose the
    // testing data more carefully that make it more suitable for half-float
    // storage.
    static void TestResidualCalculation(int random_seed);
    static void TestResidualRestriction(int random_seed);
    static void TestRestriction(int random_seed);
    static void TestZeroGuessRelaxation(int random_seed);

private:
    MultigridUnittest();
    ~MultigridUnittest();
};

#endif // _MULTIGRID_UNITTEST_H_