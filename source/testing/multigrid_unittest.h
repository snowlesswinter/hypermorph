//
// Hypermorph - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Hypermorph license (* see part 1 below)
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. Acknowledgement of the
//    original author is required if you publish this in a paper, or use it
//    in a product.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

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