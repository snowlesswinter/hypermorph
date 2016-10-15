//
// Fluid3d - Fluid Simulator for interactive applications
// Copyright (C) 2016. JIANWEN TAN(jianwen.tan@gmail.com). All rights reserved.
//
// Fluid3d license (* see part 1 below)
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

#ifndef _KERNEL_LAUNCHER_H_
#define _KERNEL_LAUNCHER_H_

#include "third_party/opengl/glew.h"

#include <cuda_runtime.h>
#include <stdint.h>

struct FlipParticles;
class AuxBufferManager;
class BlockArrangement;
enum AdvectionMethod;
enum FluidImpulse;

extern void LaunchAdvectScalarField(cudaArray* fnp1, cudaArray* fn, cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z, cudaArray* aux, float cell_size, float time_step, float dissipation, AdvectionMethod method, uint3 volume_size, bool mid_point, BlockArrangement* ba);
extern void LaunchAdvectScalarFieldStaggered(cudaArray* fnp1, cudaArray* fn, cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z, cudaArray* aux, float cell_size, float time_step, float dissipation, AdvectionMethod method, uint3 volume_size, bool mid_point, BlockArrangement* ba);
extern void LaunchAdvectVectorField(cudaArray* fnp1_x, cudaArray* fnp1_y, cudaArray* fnp1_z, cudaArray* fn_x, cudaArray* fn_y, cudaArray* fn_z, cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z, cudaArray* aux, float cell_size, float time_step, float dissipation, AdvectionMethod method, uint3 volume_size, bool mid_point, BlockArrangement* ba);
extern void LaunchAdvectVelocityStaggered(cudaArray* fnp1_x, cudaArray* fnp1_y, cudaArray* fnp1_z, cudaArray* fn_x, cudaArray* fn_y, cudaArray* fn_z, cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z, cudaArray* aux, float cell_size, float time_step, float dissipation, AdvectionMethod method, uint3 volume_size, bool mid_point, BlockArrangement* ba);
extern void LaunchAdvectVorticityStaggered(cudaArray* fnp1_x, cudaArray* fnp1_y, cudaArray* fnp1_z, cudaArray* fn_x, cudaArray* fn_y, cudaArray* fn_z, cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z, cudaArray* aux, float cell_size, float time_step, float dissipation, AdvectionMethod method, uint3 volume_size, bool mid_point, BlockArrangement* ba);
extern void LaunchApplyBuoyancy(cudaArray* vnp1_x, cudaArray* vnp1_y, cudaArray* vnp1_z, cudaArray* vn_x, cudaArray* vn_y, cudaArray* vn_z, cudaArray* temperature, cudaArray* density, float time_step, float ambient_temperature, float accel_factor, float gravity, bool staggered, uint3 volume_size, BlockArrangement* ba);
extern void LaunchComputeDivergence(cudaArray* div, cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z, float cell_size, bool outflow, bool staggered, uint3 volume_size, BlockArrangement* ba);
extern void LaunchComputeResidualDiagnosis(cudaArray* residual, cudaArray* u, cudaArray* b, float cell_size, uint3 volume_size, BlockArrangement* ba);
extern void LaunchImpulseDensity(cudaArray* dest, cudaArray* original, float3 center_point, float radius, float value, FluidImpulse impulse, uint3 volume_size, BlockArrangement* ba);
extern void LaunchImpulseScalar(cudaArray* dest, cudaArray* original, float3 center_point, float3 hotspot, float radius, float value, FluidImpulse impulse, uint3 volume_size, BlockArrangement* ba);
extern void LaunchRelax(cudaArray* unp1, cudaArray* un, cudaArray* b, bool outflow, int num_of_iterations, uint3 volume_size, BlockArrangement* ba);
extern void LaunchRoundPassed(int* dest_array, int round, int x);
extern void LaunchSubtractGradient(cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z, cudaArray* pressure, float cell_size, bool staggered, uint3 volume_size, BlockArrangement* ba);

// Multigrid.
extern void LaunchComputeResidual(cudaArray* r, cudaArray* u, cudaArray* b, uint3 volume_size, BlockArrangement* ba);
extern void LaunchProlongate(cudaArray* fine, cudaArray* coarse, uint3 volume_size_fine, BlockArrangement* ba);
extern void LaunchProlongateError(cudaArray* fine, cudaArray* coarse, uint3 volume_size_fine, BlockArrangement* ba);
extern void LaunchRelaxWithZeroGuess(cudaArray* u, cudaArray* b, uint3 volume_size, BlockArrangement* ba);
extern void LaunchRestrict(cudaArray* coarse, cudaArray* fine, uint3 volume_size, BlockArrangement* ba);

// Conjugate gradient.
extern void LaunchApplyStencil(cudaArray* aux, cudaArray* search, bool outflow, uint3 volume_size, BlockArrangement* ba);
extern void LaunchComputeAlpha(float* alpha, float* rho, cudaArray* vec0, cudaArray* vec1, uint3 volume_size, BlockArrangement* ba, AuxBufferManager* bm);
extern void LaunchComputeRho(float* rho, cudaArray* search, cudaArray* residual, uint3 volume_size, BlockArrangement* ba, AuxBufferManager* bm);
extern void LaunchComputeRhoAndBeta(float* beta, float* rho_new, float* rho, cudaArray* vec0, cudaArray* vec1, uint3 volume_size, BlockArrangement* ba, AuxBufferManager* bm);
extern void LaunchScaledAdd(cudaArray* dest, cudaArray* v0, cudaArray* v1, float* coef, float sign, uint3 volume_size, BlockArrangement* ba);

// Vorticity.
extern void LaunchAddCurlPsi(cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z, cudaArray* psi_x, cudaArray* psi_y, cudaArray* psi_z, float cell_size, uint3 volume_size, BlockArrangement* ba);
extern void LaunchApplyVorticityConfinementStaggered(cudaArray* vel_x, cudaArray* vely, cudaArray* vel_z, cudaArray* conf_x, cudaArray* conf_y, cudaArray* conf_z, uint3 volume_size, BlockArrangement* ba);
extern void LaunchBuildVorticityConfinementStaggered(cudaArray* conf_x, cudaArray* conf_y, cudaArray* conf_z, cudaArray* vort_x, cudaArray* vort_y, cudaArray* vort_z, float coeff, float cell_size, uint3 volume_size, BlockArrangement* ba);
extern void LaunchComputeCurlStaggered(cudaArray* vort_x, cudaArray* vort_y, cudaArray* vort_z, cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z, float cell_size, uint3 volume_size, BlockArrangement* ba);
extern void LaunchComputeDeltaVorticity(cudaArray* delta_x, cudaArray* delta_y, cudaArray* delta_z, cudaArray* vn_x, cudaArray* vn_y, cudaArray* vn_z, uint3 volume_size, BlockArrangement* ba);
extern void LaunchDecayVorticesStaggered(cudaArray* vort_x, cudaArray* vort_y, cudaArray* vort_z, cudaArray* div, float time_step, uint3 volume_size, BlockArrangement* ba);
extern void LaunchStretchVorticesStaggered(cudaArray* vnp1_x, cudaArray* vnp1_y, cudaArray* vnp1_z, cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z, cudaArray* vort_x, cudaArray* vort_y, cudaArray* vort_z, float cell_size, float time_step, uint3 volume_size, BlockArrangement* ba);

// Particles.
namespace kern_launcher
{
extern void AdvectParticles(const FlipParticles& particles, float time_step, float cell_size, uint3 volume_size, BlockArrangement* ba);
extern void BindParticlesToCells(const FlipParticles& particles, uint3 volume_size, BlockArrangement* ba);
extern void BuildCellOffsets(uint* cell_offsets, const uint* cell_particles_counts, int num_of_cells, BlockArrangement* ba, AuxBufferManager* bm);
extern void InterpolateDeltaVelocity(const FlipParticles& particles, cudaArray* vnp1_x, cudaArray* vnp1_y, cudaArray* vnp1_z, cudaArray* vn_x, cudaArray* vn_y, cudaArray* vn_z, BlockArrangement* ba);
extern void Resample(const FlipParticles& particles, cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z, cudaArray* density, cudaArray* temperature, uint random_seed, uint3 volume_size, BlockArrangement* ba);
extern void ResetParticles(const FlipParticles& particles, BlockArrangement* ba);
extern void SortParticles(FlipParticles particles, uint16_t* aux, uint3 volume_size, BlockArrangement* ba);
extern void TransferToGrid(cudaArray* vel_x, cudaArray* vel_y, cudaArray* vel_z, cudaArray* density, cudaArray* temperature, const FlipParticles& particles, uint3 volume_size, BlockArrangement* ba);
}

#endif // _KERNEL_LAUNCHER_H_