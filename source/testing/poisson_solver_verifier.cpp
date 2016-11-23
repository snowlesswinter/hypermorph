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

#include "stdafx.h"
#include "poisson_solver_verifier.h"

#include <tbb/tbb.h>

#include "cuda_host/cuda_volume.h"
#include "cuda_host/cuda_main.h"
#include "graphics_volume.h"
#include "opengl/gl_texture.h"
#include "poisson_solver/poisson_core_cuda.h"
#include "poisson_solver/poisson_core_glsl.h"
#include "third_party/glm/vec3.hpp"
#include "third_party/ivock/smoke_solver/AlgebraicMultigrid.h"
#include "third_party/ivock/smoke_solver/sparse_matrix.h"
#include "unittest_common.h"
#include "utility.h"

template <typename FPType>
void BuildRHSFromVolume(std::vector<FPType>* rhs,
                        const GraphicsVolume* rhs_volume, float cell_size)
{
    glm::ivec3 grid_size = rhs_volume->cuda_volume()->size();

    int element_num = grid_size.x * grid_size.y * grid_size.z;
    rhs->resize(element_num);
    rhs->assign(element_num, 0);

    CudaMain::Instance()->CopyFromVolume(&(*rhs)[0],
                                         grid_size.x * sizeof((*rhs)[0]),
                                         rhs_volume->cuda_volume());

    FPType scale = -1.0 / cell_size / cell_size;
    tbb::parallel_for(0, element_num, 1, [&](int thread_id) {
        (*rhs)[thread_id] *= scale;
    });
}

template <typename FPType>
void BuildMatrix(SparseMatrix<FPType>* matrix, const glm::ivec3& grid_size,
                 float cell_size)
{
    int element_num = grid_size.x * grid_size.y * grid_size.z;
    matrix->resize(element_num);
    matrix->zero();

    int slice = grid_size.x * grid_size.y;
    tbb::parallel_for(0, element_num, 1, [&](int thread_id) {
        int k = thread_id / slice;
        int j = (thread_id % slice) / grid_size.x;
        int i = thread_id % grid_size.x;

        FPType pos_scale = 1.0 / cell_size / cell_size;
        FPType neg_sacle = -1.0 / cell_size / cell_size;

        if (i < grid_size.x - 1) {
            matrix->add_to_element(thread_id, thread_id, pos_scale);
            matrix->add_to_element(thread_id, thread_id + 1, neg_sacle);
        }

        if (i > 0) {
            matrix->add_to_element(thread_id, thread_id, pos_scale);
            matrix->add_to_element(thread_id, thread_id - 1, neg_sacle);
        }
        
        if (j < grid_size.y - 1) {
            matrix->add_to_element(thread_id, thread_id, pos_scale);
            matrix->add_to_element(thread_id, thread_id + 1, neg_sacle);
        }

        if (j > 0) {
            matrix->add_to_element(thread_id, thread_id, pos_scale);
            matrix->add_to_element(thread_id, thread_id - 1, neg_sacle);
        }
        
        if (k < grid_size.z - 1) {
            matrix->add_to_element(thread_id, thread_id, pos_scale);
            matrix->add_to_element(thread_id, thread_id + 1, neg_sacle);
        }

        if (k > 0) {
            matrix->add_to_element(thread_id, thread_id, pos_scale);
            matrix->add_to_element(thread_id, thread_id - 1, neg_sacle);
        }
    });
}

void VerifyResult(const GraphicsVolume* pressure_volume)
{
    glm::ivec3 grid_size = pressure_volume->cuda_volume()->size();

    int element_num = grid_size.x * grid_size.y * grid_size.z;
    std::vector<double> gpu_result(element_num, 0.0);

    CudaMain::Instance()->CopyFromVolume(&gpu_result[0],
                                         grid_size.x * sizeof(gpu_result[0]),
                                         pressure_volume->cuda_volume());
    //tbb::parallel_reduce()
}

void PoissonSolverVerifier::Verify(const GraphicsVolume* pressure_volume,
                                   const GraphicsVolume* rhs_volume,
                                   float cell_size, int random_seed)
{
    srand(random_seed);

    PrintDebugString("\"%s\" test started.\n", __FUNCTION__);
    std::vector<double> rhs;
    BuildRHSFromVolume(&rhs, rhs_volume, cell_size);

    SparseMatrix<double> matrix;
    glm::ivec3 grid_size = rhs_volume->cuda_volume()->size();
    BuildMatrix(&matrix, grid_size, cell_size);

    int element_num = grid_size.x * grid_size.y * grid_size.z;
    std::vector<double> pressure(element_num, 0.0);
    double residual_result = 0.0;
    int iterations_result = 0;
    bool succeeded = AMGPCGSolve(matrix, rhs, pressure, 1e-10, 1000,
                                 residual_result, iterations_result,
                                 grid_size.x, grid_size.y, grid_size.z);
    PrintDebugString("AMGPCG solve %s: residual %e after %d iterations.\n",
                     succeeded ? "done" : "failed", residual_result,
                     iterations_result);
}
