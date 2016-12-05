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

#ifndef _PARTICLE_ADVECTION_H_
#define _PARTICLE_ADVECTION_H_

#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

struct AdvectionEuler
{
    __device__ inline float3 Advect(float3 pos_0, float3 vel_0,
                                    float time_step_over_cell_size)
    {
        float pos_x = pos_0.x + vel_0.x * time_step_over_cell_size;
        float pos_y = pos_0.y + vel_0.y * time_step_over_cell_size;
        float pos_z = pos_0.z + vel_0.z * time_step_over_cell_size;

        return make_float3(pos_x, pos_y, pos_z);
    }
};

struct AdvectionMidPoint
{
    __device__ inline float3 Advect(float3 pos_0, float3 vel_0,
                                    float time_step_over_cell_size)
    {
        float mid_x = pos_0.x + 0.5f * time_step_over_cell_size * vel_0.x;
        float mid_y = pos_0.y + 0.5f * time_step_over_cell_size * vel_0.y;
        float mid_z = pos_0.z + 0.5f * time_step_over_cell_size * vel_0.z;
        
        float v_x2 = tex3D(tex_x, mid_x + 0.5f, mid_y,        mid_z);
        float v_y2 = tex3D(tex_y, mid_x,        mid_y + 0.5f, mid_z);
        float v_z2 = tex3D(tex_z, mid_x,        mid_y,        mid_z + 0.5f);
        
        float pos_x = pos_0.x + v_x2 * time_step_over_cell_size;
        float pos_y = pos_0.y + v_y2 * time_step_over_cell_size;
        float pos_z = pos_0.z + v_z2 * time_step_over_cell_size;

        return make_float3(pos_x, pos_y, pos_z);
    }
};

struct AdvectionBogackiShampine
{
    __device__ inline float3 Advect(float3 pos_0, float3 vel_0,
                                    float time_step_over_cell_size)
    {
        float mid_x = pos_0.x + 0.5f * time_step_over_cell_size * vel_0.x;
        float mid_y = pos_0.y + 0.5f * time_step_over_cell_size * vel_0.y;
        float mid_z = pos_0.z + 0.5f * time_step_over_cell_size * vel_0.z;

        float v_x2 = tex3D(tex_x, mid_x + 0.5f, mid_y,        mid_z);
        float v_y2 = tex3D(tex_y, mid_x,        mid_y + 0.5f, mid_z);
        float v_z2 = tex3D(tex_z, mid_x,        mid_y,        mid_z + 0.5f);

        float mid_x2 = pos_0.x + 0.75f * time_step_over_cell_size * v_x2;
        float mid_y2 = pos_0.y + 0.75f * time_step_over_cell_size * v_y2;
        float mid_z2 = pos_0.z + 0.75f * time_step_over_cell_size * v_z2;

        float v_x3 = tex3D(tex_x, mid_x2 + 0.5f, mid_y2,        mid_z2);
        float v_y3 = tex3D(tex_y, mid_x2,        mid_y2 + 0.5f, mid_z2);
        float v_z3 = tex3D(tex_z, mid_x2,        mid_y2,        mid_z2 + 0.5f);

        float c1 = 2.0f / 9.0f * time_step_over_cell_size;
        float c2 = 3.0f / 9.0f * time_step_over_cell_size;
        float c3 = 4.0f / 9.0f * time_step_over_cell_size;

        float pos_x = pos_0.x + c1 * vel_0.x + c2 * v_x2 + c3 * v_x3;
        float pos_y = pos_0.y + c1 * vel_0.y + c2 * v_y2 + c3 * v_y3;
        float pos_z = pos_0.z + c1 * vel_0.z + c2 * v_z2 + c3 * v_z3;

        return make_float3(pos_x, pos_y, pos_z);
    }
};

struct AdvectionRK4
{
    __device__ inline float3 Advect(float3 pos_0, float3 vel_0,
                                    float time_step_over_cell_size)
    {
        float mid_x = pos_0.x + 0.5f * time_step_over_cell_size * vel_0.x;
        float mid_y = pos_0.y + 0.5f * time_step_over_cell_size * vel_0.y;
        float mid_z = pos_0.z + 0.5f * time_step_over_cell_size * vel_0.z;

        float v_x2 = tex3D(tex_x, mid_x + 0.5f, mid_y,        mid_z);
        float v_y2 = tex3D(tex_y, mid_x,        mid_y + 0.5f, mid_z);
        float v_z2 = tex3D(tex_z, mid_x,        mid_y,        mid_z + 0.5f);

        float mid_x2 = pos_0.x + 0.5f * time_step_over_cell_size * v_x2;
        float mid_y2 = pos_0.y + 0.5f * time_step_over_cell_size * v_y2;
        float mid_z2 = pos_0.z + 0.5f * time_step_over_cell_size * v_z2;

        float v_x3 = tex3D(tex_x, mid_x2 + 0.5f, mid_y2,        mid_z2);
        float v_y3 = tex3D(tex_y, mid_x2,        mid_y2 + 0.5f, mid_z2);
        float v_z3 = tex3D(tex_z, mid_x2,        mid_y2,        mid_z2 + 0.5f);

        float mid_x3 = pos_0.x + time_step_over_cell_size * v_x3;
        float mid_y3 = pos_0.y + time_step_over_cell_size * v_y3;
        float mid_z3 = pos_0.z + time_step_over_cell_size * v_z3;

        float v_x4 = tex3D(tex_x, mid_x3 + 0.5f, mid_y3,        mid_z3);
        float v_y4 = tex3D(tex_y, mid_x3,        mid_y3 + 0.5f, mid_z3);
        float v_z4 = tex3D(tex_z, mid_x3,        mid_y3,        mid_z3 + 0.5f);

        float c1 = 1.0f / 6.0f * time_step_over_cell_size;
        float c2 = 2.0f / 6.0f * time_step_over_cell_size;
        float c3 = 2.0f / 6.0f * time_step_over_cell_size;
        float c4 = 1.0f / 6.0f * time_step_over_cell_size;

        float pos_x = pos_0.x + c1 * vel_0.x + c2 * v_x2 + c3 * v_x3 + c4 * v_x4;
        float pos_y = pos_0.y + c1 * vel_0.y + c2 * v_y2 + c3 * v_y3 + c4 * v_y4;
        float pos_z = pos_0.z + c1 * vel_0.z + c2 * v_z2 + c3 * v_z3 + c4 * v_z4;

        return make_float3(pos_x, pos_y, pos_z);
    }
};

#endif  // _PARTICLE_ADVECTION_H_