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

#ifndef _MULTI_PRECISION_TEXTURE_H_
#define _MULTI_PRECISION_TEXTURE_H_

template <typename T>
struct TexSel
{
     __device__ static inline texture<float, cudaTextureType3D, cudaReadModeElementType> Tex(
         texture<float, cudaTextureType3D, cudaReadModeElementType> t32,
         texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> t16)
     {
         return t32;
     }
};

template <>
struct TexSel<ushort>
{
    __device__ static inline texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> Tex(
        texture<float, cudaTextureType3D, cudaReadModeElementType> t32,
        texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> t16)
    {
        return t16;
    }
};

template <typename T>
struct Tex3d
{
    __device__ inline float operator()(texture<T, cudaTextureType3D, cudaReadModeElementType> t, float x, float y, float z)
    {
        return tex3D(t, x, y, z);
    }
};

template <>
struct Tex3d<ushort>
{
    __device__ inline float operator()(texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> t, float x, float y, float z)
    {
        return tex3D(t, x, y, z);
    }
};

#endif  // _MULTI_PRECISION_TEXTURE_H_