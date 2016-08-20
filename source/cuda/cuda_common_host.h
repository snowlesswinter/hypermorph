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

#ifndef _CUDA_COMMON_HOST_H_
#define _CUDA_COMMON_HOST_H_

#include <cuda_runtime.h>
#include <helper_math.h>

template <typename SurfaceType>
cudaError_t BindCudaSurfaceToArray(SurfaceType* surf, cudaArray* cuda_array)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, cuda_array);
    cudaError_t result = cudaBindSurfaceToArray(surf, cuda_array, &desc);
    assert(result == cudaSuccess);
    return result;
}

template <typename TextureType>
cudaError_t BindCudaTextureToArray(TextureType* tex, cudaArray* cuda_array,
                                   bool normalize,
                                   cudaTextureFilterMode filter_mode,
                                   cudaTextureAddressMode addr_mode)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, cuda_array);
    tex->normalized = normalize;
    tex->filterMode = filter_mode;
    tex->addressMode[0] = addr_mode;
    tex->addressMode[1] = addr_mode;
    tex->addressMode[2] = addr_mode;
    tex->channelDesc = desc;

    cudaError_t result = cudaBindTextureToArray(tex, cuda_array, &desc);
    assert(result == cudaSuccess);
    return result;
}

template <typename TextureType>
class AutoUnbind
{
public:
    AutoUnbind(TextureType* tex, cudaArray* cuda_array, bool normalize,
               cudaTextureFilterMode filter_mode,
               cudaTextureAddressMode addr_mode)
        : tex_(tex)
        , error_(
            BindCudaTextureToArray(tex, cuda_array, normalize, filter_mode,
                                   addr_mode))
    {
    }
    AutoUnbind(AutoUnbind<TextureType>&& obj)
    {
        Take(obj);
    }

    ~AutoUnbind()
    {
        if (tex_) {
            cudaUnbindTexture(tex_);
            tex_ = nullptr;
        }
    }

    void Take(AutoUnbind<TextureType>&& obj)
    {
        if (tex_ && tex_ != obj.tex_)
            cudaUnbindTexture(tex_);

        tex_ = obj.tex_;
        error_ = obj.error_;

        obj.tex_ = nullptr;
    }

    cudaError_t error() const { return error_; }

private:
    AutoUnbind(const AutoUnbind<TextureType>& obj);
    void operator =(const AutoUnbind<TextureType>& obj);

    TextureType* tex_;
    cudaError_t error_;
};

class BindHelper
{
public:
    template <typename TextureType>
    static AutoUnbind<TextureType> Bind(TextureType* tex, cudaArray* cuda_array,
                                        bool normalize,
                                        cudaTextureFilterMode filter_mode,
                                        cudaTextureAddressMode addr_mode)
    {
        return AutoUnbind<TextureType>(tex, cuda_array, normalize, filter_mode,
                                       addr_mode);
    }
};

bool IsPow2(uint x);
bool CopyVolumeAsync(cudaArray* dest, cudaArray* source,
                     const uint3& volume_size);

#endif // _CUDA_COMMON_HOST_H_