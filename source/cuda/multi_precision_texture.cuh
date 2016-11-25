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
struct TexType
{
    typedef texture<float, cudaTextureType3D, cudaReadModeElementType> Type;
};

template <>
struct TexType<ushort>
{
    typedef texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat>
        Type;
};

template <>
struct TexType<long2>
{
    typedef texture<long2, cudaTextureType3D, cudaReadModeElementType> Type;
};

template <typename TexType>
struct TexTraits
{
};

template <>
struct TexTraits<texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat>>
{
    typedef ushort EleType;

    // NOTE: Be ware that this format initialized by the runtime is
    //       'cudaChannelFormatKindUnsigned'.
    static const cudaChannelFormatKind channel_format =
        cudaChannelFormatKindFloat;
};

template <>
struct TexTraits<texture<float, cudaTextureType3D, cudaReadModeElementType>>
{
    typedef float EleType;
    static const cudaChannelFormatKind channel_format =
        cudaChannelFormatKindFloat;
};

template <>
struct TexTraits<texture<long2, cudaTextureType3D, cudaReadModeElementType>>
{
    typedef long2 EleType;
    static const cudaChannelFormatKind channel_format =
        cudaChannelFormatKindUnsigned;
};

template <typename T>
struct TexSel
{
    __device__ static inline TexType<float>::Type Tex(TexType<ushort>::Type t16,
                                                      TexType<float>::Type t32,
                                                      TexType<long2>::Type t64)
     {
         return t32;
     }
};

template <>
struct TexSel<ushort>
{
    __device__ static inline TexType<ushort>::Type Tex(
        TexType<ushort>::Type t16, TexType<float>::Type t32,
        TexType<long2>::Type t64)
    {
        return t16;
    }
};

template <>
struct TexSel<long2>
{
    __device__ static inline TexType<long2>::Type Tex(TexType<ushort>::Type t16,
                                                      TexType<float>::Type t32,
                                                      TexType<long2>::Type t64)
    {
        return t64;
    }
};

template <typename T>
struct Tex3d
{
    typedef float ValType;
    __device__ inline float operator()(TexType<T>::Type t, float x, float y,
                                       float z)
    {
        return tex3D(t, x, y, z);
    }
};

template <>
struct Tex3d<ushort>
{
    typedef float ValType;
    __device__ inline float operator()(TexType<ushort>::Type t, float x,
                                       float y, float z)
    {
        return tex3D(t, x, y, z);
    }
};

template <>
struct Tex3d<long2>
{
    typedef double ValType;
    __device__ inline double operator()(TexType<long2>::Type t, float x,
                                        float y, float z)
    {
        long2 raw = tex3D(t, x, y, z);
        return __hiloint2double(raw.x, raw.y);
    }
};

template <typename T>
std::unique_ptr<void, std::function<void(void*)>> BindImpl(
    cudaArray* volume, bool normalize, cudaTextureFilterMode filter_mode,
    cudaTextureAddressMode addr_mode, TexType<T>::Type* t)
{
    auto auto_unbind = new AutoUnbind<typename TexType<T>::Type>(t, volume,
                                                                 normalize,
                                                                 filter_mode,
                                                                 addr_mode);
    auto release_func = [](void* ptr) {
        delete reinterpret_cast<AutoUnbind<typename TexType<T>::Type>*>(ptr);
    };
    auto r = std::unique_ptr<void, std::function<void(void*)>>(
        auto_unbind, std::move(release_func));
    if (auto_unbind->error() != cudaSuccess)
        return std::unique_ptr<void, std::function<void(void*)>>();

    return std::move(r);
}

template <typename TexType>
std::unique_ptr<void, std::function<void(void*)>>
    SelectiveBind(cudaArray* volume, bool normalize,
                  cudaTextureFilterMode filter_mode,
                  cudaTextureAddressMode addr_mode, TexType* t)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, volume);
    cudaChannelFormatDesc tex_desc = t->channelDesc;

    // Please refer to the comments in TexTraits.
    tex_desc.f = TexTraits<TexType>::channel_format;
    if (memcmp(&tex_desc, &desc, sizeof(desc)))
        return std::unique_ptr<void, std::function<void(void*)>>();

    return BindImpl<TexTraits<TexType>::EleType>(volume, normalize, filter_mode,
                                                 addr_mode, t);
}

template <typename TexType, typename... TextureTypes>
std::unique_ptr<void, std::function<void(void*)>>
    SelectiveBind(cudaArray* volume, bool normalize,
                  cudaTextureFilterMode filter_mode,
                  cudaTextureAddressMode addr_mode, TexType* t,
                  TextureTypes... textures)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, volume);
    cudaChannelFormatDesc tex_desc = t->channelDesc;

    // Please refer to the comments in TexTraits.
    tex_desc.f = TexTraits<TexType>::channel_format;
    if (memcmp(&tex_desc, &desc, sizeof(desc)))
        return SelectiveBind(volume, normalize, filter_mode, addr_mode,
                             textures...);

    return BindImpl<TexTraits<TexType>::EleType>(volume, normalize, filter_mode,
                                                 addr_mode, t);
}

#endif  // _MULTI_PRECISION_TEXTURE_H_