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

#include <cassert>

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
struct TexTraits<
    texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat>>
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

// Sanity check ================================================================

template<typename T, typename...>
struct not_same : std::true_type
{
};

template<typename T, typename U, typename... Ts>
struct not_same<T, U, Ts...>
    : std::integral_constant<
        bool, !std::is_same<T, U>::value && not_same<T, Ts...>::value>
{
};

// Auto unbind =================================================================

template <typename...>
struct TexBound
{
    bool Succeeded() const { return false; }
};

template <typename TexType, typename... TextureTypes>
struct TexBound<TexType, TextureTypes...> : public TexBound<TextureTypes...>
{
    typedef TexBound<TextureTypes...> BaseType;

    TexBound(cudaArray* volume, bool normalize,
                 cudaTextureFilterMode filter_mode,
                 cudaTextureAddressMode addr_mode, TexType* t)
        : TexBound<TextureTypes...>()
        , auto_unbind(t, volume, normalize, filter_mode, addr_mode)
    {
    }
    TexBound(TexBound<TexType, TextureTypes...>&& obj)
        : TexBound<TextureTypes...>(static_cast<BaseType&&>(obj))
        , auto_unbind(std::move(obj.auto_unbind))
    {
    }
    TexBound(TexBound<TextureTypes...>&& obj)
        : TexBound<TextureTypes...>(std::move(obj))
        , auto_unbind()
    {
    }
    TexBound()
        : TexBound<TextureTypes...>()
        , auto_unbind()
    {
    }

    bool Succeeded() const
    {
        bool r = static_cast<const BaseType*>(this)->Succeeded();
        return r || auto_unbind.tex() && auto_unbind.error() == cudaSuccess;
    }

    AutoUnbind<TexType> auto_unbind;
};

// Texture binding =============================================================

template <typename...>
TexBound<> SelectiveBindImpl(cudaArray* volume, bool normalize,
                             cudaTextureFilterMode filter_mode,
                             cudaTextureAddressMode addr_mode)
{
    return TexBound<>();
}

template <typename TexType, typename... TextureTypes>
TexBound<TexType, TextureTypes...> SelectiveBindImpl(
    cudaArray* volume, bool normalize, cudaTextureFilterMode filter_mode,
    cudaTextureAddressMode addr_mode, TexType* t, TextureTypes*... textures)
{
    // Sanity check. Not imperative, just in case the stupid things.
    static_assert(not_same<TexType, TextureTypes...>::value,
                  "same texture type detected.");

    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, volume);
    cudaChannelFormatDesc tex_desc = t->channelDesc;

    // Please refer to the comments in TexTraits.
    tex_desc.f = TexTraits<TexType>::channel_format;
    if (memcmp(&tex_desc, &desc, sizeof(desc)))
        return TexBound<TexType, TextureTypes...>(
            SelectiveBindImpl(volume, normalize, filter_mode, addr_mode,
                              textures...));

    return TexBound<TexType, TextureTypes...>(volume, normalize, filter_mode,
                                              addr_mode, t);
}

template <typename TexType, typename... TextureTypes>
TexBound<TexType, TextureTypes...>
    SelectiveBind(cudaArray* volume, bool normalize,
                  cudaTextureFilterMode filter_mode,
                  cudaTextureAddressMode addr_mode, TexType* t,
                  TextureTypes*... textures)
{
    TexBound<TexType, TextureTypes...> b = SelectiveBindImpl(volume, normalize,
                                                             filter_mode,
                                                             addr_mode, t,
                                                             textures...);
    assert(b.Succeeded());
    return b;
}

#endif  // _MULTI_PRECISION_TEXTURE_H_