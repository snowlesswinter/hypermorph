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

#ifndef _MULTI_PRECISION_H_
#define _MULTI_PRECISION_H_

#include <algorithm>
#include <cassert>
#include <tuple>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "cuda/cuda_common_host.h"

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
    typedef ushort StorageType;

    // NOTE: Be ware that this format initialized by the runtime is
    //       'cudaChannelFormatKindUnsigned'.
    static const cudaChannelFormatKind channel_format =
        cudaChannelFormatKindFloat;
};

template <>
struct TexTraits<texture<float, cudaTextureType3D, cudaReadModeElementType>>
{
    typedef float StorageType;
    static const cudaChannelFormatKind channel_format =
        cudaChannelFormatKindFloat;
};

template <>
struct TexTraits<texture<long2, cudaTextureType3D, cudaReadModeElementType>>
{
    typedef long2 StorageType;
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

    __device__ inline void Store(ValType v,
                                 surface<void, cudaSurfaceType3D> surf,
                                 uint x, uint y, uint z)
    {
        surf3Dwrite(v, surf, x * sizeof(v), y, z, cudaBoundaryModeTrap);
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

    __device__ inline void Store(ValType v,
                                 surface<void, cudaSurfaceType3D> surf,
                                 uint x, uint y, uint z)
    {
        auto r = __float2half_rn(v);
        surf3Dwrite(r, surf, x * sizeof(r), y, z, cudaBoundaryModeTrap);
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

    __device__ inline void Store(ValType v,
                                 surface<void, cudaSurfaceType3D> surf,
                                 uint x, uint y, uint z)
    {
        surf3Dwrite(v, surf, x * sizeof(v), y, z, cudaBoundaryModeTrap);
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
    typedef TexType ThisTexType;

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

    bool Bound() const
    {
        return auto_unbind.tex() && auto_unbind.error() == cudaSuccess;
    }

    bool Succeeded() const
    {
        bool r = static_cast<const BaseType*>(this)->Succeeded();
        return r || Bound();
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

// Simple scalar ===============================================================

typedef std::tuple<float*, double*> ScalarPieces;

class MemPiece;
ScalarPieces CreateScalarPieces(const MemPiece& piece);

// Reduction invoker ===========================================================

// C++ has not yet supported partial specializing function template.
// http://stackoverflow.com/questions/8992853/terminating-function-template-recursion

template <typename ScalarType, std::size_t index>
struct AssignScalarImpl;

template <typename ScalarType>
struct AssignScalarImpl<ScalarType, 0>
{
    static void Assign(ScalarType** scalar, const ScalarPieces& pack) { }
};

template <typename ScalarType, std::size_t index>
struct AssignScalarImpl
{
    static void Assign(ScalarType** scalar, const ScalarPieces& pack)
    {
        std::size_t const n = std::tuple_size<ScalarPieces>::value;
        std::size_t const i = n - index;
        using ScalarPtr = typename std::tuple_element<i, ScalarPieces>::type;
        bool same_type = std::is_same<ScalarType*, ScalarPtr>::value;
        if (std::get<i>(pack) != nullptr && same_type)
        {
            *scalar = reinterpret_cast<ScalarType*>(std::get<i>(pack));
            return;
        }

        AssignScalarImpl<ScalarType, index - 1>::Assign(scalar, pack);
    }
};

template <typename ScalarType>
void AssignScalar(ScalarType** scalar, const ScalarPieces& pack)
{
    static std::size_t const size = std::tuple_size<ScalarPieces>::value;
    AssignScalarImpl<ScalarType, size>::Assign(scalar, pack);
}

class AuxBufferManager;
class BlockArrangement;

template <template <typename S> class SchemeType, typename BoundType,
          std::size_t index, typename... MorePacks>
struct InvokeReductionImpl;

template <template <typename S> class SchemeType, typename BoundType,
          typename... MorePacks>
struct InvokeReductionImpl<SchemeType, BoundType, 0, MorePacks...>
{
    static void Invoke(const ScalarPieces& dest, const BoundType& bound,
                       uint3 volume_size, BlockArrangement* ba,
                       AuxBufferManager* bm, const MorePacks&... packs) {}
};

template <template <typename S> class SchemeType, typename BoundType,
          std::size_t index, typename... MorePacks>
struct InvokeReductionImpl
{
    static void Invoke(const ScalarPieces& dest, const BoundType& bound,
                       uint3 volume_size, BlockArrangement* ba,
                       AuxBufferManager* bm, const MorePacks&... packs)
    {
        using StorageType =
            typename TexTraits<typename BoundType::ThisTexType>::StorageType;
        using FPType = typename Tex3d<StorageType>::ValType;
        if (bound.Bound()) {
            SchemeType<StorageType> scheme;
            scheme.Init(packs...);

            FPType* dest_fp = nullptr;
            AssignScalar(&dest_fp, dest);
            assert(dest_fp);

            ReduceVolume(dest_fp, scheme, volume_size, ba, bm);
            return;
        }

        using NextBoundType = typename BoundType::BaseType;
        InvokeReductionImpl<SchemeType, NextBoundType, index - 1,
                            MorePacks...>::Invoke(dest, bound, volume_size, ba,
                                                  bm, packs...);
    }
};

template <template <typename S> class SchemeType, typename BoundType,
          typename... MorePacks>
void InvokeReduction(const ScalarPieces& dest, const BoundType& bound,
                     uint3 volume_size, BlockArrangement* ba,
                     AuxBufferManager* bm, const MorePacks&... packs)
{
    static std::size_t const n = sizeof(bound) / sizeof(bound.auto_unbind);
    InvokeReductionImpl<SchemeType, BoundType, n, MorePacks...>::Invoke(
        dest, bound, volume_size, ba, bm, packs...);
}

// Kernel invoker ==============================================================

template <template <typename S> class KernMeta, typename BoundType,
          std::size_t index, typename... Params>
struct InvokeKernelImpl;

template <template <typename S> class KernMeta, typename BoundType,
          typename... Params>
struct InvokeKernelImpl<KernMeta, BoundType, 0, Params...>
{
    static void Invoke(const BoundType& bound, Params... params) {}
};

template <template <typename S> class KernMeta, typename BoundType,
          std::size_t index, typename... Params>
struct InvokeKernelImpl
{
    static void Invoke(const BoundType& bound, Params... params)
    {
        using StorageType =
            typename TexTraits<typename BoundType::ThisTexType>::StorageType;
        if (bound.Bound()) {
            KernMeta<StorageType>::Invoke(params...);
            return;
        }

        using NextBoundType = typename BoundType::BaseType;
        InvokeKernelImpl<KernMeta, NextBoundType, index - 1, Params...>::Invoke(
            bound, params...);
    }
};

template <template <typename S> class KernMeta, typename BoundType,
          typename... Params>
void InvokeKernel(const BoundType& bound, Params... params)
{
    static std::size_t const n = sizeof(bound) / sizeof(bound.auto_unbind);
    InvokeKernelImpl<KernMeta, BoundType, n, Params...>::Invoke(bound,
                                                                params...);
}

#define MAKE_INVOKE_DECLARATION(...)                          \
    (const dim3& grid, const dim3& block, __VA_ARGS__)

#define DECLARE_KERNEL_META(kern_name, params_decl, ...)      \
template <typename StorageType>                               \
struct kern_name##Meta                                        \
{                                                             \
    static void Invoke params_decl                            \
    {                                                         \
        using FPType = typename Tex3d<StorageType>::ValType;  \
        kern_name<StorageType><<<grid, block>>>(__VA_ARGS__); \
    }                                                         \
}

#endif  // _MULTI_PRECISION_H_