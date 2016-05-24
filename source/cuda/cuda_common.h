#ifndef _CUDA_COMMON_H_
#define _CUDA_COMMON_H_

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

#endif // _CUDA_COMMON_H_