#ifndef _CUDA_MEM_PIECE_H_
#define _CUDA_MEM_PIECE_H_

#include <memory>

#include "third_party/glm/fwd.hpp"

class CudaMemPiece
{
public:
    CudaMemPiece();
    ~CudaMemPiece();

    bool Create(int size);

    void* mem() const { return mem_; }

private:
    CudaMemPiece(const CudaMemPiece&);
    void operator=(const CudaMemPiece&);

    void* mem_;
    int size_;
};

#endif // _CUDA_MEM_PIECE_H_