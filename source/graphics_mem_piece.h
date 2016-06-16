#ifndef _GRAPHICS_MEM_PIECE_H_
#define _GRAPHICS_MEM_PIECE_H_

#include <memory>

#include "graphics_lib_enum.h"

class CudaMemPiece;
class GraphicsMemPiece
{
public:
    explicit GraphicsMemPiece(GraphicsLib lib);
    ~GraphicsMemPiece();

    bool Create(int size);

    GraphicsLib graphics_lib() const { return graphics_lib_; }
    std::shared_ptr<CudaMemPiece> cuda_mem_piece() const;

private:
    GraphicsLib graphics_lib_;
    std::shared_ptr<CudaMemPiece> cuda_mem_piece_;
};

#endif // _GRAPHICS_MEM_PIECE_H_