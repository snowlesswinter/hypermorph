#ifndef _CUDA_CORE_H_
#define _CUDA_CORE_H_

struct cudaGraphicsResource;
class GraphicsResource;
class CudaCore
{
public:
    CudaCore();
    ~CudaCore();

    bool Init();
    int RegisterGLImage(unsigned int texture, unsigned int target,
                        GraphicsResource* graphics_res);
    int RegisterGLBuffer(unsigned int buffer, GraphicsResource* graphics_res);
    void UnregisterGLImage(GraphicsResource* graphics_res);
    void Absolute(GraphicsResource* graphics_res, unsigned int aa);
    void ProlongatePacked(GraphicsResource* coarse, GraphicsResource* fine,
                          GraphicsResource* out_pbo, int width);
    void AdvectVelocity(GraphicsResource* velocity, GraphicsResource* out_pbo,
                        float time_step, float dissipation, int width);
    void Advect(GraphicsResource* velocity, GraphicsResource* source,
                GraphicsResource* out_pbo, float time_step, float dissipation,
                int width);

    // For diagnosis.
    void RoundPassed(int round);

private:
};

#endif // _CUDA_CORE_H_