#include <cassert>

#include "third_party/opengl/glew.h"

#include <helper_math.h>

#include "block_arrangement.h"

surface<void, cudaSurfaceType3D> jacobi;
texture<ushort2, cudaTextureType3D, cudaReadModeNormalizedFloat> jacobi_packed;

__global__ void DampedJacobiKernel(float minus_square_cell_size,
                                   float omega_over_beta, uint3 volume_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float near =              tex3D(jacobi_packed, x, y, z - 1.0f).x;
    float south =             tex3D(jacobi_packed, x, y - 1.0f, z).x;
    float west =              tex3D(jacobi_packed, x - 1.0f, y, z).x;
    float2 packed_center =    tex3D(jacobi_packed, x, y, z);
    float east =              tex3D(jacobi_packed, x + 1.0f, y, z).x;
    float north =             tex3D(jacobi_packed, x, y + 1.0f, z).x;
    float far =               tex3D(jacobi_packed, x, y, z + 1.0f).x;

    float u = omega_over_beta * 3.0f * packed_center.x +
        (west + east + south + north + far + near + minus_square_cell_size *
        packed_center.y) * omega_over_beta;

    ushort2 raw = make_ushort2(__float2half_rn(u),
                               __float2half_rn(packed_center.y));
    surf3Dwrite(raw, jacobi, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__device__ void ReadBlockAndHalo_32x6(int z, uint tx, uint ty, float2* smem)
{
    uint linear_index = ty * blockDim.x + tx;

    const uint smem_width = 48;

    uint sx =  linear_index % smem_width;
    uint sy1 = linear_index / smem_width;
    uint sy2 = sy1 + 4;

    int ix =  static_cast<int>(blockIdx.x * blockDim.x + sx) - 8;
    int iy1 = static_cast<int>(blockIdx.y * blockDim.y + sy1) - 1;
    int iy2 = static_cast<int>(blockIdx.y * blockDim.y + sy2) - 1;

    smem[sx + sy1 * smem_width] = tex3D(jacobi_packed, ix, iy1, z);
    smem[sx + sy2 * smem_width] = tex3D(jacobi_packed, ix, iy2, z);
}

__device__ void SaveToRegisters(float2* smem, uint si, uint bw, float* south,
                                float* west, float2* center, float* east,
                                float* north)
{
    __syncthreads();

    *south =  smem[si - bw].x;
    *west =   smem[si - 1].x;
    *center = smem[si];
    *east =   smem[si + 1].x;
    *north =  smem[si + bw].x;
}

__global__ void DampedJacobiKernel_smem_25d_32x6(float minus_square_cell_size,
                                                 float omega_over_beta,
                                                 uint3 volume_size)
{
    __shared__ float2 smem[384];

    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;

    const uint bw = blockDim.x + 16;
    const uint ox = blockIdx.x * blockDim.x + tx;
    const uint oy = blockIdx.y * blockDim.y + ty;

    const uint si = (ty + 1) * bw + tx + 8;

    float  south;
    float  west;
    float2 center;
    float  east;
    float  north;

    ReadBlockAndHalo_32x6(0, tx, ty, smem);
    SaveToRegisters(smem, si, bw, &south, &west, &center, &east, &north);

    ReadBlockAndHalo_32x6(1, tx, ty, smem);

    float t1 = omega_over_beta * 4.0f * center.x +
        (west + east + south + north + minus_square_cell_size * center.y) *
        omega_over_beta;
    float b = center.y;
    float near = t1;

    ushort2 raw;
    float far;

    for (uint i = 2; i < volume_size.z; i++) {
        SaveToRegisters(smem, si, bw, &south, &west, &center, &east, &north);
        ReadBlockAndHalo_32x6(i, tx, ty, smem);

        far = center.x * omega_over_beta;
        near = t1 + far;
        raw = make_ushort2(__float2half_rn(near), __float2half_rn(b));
        if (oy < volume_size.y)
            surf3Dwrite(raw, jacobi, ox * sizeof(ushort2), oy, i - 2,
                        cudaBoundaryModeTrap);

        // t1 is now pointing to plane |i - 1|.
        t1 = omega_over_beta * 3.0f * center.x +
            (west + east + south + north + near + minus_square_cell_size *
            center.y) * omega_over_beta;
        b = center.y;
    }

    SaveToRegisters(smem, si, bw, &south, &west, &center, &east, &north);
    if (oy >= volume_size.y)
        return;

    near = center.x * omega_over_beta + t1;
    raw = make_ushort2(__float2half_rn(near),
                       __float2half_rn(b));
    surf3Dwrite(raw, jacobi, ox * sizeof(ushort2), oy, volume_size.z - 2,
                cudaBoundaryModeTrap);

    t1 = omega_over_beta * 4.0f * center.x +
        (west + east + south + north + near + minus_square_cell_size *
        center.y) * omega_over_beta;
    raw = make_ushort2(__float2half_rn(t1), __float2half_rn(center.y));
    surf3Dwrite(raw, jacobi, ox * sizeof(ushort2), oy, volume_size.z - 1,
                cudaBoundaryModeTrap);
}

__global__ void DampedJacobiKernel_smem_branch(float minus_square_cell_size,
                                               float omega_over_beta,
                                               uint3 volume_size)
{
    __shared__ float2 cached_block[1000];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int bw = blockDim.x + 2;
    int bh = blockDim.y + 2;

    int index = (threadIdx.z + 1) * bw * bh + (threadIdx.y + 1) * bw +
        threadIdx.x + 1;
    cached_block[index] = tex3D(jacobi_packed, x, y, z);

    if (threadIdx.x == 0)
        cached_block[index - 1] = x == 0 ?                       cached_block[index] : tex3D(jacobi_packed, x - 1, y, z);

    if (threadIdx.x == blockDim.x - 1)
        cached_block[index + 1] = x == volume_size.x - 1 ?       cached_block[index] : tex3D(jacobi_packed, x + 1, y, z);

    if (threadIdx.y == 0)
        cached_block[index - bw] = y == 0 ?                      cached_block[index] : tex3D(jacobi_packed, x, y - 1, z);

    if (threadIdx.y == blockDim.y - 1)
        cached_block[index + bw] = y == volume_size.y - 1 ?      cached_block[index] : tex3D(jacobi_packed, x, y + 1, z);

    if (threadIdx.z == 0)
        cached_block[index - bw * bh] = z == 0 ?                 cached_block[index] : tex3D(jacobi_packed, x, y, z - 1);

    if (threadIdx.z == blockDim.z - 1)
        cached_block[index + bw * bh] = z == volume_size.z - 1 ? cached_block[index] : tex3D(jacobi_packed, x, y, z + 1);

    __syncthreads();

    float  near =   cached_block[index - bw * bh].x;
    float  south =  cached_block[index - bw].x;
    float  west =   cached_block[index - 1].x;
    float2 center = cached_block[index];
    float  east =   cached_block[index + 1].x;
    float  north =  cached_block[index + bw].x;
    float  far =    cached_block[index + bw * bh].x;

    float u = omega_over_beta * 3.0f * center.x +
        (west + east + south + north + far + near + minus_square_cell_size *
        center.y) * omega_over_beta;
    ushort2 raw = make_ushort2(__float2half_rn(u), __float2half_rn(center.y));
    surf3Dwrite(raw, jacobi, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void DampedJacobiKernel_smem_assist_thread(
    float minus_square_cell_size, float omega_over_beta, uint3 volume_size)
{
    // Shared memory solution with halo handled by assistant threads still
    // runs a bit slower than the texture-only way(less than 3ms on my GTX
    // 660Ti doing 40 times Jacobi).
    //
    // With the bank conflicts solved, I think the difference can be narrowed
    // down to around 1ms. But, it may say that the power of shared memory is
    // not as that great as expected, for Jacobi at least. Or maybe the texture
    // cache is truely really fast.

    const int cache_size = 1000;
    const int bd = 10;
    const int bh = 10;
    const int slice_stride = cache_size / bd;
    const int bw = slice_stride / bh;

    __shared__ float2 cached_block[cache_size];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int index = (threadIdx.z + 1) * slice_stride + (threadIdx.y + 1) * bw +
        threadIdx.x + 1;

    // Kernel runs faster if we place the normal fetch prior to the assistant
    // process.
    cached_block[index] = tex3D(jacobi_packed, x, y, z);

    int inner = 0;
    int inner_x = 0;
    int inner_y = 0;
    int inner_z = 0;
    switch (threadIdx.z) {
        case 0: {
            // near
            inner = (threadIdx.y + 1) * bw + threadIdx.x + 1;
            inner_x = x;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z - 1;
            cached_block[inner] = tex3D(jacobi_packed, inner_x, inner_y,
                                        inner_z);
            break;
        }
        case 1: {
            // south
            inner = (threadIdx.y + 1) * slice_stride + threadIdx.x + 1;
            inner_x = x;
            inner_y = blockIdx.y * blockDim.y - 1;
            inner_z = blockIdx.z * blockDim.z + threadIdx.y;
            cached_block[inner] = tex3D(jacobi_packed, inner_x, inner_y,
                                        inner_z);
            break;
        }
        case 2: {
            // west
            inner = (threadIdx.x + 1) * slice_stride + (threadIdx.y + 1) * bw;

            // It's more efficient to put z in the inner-loop than y.
            inner_x = blockIdx.x * blockDim.x - 1;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z + threadIdx.x;
            cached_block[inner] = tex3D(jacobi_packed, inner_x, inner_y,
                                        inner_z);
            break;
        }
        case 5:
            // east
            inner = (threadIdx.x + 1) * slice_stride + (threadIdx.y + 1) * bw +
                blockDim.x + 1;
            inner_x = blockIdx.x * blockDim.x + blockDim.x;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z + threadIdx.x;
            cached_block[inner] = tex3D(jacobi_packed, inner_x, inner_y,
                                        inner_z);
            break;
        case 6:
            // north
            inner = (threadIdx.y + 1) * slice_stride + (blockDim.y + 1) * bw +
                threadIdx.x + 1;
            inner_x = x;
            inner_y = blockIdx.y * blockDim.y + blockDim.y;
            inner_z = blockIdx.z * blockDim.z + threadIdx.y;
            cached_block[inner] = tex3D(jacobi_packed, inner_x, inner_y,
                                        inner_z);
            break;
        case 7:
            // far
            inner = (blockDim.z + 1) * slice_stride + (threadIdx.y + 1) * bw +
                threadIdx.x + 1;
            inner_x = x;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z + blockDim.z;
            cached_block[inner] = tex3D(jacobi_packed, inner_x, inner_y,
                                        inner_z);
            break;
    }
    __syncthreads();

    float  near =   cached_block[index - slice_stride].x;
    float  south =  cached_block[index - bw].x;
    float  west =   cached_block[index - 1].x;
    float2 center = cached_block[index];
    float  east =   cached_block[index + 1].x;
    float  north =  cached_block[index + bw].x;
    float  far =    cached_block[index + slice_stride].x;

    float u = omega_over_beta * 3.0f * center.x +
        (west + east + south + north + far + near + minus_square_cell_size *
        center.y) * omega_over_beta;
    ushort2 raw = make_ushort2(__float2half_rn(u), __float2half_rn(center.y));
    surf3Dwrite(raw, jacobi, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void DampedJacobiKernel_smem_faces_assist_thread(
    float minus_square_cell_size, float omega_over_beta, uint3 volume_size)
{
    const int cache_size = 512;
    const int bd = 8;
    const int bh = 8;
    const int slice_stride = cache_size / bd;
    const int bw = slice_stride / bh;

    __shared__ float2 cached_block[cache_size];
    __shared__ float cached_face_xyz0[bw * bh];
    __shared__ float cached_face_xyz1[bw * bh];
    __shared__ float cached_face_xzy0[bw * bd];
    __shared__ float cached_face_xzy1[bw * bd];
    __shared__ float cached_face_yzx0[bh * bd];
    __shared__ float cached_face_yzx1[bh * bd];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int index = threadIdx.z * slice_stride + threadIdx.y * bw + threadIdx.x;

    cached_block[index] = tex3D(jacobi_packed, x, y, z);

    int inner_x = 0;
    int inner_y = 0;
    int inner_z = 0;
    switch (threadIdx.z) {
        case 0: {
            // near
            inner_x = x;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z - 1;
            cached_face_xyz0[blockDim.x * threadIdx.y + threadIdx.x] =
                tex3D(jacobi_packed, inner_x, inner_y, inner_z).x;
            break;
        }
        case 1: {
            // south
            inner_x = x;
            inner_y = blockIdx.y * blockDim.y - 1;
            inner_z = blockIdx.z * blockDim.z + threadIdx.y;
            cached_face_xzy0[blockDim.x * threadIdx.y + threadIdx.x] =
                tex3D(jacobi_packed, inner_x, inner_y, inner_z).x;
            break;
        }
        case 2: {
            // west
            inner_x = blockIdx.x * blockDim.x - 1;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z + threadIdx.x;
            cached_face_yzx0[blockDim.y * threadIdx.y + threadIdx.x] =
                tex3D(jacobi_packed, inner_x, inner_y, inner_z).x;
            break;
        }
        case 5:
            // east
            inner_x = blockIdx.x * blockDim.x + blockDim.x;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z + threadIdx.x;
            cached_face_yzx1[blockDim.y * threadIdx.y + threadIdx.x] =
                tex3D(jacobi_packed, inner_x, inner_y, inner_z).x;
            break;
        case 6:
            // north
            inner_x = x;
            inner_y = blockIdx.y * blockDim.y + blockDim.y;
            inner_z = blockIdx.z * blockDim.z + threadIdx.y;
            cached_face_xzy1[blockDim.x * threadIdx.y + threadIdx.x] =
                tex3D(jacobi_packed, inner_x, inner_y, inner_z).x;
            break;
        case 7:
            // far
            inner_x = x;
            inner_y = y;
            inner_z = blockIdx.z * blockDim.z + blockDim.z;
            cached_face_xyz1[blockDim.x * threadIdx.y + threadIdx.x] =
                tex3D(jacobi_packed, inner_x, inner_y, inner_z).x;
            break;
    }
    __syncthreads();

    float2 center = cached_block[index];
    float near =  threadIdx.z == 0 ?              cached_face_xyz0[blockDim.x * threadIdx.y + threadIdx.x] : cached_block[index - slice_stride].x;
    float south = threadIdx.y == 0 ?              cached_face_xzy0[blockDim.x * threadIdx.z + threadIdx.x] : cached_block[index - bw].x;
    float west =  threadIdx.x == 0 ?              cached_face_yzx0[blockDim.y * threadIdx.y + threadIdx.z] : cached_block[index - 1].x;
    float east =  threadIdx.x == blockDim.x - 1 ? cached_face_yzx1[blockDim.y * threadIdx.y + threadIdx.z] : cached_block[index + 1].x;
    float north = threadIdx.y == blockDim.y - 1 ? cached_face_xzy1[blockDim.x * threadIdx.z + threadIdx.x] : cached_block[index + bw].x;
    float far =   threadIdx.z == blockDim.z - 1 ? cached_face_xyz1[blockDim.x * threadIdx.y + threadIdx.x] : cached_block[index + slice_stride].x;

    float u = omega_over_beta * 3.0f * center.x +
        (west + east + south + north + far + near + minus_square_cell_size *
        center.y) * omega_over_beta;
    ushort2 raw = make_ushort2(__float2half_rn(u), __float2half_rn(center.y));
    surf3Dwrite(raw, jacobi, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void DampedJacobiKernel_smem_dedicated_assist_thread(
    float minus_square_cell_size, float omega_over_beta, uint3 volume_size)
{
    __shared__ float2 cached_block[1000];

    int x = blockIdx.x * (blockDim.x - 2) + threadIdx.x - 1;
    int y = blockIdx.y * (blockDim.y - 2) + threadIdx.y - 1;
    int z = blockIdx.z * (blockDim.z - 2) + threadIdx.z - 1;

    int index = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
        threadIdx.x;

    cached_block[index] = tex3D(jacobi_packed, x, y, z);

    __syncthreads();

    if (threadIdx.x < 1 || threadIdx.x > blockDim.x - 2 ||
            threadIdx.y < 1 || threadIdx.y > blockDim.y - 2 ||
            threadIdx.z < 1 || threadIdx.z > blockDim.z - 2)
        return;

    float2 center = cached_block[index];
    float near =    cached_block[index - blockDim.x * blockDim.y].x;
    float south =   cached_block[index - blockDim.x].x;
    float west =    cached_block[index - 1].x;
    float east =    cached_block[index + 1].x;
    float north =   cached_block[index + blockDim.x].x;
    float far =     cached_block[index + blockDim.x * blockDim.y].x;

    float u = omega_over_beta * 3.0f * center.x +
        (west + east + south + north + far + near + minus_square_cell_size *
        center.y) * omega_over_beta;
    ushort2 raw = make_ushort2(__float2half_rn(u), __float2half_rn(center.y));
    surf3Dwrite(raw, jacobi, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

__global__ void DampedJacobiKernel_smem_no_halo_storage(
    float minus_square_cell_size, float omega_over_beta, uint3 volume_size)
{
    __shared__ float2 cached_block[512];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int index = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
        threadIdx.x;

    cached_block[index] = tex3D(jacobi_packed, x, y, z);
    __syncthreads();

    float center = cached_block[index].x;
    float near =  threadIdx.z == 0 ?              (z == 0 ?                 center : tex3D(jacobi_packed, x, y, z - 1.0f).x) : cached_block[index - blockDim.x * blockDim.y].x;
    float south = threadIdx.y == 0 ?              (y == 0 ?                 center : tex3D(jacobi_packed, x, y - 1.0f, z).x) : cached_block[index - blockDim.x].x;
    float west =  threadIdx.x == 0 ?              (x == 0 ?                 center : tex3D(jacobi_packed, x - 1.0f, y, z).x) : cached_block[index - 1].x;
    float east =  threadIdx.x == blockDim.x - 1 ? (x == volume_size.x - 1 ? center : tex3D(jacobi_packed, x + 1.0f, y, z).x) : cached_block[index + 1].x;
    float north = threadIdx.y == blockDim.y - 1 ? (y == volume_size.y - 1 ? center : tex3D(jacobi_packed, x, y + 1.0f, z).x) : cached_block[index + blockDim.x].x;
    float far =   threadIdx.z == blockDim.z - 1 ? (z == volume_size.z - 1 ? center : tex3D(jacobi_packed, x, y, z + 1.0f).x) : cached_block[index + blockDim.x * blockDim.y].x;

    float b_center = cached_block[index].y;

    float u = omega_over_beta * 3.0f * center +
        (west + east + south + north + far + near + minus_square_cell_size *
        b_center) * omega_over_beta;
    ushort2 raw = make_ushort2(__float2half_rn(u), __float2half_rn(b_center));

    surf3Dwrite(raw, jacobi, x * sizeof(ushort2), y, z, cudaBoundaryModeTrap);
}

// =============================================================================

void LaunchDampedJacobi(cudaArray* dest_array, cudaArray* source_array,
                        float minus_square_cell_size, float omega_over_beta,
                        uint3 volume_size, BlockArrangement* ba)
{
    cudaChannelFormatDesc desc;
    cudaGetChannelDesc(&desc, dest_array);
    cudaError_t result = cudaBindSurfaceToArray(&jacobi, dest_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    cudaGetChannelDesc(&desc, source_array);
    jacobi_packed.normalized = false;
    jacobi_packed.filterMode = cudaFilterModePoint;
    jacobi_packed.addressMode[0] = cudaAddressModeClamp;
    jacobi_packed.addressMode[1] = cudaAddressModeClamp;
    jacobi_packed.addressMode[2] = cudaAddressModeClamp;
    jacobi_packed.channelDesc = desc;

    result = cudaBindTextureToArray(&jacobi_packed, source_array, &desc);
    assert(result == cudaSuccess);
    if (result != cudaSuccess)
        return;

    bool smem = false;
    bool smem_25d = true;
    if (smem_25d) {
        dim3 block(32, 6, 1);
        dim3 grid((volume_size.x + block.x - 1) / block.x,
                  (volume_size.y + block.y - 1) / block.y,
                  1);
        DampedJacobiKernel_smem_25d_32x6<<<grid, block>>>(
            minus_square_cell_size, omega_over_beta, volume_size);
    } else if (smem) {
        dim3 block(8, 8, 8);
        dim3 grid((volume_size.x + block.x - 1) / block.x,
                  (volume_size.y + block.y - 1) / block.y,
                  (volume_size.z + block.z - 1) / block.z);
        DampedJacobiKernel_smem_assist_thread<<<grid, block>>>(
            minus_square_cell_size, omega_over_beta, volume_size);
    } else {
        dim3 block;
        dim3 grid;
        ba->Arrange(&block, &grid, volume_size);
        DampedJacobiKernel<<<grid, block>>>(minus_square_cell_size,
                                            omega_over_beta, volume_size);
    }
    

    cudaUnbindTexture(&jacobi_packed);
}
