#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
                                                int row = blockIdx.y * blockDim.y + threadIdx.y;
                                                int col = blockIdx.x * blockDim.x + threadIdx.x;
                                                //
                                                //(row, col) 스레드 하나가 행렬 범위를 넘지않게 막고
                                                //C 결과값한칸을 독점해서 A 와 B의 값들을 읽고 연산
                                                if (row < M && col < K) {
                                                    float sum = 0.0f;
                                                    //A의 열, B의 행을 동시에 훑는 공통 축
                                                    for (int n = 0; n < N; n++) {
                                                        sum += A[row*N + n] * B[n*K + col];
                                                    }
                                                    C[row * K + col] = sum;
                                                    
                                                }
                                             }

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
