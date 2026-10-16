// kernels/01_sgemm/01_naive_mm.cu
// 규칙: A(M x K), B(K x N), C(M x N), 모두 row-major. K가 곱해서 없어지는 축.
#include <cuda_runtime.h>

// 스레드 하나가 결과 C의 원소 (row, col) 하나를 맡아서 계산
__global__ void naive_mm_kernel(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    // 이 스레드가 맡을 C의 좌표
    // threadIdx.x를 col로 잡기 -> warp 가 블록안의 스레드들을 threadIdx.x 기준으로 묶어서 실행하므로, col이 연속되게 잡히는 것이 좋음.
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 0 ~ M-1
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 0 ~ N-1

    // 그리드가 행렬보다 크게 잡힐 수 있으니 범위 밖 스레드는 아무것도 하지 않음
    if (row < M && col < N) {
        float sum = 0.0f;
        // 공통 축 K를 따라 A의 row번째 행과 B의 col번째 열을 곱해 더함
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// A, B, C는 GPU 메모리 포인터
void solve_naive_mm(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    // grid.x는 C의 열(N) 방향, grid.y는 행(M) 방향
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    naive_mm_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
}