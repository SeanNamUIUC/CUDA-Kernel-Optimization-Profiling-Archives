// kernels/01_sgemm/02_shared_memory_mm.cu
// 규칙: A(M x K), B(K x N), C(M x N), 모두 row-major. K가 곱해서 없어지는 축.
#include <cuda_runtime.h>

// 블록 하나가 C의 TILE x TILE 조각 하나를 계산해요.
// K 방향으로 TILE씩 이동하면서 A·B 조각을 shared memory에 올려 두고, 블록 안 스레드들이 재사용해요.
template <int TILE>
__global__ void shared_mm_kernel(const float* A, const float* B, float* C,
                                 int M, int N, int K) {
    // 1. shared memory 타일 공간. 같은 블록 안의 스레드들만 같이 씀
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int ty = threadIdx.y;  // 타일 안에서 행 위치 (0 ~ TILE-1)
    int tx = threadIdx.x;  // 타일 안에서 열 위치 (0 ~ TILE-1)

    // 2. 이 스레드가 맡을 C의 전역 좌표
    int row = blockIdx.y * TILE + ty;  // 0 ~ M-1
    int col = blockIdx.x * TILE + tx;  // 0 ~ N-1

    // 3. 레지스터에 누적할 합
    float sum = 0.0f;

    // 4. 공통 축 K를 TILE 단위로 나눠 이동
    //    A는 가로(열) 방향, B는 세로(행) 방향으로 타일을 옮겨 가며 계산
    //    t: C의 한 칸을 완성하기 위해 지나가는 타일 번호
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {

        // 4-1. A(M x K)의 타일 로드: A[row][t*TILE + tx]
        //      row는 고정, 열(K 방향)을 TILE칸씩 가로로 훑음
        int a_col = t * TILE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;  // 범위 밖은 0으로 채워서 곱해도 결과에 영향 없게
        }

        // 4-2. B(K x N)의 타일 로드: B[t*TILE + ty][col]
        //      col은 고정, 행(K 방향)을 TILE칸씩 세로로 훑음
        int b_row = t * TILE + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // 4-3. 동기화 1: 블록의 모든 스레드가 As, Bs를 다 채울 때까지 대기
        __syncthreads();

        // 4-4. shared memory에서 TILE번 곱해 더하기
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // 4-5. 동기화 2: 모든 스레드가 이번 타일 계산을 끝낼 때까지 대기
        //      (왜 필요한지 직접 한 줄로 적어 보기)
        __syncthreads();
    }

    // 5. 최종 결과를 C(M x N)에 쓰기
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// A, B, C는 GPU 메모리 포인터
// 동기화는 하지 않음 (벤치마크의 타이머가 이벤트로 동기화함).
template <int TILE>
void solve_shared_mm(const float* A, const float* B, float* C, int M, int N, int K) {
    // 블록 크기를 타일 크기와 1:1로 맞춤 (스레드 하나가 타일의 한 칸씩 로드)
    dim3 threadsPerBlock(TILE, TILE);
    // grid.x는 C의 열(N) 방향, grid.y는 행(M) 방향
    dim3 blocksPerGrid((N + TILE - 1) / TILE,
                       (M + TILE - 1) / TILE);
    shared_mm_kernel<TILE><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
}