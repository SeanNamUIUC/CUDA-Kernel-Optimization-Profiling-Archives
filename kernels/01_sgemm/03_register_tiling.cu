// kernels/01_sgemm/04_2d_blocktile_mm.cu
// 규칙: A(M x K), B(K x N), C(M x N), 모두 row-major. K가 곱해서 없어지는 축.
#include <cuda_runtime.h>

namespace blocktile2d {

// BM x BN : 블록 하나가 계산할 C 타일 행 수 와 열 수
// BK      : 공통 축 K를 몇 칸씩 잘라 shared memory에 올릴지 (8)
// TM x TN : 스레드 하나가 레지스터에 들고 계산할 C 영역 (8 x 8)
constexpr int BM = 128;
constexpr int BN = 128;

constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int NUM_THREADS = (BM / TM) * (BN / TN);  // 16 x 16 = 256

__global__ void kernel(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int M, int N, int K) {
    // 1. 블록 안에서 스레드 좌표 (0 ~ 15)
    int ty = threadIdx.y;  // 이 스레드가 맡을 TM줄 묶음 번호
    int tx = threadIdx.x;  // 이 스레드가 맡을 TN칸 묶음 번호

    // 2. shared memory 타일
    __shared__ float As[BM][BK];  // A 조각 128 x 8
    __shared__ float Bs[BK][BN];  // B 조각 8 x 128

    // 3. 레지스터: 결과 8 x 8, 그리고 As 열 하나, Bs 행 하나를 잠시 담을 그릇
    float regC[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    // 4. 협동 로딩용 1차원 번호 (0 ~ 255)와 각자 맡을 칸
    int tid = ty * blockDim.x + tx;
    int innerRowA = tid / BK;  // 0 ~ 31
    int innerColA = tid % BK;  // 0 ~ 7
    int innerRowB = tid / BN;  // 0 ~ 1
    int innerColB = tid % BN;  // 0 ~ 127
    constexpr int strideA = NUM_THREADS / BK;  // 한 번에 채우는 As 행 수 = 32
    constexpr int strideB = NUM_THREADS / BN;  // 한 번에 채우는 Bs 행 수 = 2

    // 5. 이 블록이 맡은 C 타일의 시작 좌표
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    // 6. 공통 축 K를 BK(8)칸씩 이동
    for (int k0 = 0; k0 < K; k0 += BK) {
        // 6-1. A 조각(128 x 8) 로드: 32행씩 4번
        for (int off = 0; off < BM; off += strideA) {
            int r = innerRowA + off;// A 조각에서의 행 번호 (0 ~ 127)
            int gRow = blockRow + r;      // A의 전역 행 (M 방향)   
            int gCol = k0 + innerColA;    // A의 전역 열 (K 방향)
        
            As[r][innerColA] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f; 
        }
        // 6-2. B 조각(8 x 128) 로드: 2행씩 4번
        for (int off = 0; off < BK; off += strideB) {
            int r = innerRowB + off;// B 조각에서의 행 번호 (0 ~ 7)
            int gRow = k0 + r;            // B의 전역 행 (K 방향)
            int gCol = blockCol + innerColB;  // B의 전역 열 (N 방향)
            Bs[r][innerColB] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
        }

        // 6-3. 모두 다 채울 때까지 대기
        __syncthreads();

        // 6-4. As 열 하나(8개) x Bs 행 하나(8개) 외적을 8번 누적
        #pragma unroll
        for (int d = 0; d < BK; ++d) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regA[i] = As[ty * TM + i][d];
            #pragma unroll
            for (int j = 0; j < TN; ++j) regB[j] = Bs[d][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    regC[i][j] += regA[i] * regB[j];
                }
            }
        }

        // 6-5. 모두 계산을 끝낼 때까지 대기 (다음 조각으로 덮어쓰기 전에)
        __syncthreads();
    }

    // 7. 레지스터의 8 x 8 결과를 C에 쓰기
    for (int i = 0; i < TM; ++i) {
        int row = blockRow + ty * TM + i;
        if (row >= M) continue;
        for (int j = 0; j < TN; ++j) {
            int col = blockCol + tx * TN + j;
            if (col < N) C[row * N + col] = regC[i][j];
        }
    }
    }

}  // namespace blocktile2d

// A, B, C는 GPU 메모리 포인터. 동기화는 하지 않음 (벤치 타이머가 해 줌).
void solve_2d_blocktile_mm(const float* A, const float* B, float* C, int M, int N, int K) {
    using namespace blocktile2d;
    dim3 block(BN / TN, BM / TM);                    // (16, 16) = 256 스레드
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM); // x는 열(N), y는 행(M)
    kernel<<<grid, block>>>(A, B, C, M, N, K);
}