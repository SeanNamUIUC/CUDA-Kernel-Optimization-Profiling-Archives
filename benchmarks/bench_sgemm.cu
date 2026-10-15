// benchmarks/bench_sgemm.cu  (개선판)

//  1. cuBLAS를 기준선으로 같이 측정하고, 정답 검증도 cuBLAS 결과로 함 (CPU SGEMM 제거 → 4096도 몇 초면 끝)
//  2. 상대 오차로 검증 (K가 커지면 절대 오차 1e-2는 너무 느슨하거나 빡빡해짐)
//  3. 행렬 크기를 실행 인자로 받음:  ./bench_sgemm 1024 2048 4096
//  4. 결과를 표 형식 + CSV(results/sgemm.csv)로 저장 → README 표/그래프에 바로 사용
//
// 주의: 성능 숫자는 반드시 ncu 없이 그냥 실행해서 측정할 것.
//       ncu는 커널을 여러 번 재실행(replay)해서 시간이 수십 ms씩 부풀려짐.
//
// 빌드 (RTX 5070 Ti = sm_120):
//   nvcc -O3 -arch=sm_120 -I. -Iinclude benchmarks/bench_sgemm.cu -lcublas -o bench_sgemm
//


#include "cuda_check.h"
#include "timer.h"
#include <cublas_v2.h> // 행렬 곱셈 라이브러리



//만든 커널들 
#include <kernels/01_sgemm/01_naive_mm.cu>
#include <kernels/01_sgemm/02_shared_memory_mm.cu>


#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

//cublas 오류 체크 매크로
// cuBLAS 는 cublasStatus_t 를 반환하므로, CUDA_CHECK와 비슷한 매크로를 만들어서 오류 체크
#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t st = (call);                                               \
        if (st != CUBLAS_STATUS_SUCCESS) {                                        \
            std::cerr << "[CUBLAS ERROR] " << __FILE__ << ":" << __LINE__         \
                      << " status=" << st << std::endl;                           \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

// 이 저장소의 규칙: A(M x N), B(N x K), C(M x K), 모두 row-major
// naive, shared memory, register tiling 모든 커널은 이 규칙을 따름
using SgemmFn = void (*)(const float*, const float*, float*, int, int, int);

struct KernelEntry {
    const char* name;
    SgemmFn fn;
};

static cublasHandle_t g_handle;

// cuBLAS는 column-major라서 row-major C = A*B 를 C^T = B^T * A^T 로 계산
// (row-major 행렬을 그대로 넘기면 cuBLAS 눈에는 전치 행렬로 보이는 것을 이용)
void cublas_sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             K, M, N,
                             &alpha, B, K, A, N,
                             &beta, C, K));
}

double gflops(int M, int N, int K, double ms) {
    return 2.0 * M * N * K / (ms * 1e-3) / 1e9;
}

// 평균 실행 시간(ms). 워밍업 후 iters번 실행.
float time_kernel(SgemmFn fn, const float* dA, const float* dB, float* dC,
                  int M, int N, int K, int iters, GpuTimer& timer) {
    for (int i = 0; i < 3; ++i) fn(dA, dB, dC, M, N, K);  // 워밍업 (클럭 올리기, 캐시 등)
    CUDA_CHECK_LAST_ERROR();

    timer.start();
    for (int i = 0; i < iters; ++i) fn(dA, dB, dC, M, N, K);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());
    return timer.elapsed_ms() / iters;
}

// 상대 오차 검증: |gpu - ref| <= rtol * |ref| + atol
bool verify(const std::vector<float>& ref, const std::vector<float>& out, float& max_rel) {
    const float rtol = 1e-3f, atol = 1e-3f;
    max_rel = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        float diff = std::fabs(out[i] - ref[i]);
        float rel = diff / (std::fabs(ref[i]) + 1e-6f);
        if (rel > max_rel) max_rel = rel;
        if (diff > rtol * std::fabs(ref[i]) + atol) {
            std::printf("    mismatch at %zu: ref=%f got=%f\n", i, ref[i], out[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {1024, 2048, 4096};

    std::vector<KernelEntry> kernels = {
        {"01_naive", solve_naive_mm},
        {"02_shared_tile", solve_shared_mm},
        // {"03_1d_blocktile", solve_1d_blocktile_mm},
        // {"04_2d_blocktile", solve_2d_blocktile_mm},
    };

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    CUBLAS_CHECK(cublasCreate(&g_handle));
    // TF32를 쓰지 않는 순수 FP32 기준선 (공정한 비교)
    CUBLAS_CHECK(cublasSetMathMode(g_handle, CUBLAS_PEDANTIC_MATH));

    std::FILE* csv = std::fopen("results/sgemm.csv", "w");
    if (csv) std::fprintf(csv, "kernel,size,time_ms,gflops,pct_of_cublas,correct\n");

    GpuTimer timer;
    srand(42);

    for (int S : sizes) {
        int M = S, N = S, K = S;
        size_t szA = (size_t)M * N, szB = (size_t)N * K, szC = (size_t)M * K;

        std::vector<float> hA(szA), hB(szB), hRef(szC), hOut(szC);
        for (auto& x : hA) x = (float)rand() / RAND_MAX - 0.5f;  // 평균 0 → 오차가 덜 쌓임
        for (auto& x : hB) x = (float)rand() / RAND_MAX - 0.5f;

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, szA * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, szB * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, szC * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), szA * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), szB * sizeof(float), cudaMemcpyHostToDevice));

        int iters = (S <= 1024) ? 50 : (S <= 2048 ? 20 : 10);

        // 1) cuBLAS: 기준 시간 + 정답
        float ref_ms = time_kernel(cublas_sgemm, dA, dB, dC, M, N, K, iters, timer);
        CUDA_CHECK(cudaMemcpy(hRef.data(), dC, szC * sizeof(float), cudaMemcpyDeviceToHost));
        double ref_gf = gflops(M, N, K, ref_ms);

        std::printf("=== %d x %d x %d ===\n", M, N, K);
        std::printf("%-18s %10s %10s %10s %8s\n", "kernel", "time(ms)", "GFLOPS", "%cuBLAS", "check");
        std::printf("%-18s %10.3f %10.0f %9.1f%% %8s\n", "cuBLAS", ref_ms, ref_gf, 100.0, "ref");
        if (csv) std::fprintf(csv, "cuBLAS,%d,%.4f,%.1f,100.0,ref\n", S, ref_ms, ref_gf);

        // 2) 내 커널들
        for (auto& k : kernels) {
            CUDA_CHECK(cudaMemset(dC, 0, szC * sizeof(float)));
            float ms = time_kernel(k.fn, dA, dB, dC, M, N, K, iters, timer);
            CUDA_CHECK(cudaMemcpy(hOut.data(), dC, szC * sizeof(float), cudaMemcpyDeviceToHost));

            float max_rel;
            bool ok = verify(hRef, hOut, max_rel);
            double gf = gflops(M, N, K, ms);
            double pct = 100.0 * gf / ref_gf;
            std::printf("%-18s %10.3f %10.0f %9.1f%% %8s\n", k.name, ms, gf, pct, ok ? "PASS" : "FAIL");
            if (csv) std::fprintf(csv, "%s,%d,%.4f,%.1f,%.1f,%s\n", k.name, S, ms, gf, pct, ok ? "pass" : "fail");
        }
        std::printf("\n");

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    if (csv) std::fclose(csv);
    CUBLAS_CHECK(cublasDestroy(g_handle));
    return 0;
}