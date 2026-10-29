// benchmarks/bench_sgemm.cu
//
//  1. cuBLAS를 기준선으로 같이 측정하고, 정답 검증도 cuBLAS 결과로 함
//  2. 상대 오차로 검증
//  3. 행렬 크기를 실행 인자로 받음:  ./bench_sgemm 1024 2048 4096
//  4. 결과를 표 형식 + CSV(results/sgemm.csv)로 저장
//  5. 측정 전에 정사각형이 아닌 크기로 정확성부터 확인 (M, N, K를 헷갈리는 버그는 여기서만 잡힘)
//
// 주의: 성능 숫자는 반드시 ncu 없이 그냥 실행해서 측정할 것.
//       ncu는 커널을 여러 번 재실행(replay)하고 클럭도 고정해서 시간이 달라짐.
//
// 빌드 (RTX 5070 Ti = sm_120):
//   nvcc -O3 -arch=sm_120 -I. -Iinclude benchmarks/bench_sgemm.cu -lcublas -o bench_sgemm

#include "cuda_check.h"
#include "timer.h"
#include <cublas_v2.h>

// 만든 커널들
#include <kernels/01_sgemm/01_naive_mm.cu>
#include <kernels/01_sgemm/02_shared_memory_mm.cu>
#include <kernels/01_sgemm/03_register_tiling.cu>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// cuBLAS 호출 오류 확인
#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t st = (call);                                               \
        if (st != CUBLAS_STATUS_SUCCESS) {                                        \
            std::cerr << "[CUBLAS ERROR] " << __FILE__ << ":" << __LINE__         \
                      << " status=" << st << std::endl;                           \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

// 이 저장소의 규칙 (cuBLAS·CUTLASS와 같은 표준):
//   A(M x K), B(K x N), C(M x N), 모두 row-major. K가 곱해서 없어지는 축.
// 모든 커널 함수 모양: void solve_xxx(const float* A, const float* B, float* C, int M, int N, int K)
using SgemmFn = void (*)(const float*, const float*, float*, int, int, int);

// 커널 이름 + 함수 포인터
struct KernelEntry {
    const char* name;
    SgemmFn fn;
};

// cuBLAS 핸들 전역 변수
static cublasHandle_t g_handle;

// cuBLAS는 column-major라서
// row-major C(MxN) = A(MxK) * B(KxN)  ==  column-major C^T(NxM) = B^T(NxK) * A^T(KxM)
// (row-major 행렬을 그대로 넘기면 cuBLAS 눈에는 전치 행렬로 보이는 것을 이용)
void cublas_sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    //column-major로 전치된 것처럼 넘기기 위해 A와 B를 바꿔서 호출
    CUBLAS_CHECK(cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha, B, N, A, K,
                             &beta, C, N));
}

//초당 10억 연산단위로 -> 숫자가 클수록 빠른 커널
double gflops(int M, int N, int K, double ms) {

    return 2.0 * M * N * K / (ms * 1e-3) / 1e9;
}

// 평균 실행 시간(ms). 워밍업 후 iters번 실행.
// 커널 wrapper 안에서 cudaDeviceSynchronize()를 하면 안 됨 (매 반복마다 동기화 비용이 섞임).
// timer.stop()이 이벤트로 동기화해 줌.
float time_kernel(SgemmFn fn, const float* dA, const float* dB, float* dC,
                  int M, int N, int K, int iters, GpuTimer& timer) {
    //// 워밍업                
    for (int i = 0; i < 3; ++i) fn(dA, dB, dC, M, N, K);  
    CUDA_CHECK_LAST_ERROR();
    
    //평균 1회 실행시간 측정
    timer.start();
    for (int i = 0; i < iters; ++i) fn(dA, dB, dC, M, N, K);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());
    return timer.elapsed_ms() / iters;
}

// 오차검증함수 
//atol (absolute tolerance) 절대오차-> diff = |ref - out|, 차이가 0.001 이하면 pass -> diff <= 0.001
// rtol (relative tolerance) 상대오차-> rel = |ref - out| / |ref|, 정답의 0.1% 이하면 pass -> rel <= 0.001
// 사용한 방식: 절대오차와 상대오차를 모두 고려하여, diff <= rtol * |ref| + atol 이면 pass
//ref: cuBLAS 결과, out: 내 커널 결과, max_rel: 최대 상대 오차 반환
bool verify(const std::vector<float>& ref, const std::vector<float>& out, float& max_rel) {
    const float rtol = 1e-3f, atol = 1e-3f;
    //최대 상대오차
    max_rel = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        float diff = std::fabs(out[i] - ref[i]);
        //  /0 방지용 
        float rel = diff / (std::fabs(ref[i]) + 1e-6f);
        if (rel > max_rel) max_rel = rel;
        //정답 * 0.1% + 0.001 보다 차이가 크면 틀린 것으로 간주
        if (diff > rtol * std::fabs(ref[i]) + atol) {
            std::printf("    mismatch at %zu: ref=%f got=%f\n", i, ref[i], out[i]);
            return false;
        }
    }
    return true;
}

// 정사각형이 아니고 타일 배수도 아닌 크기로 정확성만 확인.
// M = N = K일 때는 M, N, K를 바꿔 써도 결과가 맞아 버려서, 이런 버그는 여기서만 잡힘

//1.정사각형이 아닌 크기: (1000, 777, 1234), (33, 65, 17), (128, 1, 300)
//2. cuBLAS 결과를 계산해서 href에 저장
//3. 내 커널들을 실행해서 dC에 저장 후, hOut에 가져와서 cuBLAS 결과와 비교

bool check_shapes(const std::vector<KernelEntry>& kernels) {
    struct Shape { int M, N, K; };
    const Shape shapes[] = {{1000, 777, 1234}, {33, 65, 17}, {128, 1, 300}};
    bool all_ok = true;

    std::printf("=== 정확성 확인: 정사각형이 아닌 크기 ===\n");
    for (const auto& s : shapes) {
        size_t szA = (size_t)s.M * s.K, szB = (size_t)s.K * s.N, szC = (size_t)s.M * s.N;
        //host vector
        std::vector<float> hA(szA), hB(szB), hRef(szC), hOut(szC);
        //0에서 1 사이 랜덤값
        for (auto& x : hA) x = (float)rand() / RAND_MAX - 0.5f;
        for (auto& x : hB) x = (float)rand() / RAND_MAX - 0.5f;
        

        //GPU device vector
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, szA * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, szB * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, szC * sizeof(float)));
        // host -> device
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), szA * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), szB * sizeof(float), cudaMemcpyHostToDevice));

        cublas_sgemm(dA, dB, dC, s.M, s.N, s.K);
        CUDA_CHECK_LAST_ERROR();
        //GPU의 dC -> cpu의 hRef 사용하여 cuBLAS 결과를 가져옴
        CUDA_CHECK(cudaMemcpy(hRef.data(), dC, szC * sizeof(float), cudaMemcpyDeviceToHost));

        for (const auto& k : kernels) {
            CUDA_CHECK(cudaMemset(dC, 0, szC * sizeof(float)));
            k.fn(dA, dB, dC, s.M, s.N, s.K);
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaMemcpy(hOut.data(), dC, szC * sizeof(float), cudaMemcpyDeviceToHost));

            float max_rel;
            bool ok = verify(hRef, hOut, max_rel);
            all_ok = all_ok && ok;
            std::printf("  %-18s M=%-5d N=%-5d K=%-5d %s\n",
                        k.name, s.M, s.N, s.K, ok ? "PASS" : "FAIL");
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }
    std::printf("\n");
    return all_ok;
}

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {1024, 2048, 4096};

    std::vector<KernelEntry> kernels = {
        {"01_naive", solve_naive_mm},
        {"02_shared_t16", solve_shared_mm<16>},
        {"02_shared_t32", solve_shared_mm<32>},
        // {"03_1d_blocktile", solve_1d_blocktile_mm},
        {"04_2d_blocktile", solve_2d_blocktile_mm}
    };
    // GPU 정보 출력
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    CUBLAS_CHECK(cublasCreate(&g_handle));
    // TF32를 쓰지 않는 순수 FP32 기준선 (공정한 비교)
    CUBLAS_CHECK(cublasSetMathMode(g_handle, CUBLAS_PEDANTIC_MATH));
    // 시드 고정: 랜덤값이 바뀌면 정확성 검증이 달라짐
    srand(42);

    // 0) 정확성 먼저. 하나라도 틀리면 속도는 재지 않음.
    if (!check_shapes(kernels)) {
        std::printf("정확성 확인 실패: 커널을 먼저 고치세요. 속도 측정은 건너뜁니다.\n");
        CUBLAS_CHECK(cublasDestroy(g_handle));
        return 1;
    }

    std::FILE* csv = std::fopen("results/sgemm.csv", "w");
    if (csv) std::fprintf(csv, "kernel,M,N,K,time_ms,gflops,pct_of_cublas,correct\n");

    GpuTimer timer;

    for (int S : sizes) {
        int M = S, N = S, K = S;
        size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;

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
        if (csv) std::fprintf(csv, "cuBLAS,%d,%d,%d,%.4f,%.1f,100.0,ref\n", M, N, K, ref_ms, ref_gf);

        // 2) 내 커널들
        for (const auto& k : kernels) {
            CUDA_CHECK(cudaMemset(dC, 0, szC * sizeof(float)));
            float ms = time_kernel(k.fn, dA, dB, dC, M, N, K, iters, timer);
            CUDA_CHECK(cudaMemcpy(hOut.data(), dC, szC * sizeof(float), cudaMemcpyDeviceToHost));

            float max_rel;
            bool ok = verify(hRef, hOut, max_rel);
            double gf = gflops(M, N, K, ms);
            double pct = 100.0 * gf / ref_gf;
            std::printf("%-18s %10.3f %10.0f %9.1f%% %8s\n", k.name, ms, gf, pct, ok ? "PASS" : "FAIL");
            if (csv) std::fprintf(csv, "%s,%d,%d,%d,%.4f,%.1f,%.1f,%s\n",
                                  k.name, M, N, K, ms, gf, pct, ok ? "pass" : "fail");
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