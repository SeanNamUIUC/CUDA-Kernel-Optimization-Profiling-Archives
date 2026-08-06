#include "cuda_check.h"
#include "timer.h"
#include<kernels/01_sgemm/01_naive_mm.cu>
#include<kernels/01_sgemm/02_shared_memory_mm.cu>


#include <vector>

void cpu_sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

bool verify_result(const float* host_res, const float* device_res, int size) {
    for (int i = 0; i < size; ++i) {
        if (fabs(host_res[i] - device_res[i]) > 1e-2f) {
            std::cout << "Validation FAILED at index " << i 
                      << " [CPU: " << host_res[i] << ", GPU: " << device_res[i] << "]\n";
            return false;
        }
    }
    return true;
}

int main() {
    //1. Naive Matrix Multiplication testing

    // Initialize host matrices A, B, and C
    int M = 2048, N = 2048, K = 2048;
    std::vector<float> h_A(M * N), h_B(N * K), h_C_CPU(M * K), h_C_GPU(M * K);
    for (int i = 0; i < M * N; ++i) h_A[i] = (float)rand() / RAND_MAX; //returns 0.0 ~ 1.0
    for (int j = 0; j < N * K; ++j) h_B[j] = (float)rand() / RAND_MAX; //returns 0.0 ~ 1.0
    for (int k = 0; k < M * K; ++k) h_C_CPU[k] = 0.0f;
    for (int k = 0; k < M * K; ++k) h_C_GPU[k] = 0.0f;

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * K * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));

    // CPU calculation for verification
    std::cout << "Running CPU SGEMM...\n";
    cpu_sgemm(h_A.data(), h_B.data(), h_C_CPU.data(), M, N, K);

    //GPU Timer for measuring kernel execution time
    GpuTimer timer;

    // 1. Naive Matrix Multiplication kernel 호출 및 시간 측정
    std::cout << "Running GPU Naive SGEMM...\n";
    timer.start();
    solve_naive_mm(d_A, d_B, d_C, M, N, K);
    timer.stop();
    std::cout << "01_Naive Time: " << timer.elapsed_ms() << " ms\n";


    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C_GPU.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost));
    // 결과 검증
    if (verify_result(h_C_CPU.data(), h_C_GPU.data(), M * K)) {
        std::cout << "Validation PASSED! CPU and GPU results match.\n";
    }
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}