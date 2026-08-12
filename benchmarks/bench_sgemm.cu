#include "cuda_check.h"
#include "timer.h"
#include<kernels/01_sgemm/01_naive_mm.cu>
#include<kernels/01_sgemm/02_shared_memory_mm.cu>


#include <vector>
//GFLOPS calculation
double calculate_gflops(int M, int N, int K, float time_ms) {
    double operations = 2.0 * M * N * K; // Total floating-point operations
    double time_s = time_ms / 1000.0; // Convert milliseconds to seconds
    double gflops = (operations / time_s) / 1e9; // Convert to GFLOPS
    return gflops;
}

// CPU implementation of SGEMM for verification
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
// Result verification function
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

//
template <typename KernelFunc>
void run_kernel_benchmark(const char* name, KernelFunc kernel_fn, 
                         const float* d_A, const float* d_B, float* d_C, 
                         float* h_C_GPU, const float* h_C_CPU, 
                         int M, int N, int K, GpuTimer& timer){         
    CUDA_CHECK(cudaMemset(d_C, 0, M * K * sizeof(float))); // Reset output matrix on device
    
    kernel_fn(d_A, d_B, d_C, M, N, K); // Warm-up run
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure kernel has completed

    int iterations = 10; // Number of iterations for averaging      
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        kernel_fn(d_A, d_B, d_C, M, N, K);
    }
    timer.stop();
    
    float avg_time_ms = timer.elapsed_ms() / iterations; // Average time per iteration
    double gflops = calculate_gflops(M, N, K, avg_time_ms);
    std::cout << name << " Average Time: " << avg_time_ms << " ms\n";
    std::cout << name << " GFLOPS: " << gflops << "\n"; 

    // verify result with verify_result() function
    CUDA_CHECK(cudaMemcpy(h_C_GPU, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (verify_result(h_C_CPU, h_C_GPU, M * K)) {
        std::cout << "Validation: PASSED!\n";
    }

    

}
int main() {
    
    // Initialize host matrices A, B, and C
    int M = 4096, N = 4096, K = 4096;
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

    
    //1.Benchmarking the naive matrix multiplication kernel
    // std::cout << "Running Naive SGEMM Kernel...\n";
    // run_kernel_benchmark("01_naive_sgemm", solve_naive_mm, 
    //                      d_A, d_B, d_C, h_C_GPU.data(), h_C_CPU.data(), M, N, K, timer);

    
    //2.Benchmarking the shared memory matrix multiplication kernel
    std::cout << "Running Shared Memory SGEMM Kernel...\n";
    run_kernel_benchmark("02_shared_memory_sgemm", solve_shared_mm,
                         d_A, d_B, d_C, h_C_GPU.data(), h_C_CPU.data(), M, N, K, timer);

    
    //Device memory deallocation
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}