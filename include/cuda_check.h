#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <cuda_runtime.h>
#include<iostream>
#include<cstdlib>

//cuda API checking macro (ex) cudaMalloc, cudaMemcpy )
//eturns 0 if success (cudaSuccess  = 0) 
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            std::cerr << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ \
                      << " -> " << cudaGetErrorString(err)              \
                      << " (" << # call << ")" << std::endl;            \
            std::exit(EXIT_FAILURE);                                    \
        } \
    } while (0)


// 커널 실행(<<<...>>>) 직후 비동기 오류 및 동기화 오류 검증 매크로

#define CUDA_CHECK_LAST_ERROR() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA KERNEL LAUNCH ERROR] " << __FILE__ << ":" << __LINE__ \
                      << " -> " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA KERNEL EXECUTION ERROR] " << __FILE__ << ":" << __LINE__ \
                      << " -> " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

#endif 