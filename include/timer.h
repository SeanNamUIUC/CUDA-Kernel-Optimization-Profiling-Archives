#ifndef TIMER_H
#define TIMER_H

#include <cuda_runtime.h>
#include "cuda_check.h"
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event_));
        CUDA_CHECK(cudaEventCreate(&stop_event_));
    }

    ~GpuTimer() {
        CUDA_CHECK(cudaEventDestroy(start_event_));
        CUDA_CHECK(cudaEventDestroy(stop_event_));
    }

    // 측정 시작
    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_event_, stream));
    }

    // 측정 종료 및 동기화
    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_event_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));
    }

    // 측정된 경과 시간 반환 (밀리초, ms)
    float elapsed_ms() const {
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event_, stop_event_));
        return milliseconds;
    }

private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
};

#endif // TIMER_H