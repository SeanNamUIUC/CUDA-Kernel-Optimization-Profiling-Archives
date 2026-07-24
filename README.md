# CUDA Kernel Optimization & Profiling Archives
A structured journey through CUDA programming, GPU memory architecture, and LeetGPU optimizations.

C++ 로우레벨 메모리 제어 감각을 바탕으로, **CUDA 프로그래밍 개념, `ncu` 프로파일링 기반 성능 분석, LeetGPU 및 알고리즘 코딩 테스트**를 체계적인 스케줄에 맞춰 기록하는 연구 및 풀이 저장소입니다.

--

## Key Objectives

1. **CUDA & GPU Architecture:** Memory Hierarchy, Coalescing, Shared Memory Bank Conflict, Vectorized Load (`float4`) 학습 정리
2. **Profiling & Bottleneck Analysis:** NVIDIA Nsight Compute (`ncu`) 기반 DRAM Traffic 측정, Roofline Model 분석 및 성능 병목 분석 보고서 작성
3. **Problem Solving:** LeetGPU 및 알고리즘 문제 풀이 및 메모리 최적화 코드 기록

---

## Repository Structure

## 🗓️ Learning Roadmap & Timeline

### 📍 Phase 1: CUDA Fundamentals & Base Concepts (7/22 ~ 7/31)
- [x] **7/22:** Host-Device Memory Transfer Model | **LeetGPU:** Vector Addition
- [x] **7/23:** Shared Memory 기본 & Indexing | **LeetGPU:** Matrix Addition (1D)
- [ ] **7/24:** CUDA Memory Hierarchy | **LeetGPU:** Matrix Addition (2D Grid)
- [ ] **7/25:** `ncu` 기초 명령 습득 (`ncu --metrics ...`) | **NeetCode:** Sliding Window / Binary Search
- [ ] **7/26:** Memory Coalescing 분석 | **LeetGPU:** Naive Matrix Multiplication
- [ ] **7/27:** Kernel Launch & Grid/Block Tuning | **NeetCode:** Binary Search
- [ ] **7/28:** Zero-Copy DMA Memory Mapping & Pinning | **NeetCode:** Linked List
- [ ] **7/29:** INT8 Scaling & CUDA Quantization | **NeetCode:** Linked List
- [ ] **7/30:** PyTorch Weight $\rightarrow$ Binary Dump 구조 이해 | **NeetCode:** Linked List
- [ ] **7/31:** `nvcc` Compilation Flags (`-O3`, `-arch=sm_75`) 정리 | **NeetCode:** Trees

---

### 📍 Phase 2: Kernel Optimization & Profiling (8/1 ~ 8/12)
- [ ] **8/01:** `std::chrono` vs `cudaEvent_t` Timer 비교 분석 | **NeetCode:** Trees
- [ ] **8/02:** `ncu`로 Baseline Metric 수집 및 측정 방법 정리 | **NeetCode:** Trees
- [ ] **8/03:** Roofline Model 개념 정리 (Compute vs Memory Bound) | **NeetCode:** Trees
- [ ] **8/04:** Profiling Report 템플릿 및 가시화 틀 작성 | **NeetCode:** Trees
- [ ] **8/05:** Phase 1 학습 내용 중간 점검 및 복습 | **복습**
- [ ] **8/06:** Modern CUDA Programming Guide 문서화 | **NeetCode:** Trees
- [ ] **8/07:** Thread Mapping Optimization 노하우 정리 | **NeetCode:** Trees
- [ ] **8/08:** `ncu` Bank Conflict 측정 및 Padding 수정 기법 | **LeetGPU:** Tiled GEMM
- [ ] **8/09:** Operator Fusion 전/후 DRAM Traffic 비교 분석 | **NeetCode:** Tries
- [ ] **8/10:** Tiling Factor(16x16 vs 32x32)별 SOL % 비교 분석 | **NeetCode:** Heap
- [ ] **8/11:** Page-locked Memory vs Pageable Transfer Latency 측정 | **NeetCode:** Heap
- [ ] **8/12:** 1차 `ncu` Profiling Report 데이터 종합 정리 | **NeetCode:** Heap

---

### 📍 Phase 3: Advanced Tuning & Profiling Reports (8/13 ~ 8/22)
- [ ] **8/13:** Memory Leak Inspection (`Compute Sanitizer` / `cudaMemcheck`) | **NeetCode:** Greedy
- [ ] **8/14:** INT8 Tensor Core / Vectorized Load (`float4`, `int4`) 정리 | **NeetCode:** Greedy
- [ ] **8/15:** Benchmark Report (Roofline Model 분석 표) 작성 | **NeetCode:** Greedy
- [ ] **8/16:** Quantized Inference Latency 분석 | **NeetCode:** Graph
- [ ] **8/17:** `ncu` GUI/CLI 캡처 그래프 & Metric 분석 서사 작성 | **NeetCode:** Graph
- [ ] **8/18:** Latency Comparison Summary 작성 | **NeetCode:** Graph
- [ ] **8/19:** Architecture & Memory Flow 문서 보완 | **NeetCode:** Graph
- [ ] **8/20:** Colab T4 및 로컬 환경 프로파일링 데이터 재검증 | **NeetCode:** Graph
- [ ] **8/21:** Modern C++ / CUDA Best Practice 최종 점검 | **NeetCode:** DP
- [ ] **8/22:** **면접 대비:** "Profiling으로 해결한 병목 사례 3가지" 정리 완료 | **NeetCode:** DP

---

## 🛠️ Environment & Tools
* **Language:** C++17/20, CUDA C++
* **Profiling Tools:** NVIDIA Nsight Compute (`ncu`), Compute Sanitizer, NVCC
* **Target GPU:** NVIDIA Compute Capability 7.5+