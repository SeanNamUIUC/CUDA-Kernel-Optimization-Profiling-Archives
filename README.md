# CUDA Kernel Optimization & Profiling Archives

AI 연산 가속(AI Acceleration)에 쓰이는 CUDA 커널 최적화 기법을 직접 구현하고,
cuBLAS 대비 성능 측정과 Nsight Compute / Nsight Systems 프로파일링으로 병목을 분석하는 저장소.

📅 세부 학습 일정: [docs/PLAN.md](docs/PLAN.md)

---

## AI Acceleration Roadmap

GPU의 하드웨어 자원(DRAM, SRAM, Register)을 직접 제어해 Compute / Memory Bound를 극복하는 로드맵.

| 가속 영역 | 핵심 목표 | 핵심 기법 | 상태 |
| :--- | :--- | :--- | :---: |
| **1. Matrix Multiplication (SGEMM)** | Compute / Memory Bound 극복 | DRAM Coalescing, SRAM Tiling, Register Tiling, Tensor Core (WMMA) | 🚧 진행 중 |
| **2. Operator Fusion** | Kernel Launch Overhead & DRAM IO 제거 | Register, Shared Memory 재사용 | ⏳ 예정 |
| **3. FlashAttention** | Attention의 DRAM 접근 최소화 | SRAM Tiling + Online Softmax | ⏳ 예정 |
| **4. Quantization (INT8/4)** | 메모리 용량·대역폭 50~75% 절감 | Bitwise Shift, Packed Vector (float4, int4) | ⏳ 예정 |
| **5. PagedAttention** | VRAM 파편화 방지, Batch Size 극대화 | Block Table Indexing | ⏳ 예정 |

---

## 1. SGEMM

C(M×N) = A(M×K) × B(K×N), FP32, 모두 row-major.
모든 커널은 cuBLAS 결과로 정확성을 검증한 뒤 속도를 측정한다.

### 진행 단계

| 단계 | 커널 | 핵심 아이디어 | 상태 |
| :--- | :--- | :--- | :---: |
| 01 | Naive | 스레드 1개 = C 원소 1개 | ✅ |
| 02 | Shared Memory Tiling (16, 32) | 타일을 shared memory에 올려 DRAM 접근 감소 | ✅ |
| 03 | Register Tiling (2D Blocktile) | 스레드 1개가 여러 원소 계산, 레지스터 재사용 | ✅ |
| 04 | Vectorized Load (float4) | | ⏳ |
| 05 | Tensor Core (WMMA) | | ⏳ |

### cuBLAS 기준선 (column-major 처리)

cuBLAS는 Fortran 전통에 따라 **column-major**를 사용하지만, 이 저장소의 행렬은 모두 **row-major**다.
데이터를 실제로 전치(복사)하거나 `CUBLAS_OP_T`를 쓰는 대신, 다음 성질을 이용해 A와 B의 순서를 바꿔 호출했다.

- row-major로 저장된 R×C 행렬을 column-major로 읽으면 C×R인 **전치 행렬**로 보인다.
- 따라서 cuBLAS 입장에서 우리의 A는 Aᵀ, B는 Bᵀ로 보인다.
- `C = A × B` ⇔ `Cᵀ = Bᵀ × Aᵀ` 이므로, cuBLAS에 **B를 첫 번째, A를 두 번째**로 넘기고 m, n 자리를 바꾸면
  cuBLAS는 column-major Cᵀ를 계산하고, 이는 메모리상 row-major C와 정확히 같다.

```cpp
// row-major C(M×N) = A(M×K) × B(K×N)
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,          // m=N, n=M (자리 교체)
            &alpha, B, N,     // 첫 번째 자리에 B, lda = N
                    A, K,     // 두 번째 자리에 A, ldb = K
            &beta,  C, N);    // ldc = N
```

추가 데이터 복사나 전치 커널 없이 cuBLAS를 그대로 정답 및 성능 기준선으로 쓸 수 있다.
또한 `CUBLAS_PEDANTIC_MATH`로 TF32 사용을 막아 순수 FP32끼리 비교했다.

### 정확성 검증

M, N, K가 모두 다르고 타일 크기의 배수도 아닌 크기로 먼저 검증한다.
(M = N = K이면 M, N, K를 혼동하는 버그가 드러나지 않기 때문)

| M × N × K | 01_naive | 02_shared_t16 | 02_shared_t32 | 04_2d_blocktile |
| :--- | :---: | :---: | :---: | :---: |
| 1000 × 777 × 1234 | ✅ | ✅ | ✅ | ✅ |
| 33 × 65 × 17 | ✅ | ✅ | ✅ | ✅ |
| 128 × 1 × 300 | ✅ | ✅ | ✅ | ✅ |

판정 기준: `|out − ref| ≤ rtol·|ref| + atol` (rtol = atol = 1e-3)

### 벤치마크 (RTX 5070 Ti, sm_120, FP32)

**GFLOPS** (높을수록 빠름)

| kernel | 1024 | 2048 | 4096 |
| :--- | ---: | ---: | ---: |
| cuBLAS | 26,086 | 32,571 | 34,865 |
| 01_naive | 2,994 | 3,020 | 2,848 |
| 02_shared_t16 | 3,832 | 4,112 | 4,022 |
| 02_shared_t32 | 3,740 | 3,810 | 3,424 |
| **04_2d_blocktile** | **15,178** | **19,559** | **22,094** |

**% of cuBLAS**

| kernel | 1024 | 2048 | 4096 |
| :--- | ---: | ---: | ---: |
| 01_naive | 11.5% | 9.3% | 8.2% |
| 02_shared_t16 | 14.7% | 12.6% | 11.5% |
| 02_shared_t32 | 14.3% | 11.7% | 9.8% |
| **04_2d_blocktile** | **58.2%** | **60.0%** | **63.4%** |

**실행 시간 (ms, 4096 × 4096 × 4096)**

| kernel | time (ms) | naive 대비 |
| :--- | ---: | ---: |
| cuBLAS | 3.942 | 12.2× |
| 01_naive | 48.251 | 1.0× |
| 02_shared_t16 | 34.175 | 1.4× |
| 02_shared_t32 | 40.145 | 1.2× |
| 04_2d_blocktile | 6.221 | 7.8× |

### 실행 방법

```bash
mkdir -p results
nvcc -O3 -arch=sm_120 -I. -Iinclude benchmarks/bench_sgemm.cu -lcublas -o bench_sgemm
./bench_sgemm 1024 2048 4096
```

결과는 화면에 표로 출력되고 `results/sgemm.csv`에 저장된다.
성능 수치는 Nsight Compute(ncu) 없이 실행해서 측정한다. (ncu는 커널을 replay하고 클럭을 고정해 시간이 달라짐)

### 배운 점
