# CUDA Kernel Optimization & Profiling Archives
AI 연산 가속화(AI Acceleration) 및 CUDA 커널 최적화 기법을 직접 구현하고, Nsight Compute / Nsight Systems를 통해 프로파일링 및 병목 지점을 분석한 repository.
--

---

## AI Acceleration Roadmap 

GPU의 하드웨어 자원(DRAM, SRAM, Register)을 수동 제어하여 Compute & Memory Bound를 극복하는 로드맵.

| 가속 영역 | 핵심 목표 | 수동 제어하는 하드웨어 자원 및 핵심 기법 |
| :--- | :--- | :--- |
| **1. Matrix Multiplication (SGEMM)** | Compute / Memory Bound 극복 | DRAM Coalescing, SRAM Tiling, Tensor Core (WMMA) |
| **2. Operator Fusion** | Kernel Launch Overhead & DRAM IO 제거 | Register, Shared Memory (포인터 수동 연결) |
| **3. FlashAttention** | Attention 연산의 DRAM Latency 상쇄 | SRAM Tiling + Softmax Online Scaling |
| **4. Quantization (INT8/4)** | 메모리 용량 및 대역폭 50~75% 절감 | Bitwise Shift, Packed Vector Instruction (float4, int4) |
| **5. PagedAttention** | VRAM 파편화 방지 및 Batch Size 극대화 | GPU Virtual Memory Addressing, Block Table Indexing |

---