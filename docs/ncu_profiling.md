**ncu profiling**
-explanation of metric values-
**gpu__time_duration.sum**-> 커널 실행 시간 Total Kernel Execution Time.(Pure elapsed duration of kernel execution on GPU hardware)

**dram__bytes.sum** -> global memory(vram) 전송 총량. 외각의 global memory에서 gpu 칩 내부의 on-chip memory (l1,l2 cache, shared memory, register)로 가져오는 과정.  오고 간 데이터의 총 물리적 용량 (bytes)

- GPU 하드웨어 메모리 컨트롤러 -> DRAM 접근시 최소 단위인 32-byte or 128-byte transaction으로 데이터를 읽고 씀.
- 스레드가 DRAM 메모리에 접글할 때마다 발생한 트랜잭션 크기의 총합을 바이트 단위로 합산한 수치.

Q. float 로 이루어진 행렬인데 우리가 스레드 하나가 접근하면 4바이트자나. 그럼 한번에 연속된거 32개의 데이터를 가져오면 128바이트라 이렇게 가져온다는건가? 그리고 데이터를 dram에서 읽고 vram에 가져오는 행위를 지금 말하는거야? 
-> 1개의 warp(32 threads), 1thread -> 4bytes (float) 연속된 메모리주소에 접근시 GPU메모리 컨트롤러는 이를 단 1회의 memory transaction(coalesced access)으로 묶어서 효율적으로 가져옴.

**dram__throughput** -> DRAM 대역폭 활용률 (GPU가 제공하는 최대 VRAM 대역폭 대비 연산 시 사용된 비율 DRAM THROUGHPUT)(perccentage of peak memory bandwidth utilized) 

- dram__throughput이 60%-70% (높음) and sm__throughput이 30%-40%일때 
-> memory bound (ALU 연산기보다 메모리 읽기/쓰기 가 늦어서 연산기가 노는 상태)
-> TILING, Coalescing, Cache resuse method should be used to fix this

- dram__throughput이 20%-30% (낮음) and sm__throughput이 70%-80%일때 
-> compute bound (ALU 연산기가 한계치까지 일하는 상태태)
-> Vectorization(SIMD), Unrolling , Register Tiling 필요.