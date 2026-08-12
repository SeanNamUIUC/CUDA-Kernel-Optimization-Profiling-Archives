# 1. 컴파일 (-O3 최적화 및 헤더/프로젝트 경로 포함)
nvcc -O3 benchmarks/bench_sgemm.cu -I. -Iinclude -Ikernels -o bench_sgemm
nvcc: NVIDIA CUDA Compiler Driver. C/C++ 코드와 GPU CUDA 코드를 분리하여 각 컴파일러(gcc/g++ 및 GPU 컴파일러)로 넘겨 빌드합니다.

-O3: 최고 수준의 컴파일러 최적화 옵션. 루프 언롤링(Loop Unrolling), 인라인 함수화 등을 통해 CPU/GPU 실행 바이너리의 연산 속도를 대폭 높입니다.

benchmarks/bench_sgemm.cu: 메인 벤치마크 및 커널 호출 코드가 포함된 타겟 소스 파일.

-I.: 헤더 파일 탐색 경로에 현재 작업 디렉토리(Project Root)를 추가합니다. (#include "kernels/naive.cuh" 형태 인식)

-Iinclude: include/ 폴더를 탐색 경로에 추가합니다.

-Ikernels: kernels/ 폴더를 탐색 경로에 추가합니다. (#include "naive.cuh" 형태 바로 인식)

-o bench_sgemm: 최종 생성될 빌드 실행 파일(Executable)의 이름을 bench_sgemm으로 지정합니다.

# 2. 실행
./bench_sgemm
빌드된 실행 파일을 직접 작동시켜 CPU-GPU 결과 검증(Validation) 및 C++ 레벨에서의 간단한 실행 시간(ms)을 확인합니다.


# 3. 시스템 타임라인 및 요약 프로파일링 (NSYS)
nsys profile --stats=true ./bench_sgemm


# 4. 커널 세부 분석 (NCU)
A. 전체 기본 세션 프로파일링
sudo ncu ./bench_sgemm

sudo ncu: NVIDIA Nsight Compute를 실행합니다. GPU 하드웨어 레벨의 성능 카운터(Performance Counters)에 접근해야 하므로 sudo 권한이 필요할 수 있습니다.

역할: 프로그램 내에서 호출되는 모든 CUDA 커널에 대해 기본 프로파일링 세션(Memory, Compute, Occupancy 등 기본 솔루션)을 차례대로 수행합니다.


B.roofline / SpeedOfLight 병목 분석
sudo ncu --section SpeedOfLight ./bench_sgemm

--section SpeedOfLight: NCU의 핵심 분석 리포트 중 하나인 SOL(Speed of Light) 섹션만 핀포인트로 측정합니다.

역할: 해당 커널이 현재 Compute Bound(연산 성능 한계)에 걸려 있는지, 아니면 Memory Bandwidth Bound(대역폭 한계)에 걸려 있는지 비율(%)로 명확히 보여줍니다

C.특정 커널만 핀포인트 메트릭 추출 시 (추가용)
sudo ncu -k naive_mm_kernel --metrics gpu__time_duration.sum,l1tex__data_bank_conflicts_pipe_lsu.sum ./bench_sgemm

-k naive_mm_kernel: naive_mm_kernel이라는 이름을 가진 커널만 선택하여 수집합니다.

--metrics ...: 필요한 특정 하드웨어 지표(커널 실행 시간, Bank Conflict 횟수 등)만 수집하여 측정 속도를 극대화합니다.




# 기본 핵심 메트릭 덤프 명령어 (binary_name은 본인의 executable)
ncu --metrics \
gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
launch__occupancy_achieved.pct \
./bench_sgemm


# metrics profiling command I used
sudo ncu --kernel-name "naive_mm_kernel" --launch-skip 1 --launch-count 1 --section Occupancy --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu.sum,dram__bytes.sum ./bench_sgemm > results/logs/01_naive_1024.txt