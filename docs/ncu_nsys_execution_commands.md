# 1. 컴파일 (-O3 최적화 및 헤더/프로젝트 경로 포함)
nvcc -O3 benchmarks/bench_sgemm.cu -I. -Iinclude -o bench_sgemm

# 2. 실행
./bench_sgemm

nsys profile --stats=true ./bench_sgemm

# 커널 기본 프로파일링
sudo ncu ./bench_sgemm

# Memory / Compute 병목을 한눈에 보는 Roofline 분석 포함
sudo ncu --section SpeedOfLight ./bench_sgemm