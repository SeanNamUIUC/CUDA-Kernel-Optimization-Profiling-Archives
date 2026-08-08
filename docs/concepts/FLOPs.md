**FLOPs vs FLOPS**
**후에 COMPUTE-BOUND 와 연결지어서 설명할 예정**
*FLOPs* (floating point operations)
- 이 프로그램이 총 몇번의 floating-point 연산을 했는가?
-  행렬곱 문제를 해결하는 데 필요한 총 연산량
-amounts of work-

*ANALYSIS of GEMM*
-행렵 곱의 총 FLOPs 구하기-
-ex    A (M * N), B(N * K) , C(M * K)

# 1. FMA(fused multiply add): a * b + c를 하나의 명령으로 계산하는 연산
- r = a*b+c flop을 셀때 FMA하나를 2FLOPs로 계산

# 2. N 을 곱하기 (2 * N)
- 결과 행렬 C의 원소 하나 C[i][j]를 계산하려면 A의 한 행과 B의 한열에서 K개의 원소 쌍을 곱하고 누적해야함
- 즉 C[i][j]를 만들기 위한 연산 횟수가 (2 * N)번

# 3. C[i][j]를 만들기 위한 연산횟수 * 결과행렬의 전체 원소갯수
- (2 * N) * (M * K)

# 4. result -> GEMM = (2 * M * N * K)번
 



*FLOPS(floating point operations per second)*
- 1초에 몇번의 floating point 연산을 했는가?
-speed of work-

# FLOPS = FLOPs / seconds(time taken)

# GFLOS = FLOPs / (seconds 


