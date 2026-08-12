#define TILE_SIZE 32

__global__ void shared_mm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    //L1 cache 급 속도의 Shared memory 타일 공간 설정
    //블록 내부의 스레드들이 이 shared memory공간을 공용으로 사용함
    //1. (SRAM 영역) shared memory 타일 공간 할당.
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int ty = threadIdx.y; //TILE내에서 행 위치(0 ~ 15)
    int tx = threadIdx.x; //TILE내에서 열 위치(0 ~ 15)
    //현재 스레드가 결과행렬 C 에서 담당하는 (row, col) 글로벌 좌표 계산
    //각 스레드가 자신이 담당할 결과행렬 C의 좌표를 부여받음-> 스레드 한개당 C결과행렬에서의 (row,col) 좌표를 가짐
    //2. 결과 행렬인 C에서의 전역 좌표 계산
    int row = blockIdx.y * TILE_SIZE + ty;//C의 row 는 (0 ~ M-1)
    int col = blockIdx.x * TILE_SIZE + tx;//C의 col(열)은 (0 ~ K-1)

    

    //3.레지스터에 합들을 누적할 변수
    float sum = 0.0f;


    // 4. Outer Loop: 내적 차원인 N을 TILE_SIZE(16) 단위로 쪼개어 이동
    // A에서는 가로(열) 방향, B에서는 세로(행) 방향으로 타일을 옮겨가며 연산합니다.
    // Loop 횟수: (N + TILE_SIZE - 1) / TILE_SIZE
    //행렬 곱셈 C = A * B에서 내적을 수행하는 공통 길이는 N입니다.
    //t: C의 한칸을 위해 타일 (16 *16 in this case)을 갈아야하는 횟수
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {

        //행렬 A를 DRAM에서 SRAM으로의 협동 로딩 (Cooperative Load)
        //생각해야할 조건1: A의 행 (ROW)가 유효 범위인 M 이내인가?
        //생각해야할 조건2: A의 열 (t * TILE_SIZE + threadIdx.x)
        // 몇번째 타일인가?  타일안에서 몇번째 열인가?
        // -----------------------------------------------------------------
        // 4-1. 행렬 A(M x N)의 타일을 DRAM에서 SRAM(Shared Memory)으로 로딩
        // -----------------------------------------------------------------
        //조건 1: A 의 행(row) 가 조건 유효범위 M 이내인가?
        //조건 2: A 의 열(t * Tile_size + tx) 이 조건 유효범위 
        //C 결과행렬의 (row, col)의 한 값을 완성하기 위해서 
        //A행렬은 row 가 고정된채 열의 개수(N차원)을 0부터 N-1까지 가로로 tile size 만큼 훑는다.
        if(row < M and (t * TILE_SIZE + tx) < N) {
            //row-major indexing: A[row][t * Tile_SIZE + tx]
            //memory stride: N
            As[ty][tx] = A[(row * N) + (t * TILE_SIZE + tx)];
        } else {
            As[ty][tx] = 0.0f;//범위 밖은 0으로 패딩
        }
        // -----------------------------------------------------------------
        // 4-2. 행렬 B(N * K)의 타일을 DRAM에서 SRAM(Shared Memory)으로 로딩
        // -----------------------------------------------------------------
        //B행렬은 col이 고정되어있고 행 부분인 (t * TILE_SIZE + ty)를 16칸씩 세로로 훑는다.
        if((t * TILE_SIZE + ty) < N and col < K) {
            Bs[ty][tx] = B[(t*TILE_SIZE + ty)* K + col];
        } else {
            Bs[ty][tx] = 0.0f;//padding
        }

        // -----------------------------------------------------------------
        // 4-3. 동기화1 Synchronization
        // -----------------------------------------------------------------
        // 스레드가 모두 As, Bs 타일 로드를 마칠 때까지 대기
        __syncthreads();
        // -----------------------------------------------------------------
        // 4-4. SRAM 타일 데이터를 이용한 내적(Dot Product) 부분 연산
        // -----------------------------------------------------------------
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // -----------------------------------------------------------------
        // 4-5. 동기화2 Synchronization
        // -----------------------------------------------------------------
        // 다음 타일을 덮어쓰기전에 모든 스레드가 현재 타일 연산을 마쳤는지 확인.
        __syncthreads();
    }
    // [5] 최종 결과를 Global Memory의 C(M x K) 행렬에 기록
    // C의 전체 열 개수는 K이므로 인덱싱은 row * K + col
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}


extern "C" void solve_shared_mm(const float* A, const float* B, float* C, int M, int N, int K) {
    // 1. TILE_SIZE와 1:1로 매핑되는 스레드 블록 설정 
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);

    // 2. 결과 행렬 C(M x K)를 커버하기 위한 Grid 차원 계산
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    shared_mm_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}