#define TILE_SIZE 16

__global__ void mm_shared_memory(const float* A, const float* B, const float* C, int M, int N, int K) {
    //L1 cache 급 속도의 Shared memory 타일 공간 설정
    //블록 내부의 스레드들이 이 shared memory공간을 공용으로 사용함

    __shared__ As[TILE_SIZE][TILE_SIZE];
    __shared__ Bs[TILE_SIZE][TILE_SIZE];

    //현재 스레드가 결과행렬 C 에서 담당하는 (row, col) 글로벌 좌표 계산
    //각 스레드가 자신이 담당할 결과행렬 C의 좌표를 부여받음-> 스레드 한개당 C결과행렬에서의 (row,col) 좌표를 가짐
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    //레지스터에 합들을 누적할 변수
    float sum = 0.0f;


    //Outer loop: 행렬 A 에서 가로로 이동하면서, 행렬 B에서는 세로로 이동하면서 서로 곱하고 합하는 과정을 통해 sum이 완성됨
    //A의 한 행의 길이 이자 B의 한 열의 길이인 (N) 을 타일 단위로 쪼개서 이동
    //t: C의 한칸을 위해 타일 (16 *16 in this case)을 갈아야하는 횟수
    (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {

        //행렬 A를 DRAM에서 SRAM으로의 협동 로딩 (Cooperative Load)
        //생각해야할 조건1: A의 행 (ROW)가 유효 범위인 M 이내인가?
        //생각해야할 조건2: A의 열 (t * TILE_SIZE + threadIdx.x)
        //                         몇번째 타일인가?  타일안에서 몇번째 열인가?
        if(row < M and (t * TILE_SIZE + threadIdx.x) < N) {

        }
    }
}