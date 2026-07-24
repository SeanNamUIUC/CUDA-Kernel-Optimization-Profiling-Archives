**Shared Memory**
*Indexing Mapping* (global -> shared -> register)
Shared Memory: 동일한 스레드 블록내의 스레드들만 접근할 수 있는 공간. 블록 내에서 스레드들이 공유하는 메모리.
핵심: global memory에서 데이터를 끌어와서 streaming multiprocessor 내부의 초고속 L1 /SRAM공간에 얹혀두고 재사용하는것.




**Indexing**
global memory에서 데이터를 가져올때는 -> 전체영역을 가리키는 global index사용
shared memory에 데이터를 저장할때는 -> local index(threadIdx) 사용

*1d indexing*
'''
int tid = threadIdx.x; // Shared Memory 접근용 Local Index
int gid = blockIdx.x * blockDim.x + threadIdx.x; // Global Memory 접근용 Global Index

__shared__ float s_data[256];
s_data[tid] = g_data[gid]; // Global -> Shared 로드
__syncthreads();           // 모든 스레드가 로드를 마칠 때까지 동기화
'''


*2d indexing*
'''
int tx = threadIdx.x;
int ty = threadIdx.y;

__shared__ float s_tile[16][16];
s_tile[ty][tx] = g_matrix[global_row * width + global_col];
__syncthreads();
'''


**Bank Conflict**
shared_memory 내부는 32개의 Bank로 분할됨 (Warp 32개에 맞춰서)
32개의 스레드가 각각 서로 다른 0번 - 31번까지의 bank에 동시에 접근

핵심: 하나의 ward(32개의 스레드)가 동시에 shared_memory에 접근할때 모든 스레드가 각자 다른 bank 에 접근하면 1클락주기(cycle) 만에 처리됨.
**두개 이상의 스레드가 서로 다른 주소이지만 같은 bank에 접근하면 접근이 직렬화(serialized) 됨 -> 성능 급락화**
__syncthreads() -> 이게 없으면 race condition 발생

*Bank Conflict는 "서로 다른 Warp" 사이에서는 절대로 일어나지 않는다.*
*Bank Conflict는 오직 같은 Warp 내부의 32개 스레드 끼리 같은 창구(Bank)를 두고 다툴 때만 발생*
가로 읽기 (s_tile[0][tx]):

인덱스: 0, 1, 2, 3 ...

Bank ID: 0, 1, 2, 3 ...

👉 Conflict 없음! (Padding 필요 없음)

세로 읽기 (s_tile[tx][0]):

인덱스: 0, 32, 64, 96 ... (32의 배수로 껑충!)

Bank ID: 0, 0, 0, 0 ...

👉 32-way Bank Conflict 발생! (Padding 1칸 필수)

**Register**
각자 1:1로 소유하는 개인 단독 공간
[ Thread 0 ]  ──1:1 전용 통로──> [ Register R0 ]  (다른 스레드 접근 불가)
[ Thread 1 ]  ──1:1 전용 통로──> [ Register R1 ]  (다른 스레드 접근 불가)


**Shared Memory Bank**
스레드 블록내의 모든 스레드들이 (ex)256개가 공유하는 (shared)공간

[ Shared Memory (공용 공간) ]
              ┌────────┬────────┬───┬────────┐
              │ Bank 0 │ Bank 1 │...│Bank 31 │
              └───▲────┴───▲────┴───┴───▲────┘
                  │        │            │
  Warp (32 Threads) 동시 접근!
  Thread 0 ───────┘        │            │
  Thread 1 ────────────────┘            │
  ...                                   │
  Thread 31 ────────────────────────────┘

**__syncthreads() & racecondition**
'''
s_data[tid] = g_in[gid];          // [1단계] 내 데이터 쓰기
float val = s_data[(tid + 1) % 2]; // [2단계] 옆 스레드가 쓴 데이터 읽기 (위험!)
'''
__syncthreads(): 남의 데이터를 읽기 전에, 데이터를 다 써놓았는지 확인


