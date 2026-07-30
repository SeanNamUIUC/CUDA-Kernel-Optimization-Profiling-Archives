**Tiled SGEMM Indexing Architecture**

**Core Principle**
*하나의 축(X 또는 Y)을 고정으로 박아두고, 반대쪽 축만 $t$ 루프로 슬라이딩*

**DRAM memory: Y좌표(ROW) * 가로폭 + X좌표(COL)**




***** Y축만큼 이동시키고 (가로폭만큼 곱해서) + X축만큼 이동시킨다. (x축만큼 더해준다.)

### A 행렬: Y축 고정, X축 슬라이딩
- **Y축은 ROW 고정**
  - `blockIdx.y * (TILE_SIZE) 16 + threadIdx.y`로 Y축 위치를 완벽히 고정.
- **X축 가로 방향으로 슬라이딩**
  - $t$ 루프에 따라 오른쪽으로 이동: `t * 16 + threadIdx.x`
- **1D DRAM 주소:**
  - A[ROW * N + (t * 16 + threadIdx.x)]
  - y축(ROw)을 가로폭만큼(N) 이동시키고 추가적으로 x축만큼(t * 16 + threadIdx.x) 더해서 이동시킨다.
   

### B 행렬: X축 고정, Y축 슬라이딩
- **X축은 COL 고정**
  - `blockIdx.x * (TILE_SIZE) 16 + threadIdx.x`로 X축 위치를 완벽히 고정.
- **Y축 세로 방향으로 슬라이딩**
  - $t$ 루프에 따라 아래쪽으로 이동: `t * 16 + threadIdx.y`
- **1D DRAM 주소:**
  - A[(t * 16 + threadIdx.y) * K + COL]
  - y축(t* 16 + threadIdx.y)를 가로폭만큼(K) 이동시키고 추가적으로 x축만큼 (COL) 더해서 이동시킨다.

