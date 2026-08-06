**include/cuda_check.h**
*CUDA API 오류 감지후 바로 파일/라인 번호 출력*

*cuda check macro*
- 매크로를 통해서 호출된 위치의 파일명 과 줄 번홀르 컴파일 타임에 정확히 캡처하기 위해서.

*cuda check last error*

-cudaGetLastError
-GPU 커널 실행이 비동기로 동작해서 cpu는 커널을 gpu에 던져놓고 다음문장으로 넘어감. 이때 launch parameter error (ex) gird/block 잘못된 크기, shared_memory 용량 초과) -> 이 macro 를 통해 드라이버 상태 큐에서 꺼내서 확인

-cudaDeviceSynchronize
커널 내부 실행중 발생하는 에러 (out of bounds memory access, 0으로 나누기) gpu 락이 걸린후에 실행시점에 뒤늦게 터짐.
cpu가 gpu작업 종료까지 기다리게 강제 (synchronize) 한뒤에 실제 연산과정에서 오류가 있었는지 검증.


**include/timer.h**
*gpu 커널의 실행시간을 측정하기위해(std::chrono 로 cpu표준타이머로 측정하면 커널 호출시간만 측정됨)*


