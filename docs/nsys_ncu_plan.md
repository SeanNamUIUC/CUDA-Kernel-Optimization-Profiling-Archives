**1. nsys를 활용해 시스템 전체 타임라인/ 오버헤드 분석**

nsys profile --stats=true -o nsys_naive ./benchmark_sgemm
-nsys  프로파일링  summary table 출력  생성될 결과의 리포트 파일이름을 nsys_naive로  (프로파일링 실행파일)-

**2. ncu를 활용해 단일 커널 세부 분석**일