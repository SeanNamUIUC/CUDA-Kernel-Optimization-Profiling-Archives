Troubleshooting: Nsight Systems Profiling Error on RTX 5070 Ti1. Issue SummaryRTX 5070 Ti (CUDA 13.2, Ubuntu) 환경에서 nsys profile 명령어를 실행할 때, 수집 완료 후 리포트 생성 단계에서 Importer error가 발생하며 .nsys-rep 파일이 정상 출력되지 않거나 profiling이 실패함.2. EnvironmentGPU: NVIDIA GeForce RTX 5070 TiOS: Ubuntu (한글 계정 환경)CUDA Version: 13.2NVIDIA Driver Version: 595.843. Root Cause Analysis버전 불일치 (Version Mismatch): 기존 설치된 Nsight Systems (v2023.4)가 최신 아키텍처(RTX 50 시리즈)의 데이터 포맷을 파싱하지 못해 Importer 에러 발생.Path & Cache 우선순위: 독립 실행형 nsys 2026.4.1 패키지를 설치한 후에도 터미널 명령 캐시(hash) 또는 $PATH 우선순위에 의해 기존 2023.4 버전이 지속적으로 실행됨.Symbol Downloading Stalls: 수집 완료 후 링커 심볼(Symbol files) 다운로드 과정에서 터미널 응답이 지연됨.4. Step-by-Step ResolutionStep 1: 최신 Nsight Systems (.deb) 다운로드 및 디렉토리 확인한글 Ubuntu 환경의 경우 ~/Downloads가 아닌 ~/다운로드 경로로 접근 필요.와일드카드(*) 매칭 실패 시 정확한 파일명 명시.Bashcd ~/다운로드
# 패키지 명칭에 맞게 설치
sudo dpkg -i nsight-systems-2026.4.1_2026.4.1.191-1_amd64.deb
Step 2: 의존성 복구 (Dependency Resolution)dpkg 설치 중 라이브러리 미설치 오류 발생 시 apt-get 복구 실행:Bashsudo apt-get install -f
Step 3: 터미널 명령 캐시 초기화 및 버전을 2026.x로 전환패키지 설치 후에도 nsys --version 실행 시 이전 버전이 나올 경우, 터미널 명령 실행 캐시를 비우거나 절대 경로 사용:Bash# 1. 터미널 명령 캐시 비우기
hash -r

# 2. 버전을 확인하여 2026.4.1로 상향되었는지 검증
nsys --version

# (필요 시) 심볼릭 링크 수동 연결
sudo ln -sf /opt/nvidia/nsight-systems/2026.4.1/bin/nsys /usr/local/bin/nsys
Step 4: 프로파일링 수집 및 Symbol Download 지연 해결수집 후 Press Ctrl-C to stop symbol files downloading 문구에서 지연될 경우, Ctrl + C를 눌러 심볼 다운로드만 중단하고 리포트 변환을 직관적으로 완료:Bash# 프로파일링 명령 실행
nsys profile --trace=cuda,nvtx,osrt --stats=true --output=sgemm_report ./sgemm_bench

# 'Press Ctrl-C to stop symbol files downloading' 대기 시 -> Ctrl + C입력
5. Verification정상적으로 sgemm_report.nsys-rep 및 sgemm_report.sqlite 파일이 생성되고 CLI 통계 표가 출력되면 해결 완료.Bash# 타임라인 GUI 시각화 확인
nsys-ui sgemm_report.nsys-rep
