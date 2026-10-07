# Personal Page 작업 인수인계

기준일: 2026-10-07. 사용자 요청과 피드백은 이 채팅에서 받고 결과를 통합한다. 이 문서는 결정사항과 담당 범위를 보존하며, 작업이 끝날 때 갱신한다. 서브에이전트 병렬 작업은 사용자가 승인한 범위에서만 진행한다.

## 참고한 기존 채팅

- **웹페이지 로컬 미리보기 후 배포** — `01a107ae-f002-7542-8b60-be156808b7fb`: 전체 레이아웃, 소개, CV, 프로젝트 목록, ICROS 프로젝트 내용.
- **Add standalone project pages** — `01a10a89-dda4-78e3-94c5-ec986cb539e1`: 독립 HTML 템플릿, DART 페이지, 소속 표기, 논문 제출 영상.

기존 채팅의 과거 요청은 결정사항을 확인하기 위한 참고 자료다. 완료된 작업을 다시 수행하거나 새 변경 범위를 임의로 확대하지 않는다. 기존 채팅에 메시지를 보내는 작업은 포함하지 않는다.

## 현재 상태

- 현재 웹사이트는 `_jonbarron_preview/`의 Jon Barron 기반 정적 사이트다. 사용자 요청으로 이전 al-folio 파일·자료·Jekyll/Docker 설정·관련 워크플로를 제거했다. GitHub Pages 공개 주소는 `https://yeop-giraffe.github.io/`이며 `main`의 사이트 변경을 자동 배포한다.
- 로컬 미리보기 주소: `http://127.0.0.1:4173/`. 실행 방법과 생성 방법은 `_jonbarron_preview/README.md`에 있다.
- 프로젝트 8개의 내용이 준비되어 있다. DART·Depth·석사논문은 Academic Project Page Template 기반 독립 HTML이며, 나머지는 생성기가 만드는 페이지다.
- `_project_page_template/index.html`은 재사용할 빈 단일 HTML 템플릿이다. 모든 프로젝트가 이 템플릿으로 전환된 상태는 아니다.
- 이전 요청의 공개 배포와 ICROS·석사논문 영어 PDF 게시를 완료했다. ICROS는 원본 영어 제목을 유지하고 번역 노트를 생략하며, 석사논문은 전체 내용을 36페이지로 재배치하고 목차를 갱신했다.
- 홈페이지에는 Leadership와 Teaching & Mentoring이 있다. PhD 지원용 검토에서 연구 방향·개인 기여를 보강하고, 모든 프로젝트 상단에 요약을 추가했다. 웹 CV와 다운로드용 CV PDF는 같은 `content/cv.yml`에서 생성한다. 세부 평가는 `docs/research-website-review-2026-10-07.md`에 있다.

## 역할과 파일 범위

| 담당 | 범위 | 주요 파일 |
| --- | --- | --- |
| 메인 에이전트 | 우선순위, 요청 전달, 공통 파일 변경, 결과 통합과 검수 | `migrate.py`, `site/site.css`, `content/cv.yml`, README, 배포 설정, 이 문서 |
| 레이아웃 서브에이전트 | 홈 구성, 소개와 연구 관심 분야, 화면 배치 | `content/about.md`, `content/research.md`; 공통 파일 변경은 메인과 조율 |
| 프로젝트 페이지 서브에이전트 | 개별 상세 페이지와 해당 이미지·영상 | DART의 `site/projects/dart.html`, 프로젝트별 자료; 다른 페이지는 해당 `content/projects/*.md`에서 수정 |

경로는 별도 표기가 없으면 `_jonbarron_preview/`를 기준으로 한다. 홈 목록과 상세 페이지가 함께 사용하는 프로젝트 메타데이터는 메인과 조율한다. 같은 파일을 두 담당이 동시에 수정하지 않는다. 작업 중 독립적인 추가 요청은 다른 서브에이전트에 배정한다. 현재 담당들은 구현과 인수인계를 완료했다. 새 세션에서 재구성할 때는 에이전트 이름 대신 이 문서와 실제 파일을 기준으로 역할을 복원한다.

## 유지할 결정사항

### 홈과 목록

- Jon Barron 기반 800px 폭의 기본 배치를 사용한다.
- 소개는 “I'm a software engineer at Samsung Electronics, where …”로 시작하고, 고려대학교 Human-Machine Systems Lab과 Shinsuk Park 교수님 정보를 포함한다. 삼성전자·고려대학교·연구실에는 링크가 있다.
- Research Interests는 짧은 불릿으로 표현한다.
- 논문·발표가 있는 프로젝트는 **Publications & Presentations**, 나머지는 **Selected Projects**로 분리한다.
- Selected Projects는 **제목 → 소속·기간 → 카테고리 → Description** 순서다.
- 논문 서지 정보는 `content/cv.yml`에서 관리하고 프로젝트의 `publications` ID로 연결한다. CV에는 전체 목록을 유지한다.
- 논문 제목은 CV 레코드에서 관리하며, 제공된 논문 원문과 다르면 원문 표기로 정정한다. 제목에서 상세 페이지로 이동하므로 중복 Project details 링크는 두지 않는다.
- UAV 목록의 이전 묶음 제목과 중복 Paper / Paper (PDF) 링크는 제거했다.
- 홈 논문 목록에서는 소속을 생략하고 학회·연도·심사 상태·발표 표기 오른쪽에 PDF 링크를 표시한다. 상세와 웹 CV에서는 `content/cv.yml`의 `affiliation`을 유지한다. 웹 기호는 `*` 공동 기여, `†` 교신저자로 통일한다.
- 홈과 웹 CV에서는 저자 옆 기호만 표시하고 공동 기여·교신저자 설명 문구를 생략한다. 상세 페이지의 기호 설명에서는 “Corresponding author” 뒤에 이름이나 이메일 링크를 붙이지 않는다.
- 석사논문 상세의 상단 제목은 영문만 표시한다. 한글 제목 부제는 사용자 요청에 따라 제거했다.
- THESIS 항목과 상세에 지도교수 Prof. Shinsuk Park을 표시한다. 개인 연락 이메일은 `yeoplee0906@gmail.com`이다.
- 프로필은 사용자가 제공한 `lsy_profile.jpg` 원본을 사용한다. 홈 대표 이미지는 DART scene graph, 석사논문 통합 UI, ICROS 기체 사진, Depth 데이터 생성 파이프라인, LTA 경쟁 장면, Exosuit 센서 모듈이다. 나머지 Selected Projects 2개는 글자 표시를 유지한다.

### 프로젝트 상세

- 모든 프로젝트의 화면 캡션은 이미지 내용만 설명한다. 원본 사진·슬라이드·논문 그림 번호와 첨부 자료를 소개하는 문구는 표시하지 않는다. 자료 출처 기록은 내부의 `sources.txt`·`sources.json`에 보존한다.
- 프로젝트 상세는 공식적인 문체를 사용하며 I·my·we·our 등의 1인칭을 쓰지 않는다. 담당 역할은 Individual contribution으로 표시하고, 공동연구 결과와 개인 기여의 구분을 유지한다.
- 삼성전자 HIL·기구 개발 상세는 원본 CV와 현재 CV의 Industry Experience를 기준으로 작성한다. HIL은 실제 제어 보드·양산 제어 소프트웨어·Isaac Sim 센서의 통합과 Samsung R&D Institute-Delhi 협업을 설명한다. 기구 개발은 물걸레 구동 기구 설계·검증·내구성 개선·양산 및 품질 검증 지원을 구분하며, 공개 자료에 없는 성능 수치나 시험 방법을 추가하지 않는다.
- DART는 전체 논문 제목을 유지한다. DART만 별도 최상단 제목으로 표시하지 않는다.
- DART에서 “Acquire missing knowledge. Retain it. Reuse it for the next task.”와 “Robot Intelligence · Spatial AI · Persistent Task Memory” 문구는 제거했다.
- DART의 모든 저자는 공통 소속 **Samsung Electronics Co., Ltd., Suwon, Republic of Korea**를 한 줄로 표시한다. 부서는 넣지 않는다.
- DART 교신저자는 사용자 지정에 따라 Jong Jin Park이다. Depth 교신저자는 Knut Peterson이다. PDF의 기호를 바꾸지 않고 웹만 지정 기호로 통일했다.
- DART 상태는 **ICRA 2027 투고·심사 중**이다. 논문 그림 5개와 제출 오버뷰 영상 `site/assets/dart/ICRA27_7491_VI_i.mp4`가 포함되어 있다.
- ICROS 2023·2024는 각각의 논문 Description을 작성했고, 하나의 `drone-control-assist.html` 상세 페이지에서 연결한다. 2024 포스터에서 원본 PNG 4개와 네이티브 구성도를 저장해 연결했다. 2023 그림은 유지한다.
- ICROS 2024 영문 제목은 실제 논문의 `Micro UAV Autonomous Navigation System With Deep Learning based Monocular Vision Depth Estimation`으로 정정했다. 41 FPS는 깊이 영상 처리 속도다. 격자 수는 논문 본문 20셀과 논문 그림·포스터 15셀이 달라 상세의 출처 설명에서 구분한다.
- ICROS 내용에서 종전의 NASA-TLX 43% / SUS 62.5→80 평가 문구를 다시 사용하지 않는다. 현재 두 논문에 맞춘 설명을 유지한다.
- Depth는 제공된 ICCAS 논문 본문·그림 4개·전체 결과표에 기반한 독립 상세 페이지다. 저자 순서는 Knut Peterson, Seungyeop Lee, Solmaz Arezoomandan, David Han이며 Peterson 철자를 원문 기준으로 정정했다.
- 석사논문은 별도 `masters-thesis.html` 상세와 THESIS 목록 항목으로 추가했다. 단독저자 Seungyeop Lee, Korea University Mechanical Engineering, 2025년 2월 학위다. 사용자평가는 6명이며 SUS 62.5→80, NASA-TLX 정신적 부하 57.5→32.5다. 웹 CV의 기존 5명 표기도 정정했다. 다운로드용 원본 DOCX는 수정하지 않았다.
- 석사 발표자료의 원본 이미지와 PowerPoint 네이티브 도표로 상세 및 홈 대표 이미지의 화질을 개선했다. 출처·슬라이드·크기는 `site/assets/masters-thesis/sources.txt`에 기록한다. 18페이지의 구성도를 `hardware-components-en.svg`로 영어화해 드론 사진 옆에 표시하며 모바일에서는 세로로 배치한다. 내장 IMU·분배보드 표기를 번역하고, 카메라는 최종 논문의 C922, 720p/60fps로 통일했다. 구성요소별 본문 목록은 제거하고 단안 RGB 카메라·NVIDIA Jetson Nano·원격통신 중심으로 설명한다. 발표자료에 더 좋은 대응 이미지가 없는 그림은 최종 논문 원본을 유지한다.
- 석사논문의 실내 검증 그림은 사용자가 첨부한 장애물·UAV 주석 이미지 `hardware-validation-annotated.png`로 교체했다. 원본 바이트를 보존하고 캡션·대체 텍스트를 갱신했다. Showing the Intended Flight Path의 두 이미지는 4:3 표시 영역에서 contain으로 높이를 맞추며, 자르거나 원본을 변형하지 않는다.
- 경로 표시 섹션의 왼쪽은 네 기능의 영어 색상 태그를 추가한 `integrated-interface-annotated.svg`, 오른쪽은 경로 예측 그림이다. 통합 화면은 디펜스 슬라이드 23의 원본 프레임을 사용하며 홈 대표 이미지는 원본을 유지한다.
- LTA는 제공된 2024-03-04 연구실 발표자료의 원본 사진과 제어도를 사용한다. Raspberry Pi 4 영상 → 노트북 YOLOv5·Tracker → ESP32 명령 → DC 모터 4개 흐름을 설명한다. 100 g은 헬륨 부력 조건과 함께 표기한다. 출처는 `site/assets/lighter-than-air/sources.json`이며 제공 자료에 없는 STM32·PID·자율비행 성공률은 추가하지 않는다.
- Exosuit는 7개 WRL 랩미팅 자료와 개발 백업을 확인해 기초 제어 파이프라인 중심으로 구성했다. 사용자가 Jetson 프로그램·센서 모듈 제작·센서와 모터 연결을 본인 역할로 확인했고 기구 제작은 다른 학생이 담당했다. 10 ms는 센서 모듈부터 모터 명령까지의 end-to-end 처리 시간이며 반복 주기나 기계적 응답 시간으로 해석하지 않는다. 최신 CSV 제어 코드의 1 kHz 목표 설정과 구분한다. 센서 모듈·CAN 구성도·1 Nm 및 5 Nm 시험·IMU 연결 여부에 따른 지연 비교·측정 그림을 포함했다. 모터 응답 지연은 확인했으나 보행 보조까지 완료한 것으로 쓰지 않는다. 바이너리 기록과 외부 gait-tracking 예제 성능은 완료 성과에 포함하지 않는다. 자료 기록은 `site/assets/soft-exosuit/sources.json`에 있다.

## 편집과 통합 규칙

- 홈·CV·일반 프로젝트 HTML은 생성 결과다. 내용은 `content/`에서 수정하고, 공통 표현 방식은 메인이 `migrate.py` 및 `site/site.css`에서 조율한다.
- DART는 `content/projects/dart.md`의 `standalone_html: true`로 보호되어 있으며, 상세 페이지는 `site/projects/dart.html`을 직접 편집한다. 홈에 표시되는 DART 요약과 기간은 Markdown 메타데이터에서 관리한다.
- Depth와 석사논문도 `standalone_html: true`로 보호되어 상세 HTML을 직접 편집한다. 홈 정보는 각 프로젝트 Markdown과 CV 레코드를 함께 확인한다.
- 다른 프로젝트를 독립 HTML로 전환할 때는 해당 HTML을 준비하고 `standalone_html: true`를 설정해 재생성 시 덮어쓰기를 막는다.
- 생성기 실행은 다른 담당의 편집이 완료된 뒤 메인이 조율한다. 생성 후 관련 화면·링크·모바일 배치를 확인한다.
- 현재 사이트의 편집은 갱신된 `AGENTS.md`와 `.github/copilot-instructions.md`를 따른다. 기존 al-folio 소유권 문서는 제거했다.

## 다음 작업 후보

아래는 새 변경 요청이 아니라 현재 초안에서 남은 준비 항목과 선택 가능한 작업이다. 다음 사용자 요청에 맞춰 범위를 정한다.

- 나머지 Selected Projects에 적합한 대표 자료 준비.
- 원하는 프로젝트부터 독립 HTML 템플릿으로 전환.
- 페이지별 내용·자료 링크·모바일 화면 최종 검수.
- 향후 HIL·기구 개발의 공개 가능한 구성도와 정량 결과가 제공되면 근거 자료 보강.

## 작업 기록

- 2026-10-05: 기존 두 채팅과 현재 파일을 확인하고 레이아웃·프로젝트 페이지 서브에이전트로 읽기 전용 인수인계를 진행했다. 사이트 변경과 배포는 수행하지 않았다.
- 2026-10-06: Depth 상세, THESIS 목록·상세, 홈 대표 이미지 4개, 논문별 소속과 공동 기여·교신저자 표기를 병렬 구현 후 통합했다. 데스크톱·390px·320px 화면, PDF 원본 일치, 이미지·링크, 생성 시 독립 HTML 보존을 확인했다. 공개 배포는 하지 않았다.
- 2026-10-06: ICROS 2024 포스터와 석사 디펜스 PPTX에서 대응 원본 이미지·도표를 저장해 연결했다. ICROS 이미지 5개와 석사논문 핵심 그림의 화질을 개선하고 원본 출처를 기록했다. ICROS 영문 제목·실험 조건·상대 깊이·41 FPS 설명을 정리했다. 원본 PPTX와 PDF는 수정하지 않았다.
- 2026-10-06: Publications & Presentations의 논문 5편을 원문과 대조했다. DART 평가 조건·시점 지표, Depth의 혼합 결과·평가 도메인 노출, ICROS 2023 원문 제목·시뮬레이션 검증 범위, ICROS 2024 격자·깊이 처리 속도, 석사논문 실험 조건과 카메라 모드를 정리했다. 석사 UI 주석과 간결한 기체 설명, LTA 자료 기반 상세·대표 이미지, 프로필 실사진을 반영했다. 홈의 소속을 제거하고 논문별 PDF 링크 5개를 카테고리 왼쪽에 추가했다.

- 2026-10-06: 사용자 요청으로 기존 al-folio 페이지·샘플 자료·빌드 설정·Docker 설정·관련 워크플로와 문서를 제거했다. 새 사이트와 프로젝트 템플릿, 작업 기록, Git 이력은 보존했다. 생성기의 루트 PDF 복사 의존성을 제거하고 실행 안내를 현재 정적 사이트 기준으로 갱신했다.
- 2026-10-07: 공개 배포 설정, 영어 논문·포스터·석사논문 PDF, Leadership·Teaching & Mentoring을 반영했다. PhD 지원·연구실 컨택 독자를 기준으로 전체 10페이지를 검토했다. 개인 기여·핵심 결과·평가 조건을 먼저 보여 주고, Exosuit의 처리 시간 및 Depth의 비교 설계를 CV와 일치시켰다. 동기화된 CV PDF와 편집 안내, 페이지별 평가 기록을 추가했다. 1440px·390px·320px의 이미지·넘침·링크와 CV PDF 전체를 검수했다.
