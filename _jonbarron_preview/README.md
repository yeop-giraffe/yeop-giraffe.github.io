# Jon Barron template draft

최신 `Seungyeop_Lee_CV_v3.docx`와 `Seungyeop Lee_SOP_v9.docx`, 추가로 제공한 논문을 바탕으로 소개, 연구 관심 분야, CV, 논문 목록, 프로젝트 8개를 정리한 정적 홈페이지입니다. 이전 al-folio 사이트 파일과 관련 설정은 제거했습니다.

원본: https://github.com/jonbarron/jonbarron.github.io

원본 `stylesheet.css`와 800px 폭의 소개/프로젝트 배치를 사용합니다. 프로필에는 제공한 `lsy_profile.jpg`를 사용합니다. 홈 목록의 DART·석사논문·UAV·Depth·LTA·Exosuit에는 프로젝트 대표 이미지를 사용합니다. 삼성의 HW·SW 두 프로젝트는 사용자가 제공한 Samsung 파란색 워드마크 원본과 별도의 HW·SW 텍스트를 조합합니다. 로고와 출처 기록은 `site/assets/samsung/`에 있습니다.

## 로컬 미리보기

저장소 루트에서 실행:

```powershell
node _jonbarron_preview/preview.mjs
```

http://127.0.0.1:4173 에서 확인합니다. `site/` 안의 HTML과 CSS를 저장하면 약 1초 후 자동으로 새로고침됩니다. 중지는 Ctrl+C입니다.

홈페이지: `site/index.html` / 추가 스타일: `site/site.css` / CV: `site/cv.html` / 프로젝트: `site/projects/`

브라우저 탭 아이콘은 원본 프로필 사진 전체를 64×64 PNG로 축소한 `site/assets/images/profile-favicon.png`입니다. 홈·CV·일반 프로젝트에는 생성기가 아이콘 링크를 추가하며, 독립 HTML 3개는 각각 직접 연결합니다. 원본 사진은 변경하지 않았고 출처는 `site/assets/images/favicon-sources.json`에 기록합니다.

메인과 프로젝트의 공통 읽기 스타일은 `site/project-reading.css`에서 관리합니다. 본문은 16px, 프로젝트 요약은 15px이며, 상세 요약은 데스크톱에서 960px 폭, 모바일에서 한 열로 표시합니다. 모든 상세 페이지는 요약 다음에 섹션 바로가기를 배치합니다. 일반 프로젝트의 바로가기와 제목 앵커는 생성기가 만들고, 독립 HTML의 바로가기는 해당 페이지에서 관리합니다. 홈의 개인 GitHub 링크는 제거했으며, 삼성 프로젝트 대표 이미지는 Samsung 로고 아래 SW / HW 표기로 구분합니다. 템플릿 출처 링크는 유지합니다.

## 내용 수정 및 페이지 다시 생성하기

일반 프로젝트 5개의 Skills는 본문 글꼴과 16px 크기를 사용하고, 각 항목을 테두리·배경·간격이 있는 태그로 구분합니다. 스타일은 `site/project-reading.css`의 `.project-page #skills + p` 규칙에서 관리하며 좁은 화면에서는 자동으로 줄을 바꿉니다. Skills 원문과 본문 내 코드 표기는 유지합니다.

홈페이지 상단의 현재 커리어와 학력을 정리한 소개 문단은 `content/about.md`에서 수정합니다. 불릿으로 정리한 연구 관심 분야는 `content/research.md`, CV는 `content/cv.yml`, 프로젝트는 `content/projects/`에서 수정합니다. SOP의 학교별 지원 문구는 홈페이지에 포함하지 않습니다. 연구 관심과 앞으로의 목표는 완료한 성과와 구분해서 표현합니다.

웹 CV와 다운로드용 CV PDF는 `content/cv.yml`에서 관리합니다. 홈페이지와 CV 페이지의 다운로드 링크는 `site/assets/documents/Seungyeop_Lee_CV.pdf`입니다. 제공한 DOCX 원본은 `site/assets/documents/Seungyeop_Lee_CV_v3.docx`에 보존하며 수정하지 않습니다.

CV 내용을 갱신할 때는 아래 명령으로 PDF도 다시 생성하고 공개 폴더에 복사합니다. `requirements.txt`에는 PDF 생성용 ReportLab이 포함되어 있습니다.

```powershell
python _jonbarron_preview/build_cv.py
Copy-Item output/pdf/Seungyeop_Lee_CV.pdf _jonbarron_preview/site/assets/documents/Seungyeop_Lee_CV.pdf
```

홈페이지는 논문·발표가 연결된 프로젝트를 `Publications & Presentations`에, 논문이 없는 프로젝트를 `Selected Projects`에 나누어 표시합니다. 논문과 발표는 관련 프로젝트 안에 통합해 한 번씩 표시합니다. 제목·저자·학회·심사 상태는 `content/cv.yml`의 논문 목록에서 관리하고, 프로젝트 앞부분의 `publications`에 해당 논문의 `id`를 지정합니다. CV 페이지에는 전체 논문 목록이 유지됩니다.

홈의 Publications & Presentations는 소속을 생략하고, 각 논문의 학회·연도·심사 상태·발표 표기 오른쪽에 PDF 링크를 표시합니다. PDF 경로는 CV 레코드의 `pdf`에서, ICROS 논문별 카테고리는 프로젝트의 `publication_categories`에서 관리합니다. 상세와 웹 CV의 소속은 각 레코드의 `affiliation`에서 관리합니다. 웹 저자 표기는 `*` 공동 기여, `†` 교신저자입니다. DART의 교신저자는 Jong Jin Park, ICCAS Depth 논문의 교신저자는 Knut Peterson입니다. 원본 PDF는 그대로 보존하며, 웹의 기호는 사용자 요청에 따라 통일했습니다. 논문 본문에 근거해 웹 CV의 사용자평가 참가자 수를 6명으로 정정했습니다. 다운로드용 DOCX는 제공된 원본이며 이번 작업에서 수정하지 않았습니다.

```powershell
python -m pip install -r _jonbarron_preview/requirements.txt --target _jonbarron_preview/.deps
python _jonbarron_preview/migrate.py
```

생성된 홈페이지, CV와 일반 프로젝트 HTML을 덮어씁니다. HTML에서 직접 수정한 내용은 먼저 보관하세요. `content/`에 저장한 수정 내용은 다시 생성해도 유지됩니다. 프로젝트 메타데이터에 `standalone_html: true`가 있으면 해당 HTML은 재생성하지 않고 보존합니다. 이 경우 HTML 파일이 없으면 오류로 알려 줍니다.

DART·Depth·석사논문은 독립 HTML로 보존됩니다. 홈 대표 이미지 경로와 대체 텍스트는 `migrate.py`의 `thumbnails`에 있습니다.

모든 프로젝트의 상단 요약은 연구 질문/목표, 본인의 기여, 결과와 평가 조건을 짧게 정리합니다. 일반 페이지는 프로젝트 Markdown의 `project_brief`를 사용하고, 독립 페이지는 해당 HTML에도 같은 내용을 반영합니다. 홈 Publications & Presentations와 Selected Projects의 담당 범위는 **Role**로 표시합니다. 공통 역할은 프로젝트의 `role_summary`, 논문별 역할은 `publication_contributions`에서 관리합니다. 독립 페이지의 읽기 스타일은 `site/project-reading.css`에서 관리합니다. 홈페이지와 웹 CV에는 섹션 바로가기 링크가 있습니다.

프로젝트 화면에는 사진·그림의 내용 설명만 표시합니다. 슬라이드 번호·이미지 출처·첨부 자료 소개 문구는 생략하고, 출처 기록은 각 이미지 폴더의 `sources.txt`·`sources.json`에 보존합니다.

## DART 프로젝트 페이지

`site/projects/dart.html`은 Academic Project Page Template을 바탕으로 만든 독립 HTML 페이지입니다. 이 파일을 직접 수정하면 됩니다. 홈페이지의 기존 DART 링크에서 열리며, `content/projects/dart.md`의 `standalone_html: true` 설정으로 재생성 시에도 보존됩니다. 홈페이지 목록의 제목·요약·기간은 계속 `content/projects/dart.md`에서 수정합니다.

내용은 `assets/pdf/ICRA27.pdf`의 논문과 기존에 정리한 저자·기여 정보를 바탕으로 작성했습니다. 논문은 익명 심사용이므로 저자 순서와 공동 기여는 `content/projects/dart.md` 및 `content/cv.yml`을 기준으로 표시했습니다. 투고 상태는 ICRA 2027 심사 중으로, BibTeX는 2026년 제출 원고로 표기했습니다.

그림 1–5는 논문에서 추출해 `site/assets/dart/`에 넣었습니다. 원본 자료를 교체할 때는 해당 이미지 파일과 설명·결과 수치를 함께 확인하세요. CSS와 JavaScript는 HTML 안에 포함되어 있으며, 게시할 때는 HTML과 `assets/dart/`, `assets/pdf/ICRA27.pdf`의 상대 경로를 유지합니다. 공개용 상세 HTML에서는 초안의 `noindex, nofollow`를 제거하고 실제 주소를 canonical로 지정했습니다.

로컬 미리보기 주소: http://127.0.0.1:4173/projects/dart.html

논문과 함께 제출한 오버뷰 영상은 `site/assets/dart/ICRA27_7491_VI_i.mp4`에 있습니다(약 14.5 MiB). DART 페이지의 Video Overview 섹션에서 재생합니다. 원본 영상을 변환 없이 복사했으며, 배포할 때 이미지 및 PDF와 함께 포함합니다.

## Depth 논문과 석사논문 페이지

`site/projects/monocular-depth-estimation.html`은 ICCAS 2025 논문을 바탕으로 만든 독립 HTML입니다. 전체 저자·소속, 논문 그림 4개, 방법·평가 조건·전체 결과표·한계·개인 역할·BibTeX를 포함합니다. PDF와 그림은 `site/assets/monocular-depth-estimation/`에 있습니다. 홈 요약은 `content/projects/monocular-depth-estimation.md`에서, 상세는 HTML에서 수정합니다. AbsRel의 소폭 개선과 다른 지표의 혼합 결과, CycleGAN의 평가 도메인 노출을 구분해서 설명합니다.

`site/projects/masters-thesis.html`은 2025년 2월 고려대학교 기계공학 석사학위논문 상세입니다. 홈에서는 별도 **THESIS** 항목으로 표시합니다. 논문 원본 PDF와 그림은 `site/assets/masters-thesis/`, 홈 요약은 `content/projects/masters-thesis.md`에서 관리합니다. 상세는 HTML을 직접 수정합니다. 컨트롤러와 인터페이스 평가를 구분하고, 사용자평가 참가자 6명, SUS 62.5→80, NASA-TLX 정신적 부하 57.5→32.5를 원문에 맞춰 표기했습니다. 기존 ICROS 통합 UAV 페이지도 유지됩니다.

두 상세 페이지와 홈은 데스크톱 및 모바일 390px·320px에서 이미지·링크·가로 넘침을 확인했습니다. 페이지 재생성 시 독립 HTML이 보존되는 것도 확인했습니다.

## 포스터·발표자료 원본 이미지

ICROS 2024 포스터의 원본 PNG와 PowerPoint 도표를 `site/assets/images/uav/`에 저장했습니다. 상세는 `content/projects/drone-control-assist.md`에서 연결하며, 출처와 크기는 `icros2024-image-sources.json`에 기록합니다. 논문 본문과 그림·포스터의 격자 수 차이를 구분하고, 41 FPS는 깊이 영상 처리 속도로 설명했습니다. 웹 논문 제목도 실제 ICROS 논문의 영문 제목으로 정정했습니다.

석사 디펜스의 통합 인터페이스·기체·실험 장면·결과 그래프는 내장 원본을 사용하고, 흐름도·장애물 표시·경로 예측·물리 기체 검증은 기존 도표를 고해상도 출력했습니다. `site/assets/masters-thesis/sources.txt`에 원본 슬라이드와 파일별 추출 방식을 기록했습니다. 발표자료보다 최종 논문 그림이 더 적합하거나 원본 해상도가 높은 경우에는 논문 그림을 유지합니다. 첨부 PPTX는 수정하거나 웹에 포함하지 않았습니다.

디펜스 18페이지의 하드웨어 구성도는 원본 이미지를 포함한 `hardware-components-en.svg`로 영어 표기를 추가했습니다. 드론 사진과 함께 표시하며, 카메라 모델과 모드는 최종 논문에 맞춰 C922, 720p/60fps로 통일했습니다. 구성요소별 본문 목록은 삭제하고 단안 RGB 카메라, NVIDIA Jetson Nano, 원격통신 중심으로 설명했습니다.

Showing the Intended Flight Path는 왼쪽의 `integrated-interface-annotated.svg`에 네 기능의 영어 색상 태그를 표시하고 오른쪽에 경로 예측 그림을 배치합니다. 두 이미지의 표시 높이를 맞췄으며, 홈 대표 이미지는 원본 통합 화면을 유지합니다.

## Lighter-Than-Air 프로젝트

`content/projects/autonomous-lighter-than-air-vehicle.md`는 2024년 3월 4일 연구실 발표자료의 슬라이드 3–4에 기반합니다. 원본 사진·영상 대표 프레임·제어도는 `site/assets/lighter-than-air/`, 출처는 `sources.json`에 있습니다. Raspberry Pi 4 영상 → 노트북 YOLOv5·Tracker → ESP32 명령 → DC 모터 4개의 흐름을 설명합니다. 100 g은 자료에 적힌 헬륨 부력 조건을 함께 표기하며, 성능 수치는 추가하지 않았습니다. 원본 사진을 유지하고 화면에서 발표자료와 같은 방향으로 표시합니다.

## Soft Exosuit 프로젝트

`content/projects/soft-exosuit-controller.md`에서 상세와 홈 요약을 관리합니다. WRL 랩미팅 자료 7개와 개발 백업의 최신 CSV 제어 프로그램을 확인해 Jetson Orin Nano·Feather M4 CAN 센서 모듈·SocketCAN·모터 명령과 피드백·비동기 CSV 기록을 설명합니다. 상세 그림 6개와 내부 출처 기록은 `site/assets/soft-exosuit/`에 있습니다. 홈 대표 이미지는 Research Overview 7페이지 구성에 기반해 사용자가 승인한 `exosuit-control-sensing-overview.png`이며, AI 편집 방법과 원본 구성요소는 `sources.json`에 기록합니다. 상세의 원본 센서 모듈 사진과 CAN 구성도는 유지합니다. 홈 요약·상세 Overview는 하지 보조용 soft exosuit의 목적과 load cell·IMU → Jetson → 모터 제어 흐름을 먼저 설명하고, 상단 요약은 담당 업무와 10 ms 처리 결과를 강조합니다. 시험 그래프는 기존 후반 섹션에 유지합니다.

사용자가 확인한 10 ms는 센서 모듈에서 모터 명령까지의 처리 시간입니다. 코드의 1 kHz 목표 설정이나 실제 모터 응답 시간과 구분하며, 모터 응답 지연을 관찰한 기초 파이프라인 구축까지를 성과로 기술합니다. 원본 코드·실험 로그·랩미팅 자료는 웹에 복사하지 않습니다.

상세 Overview의 `exosuit-overview-layout`는 왼쪽 대표 이미지와 오른쪽 기존 설명을 약 30:70 폭으로 배치합니다. CAN 파이프라인은 Sensor Modules and Communication의 `exosuit-sensor-media`로 옮겨 왼쪽에, 기존 센서 모듈 사진은 오른쪽에 배치하고 두 이미지 높이를 맞춥니다. 800px 이하에서는 각 묶음을 세로로 배치하며, 이미지 클릭 시 원본 크기로 열립니다. 본문과 원본 이미지 파일은 유지합니다.

## 영어 PDF

ICROS 2023·2024 논문, ICROS 2024 포스터, 석사논문의 영어 번역본을 게시합니다. 논문 본문·그림·표·수식·참고문헌을 유지했습니다. 석사논문은 원본 62페이지의 전체 내용을 36페이지로 재배치했으며, 목차·그림 목록·표 목록은 새 PDF의 페이지 번호를 사용합니다. ICROS 자료는 원본 영어 제목을 사용하고 번역 노트를 생략합니다.

공개 파일은 `site/assets/pdf/ICROS2023_LSY.pdf`, `site/assets/pdf/ICROS2024_LSY.pdf`, `site/assets/pdf/ICROS2024-poster-en.pdf`, `site/assets/masters-thesis/paper.pdf`입니다. 한글 원본 사본은 `.source-documents/`에 보관하며 Git에서 제외합니다. ICROS 2024 포스터 링크는 CV 레코드의 `poster_pdf`에서 관리합니다.

## GitHub Pages 배포

`site/`의 내용만 GitHub Pages에 배포합니다. `.nojekyll`이 포함되어 있어 별도 Jekyll 빌드 없이 사용할 수 있습니다.

기존 al-folio 배포 워크플로는 정적 사이트용 `.github/workflows/deploy.yml`로 교체했습니다. GitHub의 Settings → Pages → Source는 GitHub Actions로 설정합니다. `main`에 `site/` 변경을 push하면 자동 배포하며, Actions에서 수동 실행할 수도 있습니다.

공개 주소는 https://yeop-giraffe.github.io/ 입니다. 원본 템플릿의 `CNAME`, 개인 연락처, 사진, 논문 목록은 새 사이트에 가져오지 않습니다.
