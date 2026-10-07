# Seungyeop Lee — Personal Website

현재 사이트는 `_jonbarron_preview/site/`에 있습니다. 이전 al-folio 사이트 파일과 Jekyll/Docker 설정은 제거했습니다.

## 로컬 실행

저장소 루트에서 실행합니다.

```powershell
node _jonbarron_preview/preview.mjs
```

브라우저에서 http://127.0.0.1:4173/ 를 엽니다.

## 폴더 구성

- `_jonbarron_preview/content/`: 소개·CV·논문 목록·프로젝트 내용 원본
- `_jonbarron_preview/site/`: 홈페이지·CV·프로젝트 8개와 실제 웹 자료
- `_jonbarron_preview/site/assets/`: 프로젝트별 이미지·영상·논문 PDF·CV 문서
- `_jonbarron_preview/migrate.py`: 내용 원본에서 HTML 생성
- `_project_page_template/`: 독립 프로젝트 페이지용 템플릿
- `docs/personal-page-workflow.md`: 작업 기록과 유지할 결정사항
- `output/playwright/`: 검수용 자료, Git에서 제외

## 공개 PDF

ICROS 2023·2024 논문과 ICROS 2024 포스터, 석사논문은 영어 번역 PDF를 게시합니다. 석사논문은 원본 62페이지의 전체 내용을 유지하면서 36페이지로 재배치했으며, 목차도 새 페이지 번호로 갱신했습니다. ICROS 자료는 원본 영어 제목을 사용하고 번역 노트를 생략합니다. 기존 ICROS 논문·석사논문 PDF 주소는 유지하고 파일만 영어 버전으로 교체했습니다. ICROS 2024 포스터는 논문 목록의 `Poster` 링크와 UAV 프로젝트 페이지에서 열 수 있습니다.

한글 원본 사본은 `_jonbarron_preview/.source-documents/`에 보관하며 Git과 웹 배포에서 제외합니다. DART와 ICCAS 논문은 기존 영문 원본을 유지합니다.

## 내용 편집

홈·CV·일반 프로젝트는 `content/`를 수정하고 다시 생성합니다.

```powershell
python _jonbarron_preview/migrate.py
```

최초 환경 설정과 자세한 편집 방법은 [_jonbarron_preview/README.md](_jonbarron_preview/README.md)를 참고합니다.

DART·Depth·석사논문 상세는 독립 HTML로 편집하며, 재생성해도 보존됩니다. 홈 요약은 각 프로젝트 Markdown에서 관리합니다.

홈과 프로젝트 상단에는 연구 방향, 본인의 기여와 평가 조건을 구분해 표시합니다. 프로젝트 요약은 Markdown의 `project_brief`, 논문 목록의 기여 설명은 `role_summary`·`publication_contributions`에서 관리합니다. 웹 CV와 다운로드용 CV PDF는 같은 `content/cv.yml`을 사용합니다. PDF 생성 방법은 상세 README에 있습니다.

## 공개 배포

공개 주소는 https://yeop-giraffe.github.io/ 입니다. `.github/workflows/deploy.yml`이 `main`의 사이트 변경을 GitHub Pages에 배포합니다. 게시되는 파일은 `_jonbarron_preview/site/`의 내용이며, 저장소의 Pages 게시 방식은 GitHub Actions입니다.

내용을 수정한 뒤 페이지를 다시 생성하고 Git에 commit·push하면 자동으로 반영됩니다. `output/`, `tmp/`, 로컬 의존성과 검수 자료는 Git에서 제외합니다. 배포 완료 여부는 저장소의 Actions에서 확인할 수 있습니다.
