# 단일 HTML 프로젝트 페이지 템플릿

원본: [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) (2025년 개편 버전, 2026-10-05 확인).

`index.html` 하나에 CSS와 JavaScript를 포함한 독립 템플릿입니다. Jekyll, al-folio, 별도 프로젝트 저장소, npm 설치 또는 빌드가 필요하지 않습니다. 원본의 학술 프로젝트 페이지 구성과 색상·타이포그래피를 바탕으로 단일 파일에 맞게 수정했습니다. 외부 폰트와 라이브러리는 사용하지 않으므로 원본과 글꼴·아이콘이 완전히 같지는 않습니다.

## 미리보기

`index.html`을 브라우저에서 열면 됩니다. 인터넷 연결 없이도 기본 디자인과 캐러셀을 확인할 수 있습니다. BibTeX 자동 복사가 지원되지 않는 환경에서는 텍스트를 선택해 주므로 Ctrl+C 또는 ⌘C로 복사하세요.

현재 파일은 빈 템플릿입니다. 자료 버튼은 비활성 상태이며 이미지·영상은 자리 표시자로 되어 있습니다. 실제 논문·영상·외부 서비스를 자동으로 불러오지 않습니다.

## 프로젝트마다 복사해서 사용하기

현재 별도로 준비한 개인 홈페이지를 사용할 경우:

1. 이 `index.html`을 `_jonbarron_preview/site/projects/프로젝트이름.html`로 **복사**합니다.
2. 이미지, 영상, 논문 파일을 `_jonbarron_preview/site/assets/프로젝트이름/`에 넣습니다.
3. HTML 안의 `TODO`를 검색하며 제목, 저자, 소속, 기간 또는 논문 상태, 소개, 방법, 결과, 자료 링크를 수정합니다.
4. 자료 버튼에 `href`를 추가하고 `aria-disabled="true"`를 삭제합니다. 자료가 없으면 해당 버튼을 삭제합니다.
5. 이미지·영상 자리 표시자를 삭제하고 바로 아래 주석의 예제 태그를 사용합니다. 필요 없는 영상·BibTeX 섹션은 삭제합니다. 논문 상태와 공저자 정보를 실제 내용에 맞게 수정하고, 공동 기여가 없으면 별표와 Equal contribution 문구를 삭제합니다.
6. 상단 홈페이지 링크를 `../index.html#research`로 변경합니다. 홈페이지의 프로젝트 링크는 `projects/프로젝트이름.html`로 연결합니다. 같은 이름의 기존 파일을 교체하면 링크를 바꿀 필요가 없습니다.
7. 공개할 때 `noindex, nofollow` 메타 태그를 삭제하고, 문서 제목·설명·공유 메타데이터도 수정합니다. 원본 출처와 템플릿 라이선스 표시는 유지합니다.

이미지 태그 예시:

```html
<img class="media" src="../assets/프로젝트이름/teaser.jpg" alt="이미지 내용 설명" width="1600" height="900" />
```

**생성기 주의:** `_jonbarron_preview/migrate.py`는 일반 프로젝트 HTML을 다시 생성해 덮어씁니다. 직접 편집할 페이지는 해당 `content/projects/프로젝트이름.md`의 메타데이터에 `standalone_html: true`를 추가하면 보존됩니다. 이 설정이 있는데 HTML이 없으면 오류로 알려 줍니다. DART는 이 방식을 사용합니다. 이 빈 템플릿 자체는 생성기와 연결하지 않았습니다.

다른 호스팅에서도 HTML과 실제로 사용하는 자료 파일을 함께 올리면 됩니다. HTML만으로 디자인과 기본 동작이 실행되지만, 추가한 이미지·영상·PDF는 해당 상대경로를 유지해야 합니다. 홈페이지에서 링크하는 방식이므로 개별 프로젝트를 별도 GitHub Pages 사이트로 만들 필요가 없습니다.

기존 al-folio 사이트에 연결할 경우, `_`로 시작하지 않는 공개 폴더(예: `project-pages/프로젝트이름/index.html`)에 복사하고 홈페이지에서 해당 경로를 링크하세요. HTML에 Jekyll front matter를 추가하지 않습니다. 이 `_project_page_template/` 폴더는 템플릿 보관용이며 현재 배포 설정이나 기존 프로젝트 링크를 변경하지 않습니다.

## 포함된 구성

- 제목, 부제, 저자, 소속, 기간/논문 상태, 자료 링크
- 대표 이미지/영상, Abstract, Method
- 가로 스크롤 및 버튼·키보드로 이동하는 결과 캐러셀
- 선택적으로 사용하는 결과 표, YouTube 영상, 포스터 PDF
- BibTeX 복사, 관련 프로젝트 메뉴, 맨 위로 이동
- 모바일 레이아웃, 키보드 탐색, 모션 줄이기 설정 지원

## 출처

Eliahu Horwitz의 Academic Project Page Template 및 그 기반인 [Nerfies](https://nerfies.github.io/)를 명시했습니다. 원본에서 안내하는 템플릿 디자인·코드의 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 표시를 HTML 하단에 유지했습니다.
