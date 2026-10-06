# Personal Website Agent Guidelines

This repository contains Seungyeop Lee's standalone static personal website. The former al-folio starter was removed at the user's request.

## Ownership and Editing

- Live local files: `_jonbarron_preview/site/`.
- Content sources: `_jonbarron_preview/content/`.
- Generator: `_jonbarron_preview/migrate.py`; preview: `_jonbarron_preview/preview.mjs`.
- Edit generated home, CV, and ordinary project pages through their source files.
- DART, monocular-depth-estimation, and masters-thesis have `standalone_html: true`; edit their HTML directly and preserve them when regenerating.
- Preserve original user PDFs, presentation files, profile photo, and development code. Include only selected website assets.
- Keep image provenance in asset source manifests; do not show image source, slide numbers, or uploaded-material narration in project copy.
- Use `docs/personal-page-workflow.md` and `_jonbarron_preview/README.md` for site decisions and editing details.
- Do not publish, push, or change hosting settings without a user request.
- Delegate parallel tasks only when authorized, and assign separate file ownership.

## Verification

Run the content generator when editing its sources, then check affected pages and local links at http://127.0.0.1:4173/. Check desktop and narrow mobile layouts when visual content changes. Verify standalone pages survive regeneration.

Preview command from repository root: `node _jonbarron_preview/preview.mjs`.
Generator command from repository root: `python _jonbarron_preview/migrate.py`.
No Jekyll, Ruby, Docker, or theme plugins are required.
