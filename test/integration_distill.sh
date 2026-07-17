#!/usr/bin/env bash
set -euo pipefail

tmp_dir="$(mktemp -d)"
tmp_override="${tmp_dir}/distill-override.yml"
tmp_site="${tmp_dir}/site"
distill_fixture="_posts/2021-07-04-distill.md"
created_distill_fixture=false

cleanup() {
  if [ "${created_distill_fixture}" = true ]; then
    rm -f "${distill_fixture}"
  fi
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

cat >"${tmp_override}" <<'YAML'
giscus:
  repo: alshedivat/al-folio
  repo_id: R_kgDOExample
  category: Comments
  category_id: DIC_kwDOExample
YAML

if [ ! -e "${distill_fixture}" ]; then
  cat >"${distill_fixture}" <<'MARKDOWN'
---
layout: distill
title: Distill
date: 2021-07-04 11:59:00-0400
description: an example of a distill-style blog post
tags: formatting distill
categories: sample-posts
giscus_comments: true
mermaid:
  enabled: true
  zoomable: true
tikzjax: true
related_posts: false

authors:
  - name: Test Author
    affiliations:
      name: al-folio
---

<d-front-matter>
<script type="text/json">
{
  "title": "Distill"
}
</script>
</d-front-matter>

This post exercises Distill, Mermaid, TikZJax, and Giscus integration.

```mermaid
graph TD;
  A-->B;
```

<script type="text/tikz">
\begin{tikzpicture}
\draw (0,0) -- (1,1);
\end{tikzpicture}
</script>
MARKDOWN
  created_distill_fixture=true
fi

bundle exec jekyll build --config "_config.yml,${tmp_override}" -d "${tmp_site}" >/dev/null

distill_page="${tmp_site}/blog/2021/distill/index.html"

if [ ! -f "${distill_page}" ]; then
  echo "distill page was not generated at ${distill_page}" >&2
  exit 1
fi

check_contains() {
  local pattern="$1"
  local description="$2"
  if ! grep -q "${pattern}" "${distill_page}"; then
    echo "distill page is missing ${description}: ${pattern}" >&2
    exit 1
  fi
}

check_contains 'd-front-matter' 'front matter output'
check_contains '/assets/js/distillpub/template.v2.js' 'distill template runtime'
check_contains '/assets/js/distillpub/transforms.v2.js' 'distill transforms runtime'
check_contains '/assets/js/distillpub/overrides.js' 'distill overrides runtime'
check_contains '/assets/al_charts/js/mermaid-setup.js' 'Mermaid setup runtime'
check_contains 'https://cdn.jsdelivr.net/npm/@planktimerr/tikzjax@1.0.8/dist/fonts.css' 'TikZJax stylesheet'
check_contains 'https://cdn.jsdelivr.net/npm/@planktimerr/tikzjax@1.0.8/dist/tikzjax.js' 'TikZJax script'
check_contains 'id="giscus_thread"' 'Giscus comments container'
transforms_runtime="${tmp_site}/assets/js/distillpub/transforms.v2.js"
distill_runtime="$(PATH="$HOME/.rbenv/shims:$PATH" bundle exec ruby -e 'spec = Gem.loaded_specs["al_folio_distill"]; puts(spec ? File.join(spec.full_gem_path, "assets/js/distillpub/transforms.v2.js") : "")')"
if [ -f "${distill_runtime}" ]; then
  # Prefer the packaged gem runtime for deterministic parity checks.
  transforms_runtime="${distill_runtime}"
elif [ ! -f "${transforms_runtime}" ]; then
  echo "distill transforms runtime missing at ${transforms_runtime} (and not found in installed al_folio_distill gem)" >&2
  exit 1
fi

expected_transforms_hash="70e3f488e23ec379d33a10a60311ec60b570b3b2d5f1823e9159f661c315184e"
actual_transforms_hash="$(ruby -rdigest -e 'print Digest::SHA256.file(ARGV[0]).hexdigest' "${transforms_runtime}")"
if [ "${actual_transforms_hash}" != "${expected_transforms_hash}" ]; then
  echo "unexpected distill transforms runtime hash: ${actual_transforms_hash}" >&2
  exit 1
fi

echo "distill integration checks passed"
