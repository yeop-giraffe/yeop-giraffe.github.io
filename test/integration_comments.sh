#!/usr/bin/env bash
set -euo pipefail

tmp_dir="$(mktemp -d)"
tmp_override="${tmp_dir}/comments-test-override.yml"
tmp_site="${tmp_dir}/site"
giscus_fixture="_posts/2022-12-10-giscus-comments.md"
disqus_fixture="_posts/2015-10-20-disqus-comments.md"
created_giscus_fixture=false
created_disqus_fixture=false

cleanup() {
  if [ "${created_giscus_fixture}" = true ]; then
    rm -f "${giscus_fixture}"
  fi
  if [ "${created_disqus_fixture}" = true ]; then
    rm -f "${disqus_fixture}"
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
disqus_shortname: al-folio-test
YAML

if [ ! -e "${giscus_fixture}" ]; then
  cat >"${giscus_fixture}" <<'MARKDOWN'
---
layout: post
title: a post with giscus comments
date: 2022-12-10 11:59:00-0400
description: an example of a blog post with giscus comments
tags: comments
categories: sample-posts external-services
giscus_comments: true
related_posts: false
---

This post shows how to add Giscus comments.
MARKDOWN
  created_giscus_fixture=true
fi

if [ ! -e "${disqus_fixture}" ]; then
  cat >"${disqus_fixture}" <<'MARKDOWN'
---
layout: post
title: a post with disqus comments
date: 2015-10-20 11:59:00-0400
description: an example of a blog post with disqus comments
tags: comments
categories: sample-posts external-services
disqus_comments: true
related_posts: false
---

This post shows how to add Disqus comments.
MARKDOWN
  created_disqus_fixture=true
fi

bundle exec jekyll build --config "_config.yml,${tmp_override}" -d "${tmp_site}" >/dev/null

giscus_page="${tmp_site}/blog/2022/giscus-comments/index.html"
disqus_page="${tmp_site}/blog/2015/disqus-comments/index.html"

grep -q 'https://giscus.app/client.js' "${giscus_page}"
if grep -q 'giscus comments misconfigured' "${giscus_page}"; then
  echo "unexpected giscus misconfiguration warning in ${giscus_page}" >&2
  exit 1
fi

grep -q 'id="disqus_thread"' "${disqus_page}"
grep -q '.disqus.com/embed.js' "${disqus_page}"

echo "comments integration checks passed"
