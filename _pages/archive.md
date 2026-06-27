---
layout: page
permalink: /archive/
title: archive
description: Archived posts from the previous yeop-giraffe.github.io site.
nav: true
nav_order: 4
---

These posts were migrated from the previous `yeop-giraffe.github.io` site.

{% assign archived_posts = site.posts | where_exp: "post", "post.categories contains 'archive'" %}

{% if archived_posts.size > 0 %}
<div class="post-list">
  {% for post in archived_posts %}
    <article class="post-preview">
      <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
      <p class="post-meta">{{ post.date | date: "%B %-d, %Y" }}</p>
      {% if post.description %}
        <p>{{ post.description }}</p>
      {% endif %}
    </article>
  {% endfor %}
</div>
{% else %}
No archived posts yet.
{% endif %}

The original source archive is preserved at [assets/archive/yeop-giraffe.github.io-legacy-master.zip]({{ "/assets/archive/yeop-giraffe.github.io-legacy-master.zip" | relative_url }}).
