---
layout: page
title: Projects
permalink: /projects/
description: Projects and selected work.
nav: true
nav_order: 3
---

{% assign sorted_projects = site.projects | sort: "importance" %}

<div class="project-list">
  {% for project in sorted_projects %}
    <article class="project-list-item">
      <h2 class="project-list-title">
        <a href="{{ project.url | relative_url }}">{{ project.title }}</a>
      </h2>
      {% if project.period %}
        <p><strong>Period:</strong> {{ project.period }}</p>
      {% endif %}
      <p><strong>Category:</strong> {{ project.display_category | default: project.category | capitalize }}</p>
      <p><strong>Summary:</strong> {{ project.summary | default: project.description }}</p>
    </article>
  {% endfor %}
</div>

<style>
  .project-list {
    display: grid;
    gap: 1.9rem;
  }

  .project-list-item {
    padding-bottom: 1.6rem;
    border-bottom: 1px solid var(--global-divider-color, #e5e7eb);
  }

  .project-list-title {
    margin-bottom: 0.55rem;
    font-size: 1.65rem;
    line-height: 1.25;
  }

  .project-list-title a {
    color: inherit;
    text-decoration: underline;
    text-underline-offset: 0.12em;
  }

  .project-list-item p {
    margin: 0.18rem 0;
    font-size: 1.05rem;
    line-height: 1.5;
  }
</style>
