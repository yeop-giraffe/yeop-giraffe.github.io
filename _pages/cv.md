---
layout: page
permalink: /cv/
title: CV
nav: true
nav_order: 5
cv_pdf: /assets/pdf/seungyeop_cv.pdf # you can also use external links here
description: Education, research experience, publications, projects, awards, and skills.
---

{% assign cv = site.data.cv.cv %}
{% assign sections = cv.sections %}

<div class="cv-page">
  <header class="cv-header">
    <div>
      <h1>{{ cv.name }}</h1>
      <p class="cv-label">{{ cv.label }}</p>
      <p class="cv-contact">{{ cv.email }} | {{ cv.location }}</p>
    </div>
    <a class="cv-pdf-link" href="{{ page.cv_pdf | relative_url }}" target="_blank" rel="noopener noreferrer">
      <i class="fa-solid fa-file-pdf"></i> PDF
    </a>
  </header>

  <section class="cv-section">
    <h2>Professional Summary</h2>
    <p>{{ cv.summary }}</p>
  </section>

  <section class="cv-section">
    <h2>Experience</h2>
    {% for item in sections.Experience %}
      <article class="cv-entry">
        <h3>{{ item.position }}</h3>
        <p class="cv-meta">
          {{ item.company }} | {{ item.start_date }} - {{ item.end_date }} | {{ item.location }}
        </p>
        {% if item.highlights %}
          <ul>
            {% for highlight in item.highlights %}
              <li>{{ highlight }}</li>
            {% endfor %}
          </ul>
        {% endif %}
      </article>
    {% endfor %}
  </section>

  <section class="cv-section">
    <h2>Education</h2>
    {% for item in sections.Education %}
      <article class="cv-entry">
        <h3>{{ item.studyType }}, {{ item.area }}</h3>
        <p class="cv-meta">
          {{ item.institution }} | {{ item.start_date }} - {{ item.end_date }} | {{ item.location }}
        </p>
        {% if item.score %}
          <p>{{ item.score }}</p>
        {% endif %}
        {% if item.highlights %}
          <ul>
            {% for highlight in item.highlights %}
              <li>{{ highlight }}</li>
            {% endfor %}
          </ul>
        {% endif %}
      </article>
    {% endfor %}
  </section>

  <section class="cv-section">
    <h2>Publications</h2>
    {% for item in sections.Publications %}
      <article class="cv-entry">
        <h3>{{ item.title }}</h3>
        <p class="cv-meta">{{ item.authors | join: ", " }} | {{ item.publisher }} | {{ item.releaseDate }}</p>
      </article>
    {% endfor %}
  </section>

  <section class="cv-section">
    <h2>Projects</h2>
    {% for item in sections.Projects %}
      <article class="cv-entry">
        <h3>{{ item.name }}</h3>
        <p class="cv-meta">{{ item.start_date }} - {{ item.end_date }} | {{ item.location }}</p>
        <p>{{ item.summary }}</p>
        {% if item.highlights %}
          <ul>
            {% for highlight in item.highlights %}
              <li>{{ highlight }}</li>
            {% endfor %}
          </ul>
        {% endif %}
      </article>
    {% endfor %}
  </section>

  <section class="cv-section">
    <h2>Awards</h2>
    {% for item in sections.Awards %}
      <article class="cv-entry">
        <h3>{{ item.title }}</h3>
        <p class="cv-meta">{{ item.awarder }} | {{ item.date }}</p>
        <p>{{ item.summary }}</p>
      </article>
    {% endfor %}
  </section>

  <section class="cv-section">
    <h2>Skills</h2>
    {% for item in sections.Skills %}
      <p><strong>{{ item.name }}:</strong> {{ item.keywords | join: ", " }}</p>
    {% endfor %}
  </section>
</div>

<style>
  .cv-page,
  .cv-page * {
    color: #000000;
  }

  .cv-header {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: flex-start;
    margin-bottom: 1.8rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid #000000;
  }

  .cv-header h1 {
    margin-bottom: 0.2rem;
    font-size: 2rem;
    line-height: 1.2;
  }

  .cv-label,
  .cv-contact,
  .cv-meta {
    margin: 0.15rem 0;
  }

  .cv-pdf-link {
    white-space: nowrap;
    font-weight: 600;
    text-decoration: none;
  }

  .cv-section {
    margin-top: 1.7rem;
  }

  .cv-section h2 {
    margin-bottom: 0.8rem;
    font-size: 1.35rem;
    font-weight: 700;
  }

  .cv-section > :not(h2) {
    padding: 1.1rem 1.25rem;
    border-right: 1px solid #d1d5db;
    border-left: 1px solid #d1d5db;
    background: #ffffff;
  }

  .cv-section > :not(h2):nth-child(2) {
    border-top: 1px solid #d1d5db;
    border-top-left-radius: 0.25rem;
    border-top-right-radius: 0.25rem;
  }

  .cv-section > :not(h2):last-child {
    border-bottom: 1px solid #d1d5db;
    border-bottom-right-radius: 0.25rem;
    border-bottom-left-radius: 0.25rem;
  }

  .cv-entry {
    margin-bottom: 0;
    padding-top: 1rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #e5e7eb;
  }

  .cv-entry:last-child {
    margin-bottom: 0;
    border-bottom-color: #d1d5db;
  }

  .cv-entry h3 {
    margin-bottom: 0.2rem;
    font-size: 1.05rem;
    font-weight: 700;
  }

  .cv-entry ul {
    margin-top: 0.35rem;
    padding-left: 1.25rem;
  }

  .cv-entry li {
    margin-bottom: 0.2rem;
  }
</style>
