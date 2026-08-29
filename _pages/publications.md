---
layout: page
permalink: /publications/
title: Publications
description: Publications and conference papers.
nav: true
nav_order: 2
---

<!-- _pages/publications.md -->

<div class="publications">

{% bibliography %}

</div>

<style>
  .publications ol.bibliography li .abbr {
    display: none;
  }

  .publications ol.bibliography li .abbr + .col-sm-8 {
    flex: 0 0 100%;
    max-width: 100%;
  }
</style>
