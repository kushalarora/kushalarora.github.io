---
layout: default
title: summaries
permalink: /summaries
nav: false
---
<ul class="post-list">
  {% for post in site.summaries %}
    <li>
      <h2><a class="post-title" href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a></h2>
      <p class="post-meta">{{ post.date | date: '%B %-d, %Y — %H:%M' }}</p>
      <p>{{ post.description }}</p>
    </li>
  {% endfor %}
</ul>
