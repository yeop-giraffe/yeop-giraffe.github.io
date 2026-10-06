"""Build the standalone draft from its reviewed CV and SOP-based content."""
from pathlib import Path
import sys
import html
import re

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '.deps'))
import yaml
import markdown

CONTENT = HERE / 'content'
SITE = HERE / 'site'
SITE.mkdir(exist_ok=True)
esc = lambda value: html.escape(str(value), quote=True)


def author_html(author, highlight=False):
    name = author.rstrip('*†')
    markers = author[len(name):]
    label = f'<strong>{esc(name)}</strong>' if highlight else esc(name)
    return label + (f'<sup>{esc(markers)}</sup>' if markers else '')


def page(title, content, prefix=''):
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="author" content="Seungyeop Lee">
  <title>{esc(title)}</title>
  <link rel="stylesheet" href="{prefix}stylesheet.css">
  <link rel="stylesheet" href="{prefix}site.css">
</head>
<body>
<main>{content}</main>
<footer>Website template by <a href="https://github.com/jonbarron/jonbarron.github.io">Jon Barron</a>.</footer>
</body>
</html>
'''


def frontmatter(path):
    _, head, body = path.read_text(encoding='utf-8-sig').split('---', 2)
    return yaml.safe_load(head), body.strip()


cv = yaml.safe_load((CONTENT / 'cv.yml').read_text(encoding='utf-8'))['cv']
intro = markdown.markdown((CONTENT / 'about.md').read_text(encoding='utf-8'))
research = markdown.markdown((CONTENT / 'research.md').read_text(encoding='utf-8'))
download = cv['download']
if not (SITE / download).is_file():
    raise FileNotFoundError(f'Latest CV download missing: {SITE / download}')

(SITE / 'assets/pdf').mkdir(parents=True, exist_ok=True)

cv_content = f'<nav><a href="index.html">← Home</a> / <a href="{esc(download)}" download>Download CV (DOCX)</a></nav><h1>{esc(cv["name"])}</h1><p>{esc(cv["label"])}<br><a href="mailto:{esc(cv["email"])}">{esc(cv["email"])}</a> · {esc(cv["phone"])}<br>{esc(cv["location"])}</p>'
for section, items in cv['sections'].items():
    cv_content += f'<section class="cv-section"><h2>{esc(section)}</h2>'
    for item in items:
        heading = item.get('title') or item.get('name') or item.get('position') or f'{item.get("studyType", "")}, {item.get("area", "")}'
        metadata = [item.get(k) for k in ['company', 'institution', 'publisher', 'awarder', 'location'] if item.get(k)]
        if item.get('name') and item.get('position'):
            metadata.insert(0, item['position'])
        if item.get('advisor'):
            metadata.append('Advisor: ' + item['advisor'])
        if item.get('start_date'):
            metadata.append(f'{item["start_date"]} – {item.get("end_date", "Present")}')
        metadata += [item[k] for k in ['date', 'dates', 'releaseDate', 'score', 'status'] if item.get(k)]
        heading_text = f'“{esc(heading)}”' if section == 'Publications & Presentations' else esc(heading)
        cv_content += f'<article class="cv-entry"><h3>{heading_text}</h3>'
        if item.get('authors'):
            cv_content += '<p>' + ', '.join(author_html(author, author.rstrip('*†') == cv['name']) for author in item['authors']) + '</p>'
        if item.get('affiliation'):
            cv_content += f'<p class="publication-affiliation">{esc(item["affiliation"])}</p>'
        if metadata:
            note = ' ' + esc(item['note']) if item.get('note') else ''
            cv_content += f'<p class="muted">{esc(" · ".join(map(str, metadata)))}{note}</p>'
        if item.get('summary'):
            cv_content += f'<p>{esc(item["summary"])}</p>'
        if item.get('note') and not metadata:
            cv_content += f'<p>{esc(item["note"])}</p>'
        if item.get('pdf'):
            cv_content += f'<p><a href="{esc(item["pdf"])}">Manuscript (PDF)</a></p>'
        if item.get('email'):
            cv_content += f'<p><a href="mailto:{esc(item["email"])}">{esc(item["email"])}</a> · {esc(item.get("phone", ""))}</p>'
        details = item.get('highlights') or item.get('keywords')
        if details:
            cv_content += '<ul>' + ''.join(f'<li>{esc(x)}</li>' for x in details) + '</ul>'
        cv_content += '</article>'
    cv_content += '</section>'
(SITE / 'cv.html').write_text(page('CV | Seungyeop Lee', cv_content), encoding='utf-8')

projects = [(*frontmatter(p), p.stem) for p in (CONTENT / 'projects').glob('*.md')]
projects.sort(key=lambda x: x[0].get('importance', 99))
publication_records = cv['sections'].get('Publications & Presentations', [])
publication_by_id = {item['id']: item for item in publication_records}
if len(publication_by_id) != len(publication_records):
    raise ValueError('Publication IDs must be unique.')
used_publications = [publication_id for data, _, _ in projects for publication_id in data.get('publications', [])]
if len(used_publications) != len(set(used_publications)):
    raise ValueError('Each publication must be associated with one project.')
if set(used_publications) != set(publication_by_id):
    raise ValueError('Associate every publication with its project before generating the homepage.')


def publication_html(item, show_title=True, project_link=None, description=None, show_affiliation=True, category=None):
    title_text = f'“{esc(item["title"])}”'
    if project_link:
        title_text = f'<a class="papertitle" href="{esc(project_link)}">{title_text}</a>'
    title = f'<h3 class="papertitle">{title_text}</h3>' if show_title else ''
    authors = ', '.join(author_html(author, author.rstrip('*†') == cv['name']) for author in item['authors'])
    affiliation = f'<p class="publication-affiliation">{esc(item["affiliation"])}</p>' if show_affiliation and item.get('affiliation') else ''
    advisor = f'<p class="publication-advisor muted">Advisor: {esc(item["advisor"])}</p>' if item.get('advisor') else ''
    venue = item.get('short_venue', item['publisher'])
    status = ' · ' + esc(item['status']) if item.get('status') else ''
    note = f' <span class="publication-note">{esc(item["note"])}</span>' if item.get('note') else ''
    description_html = f'<p class="publication-description">{esc(description)}</p>' if description else ''
    category_html = ''
    pdf_link = ''
    if category is not None:
        if item.get('pdf'):
            if not (SITE / item['pdf']).is_file():
                raise FileNotFoundError(f'Publication PDF missing: {item["pdf"]}')
            pdf_link = f' / <a class="publication-pdf" href="{esc(item["pdf"])}" target="_blank" rel="noopener">PDF</a>'
        category_html = f'<p class="project-meta">{esc(category)}</p>'
    return f'<div class="publication-meta" data-publication-id="{esc(item["id"])}">{title}<p>{authors}</p>{affiliation}{advisor}<p class="publication-venue"><em>{esc(venue)}</em>{status}{note}{pdf_link}</p>{category_html}{description_html}</div>'


(SITE / 'projects').mkdir(exist_ok=True)
publication_rows = ''
project_rows = ''
labels = {'dart': 'DART', 'drone-control-assist': 'UAV', 'soft-exosuit-controller': 'Exosuit', 'monocular-depth-estimation': 'Depth', 'hardware-in-the-loop': 'HIL', 'robot-vacuum-mop-module': 'Robotics', 'autonomous-lighter-than-air-vehicle': 'Airship'}
# Use figures from each project's own paper; projects without verified images keep their labels.
thumbnails = {
    'dart': ('assets/dart/scene-graph.png', 'DART scene graph connecting indoor regions, objects, and observation viewpoints.'),
    'masters-thesis': ('assets/masters-thesis/integrated-interface.png', 'Monocular vision-based drone interface showing spatial reconstruction, obstacles, and predicted flight paths.'),
    'drone-control-assist': ('assets/images/uav/icros2023-platform.jpg', 'Indoor quadcopter with a stereo camera and onboard flight-control electronics.'),
    'monocular-depth-estimation': ('assets/monocular-depth-estimation/pipeline.png', 'CycleGAN synthetic-to-real data generation and monocular depth model training pipeline.'),
    'autonomous-lighter-than-air-vehicle': ('assets/lighter-than-air/competition-demo.png', 'Helium-supported competition robot with its lightweight frame and onboard control hardware.'),
    'soft-exosuit-controller': ('assets/soft-exosuit/sensor-module.jpg', 'Assembled soft-exosuit sensor module with a Feather M4 CAN board, IMU, load-cell interface, and power components.'),
}
for data, body, slug in projects:
    related = [publication_by_id[publication_id] for publication_id in data.get('publications', [])]
    # Resolve the existing Jekyll PDF helper into a standalone relative link.
    body = re.sub(r"\{\{\s*['\"](/assets/[^'\"]+)['\"]\s*\|\s*relative_url\s*\}\}", lambda m: '..' + m[1], body)
    group_anchor = 'research' if related else 'projects'
    group_title = 'Publications & Presentations' if related else 'Selected Projects'
    detail = f'<nav><a href="../index.html#{group_anchor}">← {esc(group_title)}</a></nav><h1>{esc(data["title"])}</h1><p class="muted">{esc(data.get("display_category", ""))}</p>'
    detail += markdown.markdown(body, extensions=['tables', 'fenced_code'])
    if related:
        detail += '<section class="related-publications"><h2>Publications &amp; Presentations</h2>'
        detail += ''.join(publication_html(item) for item in related) + '</section>'
    target = SITE / 'projects' / f'{slug}.html'
    if data.get('standalone_html'):
        # Academic project pages are edited directly and must survive regeneration.
        if not target.is_file():
            raise FileNotFoundError(f'Standalone project page missing: {target}')
    else:
        target.write_text(page(data['title'] + ' | Seungyeop Lee', detail, '../'), encoding='utf-8')
    link = f'projects/{slug}.html'
    heading = related[0]['title'] if len(related) == 1 else data['title']
    heading_text = f'“{esc(heading)}”' if len(related) == 1 else esc(heading)
    heading_html = f'<a class="papertitle" href="{link}">{heading_text}</a>' if len(related) <= 1 else ''
    descriptions = data.get('publication_descriptions', {})
    categories = data.get('publication_categories', {})
    publication_metadata = ''.join(publication_html(item, show_title=len(related) > 1, project_link=link,
        description=descriptions.get(item['id'], data.get('summary', data.get('description', ''))),
        show_affiliation=False, category=categories.get(item['id'], data.get('display_category', ''))) for item in related)
    affiliation_html = ''
    if not related:
        affiliation = data.get('affiliation', '')
        period = data.get('period', '')
        affiliation_html = f'<p class="project-affiliation">{esc(affiliation)}{(", " + esc(period)) if period else ""}</p>'
    # Each paper's description takes the place of a shared project summary.
    summary_html = ''
    if not related:
        summary_html = f'<p class="project-meta">{esc(data.get("display_category", ""))}</p><p>{esc(data.get("summary", data.get("description", "")))}</p>'
    thumbnail = thumbnails.get(slug)
    visual_class = 'project-label'
    visual_html = esc(labels.get(slug, 'Project'))
    if thumbnail:
        image_path, image_alt = thumbnail
        if not (SITE / image_path).is_file():
            raise FileNotFoundError(f'Project thumbnail missing: {SITE / image_path}')
        visual_class += ' project-thumbnail'
        visual_html = f'<img src="{esc(image_path)}" alt="{esc(image_alt)}" width="160" height="160" loading="lazy" decoding="async">'
    card = f'''<article class="project{' highlighted' if slug == 'dart' else ''}" data-project="{esc(slug)}">
  <a class="{visual_class}" href="{link}" aria-label="{esc(data['title'])}">{visual_html}</a>
  <div>{heading_html}
  {publication_metadata}
  {affiliation_html}
  {summary_html}</div>
</article>'''
    if related:
        publication_rows += card
    else:
        project_rows += card

content = f'''<section class="intro">
  <div><h1 class="name">{esc(cv['name'])}</h1>{intro}
  <nav class="contact"><a href="mailto:{esc(cv['email'])}">Email</a> / <a href="cv.html">CV</a> / <a href="{esc(download)}" download>DOCX</a> / <a href="https://github.com/yeop-giraffe">GitHub</a></nav></div>
  <div class="profile"><img class="profile-photo" src="assets/images/lsy-profile.jpg" width="800" height="800" alt="Portrait of Seungyeop Lee by the sea" fetchpriority="high" decoding="async"></div>
</section>
<section class="research-interests"><h2>Research Interests</h2>{research}</section>
<section id="research"><h2>Publications &amp; Presentations</h2>{publication_rows}</section>
<section id="projects"><h2>Selected Projects</h2>{project_rows}</section>'''
(SITE / 'index.html').write_text(page('Seungyeop Lee', content), encoding='utf-8')
(SITE / '.nojekyll').touch()
print(f'Imported {len(projects)} projects and complete CV into {SITE}')
