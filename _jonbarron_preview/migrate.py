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


def page(title, content, prefix='', description='', body_class='', canonical_path=''):
    reading_css = f'\n  <link rel="stylesheet" href="{prefix}project-reading.css">' if body_class in ('home-page', 'project-page') else ''
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="author" content="Seungyeop Lee">
  <meta name="description" content="{esc(description)}">
  <link rel="icon" type="image/png" sizes="64x64" href="{prefix}assets/images/profile-favicon.png">
  <title>{esc(title)}</title>
  <link rel="canonical" href="https://yeop-giraffe.github.io/{esc(canonical_path)}">
  <link rel="stylesheet" href="{prefix}stylesheet.css">
  <link rel="stylesheet" href="{prefix}site.css">{reading_css}
</head>
<body class="{esc(body_class)}">
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
source_download = cv.get('source_download', download)
if not (SITE / source_download).is_file():
    raise FileNotFoundError(f'Original CV document missing: {SITE / source_download}')

(SITE / 'assets/pdf').mkdir(parents=True, exist_ok=True)

cv_content = f'<nav><a href="index.html">← Home</a> / <a href="{esc(download)}" download>Download CV (PDF)</a></nav><h1>{esc(cv["name"])}</h1><p>{esc(cv["label"])}<br><a href="mailto:{esc(cv["email"])}">{esc(cv["email"])}</a> · {esc(cv["phone"])}<br>{esc(cv["location"])}</p>'
cv_anchors = {section: re.sub(r'[^a-z0-9]+', '-', section.lower()).strip('-') for section in cv['sections']}
cv_content += '<nav class="page-nav" aria-label="CV sections">' + ''.join(f'<a href="#{cv_anchors[section]}">{esc(section)}</a>' for section in cv['sections']) + '</nav>'
for section, items in cv['sections'].items():
    cv_content += f'<section class="cv-section" id="{cv_anchors[section]}"><h2>{esc(section)}</h2>'
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
        heading_text = esc(heading)
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
            resource_label = 'Manuscript' if item.get('status') else 'Thesis' if item.get('id') == 'lee2025thesis' else 'Paper'
            poster = f' / <a href="{esc(item["poster_pdf"])}">Poster (PDF)</a>' if item.get('poster_pdf') else ''
            cv_content += f'<p><a href="{esc(item["pdf"])}">{resource_label} (PDF)</a>{poster}</p>'
        if item.get('email'):
            cv_content += f'<p><a href="mailto:{esc(item["email"])}">{esc(item["email"])}</a> · {esc(item.get("phone", ""))}</p>'
        details = item.get('highlights') or item.get('keywords')
        if details:
            cv_content += '<ul>' + ''.join(f'<li>{esc(x)}</li>' for x in details) + '</ul>'
        cv_content += '</article>'
    cv_content += '</section>'
(SITE / 'cv.html').write_text(page('CV | Seungyeop Lee', cv_content, description='Education, research, publications and engineering experience of Seungyeop Lee.', body_class='cv-page', canonical_path='cv.html'), encoding='utf-8')

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


def publication_html(item, show_title=True, project_link=None, description=None, show_affiliation=True, category=None, contribution=None):
    title_text = esc(item['title'])
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
    contribution_html = f'<p class="project-contribution"><strong>Role:</strong> {esc(contribution)}</p>' if contribution else ''
    category_html = ''
    pdf_link = ''
    if category is not None:
        if item.get('pdf'):
            if not (SITE / item['pdf']).is_file():
                raise FileNotFoundError(f'Publication PDF missing: {item["pdf"]}')
            pdf_link = f' / <a class="publication-pdf" href="{esc(item["pdf"])}" target="_blank" rel="noopener">PDF</a>'
        if item.get('poster_pdf'):
            if not (SITE / item['poster_pdf']).is_file():
                raise FileNotFoundError(f'Publication poster PDF missing: {item["poster_pdf"]}')
            pdf_link += f' / <a class="publication-poster" href="{esc(item["poster_pdf"])}" target="_blank" rel="noopener">Poster</a>'
        category_html = f'<p class="project-meta">{esc(category)}</p>'
    return f'<div class="publication-meta" data-publication-id="{esc(item["id"])}">{title}<p>{authors}</p>{affiliation}{advisor}<p class="publication-venue"><em>{esc(venue)}</em>{status}{note}{pdf_link}</p>{category_html}{description_html}{contribution_html}</div>'


def project_brief_html(fields):
    entries = ''.join(f'<dt>{esc(label)}</dt><dd>{esc(value)}</dd>' for label, value in fields.items())
    return f'<section class="project-summary" id="summary" aria-labelledby="summary-title"><h2 id="summary-title">Project Summary</h2><dl class="summary-facts">{entries}</dl></section>'


def project_sections(body_html):
    """Give generated pages the same summary-first navigation as research pages."""
    links = [('summary', 'Summary')]
    used_ids = set(re.findall(r'\bid="([^"]+)"', body_html)) | {'summary', 'summary-title'}
    # Fold the existing UAV study shortcuts into the page-wide navigation.
    study_labels = dict(re.findall(r'<a href="#([^"]+)">([^<]+)</a>', body_html))
    body_html = re.sub(r'<nav class="uav-section-nav".*?</nav>', '', body_html, flags=re.S)

    def heading(match):
        attrs, title = match.groups()
        text = html.unescape(re.sub(r'<[^>]+>', '', title)).strip()
        existing = re.search(r'\bid="([^"]+)"', attrs)
        if existing:
            anchor = existing[1]
        else:
            base = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-') or 'section'
            anchor = base
            suffix = 2
            while anchor in used_ids:
                anchor = f'{base}-{suffix}'
                suffix += 1
            used_ids.add(anchor)
            attrs += f' id="{anchor}"'
        links.append((anchor, html.unescape(study_labels.get(anchor, text))))
        return f'<h2{attrs}>{title}</h2>'

    body_html = re.sub(r'<h2([^>]*)>(.*?)</h2>', heading, body_html, flags=re.S)
    navigation = '<nav class="section-nav" aria-label="On this page">' + ''.join(
        f'<a href="#{anchor}">{esc(label)}</a>' for anchor, label in links) + '</nav>'
    return navigation + body_html


(SITE / 'projects').mkdir(exist_ok=True)
publication_rows = ''
project_rows = ''
labels = {'dart': 'DART', 'drone-control-assist': 'UAV', 'soft-exosuit-controller': 'Exosuit', 'monocular-depth-estimation': 'Depth', 'hardware-in-the-loop': 'Samsung SW', 'robot-vacuum-mop-module': 'Samsung HW', 'autonomous-lighter-than-air-vehicle': 'Airship'}
# Use project figures or user-provided representative images.
thumbnails = {
    'dart': ('assets/dart/scene-graph.png', 'DART scene graph connecting indoor regions, objects, and observation viewpoints.'),
    'masters-thesis': ('assets/masters-thesis/integrated-interface.png', 'Monocular vision-based drone interface showing spatial reconstruction, obstacles, and predicted flight paths.'),
    'drone-control-assist': ('assets/images/uav/icros2023-platform.jpg', 'Indoor quadcopter with a stereo camera and onboard flight-control electronics.'),
    'monocular-depth-estimation': ('assets/monocular-depth-estimation/pipeline.png', 'CycleGAN synthetic-to-real data generation and monocular depth model training pipeline.'),
    'autonomous-lighter-than-air-vehicle': ('assets/lighter-than-air/competition-demo.png', 'Helium-supported competition robot with its lightweight frame and onboard control hardware.'),
    'soft-exosuit-controller': ('assets/soft-exosuit/exosuit-control-sensing-overview.png', 'Tendon-driven lower-limb soft exosuit alongside a Jetson controller, motor, and load-cell and IMU sensor interfaces.'),
    'hardware-in-the-loop': ('assets/samsung/samsung-wordmark-blue.png', 'Samsung'),
    'robot-vacuum-mop-module': ('assets/samsung/samsung-wordmark-blue.png', 'Samsung'),
}
company_project_labels = {'hardware-in-the-loop': 'SW', 'robot-vacuum-mop-module': 'HW'}
for data, body, slug in projects:
    related = [publication_by_id[publication_id] for publication_id in data.get('publications', [])]
    # Resolve the existing Jekyll PDF helper into a standalone relative link.
    body = re.sub(r"\{\{\s*['\"](/assets/[^'\"]+)['\"]\s*\|\s*relative_url\s*\}\}", lambda m: '..' + m[1], body)
    group_anchor = 'research' if related else 'projects'
    group_title = 'Publications & Presentations' if related else 'Selected Projects'
    detail = f'<nav><a href="../index.html#{group_anchor}">← {esc(group_title)}</a></nav><h1>{esc(data["title"])}</h1><p class="muted">{esc(data.get("display_category", ""))}</p>'
    context = ' · '.join(data[key] for key in ('affiliation', 'period') if data.get(key))
    if context:
        detail += f'<p class="project-context">{esc(context)}</p>'
    if data.get('project_brief'):
        detail += project_brief_html(data['project_brief'])
    body_html = markdown.markdown(body, extensions=['tables', 'fenced_code'])
    if related:
        body_html += '<section class="related-publications"><h2>Publications &amp; Presentations</h2>'
        body_html += ''.join(publication_html(item) for item in related) + '</section>'
    detail += project_sections(body_html)
    target = SITE / 'projects' / f'{slug}.html'
    if data.get('standalone_html'):
        # Academic project pages are edited directly and must survive regeneration.
        if not target.is_file():
            raise FileNotFoundError(f'Standalone project page missing: {target}')
    else:
        target.write_text(page(data['title'] + ' | Seungyeop Lee', detail, '../', data.get('description', ''), 'project-page', f'projects/{slug}.html'), encoding='utf-8')
    link = f'projects/{slug}.html'
    heading = related[0]['title'] if len(related) == 1 else data['title']
    heading_text = esc(heading)
    heading_html = f'<a class="papertitle" href="{link}">{heading_text}</a>' if len(related) <= 1 else ''
    descriptions = data.get('publication_descriptions', {})
    categories = data.get('publication_categories', {})
    contributions = data.get('publication_contributions', {})
    publication_metadata = ''.join(publication_html(item, show_title=len(related) > 1, project_link=link,
        description=descriptions.get(item['id'], data.get('summary', data.get('description', ''))),
        show_affiliation=False, category=categories.get(item['id'], data.get('display_category', '')),
        contribution=contributions.get(item['id'], data.get('role_summary'))) for item in related)
    affiliation_html = ''
    if not related:
        affiliation = data.get('affiliation', '')
        period = data.get('period', '')
        affiliation_html = f'<p class="project-affiliation">{esc(affiliation)}{(", " + esc(period)) if period else ""}</p>'
    # Each paper's description takes the place of a shared project summary.
    summary_html = ''
    if not related:
        summary_html = f'<p class="project-meta">{esc(data.get("display_category", ""))}</p><p>{esc(data.get("summary", data.get("description", "")))}</p>'
        if data.get('role_summary'):
            summary_html += f'<p class="project-contribution"><strong>Role:</strong> {esc(data["role_summary"])}</p>'
    thumbnail = thumbnails.get(slug)
    visual_class = 'project-label'
    visual_html = esc(labels.get(slug, 'Project'))
    if thumbnail:
        image_path, image_alt = thumbnail
        if not (SITE / image_path).is_file():
            raise FileNotFoundError(f'Project thumbnail missing: {SITE / image_path}')
        visual_class += ' project-thumbnail'
        visual_html = f'<img src="{esc(image_path)}" alt="{esc(image_alt)}" width="160" height="160" loading="lazy" decoding="async">'
        if slug in company_project_labels:
            visual_class += ' company-thumbnail'
            visual_html = f'<img src="{esc(image_path)}" alt="{esc(image_alt)}" width="1600" height="425" loading="lazy" decoding="async"><span class="company-project-label">{esc(company_project_labels[slug])}</span>'
    card = f'''<article class="project" data-project="{esc(slug)}">
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

def activity_html(item):
    title = item.get('name', item.get('title', ''))
    dates = item.get('dates') or item.get('date') or ' – '.join(str(item[key]) for key in ('start_date', 'end_date') if item.get(key))
    context = ' · '.join(str(item[key]) for key in ('position', 'company') if item.get(key))
    details = ''.join(f'<p>{esc(detail)}</p>' for detail in item.get('highlights', []))
    links = ' / '.join(f'<a href="{esc(link["url"])}" target="_blank" rel="noopener noreferrer">{esc(link["label"])}</a>' for link in item.get('homepage_links', []))
    links_html = f' <span class="activity-links">· {links}</span>' if links and context else f'<span class="activity-links">{links}</span>' if links else ''
    context_html = f'<p class="activity-meta">{esc(context)}{links_html}</p>' if context or links else ''
    return f'<article class="activity-entry"><header><h3>{esc(title)}</h3><p class="activity-date">{esc(dates)}</p></header>{context_html}{details}</article>'


leadership_names = [
    'Horang-Nabi Model Aircraft Club Foundation',
    'National University Student Model Aircraft Competition',
    'Pi Village Start-up Team – DRONEDU',
]
leadership_records = {item['name']: item for item in cv['sections']['Selected Projects & Leadership']}
leadership_html = ''.join(activity_html(leadership_records[name]) for name in leadership_names)
teaching_html = ''.join(activity_html(item) for item in cv['sections']['Teaching and Mentoring'])

content = f'''<section class="intro">
  <div><h1 class="name">{esc(cv['name'])}</h1>{intro}
  <nav class="contact" aria-label="Contact and CV"><a href="mailto:{esc(cv['email'])}">Email</a> / <a href="cv.html">CV</a> / <a href="{esc(download)}" download>CV PDF</a></nav></div>
  <div class="profile"><img class="profile-photo" src="assets/images/lsy-profile.jpg" width="800" height="800" alt="Portrait of Seungyeop Lee by the sea" fetchpriority="high" decoding="async"></div>
</section>
<section class="research-interests"><h2>Research Interests</h2>{research}</section>
<nav class="page-nav" aria-label="Homepage sections"><a href="#research">Publications</a><a href="#projects">Projects</a><a href="#leadership">Leadership</a><a href="#teaching">Teaching &amp; Mentoring</a></nav>
<section id="research"><h2>Publications &amp; Presentations</h2>{publication_rows}</section>
<section id="projects"><h2>Selected Projects</h2>{project_rows}</section>
<section id="leadership" class="activities"><h2>Leadership</h2>{leadership_html}</section>
<section id="teaching" class="activities"><h2>Teaching &amp; Mentoring</h2>{teaching_html}</section>'''
(SITE / 'index.html').write_text(page('Seungyeop Lee | Robotics Research', content, description='Seungyeop Lee: robot perception, embedded control and human-robot interaction, with research interests in adaptive wearable assistance.', body_class='home-page'), encoding='utf-8')
(SITE / '.nojekyll').touch()
print(f'Imported {len(projects)} projects and complete CV into {SITE}')
