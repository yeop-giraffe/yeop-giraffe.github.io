"""Build the downloadable CV from the same data as the website."""
from pathlib import Path
from html import escape
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '.deps'))
import yaml
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether

cv = yaml.safe_load((HERE / 'content/cv.yml').read_text(encoding='utf-8'))['cv']
OUT = HERE.parent / 'output/pdf/Seungyeop_Lee_CV.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
font_dir = Path('C:/Windows/Fonts')
if (font_dir / 'arial.ttf').is_file():
    for name, filename in [('CV', 'arial.ttf'), ('CV-Bold', 'arialbd.ttf'), ('CV-Italic', 'ariali.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily('CV', normal='CV', bold='CV-Bold', italic='CV-Italic', boldItalic='CV-Bold')
    font, bold = 'CV', 'CV-Bold'
else:
    font, bold = 'Helvetica', 'Helvetica-Bold'

def normalized(value):
    return str(value).replace('–', '-').replace('—', '-').replace('‑', '-').replace('→', 'to')

def text(value):
    return escape(normalized(value))

body = ParagraphStyle('body', fontName=font, fontSize=10, leading=14, spaceAfter=5, allowWidows=0, allowOrphans=0)
meta_style = ParagraphStyle('meta', parent=body, fontSize=9.3, leading=12.8, textColor=colors.HexColor('#475569'))
title_style = ParagraphStyle('item', parent=body, fontName=bold, fontSize=10.5, leading=14, keepWithNext=True)
section_style = ParagraphStyle('section', parent=body, fontName=bold, fontSize=12.5, leading=17, spaceBefore=14, spaceAfter=8, keepWithNext=True, textColor=colors.HexColor('#17558c'))
bullet_style = ParagraphStyle('bullet', parent=body, leftIndent=11, firstLineIndent=-9, spaceAfter=4)

def para(value, style=body):
    return Paragraph(value, style)

story = [para(text(cv['name']), ParagraphStyle('name', parent=body, fontName=bold, fontSize=24, leading=29, spaceAfter=7)),
         para(text(cv['label'])),
         para(f'<link href="mailto:{text(cv["email"])}">{text(cv["email"])}</link> | {text(cv["phone"])} | {text(cv["location"])}', meta_style),
         para('<link href="https://yeop-giraffe.github.io/" color="#17558c">yeop-giraffe.github.io</link>', meta_style), Spacer(1, 3)]

for section, items in cv['sections'].items():
    for index, item in enumerate(items):
        heading = item.get('title') or item.get('name') or item.get('position') or f'{item.get("studyType", "")}, {item.get("area", "")}'
        entry = [para(text(section), section_style)] if index == 0 else []
        entry.append(para(text(heading), title_style))
        if item.get('authors'):
            authors = []
            for author in item['authors']:
                name = author.rstrip('*†')
                markers = author[len(name):]
                value = '<b>'+text(name)+'</b>' if name==cv['name'] else text(name)
                authors.append(value+('<super>'+text(markers)+'</super>' if markers else ''))
            entry.append(para(', '.join(authors)))
        if item.get('affiliation'):
            entry.append(para(text(item['affiliation']), meta_style))
        metadata = [item[k] for k in ['company','institution','publisher','awarder','location'] if item.get(k)]
        if item.get('name') and item.get('position'):
            metadata.insert(0,item['position'])
        if item.get('advisor'):
            metadata.append('Advisor: '+item['advisor'])
        if item.get('start_date'):
            metadata.append(item['start_date']+' - '+item.get('end_date','Present'))
        metadata += [item[k] for k in ['date','dates','releaseDate','score','status'] if item.get(k)]
        if item.get('note'):
            metadata.append(item['note'])
        if metadata:
            entry.append(para(' | '.join(text(value) for value in metadata), meta_style))
        if item.get('summary'):
            entry.append(para(text(item['summary'])))
        for value in item.get('highlights') or item.get('keywords') or []:
            entry.append(para('• '+text(value), bullet_style))
        resources = []
        for key, label in [('pdf','PDF'),('poster_pdf','Poster')]:
            if item.get(key):
                url='https://yeop-giraffe.github.io/'+item[key]
                resources.append(f'<link href="{text(url)}" color="#17558c">{label}</link>')
        if resources:
            entry.append(para(' | '.join(resources), meta_style))
        if item.get('email'):
            entry.append(para(f'<link href="mailto:{text(item["email"])}">{text(item["email"])}</link> | {text(item.get("phone", ""))}', meta_style))
        entry.append(Spacer(1, 7))
        story.append(KeepTogether(entry))

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(font, 8)
    canvas.setFillColor(colors.HexColor('#64748b'))
    canvas.drawString(48, 25, cv['name']+' | Curriculum Vitae')
    canvas.drawRightString(A4[0]-48, 25, str(doc.page))
    canvas.restoreState()

doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=43,title='Seungyeop Lee - Curriculum Vitae',author=cv['name'])
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f'Built CV PDF: {OUT}')
