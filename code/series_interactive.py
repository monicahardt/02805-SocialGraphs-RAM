"""Self-contained character polarity explorer and a local figure collection."""

import html
import json

from .series_comparison import BOOK, FILM, INK, PAPER
from .vocabulary_map import COLORS, ROOT


def render_polarity(data, out):
    import plotly.graph_objects as go
    import plotly.io as pio

    nodes = [dict(row) for row in data['polarity']]
    for node in nodes:
        node['terms'] = {}
        for polarity in ('positive', 'negative'):
            node['terms'][polarity] = []
            for row in [r for r in data['contributions'] if r['id']==node['id'] and r['polarity']==polarity][:6]:
                node['terms'][polarity].append({'word':row['word'],'count':row['count'],
                    'per_1000':row['per_1000'],'examples':data['examples'][node['id']][row['word']]})
    traces = [go.Scatter(x=[],y=[],mode='lines',line={'width':1,'color':'#b8c5b6'},opacity=.32,
                         hoverinfo='skip',showlegend=False),
              go.Scatter(x=[],y=[],mode='lines',line={'width':2.5,'color':INK},
                         hoverinfo='skip',showlegend=False)]
    groups=[]
    for role,color in COLORS.items():
        for medium,symbol in (('books','circle-open'),('movies','triangle-up')):
            subset=[n for n in nodes if n['role']==role and n['medium']==medium]
            groups.append({'trace':len(traces),'role':role,'ids':[n['id'] for n in subset]})
            traces.append(go.Scatter(x=[n['negative_per_1000'] for n in subset],y=[n['positive_per_1000'] for n in subset],
                customdata=[n['id'] for n in subset],mode='markers',
                marker={'color':color,'size':[10 if n['low_word_count'] else 14 for n in subset],
                        'symbol':symbol,'line':{'width':1.5}},
                text=[f"<b>{html.escape(n['character'])}</b> · {'Book' if medium=='books' else 'Film'}<br>"
                      f"{n['words']:,} words · stories {n['installments']}<br>"
                      f"Positive: {n['positive_per_1000']:.1f} / 1,000<br>Negative: {n['negative_per_1000']:.1f} / 1,000<br>"
                      f"{n['role'].title()}" for n in subset],hovertemplate='%{text}<extra></extra>',showlegend=False))
    max_x=max(n['negative_per_1000'] for n in nodes)*1.18
    max_y=max(n['positive_per_1000'] for n in nodes)*1.18
    fig=go.Figure(traces)
    fig.update_layout(height=650,margin={'l':65,'r':25,'t':30,'b':65},paper_bgcolor=PAPER,plot_bgcolor=PAPER,
        font={'family':'Arial, sans-serif','size':13,'color':INK},hovermode='closest',dragmode='pan',
        xaxis={'title':'Negative words / 1,000 →','range':[0,max_x],'gridcolor':'#dfe2d7','zeroline':False,'fixedrange':False},
        yaxis={'title':'Positive words / 1,000 →','range':[0,max_y],'gridcolor':'#dfe2d7','zeroline':False,'fixedrange':False})
    plot=pio.to_html(fig,full_html=False,include_plotlyjs=True,div_id='polarity',
        config={'responsive':True,'displaylogo':False,'scrollZoom':True,'modeBarButtonsToRemove':['select2d','lasso2d']})
    template=(ROOT/'code/templates/character_polarity.html').read_text(encoding='utf-8')
    script=(ROOT/'code/templates/character_polarity.js').read_text(encoding='utf-8')
    encoded=json.dumps({'nodes':nodes,'groups':groups,'colors':COLORS},ensure_ascii=False).replace('<','\\u003c')
    options=''.join(f'<option value="{n["id"]}">{html.escape(n["character"])}</option>' for n in nodes[::2])
    (out/'character_word_map.html').write_text(template.replace('__NODES__',encoded).replace('__OPTIONS__',options)
        .replace('__SCRIPT__',script).replace('__PLOT__',plot),encoding='utf-8')


def render_gallery(out):
    cards=[('01','Word counts','01_dialogue_volume','Seven stories, two dialogue budgets',
            'All seven books and eight films. Compare the amount of attributed dialogue before interpreting changes in vocabulary.'),
           ('02','LDA','02_topic_journeys','Follow the topics through the series',
            'Eight shared topics across every story. Book and film averages match characters within each installment.'),
           ('02B','LDA','02b_all_character_topics','Every matched voice, side by side',
            'All 51 characters with matched LDA documents. Topic fingerprints use one color scale and shared topic IDs.'),
           ('03','TF-IDF','03_vocabulary_retention','Who sounds most like their book version?',
            'All 59 matched characters, ranked by full-vector cosine similarity. No character is selected for having a striking result.'),
           ('04','PPMI','04_word_contexts','Same word, different company',
            'Books and films now have separate context matrices. Shared vocabulary, five-word windows and visible pair counts.'),
           ('05','Word-list polarity','05_character_word_map','Where do the characters’ words place them?',
            'Positive and negative word densities for all 59 character pairs. Story roles supply colors; measured words supply coordinates.')]
    content=''.join(f'''<article><p class="eyebrow">{number} / {method}</p><h2>{title}</h2><p>{description}</p>
        <a href="{stem}.png"><img src="{stem}.png" alt="{html.escape(title)}" loading="lazy"></a>
        <p class="links"><a href="{stem}.png">Full image</a> <a href="{stem}.svg">Vector SVG</a>
        {('<a href="character_word_map.html">Explore interactively →</a>' if number=='05' else '')}</p></article>'''
        for number,method,stem,title,description in cards)
    document='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>From page to screen · Series comparisons</title><style>
    *{box-sizing:border-box}body{margin:0;color:#243e3b;background:#f7f4ec;font:16px/1.6 Arial,sans-serif}
    main{max-width:1300px;margin:auto;padding:45px 30px}.eyebrow{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#66796f}
    h1{font:normal clamp(34px,5vw,60px)/1.12 Georgia,serif;max-width:1000px;margin:10px 0 25px}h2{font:normal 28px/1.2 Georgia,serif}
    header{border-bottom:1px solid #ccd3c4;padding-bottom:35px;margin-bottom:35px}header p{max-width:850px}.actions,.links{display:flex;gap:20px;flex-wrap:wrap}
    a{color:#287c83;text-underline-offset:4px}article{padding:15px 0 30px;border-bottom:1px solid #ccd3c4;margin-bottom:30px}article>p{max-width:900px}
    img{width:100%;max-height:820px;object-fit:contain;object-position:left center;border:1px solid #e0e1d5}.note{color:#66796f;font-size:14px;max-width:1000px}
    </style></head><body><main><header><p class="eyebrow">Harry Potter / books & films</p>
    <h1>Seven stories.<br>Many voices. Two versions.</h1>
    <p>A series-wide collection of word counts, shared topics, vocabulary similarities and word contexts, followed by a transparent positive/negative vocabulary map.</p>
    <p class="actions"><a href="character_word_map.html">Explore the character word map →</a>
    <a href="../vocabulary_map/vocabulary_map.html">Explore the vocabulary network →</a><a href="../spells/index.html">Explore spell language →</a><a href="README.md">Methods, coverage & results</a></p>
    <p class="note">All stories contribute. Character comparisons use matched installments and minimum word thresholds, so sparse or unmatched voices are excluded and listed in the data.
    Word polarity measures vocabulary; it does not assign moral character. Earlier Umbridge figures remain available as case studies.</p></header>
    '''+content+'''<p class="note">Each image has a corresponding CSV in this folder. The original datasets and the earlier LDA, TF-IDF, PPMI and interactive network are preserved.</p></main></body></html>'''
    (out/'index.html').write_text(document,encoding='utf-8')
