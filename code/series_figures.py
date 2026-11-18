"""Publication figures for the series comparison; numerical work is separate."""

import html
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np

from .series_comparison import BOOK, FILM, INK, PAPER, MUTED, SHORT, TOPIC_NAMES
from .vocabulary_map import COLORS, ROOT


def setup():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': INK, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.edgecolor': '#cbd0c5',
                         'axes.facecolor': PAPER, 'figure.facecolor': PAPER,
                         'svg.fonttype': 'none', 'savefig.facecolor': PAPER})


def heading(fig, number, title, subtitle):
    fig.text(.055, .969, f'FROM PAGE TO SCREEN    /    {number}', fontsize=10,
             color=MUTED, va='top')
    fig.text(.055, .935, title, fontsize=27, fontfamily='DejaVu Serif', va='top')
    fig.text(.055, .888, subtitle, fontsize=10.5, color=MUTED, va='top', linespacing=1.5)


def finish(fig, out, stem, note):
    fig.text(.055, .022, note, fontsize=9, color=MUTED, va='bottom', linespacing=1.5)
    for suffix in ('png', 'svg'):
        fig.savefig(out / f'{stem}.{suffix}', dpi=180, facecolor=PAPER)
    plt.close(fig)


def clean(ax, grid='x'):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(length=0, pad=7)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color='#dce0d5', linewidth=.65)


def volume_plot(data, out):
    rows = {(r['installment'], r['medium']): r for r in data['volume']}
    fig, ax = plt.subplots(figsize=(15, 9))
    fig.subplots_adjust(left=.21, right=.92, top=.77, bottom=.13)
    heading(fig, '01 · WORD COUNTS', 'Seven stories. Much less dialogue on screen.',
            'Every supplied dialogue line, including unknown speakers. The final two films are combined as story 7.\n'
            'Bar lengths show word tokens; the percentage is film words ÷ book words.')
    max_words = max(r['words'] for r in data['volume'])
    for i in range(1, 8):
        b, f = rows[i, 'books']['words'], rows[i, 'movies']['words']
        y = 7-i
        ax.plot([0, b], [y+.12]*2, color=BOOK, lw=15, solid_capstyle='round', zorder=2)
        ax.plot([0, f], [y-.15]*2, color=FILM, lw=7, solid_capstyle='round', zorder=3)
        ax.text(b+max_words*.018, y+.12, f'{b:,}', va='center', fontsize=10, color=BOOK)
        ax.text(f+max_words*.018, y-.17, f'{f:,}', va='center', fontsize=9, color=FILM)
        ax.text(max_words*1.22, y, f'{f/b:.0%}', fontsize=17, fontfamily='DejaVu Serif',
                ha='right', va='center', color=INK)
    ax.set_yticks(range(7), [f'{i}   {SHORT[i-1]}' for i in range(7, 0, -1)])
    ax.set_ylim(-.6, 6.7)
    ax.set_xlim(-max_words*.01, max_words*1.25)
    ax.set_xticks(np.arange(0, max_words+1, 20000), [f'{x//1000}k' for x in np.arange(0, max_words+1, 20000)])
    ax.set_xlabel('Dialogue word tokens', labelpad=12)
    clean(ax)
    ax.legend(handles=[Line2D([], [], lw=9, color=BOOK, label='Book'),
                       Line2D([], [], lw=5, color=FILM, label='Film')], frameon=False,
              loc='upper left', bbox_to_anchor=(0, 1.13), ncol=2)
    finish(fig, out, '01_dialogue_volume',
           'All 7 books and all 8 films are represented. This measures the supplied dialogue extraction, not full-book length.\n'
           'Word-rate comparisons elsewhere divide by the relevant dialogue word total.')


def topic_plot(data, out):
    lookup = {(r['installment'], r['medium']): r for r in data['topics']}
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    fig.subplots_adjust(left=.1, right=.96, top=.825, bottom=.09, wspace=.28, hspace=.65)
    heading(fig, '02 · SHARED LDA TOPICS', 'What changes from page to screen?',
            'Each card follows one shared topic across all seven stories. Circle = book; triangle = film.\n'
            'Within each story, both averages use the same eligible characters, weighted by their modeled word counts.')
    for t, ax in enumerate(axes.flat, 1):
        book = [100 * lookup[i, 'books'][f'topic_{t}'] for i in range(1, 8)]
        film = [100 * lookup[i, 'movies'][f'topic_{t}'] for i in range(1, 8)]
        y = np.arange(7)
        for index, (b, f) in enumerate(zip(book, film)):
            ax.plot([b, f], [index]*2, color=FILM if f > b else BOOK, lw=3, alpha=.45)
        ax.scatter(book, y, facecolors=PAPER, edgecolors=BOOK, marker='o', s=55, linewidth=1.5, zorder=3)
        ax.scatter(film, y, color=FILM, marker='^', s=40, zorder=4)
        ax.set_yticks(y, [str(i) for i in range(1, 8)])
        ax.invert_yaxis()
        ax.set_xlim(-2, 80)
        ax.set_xticks([0, 20, 40, 60, 80], ['0', '20', '40', '60', '80%'])
        ax.set_title(f'{t:02}   {TOPIC_NAMES[t-1]}', fontsize=12, loc='left', pad=14)
        if t == 2:
            ax.text(0,-.23,'Film share is higher in all seven stories',transform=ax.transAxes,fontsize=9,color=FILM)
        clean(ax)
    finish(fig, out, '02_topic_journeys',
           'Topic titles are interpretations of the existing eight-topic LDA. Numbers 1–7 indicate story installment.\n'
           'Matched character coverage can vary by story; this is not a same-scene comparison. Exact coverage: series_topics.csv.')


def topic_matrix(data, out):
    names = sorted({r['character'] for r in data['character_topics']},
                   key=lambda name: (list(COLORS).index(data['roles'].get(name, 'unassigned')), name))
    lookup = {(r['character'], r['medium']): r for r in data['character_topics']}
    fig, axes = plt.subplots(1, 2, figsize=(15, 18))
    fig.subplots_adjust(left=.23, right=.92, top=.83, bottom=.105, wspace=.13)
    heading(fig, '02B · EVERY MATCHED LDA CHARACTER', 'A topic fingerprint for every voice',
            f'{len(names)} matched characters · eight shared topics · one color scale. Each row sums to 100% in each medium.\n'
            'Characters are grouped by provisional story role; those labels never enter the LDA model.')
    cmap = mcolors.LinearSegmentedColormap.from_list('ink_paper', [PAPER, '#a5c6b5', '#22645e', '#123c3b'])
    for ax, medium in zip(axes, ('books', 'movies')):
        values = np.array([[lookup[name, medium][f'topic_{i}'] for i in range(1, 9)] for name in names])
        image = ax.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(8), [f'{i}' for i in range(1, 9)])
        ax.set_title('BOOKS' if medium == 'books' else 'FILMS', loc='left', fontsize=12, pad=15, color=BOOK if medium=='books' else FILM)
        ax.set_yticks(range(len(names)), names if medium == 'books' else ['']*len(names))
        ax.tick_params(length=0, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for label, name in zip(ax.get_yticklabels(), names):
            label.set_color(COLORS[data['roles'].get(name, 'unassigned')])
        for y in np.arange(.5, len(names)-1, 1):
            ax.axhline(y, color=PAPER, lw=.7)
        ax.set_xlabel('Shared topic ID', labelpad=10)
    cax = fig.add_axes([.37, .065, .4, .008])
    fig.colorbar(image, cax=cax, orientation='horizontal', ticks=[0, .25, .5, .75, 1], format=lambda v, pos:f'{v:.0%}')
    finish(fig, out, '02b_all_character_topics',
           'Within each character, only installments represented in both LDA outputs are included. Averages use modeled word counts.\n'
           'The 100-word document threshold excludes some sparse voices. Exact profiles and coverage: character_topics.csv and matched_topic_documents.csv.')


def vocabulary_plot(data, out):
    rows = sorted(data['vocabulary'], key=lambda r: (-r['cosine'], r['character']))
    fig, axes = plt.subplots(1, 3, figsize=(20, 12))
    fig.subplots_adjust(left=.15, right=.97, top=.805, bottom=.16, wspace=.74)
    heading(fig, '03 · TF-IDF & COSINE', 'Whose vocabulary survives the adaptation?',
            'Every matched character, ordered by book–film vocabulary similarity. One point compares two full TF-IDF vectors.\n'
            'The characters’ book and film profiles cover the same story installments; each contains at least 100 words.')
    count = int(np.ceil(len(rows)/3))
    for column, ax in enumerate(axes):
        subset = rows[column*count:(column+1)*count]
        for index, row in enumerate(subset):
            color = COLORS[row['role']]
            ax.plot([0, 1], [index]*2, color='#e3e3d7', lw=1.3)
            ax.plot([0, row['cosine']], [index]*2, color=color, lw=3, alpha=.45)
            ax.scatter(row['cosine'], index, s=30 if min(row['book_words'],row['film_words'])<300 else 55, color=color, zorder=4)
            ax.text(1.04, index, f"{row['cosine']:.2f}", fontsize=10, va='center', ha='right')
        ax.set_yticks(range(len(subset)), [f"{column*count+i+1:02}  {r['character']}"+(' *' if min(r['book_words'],r['film_words'])<300 else '') for i, r in enumerate(subset)])
        ax.set_ylim(count-.4, -.8)
        ax.set_xlim(-.025, 1.06)
        ax.set_xticks([0, .5, 1], ['0', '0.5', '1'])
        ax.tick_params(axis='y', labelsize=9)
        ax.set_xlabel('Different vocabulary  ←  cosine  →  Similar vocabulary', fontsize=9, labelpad=12)
        clean(ax, None)
    fig.legend(handles=[Line2D([], [], marker='o', color=c, label=r.title(), linestyle='none') for r,c in COLORS.items()],
               loc='lower center', bbox_to_anchor=(.52,.07), ncol=4, frameon=False, fontsize=10)
    finish(fig, out, '03_vocabulary_retention',
           'Full vectors and preprocessing match the existing interactive network. Similarity measures word use, not preserved meaning or personality.\n'
           '* Small marker: fewer than 300 words in at least one medium. Names remain; exact lengths and coverage: character_vocabulary.csv.')


def context_plot(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(left=.065, right=.955, top=.755, bottom=.13, hspace=.48, wspace=.15)
    heading(fig, '04 · PPMI IN EACH MEDIUM', 'The company a word keeps',
            'The same nearby-word vocabulary, with separate book and film association scores across the series.\n'
            'Longer bars mean stronger-than-expected association. n is the number of nearby word pairs, not independent scenes.')
    for ax, target in zip(axes.flat, ('blood', 'power', 'death', 'love')):
        rows = [r for r in data['contexts'] if r['word'] == target]
        for i, row in enumerate(rows):
            for medium, sign, color in (('books', -1, BOOK), ('movies', 1, FILM)):
                value, count = row[f'{medium}_ppmi'], row[f'{medium}_count']
                start = sign*2.5
                end = sign*(2.5+value)
                ax.plot([start, end], [i]*2, color=color, alpha=1 if count>=3 else .22,
                        lw=5, solid_capstyle='round')
                ax.text(end+sign*.3, i, f'n={count}', va='center', ha='right' if sign<0 else 'left',
                        fontsize=8, color=color if count>=3 else '#9b9c90')
            ax.text(0, i, row['context'], ha='center', va='center', fontsize=11, color=INK,
                    bbox={'facecolor':PAPER,'edgecolor':'none','pad':2},zorder=4)
        ax.set_ylim(max(8,len(rows))-.5, -.9)
        ax.set_xlim(-13, 13)
        ax.set_xticks([-10.5,-6.5,-2.5,2.5,6.5,10.5], ['8','4','0','0','4','8'])
        ax.set_yticks([])
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(length=0, labelsize=8)
        ax.set_title(target, fontfamily='DejaVu Serif', fontsize=26, loc='center', pad=23)
        ax.text(.03, 1.025, '← BOOK PPMI', transform=ax.transAxes, fontsize=9, color=BOOK)
        ax.text(.97, 1.025, 'FILM PPMI →', transform=ax.transAxes, fontsize=9, color=FILM, ha='right')
    finish(fig, out, '04_word_contexts',
           'Contexts rank by pooled PPMI among pairs seen ≥3 times overall; only alphabetic non-stopwords are displayed. Pale bars have fewer than 3 pairs in that medium.\n'
           'Window: ±5 original token positions within one dialogue line. PPMI is an association weight; these differences are descriptive, not significance tests.')


def label_points(ax, positions, selected):
    """Place a small set of named labels without covering other labels or points."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    occupied = []
    for point in positions:
        x,y = ax.transData.transform(point)
        from matplotlib.transforms import Bbox
        occupied.append(Bbox.from_extents(x-6,y-6,x+6,y+6))
    offsets = [(10,10), (10,-17), (-10,10), (-10,-17), (15,29), (-15,29), (15,-35),
               (-15,-35), (38,0),(-38,0),(0,48),(0,-48),(65,25),(-65,25),(65,-25),(-65,-25)]
    for name, point, color in selected:
        best = None
        for dx,dy in offsets:
            annotation = ax.annotate(name, point, xytext=(dx,dy), textcoords='offset points',
                                     ha='left' if dx>0 else 'right' if dx<0 else 'center', fontsize=9)
            box = annotation.get_window_extent(renderer).expanded(1.06,1.2)
            overlap = sum(box.overlaps(old) for old in occupied)
            outside = not ax.get_window_extent(renderer).contains(box.x0,box.y0) or not ax.get_window_extent(renderer).contains(box.x1,box.y1)
            score = overlap + 100*outside
            annotation.remove()
            if best is None or score<best[0]:
                best = (score,dx,dy,box)
            if score == 0:
                break
        _, dx,dy,box = best
        occupied.append(box)
        ax.annotate(name, point, xytext=(dx,dy), textcoords='offset points',
                    ha='left' if dx>0 else 'right' if dx<0 else 'center', color=color, fontsize=9,
                    bbox={'facecolor':PAPER,'edgecolor':'none','alpha':.95,'pad':2.5},
                    arrowprops={'arrowstyle':'-','lw':.6,'color':'#a2aca0'}, zorder=6)


def polarity_plot(data, out):
    rows = data['polarity']
    fig, ax = plt.subplots(figsize=(15, 12))
    fig.subplots_adjust(left=.105,right=.95,top=.80,bottom=.16)
    heading(fig, '05 · POSITIVE & NEGATIVE VOCABULARY', 'Where do the characters’ words place them?',
            'A transparent word-list map: up means more positive-word matches; right means more negative-word matches.\n'
            'Every matched character appears twice. Arrows connect book circles to film triangles. Colors show story roles.')
    xmax = max(r['negative_per_1000'] for r in rows)*1.17
    ymax = max(r['positive_per_1000'] for r in rows)*1.16
    xx,yy=np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
    rgba=np.zeros((100,100,4));rgba[:,:,:3]=[.84,.87,.81];rgba[:,:,3]=.1+.15*np.abs(xx-yy)
    ax.imshow(rgba,extent=[0,xmax,0,ymax],origin='lower',aspect='auto',zorder=0)
    for i in range(0,len(rows),2):
        b,f=rows[i:i+2]
        color=COLORS[b['role']]
        ax.annotate('',(f['negative_per_1000'],f['positive_per_1000']),
                    (b['negative_per_1000'],b['positive_per_1000']),
                    arrowprops={'arrowstyle':'->','color':color,'lw':.8,'alpha':.28,'shrinkA':5,'shrinkB':5},zorder=2)
    for role,color in COLORS.items():
        for medium,marker in (('books','o'),('movies','^')):
            subset=[r for r in rows if r['role']==role and r['medium']==medium]
            ax.scatter([r['negative_per_1000'] for r in subset],[r['positive_per_1000'] for r in subset],
                       s=[48 if r['low_word_count'] else 80 for r in subset],marker=marker,
                       facecolors=PAPER if medium=='books' else color,edgecolors=color,linewidths=1.4,alpha=.86,zorder=4)
    ax.set(xlim=(0,xmax),ylim=(0,ymax),xlabel='Negative-word matches per 1,000 dialogue words  →',
           ylabel='Positive-word matches per 1,000 dialogue words  →')
    ax.text(.02,.97,'MORE POSITIVE WORDS',transform=ax.transAxes,color=BOOK,fontsize=10,va='top')
    ax.text(.98,.03,'MORE NEGATIVE WORDS',transform=ax.transAxes,color=FILM,fontsize=10,ha='right')
    ax.plot([0,min(xmax,ymax)],[0,min(xmax,ymax)],color='#b6c0b3',lw=.8,ls=(0,(3,5)))
    ax.text(.98,.97,'Both can be high.\nBoth can be low.',transform=ax.transAxes,ha='right',va='top',fontsize=10,color=MUTED)
    clean(ax,'both')
    labels={'Voldemort','Dolores Umbridge','Bellatrix Lestrange','Lucius Malfoy','Harry Potter',
            'Albus Dumbledore','Rubeus Hagrid','Severus Snape','Draco Malfoy',
            'Peter Pettigrew','Nearly Headless Nick','George Weasley'}
    selected=[(r['character']+(' · film' if r['medium']=='movies' else ' · book'),
               (r['negative_per_1000'],r['positive_per_1000']),COLORS[r['role']])
              for r in rows if r['character'] in labels and r['medium']=='movies']
    positions=[(r['negative_per_1000'],r['positive_per_1000']) for r in rows]
    label_points(ax,positions,selected)
    fig.legend(handles=[Line2D([],[],marker='o',linestyle='none',color=c,label=r.title()) for r,c in COLORS.items()],
               loc='lower center',bbox_to_anchor=(.53,.095),ncol=4,frameon=False)
    finish(fig,out,'05_character_word_map',
           'Hu–Liu opinion lexicon (KDD 2004), exact token matches. Small markers: fewer than 300 words; every profile has ≥100.\n'
           'This measures vocabulary polarity, not goodness or evil. Negation, sarcasm and fantasy meanings are not resolved. Inspect words and lines in the interactive version.')


def render_all(data,out):
    setup()
    volume_plot(data,out)
    topic_plot(data,out)
    topic_matrix(data,out)
    vocabulary_plot(data,out)
    context_plot(data,out)
    polarity_plot(data,out)
    from .series_interactive import render_polarity, render_gallery
    render_polarity(data,out)
    render_gallery(out)
