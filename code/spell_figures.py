"""Spell-analysis figures, findings and searchable evidence page."""

import hashlib
import html
import json
import textwrap
from collections import Counter,defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .series_comparison import BOOK,FILM,INK,PAPER,MUTED,SHORT
from .series_figures import setup,heading,finish,clean
from .spell_analysis import primary
from .vocabulary_map import ROOT


def common_families(data,limit=20):
    counts=Counter()
    for row in data['totals']:counts[row['family']]+=row['count_with_variants']
    return sorted((family for family,n in counts.items() if n),key=lambda family:(-counts[family],family))[:limit]


def frequency_plot(data,out):
    families=common_families(data)
    lookup={(r['family'],r['medium']):r for r in data['totals']}
    fig,ax=plt.subplots(figsize=(15,12))
    fig.subplots_adjust(left=.235,right=.92,top=.795,bottom=.135)
    heading(fig,'06 · SPELL REFERENCES','Which incantations make it to the screen?',
            'The 20 most frequent spell families across all seven books and eight films. Rates adjust for dialogue length.\n'
            'Filled circles: books with spelling variants. Small open circles: strict book spelling. Triangles: films with variants.')
    max_rate=max(r['per_10000'] for r in data['totals'])
    for i,family in enumerate(families):
        b,f=lookup[family,'books'],lookup[family,'movies']
        ax.plot([b['per_10000'],f['per_10000']],[i]*2,color='#c9cfc0',lw=1.5)
        ax.plot([b['strict_per_10000'],b['per_10000']],[i]*2,color=BOOK,lw=4,alpha=.35)
        ax.scatter(b['per_10000'],i,color=BOOK,s=55,zorder=3)
        ax.scatter(b['strict_per_10000'],i,facecolors=PAPER,edgecolors=BOOK,s=19,zorder=4)
        ax.scatter(f['per_10000'],i,color=FILM,s=50,marker='^',zorder=5)
        ax.text(max_rate*1.14,i,f"{b['count_with_variants']:>2} / {f['count_with_variants']:<2}",
                fontsize=9,va='center',ha='right',fontfamily='DejaVu Sans Mono',color=MUTED)
    ax.text(max_rate*1.14,-1,'n book / film',fontsize=9,ha='right',color=MUTED)
    ax.set_yticks(range(len(families)),families)
    ax.set_ylim(len(families)-.3,-1.4)
    ax.set_xlim(-.06,max_rate*1.17)
    ax.set_xlabel('Incantation mentions per 10,000 dialogue words',labelpad=13)
    clean(ax)
    finish(fig,out,'01_spell_frequencies',
           'A mention may be casting, an attempt, a lesson or a quotation. It is not a count of successful spells or all on-screen magic.\n'
           'Lumos, Protego, Reparo, Finite and Bombarda variants are grouped into documented families. Counts and strict sensitivity: spell_totals.csv.')


def timeline_plot(data,out):
    families=common_families(data,18)
    lookup={(r['family'],r['medium'],r['installment']):r for r in data['series']}
    fig,ax=plt.subplots(figsize=(16,12))
    fig.subplots_adjust(left=.205,right=.96,top=.78,bottom=.17)
    heading(fig,'07 · SEVEN-STORY SPELL SCORE','A changing vocabulary of magic',
            'One row per common spell family; one column per story. Circle = book; triangle = film.\n'
            'Marker area represents incantation mentions per 10,000 dialogue words. Empty space means no detected full-form mention.')
    for x in range(7):
        if x%2==0:ax.axvspan(x-.47,x+.47,color='#e9eade',alpha=.65,zorder=0)
    for i,family in enumerate(families):
        for story in range(1,8):
            for m,dx,marker,color in [('books',-.13,'o',BOOK),('movies',.13,'^',FILM)]:
                row=lookup[family,m,story]
                if row['count_with_variants']:
                    ax.scatter(story-1+dx,i,s=35*row['per_10000'],marker=marker,
                               color=color,alpha=.78,edgecolors=PAPER,linewidth=.8,zorder=3)
    ax.set_xticks(range(7),[f'{i+1}\n{title}' for i,title in enumerate(SHORT)],fontsize=10)
    ax.set_yticks(range(len(families)),families)
    ax.set_ylim(len(families)-.4,-1)
    ax.set_xlim(-.55,6.55)
    clean(ax,None)
    ax.spines['bottom'].set_visible(False)
    legend=[Line2D([],[],marker='o',color=BOOK,linestyle='none',label='Book',markersize=9),
            Line2D([],[],marker='^',color=FILM,linestyle='none',label='Film',markersize=9)]
    fig.legend(handles=legend,loc='lower left',bbox_to_anchor=(.205,.085),ncol=2,frameon=False)
    for x,value in zip([.56,.65,.75],[1,3,6]):
        fig.text(x,.104,f'{value} / 10k',fontsize=9,color=MUTED)
    # Show actual area-sized reference dots in a small independent legend axis.
    lax=fig.add_axes([.54,.078,.3,.033]);lax.axis('off')
    lax.scatter([.08,.38,.71],[.45]*3,s=np.array([1,3,6])*35,color=BOOK)
    lax.set(xlim=(0,1),ylim=(0,1))
    finish(fig,out,'02_spell_timeline',
           'All seven story installments are represented; the two final films form column 7. Spell dictionary and OCR adjustments are explicit.\n'
           'Data describe the supplied dialogue, including unknown speakers. Silent magic and omitted or truncated incantations are not counted.')


def context_plot(data,out):
    cases=[('books',10623,'Avada Kedavra','A recollection, not a casting',
            'Ron describes what Moody did to a spider. Counting the incantation in this line as Ron casting it would be wrong.'),
           ('books',33745,'Imperio','Harry uses an Unforgivable Curse',
            'The original book identifies Harry casting at the Gringotts goblin. The supplied dialogue labels the speaker Ted Tonks.'),
           ('books',22845,'Crucio','The victim was labeled as the speaker',
            'The book identifies Bellatrix as the caster and Neville as the victim. This is a confirmed source-attribution error.'),
           ('books',34275,'Imperio','McGonagall also uses one',
            'The book identifies McGonagall casting at Amycus. The supplied dialogue labels this line Ted Tonks.')]
    fig,axes=plt.subplots(2,2,figsize=(15,11))
    fig.subplots_adjust(left=.06,right=.95,top=.77,bottom=.12,hspace=.38,wspace=.15)
    heading(fig,'08 · READ THE CONTEXT','Saying the words is only the beginning',
            'Four checked passages show why word counts need context. The Unforgivable Curses are Avada Kedavra, Crucio and Imperio.\n'
            'Review labels are local annotations; the original datasets have not been edited.')
    for ax,(medium,index,spell,title,note) in zip(axes.flat,cases):
        event=next(e for e in data['events'] if e['medium']==medium and e['record_index']==index and e['spell']==spell)
        ax.set_facecolor('#eeeee2');ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
        ax.text(.05,.91,title,transform=ax.transAxes,fontfamily='DejaVu Serif',fontsize=18,va='top')
        excerpt=event['text'].strip()
        if len(excerpt)>180:
            position=excerpt.lower().find(event['matched_text'].lower())
            start=max(0,position-50)
            excerpt=('…' if start else '')+excerpt[start:start+180]+('…' if start+180<len(excerpt) else '')
        quote='“'+excerpt+'”'
        lines=textwrap.fill(quote,65)
        ax.text(.05,.72,lines,transform=ax.transAxes,fontsize=12,color=BOOK,va='top',linespacing=1.5)
        ax.text(.05,.39,textwrap.fill(note,75),transform=ax.transAxes,fontsize=10,color=INK,va='top',linespacing=1.5)
        ax.text(.05,.11,f"Story {event['installment']} · parsed record {index}\nReviewed speaker: {event['reviewed_speaker']}",
                transform=ax.transAxes,fontsize=9,color=MUTED,va='bottom')
    finish(fig,out,'03_spell_context_cases',
           'These four reviewed examples are selected to explain interpretation problems. They are not an exhaustive casting inventory or a hero/villain comparison.\n'
           'Original lines, reviewed actors and references to the surrounding book text are in manual_reviews.csv; every detected mention is in spell_mentions.csv.')


def explorer(data,out):
    template=(ROOT/'code/templates/spell_explorer.html').read_text(encoding='utf-8')
    script=(ROOT/'code/templates/spell_explorer.js').read_text(encoding='utf-8')
    events=[dict(row,neighbors=e['neighbors']) for row,e in zip(data['public'],data['events'])]
    encoded=json.dumps(events,ensure_ascii=False).replace('<','\\u003c')
    options=''.join(f'<option>{html.escape(s)}</option>' for s in sorted({e['spell'] for e in events}))
    budgets={m:sum(r['words'] for r in data['coverage'] if r['medium']==m) for m in ('books','movies')}
    doc=template.replace('__SPELL_OPTIONS__',options).replace('__DATA__',encoded).replace('__SCRIPT__',script).replace('__BUDGETS__',json.dumps(budgets))
    (out/'index.html').write_text(doc,encoding='utf-8')


def write_report(data,out):
    summary={}
    for medium in ('books','movies'):
        es=[e for e in data['events'] if e['medium']==medium and primary(e)]
        words=sum(r['words'] for r in data['coverage'] if r['medium']==medium)
        summary[medium]={'words':words,'incantation_mentions':len(es),'strict_mentions':sum(primary(e,True) for e in es),
                         'per_10000':10000*len(es)/words,'strict_per_10000':10000*sum(primary(e,True) for e in es)/words,
                         'detected_incantations':len({e['spell'] for e in es}),
                         'incantation_only_lines':len({e['record_index'] for e in es if e['utterance_shape']=='incantation_only'}),
                         'lines_with_incantations':len({e['record_index'] for e in es})}
    summary['film_over_book_rate_ratio']=summary['movies']['per_10000']/summary['books']['per_10000']
    summary['strict_film_over_book_rate_ratio']=summary['movies']['strict_per_10000']/summary['books']['strict_per_10000']
    summary['dictionary_incantations']=len(data['config']['incantations'])
    summary['manual_review_rows']=data['review_count']
    summary['context_pairs_with_three_distinct_lines']=data['contexts']
    summary['parameters']={'context_window':5,'context_min_word_count':3,'context_min_distinct_lines':3,
                            'context_scope':'one dialogue line; full spell spans and own partial components excluded',
                            'rates':'mentions per 10,000 full raw dialogue tokens; unknown speakers retained',
                            'matching':'longest non-overlapping token patterns; explicit variants; no fuzzy match'}
    summary['inputs']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [ROOT/'book_data/all_books.csv',ROOT/'movie_data/all_movies.csv',
                                 ROOT/'config/spells.json',ROOT/'config/spell_reviews.csv']}
    summary['review_source_hashes']={f'book_data/Book{i}.txt':hashlib.sha256((ROOT/f'book_data/Book{i}.txt').read_bytes()).hexdigest() for i in (4,5,7)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    books,films=summary['books'],summary['movies']
    contexts='; '.join(f"{r['spell']} / {r['word']}: PPMI {r['ppmi']:.2f}, {r['pair_count']} pairs across {r['supporting_lines']} lines" for r in data['contexts']) or 'No pairs pass the three-distinct-line reporting threshold.'
    text=f'''# Spell analysis: books versus films

[Open the spell explorer and figures](index.html)

## What is covered

All supplied dialogue from seven books and eight films. No character word threshold is imposed: a one-line spell still counts. Unknown speakers remain included. The two final films form installment seven. The dictionary contains {summary['dictionary_incantations']} curated incantations, explicit spell-family groupings, separate named references and spelling variants. This is a study of spell language in the supplied dialogue, not an exhaustive count of magic performed in the stories.

## Results you can present

- Book dialogue contains {books['incantation_mentions']} full incantation mentions after documented OCR variants, versus {films['incantation_mentions']} in films. Normalizing by all dialogue words gives **{books['per_10000']:.2f} versus {films['per_10000']:.2f} per 10,000 words**: films are **{summary['film_over_book_rate_ratio']:.2f} times as dense** under this dictionary.
- Strict spellings give {books['strict_mentions']} book mentions and {films['strict_mentions']} film mentions. The corresponding rate ratio is {summary['strict_film_over_book_rate_ratio']:.2f}; the direction of the density result survives including the documented spelling variants. The precise magnitude depends strongly on spelling treatment. Films also contain the documented Avada Kadavra transcription variant, included in the expanded count.
- The observed inventory contains {books['detected_incantations']} book incantation types and {films['detected_incantations']} film types. Do not infer lower intrinsic diversity from these raw type counts: the supplied book dialogue is much larger.
- Harry and McGonagall use Unforgivable Curses in checked book passages. For example, the book identifies Harry casting Imperio on the Gringotts goblin and McGonagall casting Imperio on Amycus. Both lines are incorrectly labeled Ted Tonks in the supplied dialogue. Bellatrix casting Crucio on Neville is labeled Neville. These cases establish that a speaker-to-spell count is not automatically a caster-to-spell count.
- A spell name can be a recollection: Ron describing Moody's Avada Kedavra is not Ron casting it. Manual reviews distinguish selected discussion/reported-casting cases from attempts indicated by surrounding text. Attempt does not assert success.
- Spell/word PPMI is sparse because many incantations occupy whole dialogue lines. With a ±5-token window confined to a line and at least three distinct supporting lines, the reported pairs are: {contexts}. No unsupported context graph or topic model is inferred from the sparse spell subset.

## Figures

- 01_spell_frequencies.png/.svg: the 20 most frequent spell families across the whole corpus, normalized by dialogue length; strict book counts are shown as a sensitivity check.
- 02_spell_timeline.png/.svg: 18 frequent families across all seven story installments; marker area encodes normalized frequency. Genuine zero cells are retained in the CSV.
- 03_spell_context_cases.png/.svg: four checked interpretation examples, including a recollection and misattributed casting attempts. These are illustrative cases, not a casting census.

## Matching and context

Dictionary matching uses the existing English-letter tokenizer, Unicode/apostrophe normalization and longest non-overlapping token sequences. Lumos Maxima is not counted a second time as Lumos; it is later grouped into the Lumos family for visualization. Finite Incantatem, Protego variants, Oculus Reparo and Bombarda Maxima are handled similarly. Names such as Killing Curse, Cruciatus and Imperius are exported as named references and excluded from primary incantation counts.

Explicit OCR variants such as Stupefyl and ExpelliarmusV contribute only to the variant-inclusive book count. The observed film transcription Avada Kadavra is explicitly mapped to Avada Kedavra in the expanded film count. Their original forms, lines and match status are retained. These are exploratory text repairs; they do not fix speaker attribution. No global fuzzy matching or automatic correction of the ordinary word crucial is used. Deliberate movie pronunciation variants, partial Expecto/Avada starts and other unmatched candidates are separately exported and excluded from primary counts. Pack and Point Me are omitted because ordinary language can collide with them. Unspoken spells, stage action absent from the dialogue, truncations and names outside this curated inventory can be missed. Absence here never proves a spell is absent from a book or film.

The shape label incantation_only means that the line consists entirely of recognized spell forms; it is not an automatic casting label. embedded_in_dialogue also does not mean discussion: a character can utter a spell and continue speaking. Review labels apply only to selected records explicitly listed in config/spell_reviews.csv. Book reviews reference surrounding original narration. Film reviews are transcript-based inferences; video was not independently reviewed. All remaining mentions are unreviewed. Context/actor reviews are annotations for this analysis, and do not modify original inputs or other project outputs.

KWIC left/right snippets use up to eight neighboring word tokens in the same line. The explorer additionally shows two preceding and two following dialogue records within the same source/chapter, with supplied speaker labels. These are reading aids, not confirmed casting targets or interaction edges.

Spell/context PPMI uses a rectangular incantation-by-context-word matrix pooled across the two media. Whole spell spans and partial repetitions of the target's own components are excluded from context. A window extends five original token positions on either side of the whole phrase, without crossing a dialogue line. A repeated context token at a different position counts again. Context-word eligibility requires at least three occurrences across incantation-bearing records (each record counted once). PPMI uses complete row/column marginals; the three-distinct-line and English-stopword filters apply only to the displayed report. This spell-conditioned matrix is a separate representation from the earlier ordinary word/word PPMI. It is not used to claim a medium-specific context difference.

## Networks relevance

character_spell_network.graphml is a bipartite graph: one node set is canonical supplied speaker labels and the other is spell families. An edge means that a speaker's attributed dialogue references an incantation. Edge attributes retain book and film counts; weight is their sum. This is not a confirmed casting network and has no inferred targets. Unknown and misattributed labels remain visible. network_edges.csv and character_spell_mentions.csv expose these data; per-character rates use all their dialogue and are descriptive, without matched-installment control.

## Files and reproducibility

spell_mentions.csv preserves every match and its original text, source, parsed-record index, speaker label, form, detection status, contexts and any review. spell_totals.csv and spell_by_installment.csv include counts, strict counts, denominators and normalized rates. source_coverage.csv exposes all seven stories' word budgets. spelling_variant_evidence.csv and partial_or_pronunciation_candidates.csv expose detection choices. manual_reviews.csv contains the reviewed matches. spell_context_counts.npz and spell_context_ppmi.npz use context_spell_index.csv and context_word_index.csv; spell_contexts.csv contains the supported report. summary.json records input hashes, settings and findings.

Run `.venv\\Scripts\\python.exe -m code.spell_analysis` with requirements-map.txt. Earlier source files, analysis outputs and interactive maps are preserved.

The spell vocabulary is curated against the supplied text and the [official spell guide](https://www.harrypotter.com/features/your-guide-to-the-best-spells-in-harry-potter). The three Unforgivable Curses follow the [official encyclopedia](https://www.harrypotter.com/fact-file/spells/the-unforgivable-curses). Exact incantations and source references are in config/spells.json.
'''
    (out/'README.md').write_text(text,encoding='utf-8')


def render_all(data,out):
    setup()
    frequency_plot(data,out)
    timeline_plot(data,out)
    context_plot(data,out)
    explorer(data,out)
