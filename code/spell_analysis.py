"""Series-wide spell references, normalized counts, concordances and PPMI.

Run: python -m code.spell_analysis
Spell names in dialogue are not automatically evidence of spell casting.
Original data and previous analysis outputs are read only.
"""

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .dialogue_csv_reader import read_dialogue_records
from .ppmi import positive_pmi
from .vocabulary_map import ALIASES, ROOT, TITLES, installment, read_csv, write_csv
from .word_counts import WORD_RE, tokenize


def load_patterns(path):
    config=json.loads(path.read_text(encoding='utf-8'))
    entries=[]
    def add(surface, spell, kind, status, medium='both'):
        family=config['families'].get(spell,spell)
        entries.append({'surface':surface,'spell':spell,'family':family,'kind':kind,'status':status,
                        'medium':medium,'unforgivable':spell in config['unforgivable'],
                        'tokens':tuple(tokenize(surface))})
    for spell in config['incantations']:
        add(spell,spell,'incantation','exact')
    for surface,spell in config['references'].items():
        add(surface,spell,'name','exact')
    for spell,variants in config['ocr_variants'].items():
        for variant in variants:
            add(variant,spell,'incantation','ocr_variant','books')
    for spell,variants in config['reference_variants'].items():
        for variant in variants:
            add(variant,spell,'name','transcript_variant')
    for spell,variants in config.get('incantation_transcript_variants',{}).items():
        for variant in variants:
            add(variant,spell,'incantation','transcript_variant','movies')
    for spell,variants in config['pronunciation_variants'].items():
        for variant in variants:
            add(variant,spell,'pronunciation','pronunciation_variant','movies')
    lookup=defaultdict(list)
    seen=set()
    for entry in entries:
        key=(entry['tokens'],entry['medium'])
        if key in seen:
            raise ValueError(f'Duplicate spell pattern: {entry["surface"]}')
        seen.add(key)
        lookup[entry['tokens'][0]].append(entry)
    for values in lookup.values():
        values.sort(key=lambda row:(-len(row['tokens']),row['status']!='exact',row['surface']))
    return config,lookup


def match_spells(text,medium,lookup):
    """Longest, non-overlapping token matches; preserve original normalized spans."""
    normalized=unicodedata.normalize('NFKC',text).replace('\u2019',"'").replace('\u2018',"'")
    tokens=[match.group() for match in WORD_RE.finditer(normalized.lower())]
    spans=[match.span() for match in WORD_RE.finditer(normalized.lower())]
    matches=[]
    start=0
    while start<len(tokens):
        candidate=None
        for entry in lookup.get(tokens[start],[]):
            stop=start+len(entry['tokens'])
            if entry['medium'] in ('both',medium) and tuple(tokens[start:stop])==entry['tokens']:
                candidate=entry;break
        if candidate is None:
            start+=1;continue
        stop=start+len(candidate['tokens'])
        matches.append({**candidate,'start':start,'stop':stop,'matched_text':normalized[spans[start][0]:spans[stop-1][1]]})
        start=stop
    return tokens,matches


def spell_context_matrix(events,window=5,min_word_count=3):
    """Count spell/word pairs within the SAME dialogue line, around the whole phrase.

    All detected spell spans are masked out of context words. Each incantation
    mention contributes at most once per neighboring token position. Repeated
    words at different positions contribute repeatedly. English stopwords remain
    in the calculation; display filtering is separate.
    """
    eligible=[event for event in events if event['kind']=='incantation']
    spells=sorted({event['spell'] for event in eligible})
    unique_records={(event['medium'],event['record_index']):event['tokens'] for event in eligible}
    occurrences=Counter(word for tokens in unique_records.values() for word in tokens)
    words=sorted(word for word,count in occurrences.items() if count>=min_word_count)
    si,wi={s:i for i,s in enumerate(spells)},{w:i for i,w in enumerate(words)}
    pairs=Counter()
    supports=defaultdict(set)
    for event in eligible:
        own_components=set(tokenize(event['spell']))
        indices=list(range(max(0,event['start']-window),event['start']))+list(range(event['stop'],min(len(event['tokens']),event['stop']+window)))
        for i in indices:
            word=event['tokens'][i]
            if i not in event['masked'] and word in wi and word not in own_components:
                pairs[si[event['spell']],wi[word]]+=1
                supports[event['spell'],word].add((event['medium'],event['record_index']))
    if pairs:
        indices=list(pairs)
        matrix=sparse.csr_matrix(([pairs[pair] for pair in indices],
                                 ([p[0] for p in indices],[p[1] for p in indices])),shape=(len(si),len(wi)),dtype=np.int64)
    else:
        matrix=sparse.csr_matrix((len(si),len(wi)),dtype=np.int64)
    return matrix,spells,words,supports


def primary(event,strict=False):
    return event['kind']=='incantation' and (not strict or event['status']=='exact')


def build_analysis(out,config_path,reviews_path):
    config,lookup=load_patterns(config_path)
    roles={r['character']:r['role'] for r in read_csv(ROOT/'config/map_roles.csv')}
    reviews={(r['medium'],int(r['record_index']),r['spell']):r for r in read_csv(reviews_path)}
    events,partial=[],[]
    budgets=defaultdict(Counter)
    character_words=Counter()
    for medium,path in (('books',ROOT/'book_data/all_books.csv'),('movies',ROOT/'movie_data/all_movies.csv')):
        records=read_dialogue_records(path)
        for record_index,record in enumerate(records,1):
            tokens,matches=match_spells(record.line,medium,lookup)
            story=installment(record.book)
            character=ALIASES.get(record.character,record.character)
            budgets[medium,story].update(tokens)
            character_words[medium,story,character]+=len(tokens)
            masked={i for hit in matches for i in range(hit['start'],hit['stop'])}
            spell_only=bool(matches) and all(i in masked for i in range(len(tokens)))
            for number,hit in enumerate(matches,1):
                review=reviews.get((medium,record_index,hit['spell']),{})
                event={**hit,'id':f'{medium}:{record_index}:{number}', 'medium':medium,'source':record.book,
                       'installment':story,'chapter':record.chapter,'record_index':record_index,
                       'source_speaker':record.character,'character':character,'role':roles.get(character,'unassigned'),
                       'utterance_shape':'incantation_only' if spell_only else 'embedded_in_dialogue',
                       'reviewed_context':review.get('context','unreviewed'),
                       'reviewed_speaker':review.get('speaker',''),'review_evidence':review.get('evidence',''),
                       'left_context':' '.join(tokens[max(0,hit['start']-8):hit['start']]),
                       'right_context':' '.join(tokens[hit['stop']:hit['stop']+8]),
                       'text':record.line,'tokens':tokens,'masked':masked,
                       'neighbors':[{'record_index':i+1,'speaker':records[i].character,'text':records[i].line}
                                    for i in range(max(0,record_index-3),min(len(records),record_index+2))
                                    if i!=record_index-1 and records[i].book==record.book and records[i].chapter==record.chapter]}
                events.append(event)
            for i,token in enumerate(tokens):
                if i not in masked and token in ('avada','expecto','wingardium','wingardrium','leviosar','leviosaaaar'):
                    partial.append({'medium':medium,'source':record.book,'record_index':record_index,
                                    'source_speaker':record.character,'token':token,'text':record.line})
    fields=['id','medium','source','installment','chapter','record_index','source_speaker','character','role',
            'spell','family','kind','status','unforgivable','surface','matched_text','start','stop',
            'utterance_shape','reviewed_context','reviewed_speaker','review_evidence','left_context','right_context','text']
    public=[{field:event[field] for field in fields} for event in events]
    write_csv(out/'spell_mentions.csv',public)
    write_csv(out/'partial_or_pronunciation_candidates.csv',partial)
    write_csv(out/'spelling_variant_evidence.csv',[row for row in public if row['status']!='exact'])
    write_csv(out/'manual_reviews.csv',[row for row in public if row['reviewed_context']!='unreviewed'])
    # Include every dictionary family, every story and genuine zero cells.
    families=sorted({config['families'].get(s,s) for s in config['incantations']})
    counts=Counter((e['family'],e['medium'],e['installment']) for e in events if primary(e))
    strict_counts=Counter((e['family'],e['medium'],e['installment']) for e in events if primary(e,True))
    series=[]
    for family in families:
        for medium in ('books','movies'):
            for story in range(1,8):
                count,strict=counts[family,medium,story],strict_counts[family,medium,story]
                words=budgets[medium,story].total()
                series.append({'family':family,'medium':medium,'installment':story,'words':words,
                               'strict_count':strict,'count_with_variants':count,'variant_additions':count-strict,
                               'per_10000':10000*count/words,'strict_per_10000':10000*strict/words})
    write_csv(out/'spell_by_installment.csv',series)
    totals=[]
    for family in families:
        for medium in ('books','movies'):
            rows=[r for r in series if r['family']==family and r['medium']==medium]
            words=sum(r['words'] for r in rows)
            count=sum(r['count_with_variants'] for r in rows)
            strict=sum(r['strict_count'] for r in rows)
            totals.append({'family':family,'medium':medium,'words':words,'strict_count':strict,
                           'count_with_variants':count,'variant_additions':count-strict,
                           'per_10000':10000*count/words,'strict_per_10000':10000*strict/words})
    write_csv(out/'spell_totals.csv',totals)
    character_counts=Counter((e['character'],e['family'],e['medium']) for e in events if primary(e))
    character_rows=[]
    graph=nx.Graph(relation='bipartite character/spell incantation references; not confirmed casting')
    for (name,family,medium),count in sorted(character_counts.items()):
        words=sum(n for (m,story,char),n in character_words.items() if m==medium and char==name)
        character_rows.append({'character':name,'family':family,'medium':medium,'count':count,'words':words,
                               'per_10000':10000*count/words,'role':roles.get(name,'unassigned')})
        graph.add_node('character:'+name,bipartite=0,label=name,role=roles.get(name,'unassigned'))
        graph.add_node('spell:'+family,bipartite=1,label=family)
        a,b='character:'+name,'spell:'+family
        if not graph.has_edge(a,b):graph.add_edge(a,b,books=0,movies=0,weight=0)
        graph[a][b][medium]+=count;graph[a][b]['weight']+=count
    write_csv(out/'character_spell_mentions.csv',character_rows)
    nx.write_graphml(graph,out/'character_spell_network.graphml')
    write_csv(out/'network_edges.csv',[{'character':graph.nodes[a]['label'],'spell':graph.nodes[b]['label'],**weights}
                                      if a.startswith('character:') else {'character':graph.nodes[b]['label'],'spell':graph.nodes[a]['label'],**weights}
                                      for a,b,weights in graph.edges(data=True)])
    matrix,spells,words,supports=spell_context_matrix(events)
    scores=positive_pmi(matrix)
    sparse.save_npz(out/'spell_context_counts.npz',matrix)
    sparse.save_npz(out/'spell_context_ppmi.npz',scores)
    write_csv(out/'context_spell_index.csv',[{'row':i,'spell':spell} for i,spell in enumerate(spells)])
    write_csv(out/'context_word_index.csv',[{'column':i,'word':word} for i,word in enumerate(words)])
    context_rows=[]
    for i,spell in enumerate(spells):
        candidates=[j for j in scores.getrow(i).indices if len(words[j])>=3 and words[j] not in ENGLISH_STOP_WORDS
                    and len(supports[spell,words[j]])>=3]
        candidates.sort(key=lambda j:(-float(scores[i,j]),words[j]))
        for rank,j in enumerate(candidates[:8],1):
            context_rows.append({'spell':spell,'rank':rank,'word':words[j],'pair_count':int(matrix[i,j]),
                                 'supporting_lines':len(supports[spell,words[j]]),'ppmi':float(scores[i,j])})
    write_csv(out/'spell_contexts.csv',context_rows)
    coverage=[{'medium':m,'installment':i,'title':TITLES[i-1],'words':budgets[m,i].total(),
               'incantation_mentions':sum(n for (s,med,story),n in counts.items() if med==m and story==i)}
              for i in range(1,8) for m in ('books','movies')]
    write_csv(out/'source_coverage.csv',coverage)
    return {'config':config,'events':events,'public':public,'series':series,'totals':totals,'contexts':context_rows,
            'coverage':coverage,'graph':graph,'review_count':sum(e['reviewed_context']!='unreviewed' for e in events)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/spells')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    data=build_analysis(args.output_dir,ROOT/'config/spells.json',ROOT/'config/spell_reviews.csv')
    from .spell_figures import render_all,write_report
    render_all(data,args.output_dir)
    write_report(data,args.output_dir)
    print(json.dumps({'mentions':len(data['events']),'incantations':sum(primary(e) for e in data['events']),
                      'reviewed':data['review_count'],'network_nodes':len(data['graph'])}))


if __name__=='__main__':main()
