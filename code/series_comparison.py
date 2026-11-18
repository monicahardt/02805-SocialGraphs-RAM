"""Series-wide, source-matched NLP comparisons and a transparent polarity map.

Run: python -m code.series_comparison
Earlier method outputs and original dialogue are read only. Sentiment measures
word-list matches, not a speaker's intent, morality or psychological state.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .dialogue_csv_reader import read_dialogue_records
from .ppmi import build_vocabulary, cooccurrence, positive_pmi
from .vocabulary_map import (ALIASES, COLORS, ROOT, TITLES, installment,
                             matched_profiles, read_csv, vocabulary_vectors, write_csv)
from .word_counts import tokenize

BOOK, FILM, INK, PAPER, MUTED = '#287c83', '#bf6846', '#243e3b', '#f7f4ec', '#66796f'
SHORT = ['Stone', 'Chamber', 'Azkaban', 'Goblet', 'Phoenix', 'Half-Blood Prince', 'Hallows']
TOPIC_NAMES = ['Everyday exchanges', 'The Dark Lord & authority', 'Conversation & reaction',
               'Sirius & school', 'Dobby & Crouch', 'Wands & the Hallows',
               'Dumbledore & the Ministry', 'Hagrid’s written dialect']


def load_lexicon(directory):
    """Read the published lists unchanged. No role labels inform these scores."""
    entries = defaultdict(set)
    for polarity in ('positive', 'negative'):
        text = (directory / f'{polarity}-words.txt').read_text(encoding='utf-8')
        for line in text.splitlines():
            word = line.strip().lower()
            if not word or word.startswith(';'):
                continue
            entries[word].add(polarity)
    # The published lists overlap on "envious"; do not resolve it arbitrarily.
    return {word: next(iter(values)) for word, values in entries.items() if len(values) == 1}


def polarity_counts(counts, lexicon):
    """Count tokens, normalizing both densities by ALL dialogue words."""
    words = sum(counts.values())
    if words <= 0:
        raise ValueError('A polarity profile needs positive token mass.')
    if any(value < 0 for value in counts.values()):
        raise ValueError('Counts cannot be negative.')
    positive = sum(n for word, n in counts.items() if lexicon.get(word) == 'positive')
    negative = sum(n for word, n in counts.items() if lexicon.get(word) == 'negative')
    return {'words': words, 'positive': positive, 'negative': negative,
            'positive_per_1000': 1000 * positive / words,
            'negative_per_1000': 1000 * negative / words,
            'balance_per_1000': 1000 * (positive - negative) / words,
            'lexicon_coverage': (positive + negative) / words}


def paired_topics(rows):
    """Match canonical speakers per installment; combine the two final films.

    Average mixtures by modeled token count, retaining their original unit and
    original shared topic IDs. Only common speaker/installment groups contribute.
    """
    groups = {}
    for row in rows:
        key = (ALIASES.get(row['character'], row['character']), installment(row['source']), row['medium'])
        entry = groups.setdefault(key, {'tokens': 0, 'mass': np.zeros(8), 'documents': 0})
        tokens = int(row['modeled_words'])
        entry['tokens'] += tokens
        entry['documents'] += 1
        entry['mass'] += tokens * np.array([float(row[f'topic_{i}']) for i in range(1, 9)])
    common = {(name, number) for name, number, medium in groups
              if (name, number, 'books') in groups and (name, number, 'movies') in groups}
    output = []
    for name, number in sorted(common):
        for medium in ('books', 'movies'):
            entry = groups[name, number, medium]
            output.append({'character': name, 'installment': number, 'medium': medium,
                           'modeled_words': entry['tokens'], 'original_documents': entry['documents'],
                           **{f'topic_{i+1}': float(value / entry['tokens']) for i, value in enumerate(entry['mass'])}})
    return output


def aggregate_topics(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key], row['medium']].append(row)
    return [{key: value, 'medium': medium, 'modeled_words': sum(r['modeled_words'] for r in items),
             'matched_groups': len(items),
             **{f'topic_{i}': sum(r['modeled_words'] * r[f'topic_{i}'] for r in items)
                / sum(r['modeled_words'] for r in items) for i in range(1, 9)}}
            for (value, medium), items in sorted(groups.items())]


def split_contexts(sequences, vocabulary, window=5):
    """Use a shared vocabulary and separate, complete medium-specific marginals."""
    counts = {medium: cooccurrence(lines, vocabulary, window) for medium, lines in sequences.items()}
    return counts, {medium: positive_pmi(matrix) for medium, matrix in counts.items()}


def collect_data(out, lexicon):
    paths = {medium: ROOT / f'results/word_counts/{medium}/character_word_counts.csv'
             for medium in ('books', 'movies')}
    profiles, excluded = matched_profiles(paths, 100)
    roles = {r['character']: r['role'] for r in read_csv(ROOT / 'config/map_roles.csv')}
    vectors, cosine, vocabulary, _ = vocabulary_vectors(profiles)
    polarity, contributions, pair_rows = [], [], []
    for index, profile in enumerate(profiles):
        score = polarity_counts(profile['counts'], lexicon)
        row = {'id': index, 'character': profile['character'], 'medium': profile['medium'],
               'role': roles.get(profile['character'], 'unassigned'), 'installments': profile['installments'],
               'low_word_count': score['words'] < 300, **score}
        polarity.append(row)
        for word, count in sorted(profile['counts'].items(), key=lambda item: (-item[1], item[0])):
            if word in lexicon:
                contributions.append({'id': index, 'character': profile['character'], 'medium': profile['medium'],
                                      'word': word, 'polarity': lexicon[word], 'count': count,
                                      'per_1000': 1000 * count / score['words']})
        if index % 2 == 0:
            b, f = vectors.getrow(index), vectors.getrow(index + 1)
            words = []
            for vector in (b, f):
                order = np.lexsort((vector.indices, -vector.data))[:4]
                words.append(', '.join(str(vocabulary[j]) for j in vector.indices[order]))
            pair_rows.append({'character': profile['character'], 'role': row['role'],
                              'installments': profile['installments'], 'book_words': profiles[index]['words'],
                              'film_words': profiles[index+1]['words'], 'cosine': float(cosine[index, index+1]),
                              'book_fingerprint': words[0], 'film_fingerprint': words[1]})
    write_csv(out / 'character_polarity.csv', polarity)
    write_csv(out / 'word_contributions.csv', contributions)
    write_csv(out / 'character_vocabulary.csv', pair_rows)
    write_csv(out / 'excluded_characters.csv', excluded)
    write_csv(out / 'sentiment_lexicon.csv', [{'word': w, 'polarity': p} for w, p in sorted(lexicon.items())])

    eligible = {(r['medium'], r['source'], r['character'])
                for r in read_csv(ROOT / 'results/ppmi/documents.csv')}
    sequences, volume, evidence = defaultdict(list), defaultdict(Counter), []
    profile_lookup = {(p['character'], p['medium']): i for i, p in enumerate(profiles)}
    examples = defaultdict(lambda: defaultdict(list))
    for medium, path in (('books', ROOT / 'book_data/all_books.csv'), ('movies', ROOT / 'movie_data/all_movies.csv')):
        for record_index, record in enumerate(read_dialogue_records(path), 1):
            tokens = tokenize(record.line)
            number = installment(record.book)
            volume[number, medium].update(tokens)
            if (medium, record.book, record.character) in eligible:
                sequences[medium].append(tokens)
            name = ALIASES.get(record.character, record.character)
            profile_id = profile_lookup.get((name, medium))
            if profile_id is None or str(number) not in profiles[profile_id]['installments'].split():
                continue
            hits = Counter(word for word in tokens if word in lexicon)
            if not hits:
                continue
            entry = {'id': profile_id, 'character': name, 'medium': medium, 'source': record.book,
                     'chapter': record.chapter, 'record_index': record_index,
                     'positive': sum(n for w, n in hits.items() if lexicon[w] == 'positive'),
                     'negative': sum(n for w, n in hits.items() if lexicon[w] == 'negative'),
                     'matched_words': ', '.join(f'{w}:{n}' for w, n in sorted(hits.items())), 'text': record.line}
            evidence.append(entry)
            for word in hits:
                if len(examples[profile_id][word]) < 3:
                    examples[profile_id][word].append({'source': record.book, 'chapter': record.chapter,
                                                      'record_index': record_index, 'text': record.line})
    volume_rows = [{'installment': n, 'title': TITLES[n-1], 'medium': m,
                    'words': volume[n, m].total(), 'unique_words': len(volume[n, m])}
                   for n in range(1, 8) for m in ('books', 'movies')]
    for medium in ('books','movies'):
        original=Counter({r['word']:int(r['count']) for r in read_csv(ROOT/f'results/word_counts/{medium}/word_counts.csv')})
        assert sum((volume[i,medium] for i in range(1,8)),Counter()) == original
    write_csv(out / 'series_word_counts.csv', volume_rows)
    write_csv(out / 'sentiment_evidence.csv', evidence)

    topic_pairs = paired_topics(read_csv(ROOT / 'results/lda/document_topics.csv'))
    topics = aggregate_topics(topic_pairs, 'installment')
    character_topics = aggregate_topics(topic_pairs, 'character')
    write_csv(out / 'matched_topic_documents.csv', topic_pairs)
    write_csv(out / 'series_topics.csv', topics)
    write_csv(out / 'character_topics.csv', character_topics)
    context_vocabulary, frequencies = build_vocabulary(sequences['books'] + sequences['movies'], 10, 5000)
    counts, ppmi = split_contexts(sequences, context_vocabulary)
    pooled_counts = counts['books'] + counts['movies']
    pooled_ppmi = positive_pmi(pooled_counts)
    context_rows = []
    for target in ('blood', 'power', 'death', 'love'):
        first = context_vocabulary[target]
        candidates = [column for word, column in context_vocabulary.items()
                      if column != first and len(word) >= 3 and word.isalpha() and word not in ENGLISH_STOP_WORDS
                      and pooled_counts[first, column] >= 3
                      and pooled_ppmi[first, column] > 0]
        candidates.sort(key=lambda column: (-float(pooled_ppmi[first, column]), column))
        inverse = {index: word for word, index in context_vocabulary.items()}
        for rank, second in enumerate(candidates[:8], 1):
            context_rows.append({'word': target, 'context': inverse[second], 'rank': rank,
                                 **{f'{m}_count': int(counts[m][first, second]) for m in ('books', 'movies')},
                                 **{f'{m}_ppmi': float(ppmi[m][first, second]) for m in ('books', 'movies')}})
    for medium in ('books', 'movies'):
        sparse.save_npz(out / f'context_counts_{medium}.npz', counts[medium])
        sparse.save_npz(out / f'ppmi_{medium}.npz', ppmi[medium])
    # Exact reconstruction verifies the split is equivalent to the original pooled run.
    previous_counts = sparse.load_npz(ROOT / 'results/ppmi/counts.npz')
    previous_ppmi = sparse.load_npz(ROOT / 'results/ppmi/ppmi.npz')
    assert {r['word']:int(r['index']) for r in read_csv(ROOT/'results/ppmi/vocabulary.csv')} == context_vocabulary
    assert (pooled_counts != previous_counts).nnz == 0
    difference = pooled_ppmi - previous_ppmi
    assert not difference.nnz or np.max(np.abs(difference.data)) < 1e-12
    write_csv(out / 'context_vocabulary.csv', [{'index': i, 'word': w, 'occurrences': frequencies[w]}
                                             for w, i in sorted(context_vocabulary.items(), key=lambda item: item[1])])
    write_csv(out / 'context_comparison.csv', context_rows)
    return {'profiles': profiles, 'polarity': polarity, 'contributions': contributions, 'examples': examples,
            'vocabulary': pair_rows, 'volume': volume_rows, 'topics': topics, 'character_topics': character_topics,
            'contexts': context_rows, 'roles': roles, 'evidence': evidence,
            'context_vocabulary_size':len(context_vocabulary), 'context_documents':len(eligible)}


def write_report(data, lexicon, out):
    pairs = []
    for b, f in zip(data['polarity'][::2], data['polarity'][1::2]):
        pairs.append({'character':b['character'],'role':b['role'],'book_words':b['words'],'film_words':f['words'],
                      'book_balance':b['balance_per_1000'],'film_balance':f['balance_per_1000'],
                      'change':f['balance_per_1000']-b['balance_per_1000']})
    write_csv(out/'polarity_changes.csv',pairs)
    robust=[r for r in pairs if min(r['book_words'],r['film_words'])>=300]
    means={role:{medium:float(np.mean([r['balance_per_1000'] for r in data['polarity']
                                     if r['role']==role and r['medium']==medium]))
                 for medium in ('books','movies')} for role in ('hero','villain')}
    retained=sorted([r for r in data['vocabulary'] if min(r['book_words'],r['film_words'])>=300],key=lambda r:r['cosine'])
    topics={(r['installment'],r['medium']):r for r in data['topics']}
    topic_changes=sorted([{'installment':i,'topic':t,
                           'book_proportion':topics[i,'books'][f'topic_{t}'],
                           'film_proportion':topics[i,'movies'][f'topic_{t}'],
                           'difference':topics[i,'movies'][f'topic_{t}']-topics[i,'books'][f'topic_{t}']}
                          for i in range(1,8) for t in range(1,9)],key=lambda r:-abs(r['difference']))
    lexicon_path=ROOT/'config/opinion_lexicon'
    inputs=[ROOT/'book_data/all_books.csv', ROOT/'movie_data/all_movies.csv',
            ROOT/'results/lda/document_topics.csv', ROOT/'config/map_roles.csv',
            *[ROOT/f'results/word_counts/{m}/character_word_counts.csv' for m in ('books','movies')],
            lexicon_path/'positive-words.txt',lexicon_path/'negative-words.txt']
    summary={'matched_characters':len(pairs),'matched_lda_characters':len(data['character_topics'])//2,
             'installments':7,'films':8,'lexicon_words':dict(Counter(lexicon.values())),
             'excluded_overlapping_entry':['envious'],'minimum_profile_words':100,'small_profile_flag_below':300,
             'sentiment_formula':'1,000 * exact polarity token matches / all profile word tokens',
             'sentiment_role_mean_balance_equal_character_weight':means,
             'sentiment_changes_300_words_each':sorted(robust,key=lambda r:-abs(r['change']))[:8],
             'lowest_cosine_300_words_each':retained[:5],'highest_cosine_300_words_each':retained[-5:][::-1],
             'largest_lda_shifts':topic_changes[:8],
             'ppmi':{'vocabulary_words':data['context_vocabulary_size'],'window':5,'marginals':'separate by medium',
                     'eligible_documents':data['context_documents'],
                     'eligibility':'same character/source groups as the previous pooled run',
                     'display':'top 8 pooled-PPMI contexts per target with >=3 total pairs; alphabetic non-stopwords only',
                     'pooled_reconstruction_verified':True},
             'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # A reproducible qualitative audit set, not a held-out accuracy evaluation.
    rng=np.random.default_rng(42)
    sample=[data['evidence'][int(i)] for i in rng.choice(len(data['evidence']),size=20,replace=False)]
    write_csv(out/'sentiment_audit_sample.csv',sample)
    shift=topic_changes[0]
    shifts='\n'.join(f"- {r['character']}: {r['book_balance']:.1f} → {r['film_balance']:.1f} matches per 1,000 (change {r['change']:+.1f}; {r['book_words']:,}/{r['film_words']:,} book/film words)." for r in summary['sentiment_changes_300_words_each'][:4])
    topic2_higher=sum(topics[i,'movies']['topic_2']>topics[i,'books']['topic_2'] for i in range(1,8))
    report=f'''# From page to screen: the series-wide figures

[Open the figure collection](index.html) · [Interactive character polarity map](character_word_map.html) · [Existing vocabulary network](../vocabulary_map/vocabulary_map.html)

These figures replace the earlier selected examples as the main project overview. All seven book installments and all eight films contribute. Umbridge remains an optional case study in ../comparison/umbridge_phoenix.

## What each figure covers

| Figure | Scope | Underlying data |
|---|---|---|
| 01_dialogue_volume | Every dialogue row, including unknown speakers; final films grouped into installment 7 | series_word_counts.csv |
| 02_topic_journeys | All 7 stories, all 8 original shared LDA topics, matched eligible speakers within each story | series_topics.csv; matched_topic_documents.csv |
| 02b_all_character_topics | All {len(data['character_topics'])//2} characters with common eligible LDA installments | character_topics.csv |
| 03_vocabulary_retention | All {len(pairs)} matched character pairs, including every eligible villain | character_vocabulary.csv |
| 04_word_contexts | Series-wide book/film PPMI for blood, power, death and love | context_comparison.csv; context_vocabulary.csv; context_counts_*.npz; ppmi_*.npz |
| 05_character_word_map | All {len(pairs)} matched character pairs, positive/negative word densities | character_polarity.csv; polarity_changes.csv; word_contributions.csv |

Every static figure is saved as PNG and editable SVG. The HTML map embeds Plotly and works locally without a server. Read the figures as exploratory comparisons of the supplied dialogue, with source attribution and transcript limitations.

## Concrete results

- Across the supplied dialogue, books contain {sum(r['words'] for r in data['volume'] if r['medium']=='books'):,} word tokens and films contain {sum(r['words'] for r in data['volume'] if r['medium']=='movies'):,}. The volume plot exposes this compression for every story.
- Among characters with at least 300 words in each medium, the largest vocabulary differences include {', '.join(r['character']+' ('+format(r['cosine'],'.3f')+')' for r in retained[:3])}. The closest vocabularies include {', '.join(r['character']+' ('+format(r['cosine'],'.3f')+')' for r in retained[-3:][::-1])}. These are full TF-IDF cosine scores, not morality or personality similarity.
- The largest matched story-level LDA shift in this fitted model is topic {shift['topic']} in installment {shift['installment']}: {shift['book_proportion']:.1%} in books and {shift['film_proportion']:.1%} in films. Names and scenes contribute to topics. This is one existing eight-topic fit, without a topic-stability claim.
- Topic 2 (lord, severus, dark, draco, professor) has a higher film proportion in {topic2_higher} of 7 matched story comparisons. It includes names and references to authority; this is not a direct measure of villains being more negative.
- Positive/negative word counts do not establish a clean hero/villain boundary. For the {sum(r['role']=='hero' for r in pairs)} heroes and {sum(r['role']=='villain' for r in pairs)} villains, equal-character mean positive-minus-negative densities are: books {means['hero']['books']:.2f} vs {means['villain']['books']:.2f}; films {means['hero']['movies']:.2f} vs {means['villain']['movies']:.2f} per 1,000 words. These are descriptive group means for this selected set, not an accuracy score or significance test.

Largest polarity-density changes among profiles with at least 300 words in both media:

{shifts}

Positive balance means more positive than negative dictionary matches under this particular instrument. The unequal list sizes and domain limitations prevent treating zero as a calibrated neutral point or the balance as a morality score.

## Source matching and exclusions

Character TF-IDF and polarity use the existing vocabulary-network matching rule: pool each character only over installments with dialogue in both media; require >=100 total raw words in each medium. The two final films form installment seven. The map includes {len(pairs)} characters; excluded characters and word budgets are in excluded_characters.csv. Small markers flag fewer than 300 words; robust examples above require at least 300 each. Coverage is all eligible characters across the series, not every named person or every scene.

The existing alias table is reused locally, including Tom Riddle → Voldemort. Original speaker labels remain unchanged. Moody impersonation and Crouch identity ambiguity remain provisional. Matching installments does not imply matching scenes or equal word budgets. Longer installments contribute more to pooled character profiles.

LDA reuses original document-topic probabilities rather than training a new model. Each original document needed >=100 raw words. Canonical character/installment groups must be present in both media. Topic proportions are aggregated with modeled-token weights, and each medium retains its own word budget. Matched speaker counts and all contributing documents are exported. Sparse voices can have a pooled vocabulary profile but no matched eligible LDA document, explaining {len(pairs)} versus {len(data['character_topics'])//2} characters.

TF-IDF uses the same pooled-profile vectors as the current network: English stopwords and short tokens removed; min_df=2; TF=count/retained document length, IDF=ln(N/df); cosine normalizes vector length. Word-count and polarity rates instead use full raw dialogue word totals.

## PPMI: now a real medium comparison

The original PPMI eligibility and shared 2,506-word vocabulary are retained. Book and film co-occurrence counts are computed separately, with separate marginal totals and PPMI calculations. Their raw matrices add exactly to the original pooled matrix; recomputed pooled PPMI agrees to numerical precision. The previous pooled files are preserved.

The four target words are the previous thematic probes, not the entire vocabulary. Full matrices for every retained word are saved. The figure selects up to 8 contexts per target by pooled PPMI, requiring >=3 overall pair observations. Only alphabetic words of at least three letters outside the English stopword list are displayed. These display filters never change the matrices. Pairs with <3 observations in one medium are pale, including zeros. Scores use complete within-medium marginals, with no smoothing or support pruning. Zero PPMI means no positive association under this instrument; it is not proof of no semantic relationship. Media differ in text amount, characters and scenes, so score differences do not isolate a causal adaptation effect.

## Positive/negative word categorization

The [Hu–Liu opinion lexicon](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html), distributed by [NLTK](https://www.nltk.org/howto/corpus.html), supplies word categories. Cite: Minqing Hu and Bing Liu, *Mining and Summarizing Customer Reviews*, KDD 2004. Original files, headers, CC BY 4.0 notice and provenance are in config/opinion_lexicon. The entry envious appears in both lists and is omitted; the resulting dictionary has {len(lexicon):,} unique entries. No Harry Potter role labels enter the word categories, coordinates or TF-IDF.

Coordinates are exact token counts per 1,000 full raw dialogue words. Matching uses the project tokenizer, with lowercase/apostrophe normalization; no stemming or negation reversal. Unmatched words have no polarity label. The whole dictionary and every observed word contribution are exported. A term such as well can be a discourse filler; a hero can describe a murder; a villain can say good. The method does not measure intent or morality. It does not resolve negation, sarcasm, mixed sentiment or fantasy senses. The interactive view exposes up to three original lines for each displayed term; sentiment_evidence.csv contains all matching dialogue lines, and sentiment_audit_sample.csv is a reproducible 20-line audit sample, not a held-out accuracy evaluation.

The map is a transparent descriptive baseline inspired by [course week 7: sentiment as a measurement](https://sunelehmann.com/socialgraphs2026-web/weeks/week7.html). The original four methods follow [course week 6](https://sunelehmann.com/socialgraphs2026-web/weeks/week6.html). No sentiment-classification performance claim is made for this domain.

Read [the qualitative measurement audit](MEASUREMENT_AUDIT.md) for observed dictionary/context problems and a confirmed speaker-attribution error. These limit interpretation even when the arithmetic is correct.

## Reproduce

Run `.venv\\Scripts\\python.exe -m code.series_comparison`. Uses requirements-map.txt and the saved original method outputs. summary.json records results, thresholds and input hashes. Original datasets, earlier method outputs and the existing interactive network are preserved.
'''
    (out/'README.md').write_text(report,encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/series_comparison')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lexicon = load_lexicon(ROOT / 'config/opinion_lexicon')
    data = collect_data(args.output_dir, lexicon)
    from .series_figures import render_all
    render_all(data, args.output_dir)
    write_report(data, lexicon, args.output_dir)
    print(f"Built series-wide comparisons for {len(data['profiles'])//2} matched characters: {args.output_dir}")


if __name__ == '__main__':
    main()
