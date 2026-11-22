"""Reproduce the matched Harry Potter dialogue analysis from the project CSVs."""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


MEDIUMS = ('Books', 'Films')
SHORT_TITLES = ('Philosopher’s Stone', 'Chamber of Secrets', 'Prisoner of Azkaban',
                'Goblet of Fire', 'Order of the Phoenix', 'Half-Blood Prince', 'Deathly Hallows')
MIN_WORDS = 100
SHUFFLES = 1999
THINNINGS = 1000
SEED = 2805
STAGE_DIRECTIONS = re.compile(r'\[[^\[\]]*\]')


@dataclass(frozen=True)
class Line:
    speaker: str | None
    words: tuple
    row: int
    text: str = ''


@dataclass(frozen=True)
class Block:
    source: str
    section: str
    lines: tuple


def read_project(repo):
    sys.path.insert(0, str(repo / 'code'))
    from dialogue_csv_reader import read_dialogue_records
    from interaction_graph import NON_SPECIFIC_MOVIE_LABELS
    from vocabulary_map import ALIASES, installment
    from word_counts import tokenize

    blocks, counts, audit = {}, {}, {}
    names = set()
    for medium, relative in zip(MEDIUMS, ('book_data/all_books.csv', 'movie_data/all_movies.csv')):
        records = read_dialogue_records(repo / relative)
        audit[medium] = {'file': relative, 'sha256': hashlib.sha256((repo / relative).read_bytes()).hexdigest(),
                         'rows': len(records), 'unknown_rows': sum(r.character.casefold() == 'unknown' for r in records),
                         'count_by_source': dict(Counter(r.book for r in records)),
                         'bracketed_rows': sum(bool(STAGE_DIRECTIONS.search(r.line)) for r in records)}
        generic = NON_SPECIFIC_MOVIE_LABELS | {'unknown', 'wakanda'}
        for (source, section), group in groupby(enumerate(records, 1), lambda item: (item[1].book, item[1].chapter)):
            story = installment(source)
            key = (story, medium)
            profile = counts.setdefault(key, defaultdict(Counter))
            lines = []
            for row_number, row in group:
                spoken = STAGE_DIRECTIONS.sub(' ', row.line)
                words = tuple(tokenize(spoken))
                speaker = ALIASES.get(row.character, row.character)
                if speaker.casefold() in generic or not words:
                    lines.append(Line(None, (), row_number))
                    continue
                names.add(speaker)
                lines.append(Line(speaker, words, row_number, spoken))
                profile[speaker].update(words)
            blocks.setdefault(key, []).append(Block(source, section, tuple(lines)))

        audit[medium]['recorded_blocks_by_story'] = {
            story: len(blocks[story, medium]) for story in range(1, 8)
        }
    name_words = {word for name in names for word in tokenize(name) if len(word) >= 3}
    return blocks, counts, name_words, audit


def content_counts(counts, name_words):
    stop = ENGLISH_STOP_WORDS | name_words
    return {word: count for word, count in counts.items()
            if len(word) >= 3 and word not in stop and word.removesuffix("'s") not in stop}


def omit_ambiguous_repeats(blocks, counts):
    """Remove adjacent identical lines credited to different named speakers."""
    filtered, updated = {}, {}
    removed = Counter()
    for key, source_blocks in blocks.items():
        updated[key] = {name: profile.copy() for name, profile in counts[key].items()}
        new_blocks = []
        for block in source_blocks:
            lines = block.lines
            ambiguous = {position for i, (a, b) in enumerate(zip(lines, lines[1:]))
                         if a.speaker and b.speaker and a.speaker != b.speaker
                         and len(a.words) >= 4 and a.words == b.words
                         for position in (i, i + 1)}
            if not ambiguous:
                new_blocks.append(block)
                continue
            new_lines = list(lines)
            for i in ambiguous:
                line = lines[i]
                updated[key][line.speaker].subtract(line.words)
                updated[key][line.speaker] = +updated[key][line.speaker]
                new_lines[i] = Line(None, (), line.row)
                removed[key[1]] += 1
            new_blocks.append(Block(block.source, block.section, tuple(new_lines)))
        filtered[key] = new_blocks
    return filtered, updated, dict(removed)


def vocabulary_matrices(counts, story, names, name_words):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.preprocessing import normalize
    from tfidf import course_tfidf

    profiles = [content_counts(counts[story, medium][name], name_words)
                for medium in MEDIUMS for name in names]
    vectorizer = DictVectorizer(sort=True)
    raw = vectorizer.fit_transform(profiles).tocsr()
    raw.sort_indices()
    weighted, _, idf, _ = course_tfidf(raw)
    unit = normalize(weighted).tocsr()
    if np.any(np.diff(unit.indptr) == 0):
        raise ValueError('A matched dialogue profile has no weighted terms')
    n = len(names)
    matrices = [(unit[offset:offset + n] @ unit[offset:offset + n].T).toarray()
                for offset in (0, n)]
    return matrices, raw, idf


def capture_edges(blocks, names, window=1, shuffles=0, rng=None, min_turn_words=1):
    """Count nearby speaker changes and shuffle known labels within each block."""
    n = len(names)
    ids = {name: i for i, name in enumerate(names)}
    first, second = np.triu_indices(n, k=1)
    pair_id = np.full((n + 1, n + 1), len(first), dtype=np.int32)
    pair_id[first, second] = pair_id[second, first] = np.arange(len(first))
    observed = np.zeros(len(first), dtype=int)
    expected = np.zeros(len(first))
    null = np.zeros((shuffles, len(first)), dtype=int)
    row_offset = np.arange(shuffles, dtype=np.int32)[:, None] * (len(first) + 1)

    for block in blocks:
        # Block boundaries and unknown rows break adjacency. A known, non-cohort
        # speaker keeps their slot and is never compressed out of the sequence.
        ids_in_order = np.array([ids.get(line.speaker, n) for line in block.lines])
        known = np.array([i for i, line in enumerate(block.lines) if len(line.words) >= min_turn_words], dtype=int)
        if len(known) < 2:
            continue
        labels = ids_in_order[known]
        span = np.concatenate([np.arange(len(known) - d) for d in range(1, window + 1)])
        end = np.concatenate([np.arange(d, len(known)) for d in range(1, window + 1)])
        contiguous = known[end] - known[span] == end - span
        left, right = span[contiguous], end[contiguous]
        sizes = np.bincount(labels, minlength=n + 1)
        if not len(left) or np.count_nonzero(sizes[:n]) < 2:
            continue
        observed += np.bincount(pair_id[labels[left], labels[right]], minlength=len(first) + 1)[:-1]
        expected += 2 * len(left) * sizes[first] * sizes[second] / (len(known) * (len(known) - 1))
        if shuffles:
            shuffled = rng.permuted(np.broadcast_to(labels, (shuffles, len(known))), axis=1)
            keys = pair_id[shuffled[:, left], shuffled[:, right]] + row_offset
            totals = np.bincount(keys.ravel(), minlength=shuffles * (len(first) + 1))
            null += totals.reshape(shuffles, -1)[:, :-1]
    return observed, expected, null


def unit_residuals(values, first, second, n, adjusted=True):
    """Midrank, then remove additive speaker effects in a complete dyad set."""
    ranks = rankdata(values, axis=-1)
    centered = ranks - ranks.mean(axis=-1, keepdims=True)
    if adjusted:
        membership = np.eye(n)[first] + np.eye(n)[second]
        effects = centered @ membership / (n - 2)
        centered -= effects[..., first] + effects[..., second]
    norm = np.linalg.norm(centered, axis=-1, keepdims=True)
    return np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 1e-12)


def settings_result(blocks, counts, name_words, *, minimum=MIN_WORDS, window=1,
                    shuffles=0, rng=None, exclude=(), min_turn_words=1):
    results = []
    for story in range(1, 8):
        names = sorted(set(counts[story, 'Books']) & set(counts[story, 'Films']))
        names = [name for name in names if name not in exclude and all(
            counts[story, medium][name].total() >= minimum for medium in MEDIUMS)]
        if len(names) < 4:
            raise ValueError(f'Story {story} needs at least four matched speakers')
        n = len(names)
        first, second = np.triu_indices(n, k=1)
        similarities, profile_counts, idf = vocabulary_matrices(counts, story, names, name_words)
        for medium, similarity in zip(MEDIUMS, similarities):
            edges, expected, simulations = capture_edges(blocks[story, medium], names, window, shuffles,
                                                          rng, min_turn_words=min_turn_words)
            cosine = np.round(similarity[first, second], 12)
            target = unit_residuals(cosine, first, second, n)
            edge_ranks = unit_residuals(edges, first, second, n)
            score = edge_ranks @ target
            raw_score = unit_residuals(edges, first, second, n, adjusted=False) @ unit_residuals(
                cosine, first, second, n, adjusted=False)
            null_scores = unit_residuals(simulations, first, second, n) @ target if shuffles else np.empty(0)
            excess = np.log((edges + 1) / (expected + 1)).round(12)
            opportunity_adjusted = unit_residuals(excess, first, second, n) @ target
            results.append(dict(story=story, medium=medium, n=n, names=names, first=first, second=second,
                                edges=edges, expected=expected, cosine=cosine, score=float(score), raw=float(raw_score),
                                opportunity_adjusted=float(opportunity_adjusted),
                                null=null_scores, edge_ranks=edge_ranks, profiles=profile_counts, idf=idf))
    return results


def equal_length_benchmark(primary, repetitions, rng):
    """Thin each longer dialogue profile to its counterpart's token count.

    Every draw uses the original story vocabulary, IDF, casts and interactions.
    """
    from scipy.sparse import csr_matrix
    from sklearn.preprocessing import normalize

    estimated = np.zeros((len(primary) // 2, repetitions, len(MEDIUMS)))
    for story_index, (book, film) in enumerate(zip(primary[::2], primary[1::2])):
        profiles, idf = book['profiles'], book['idf']
        n, first, second = book['n'], book['first'], book['second']
        totals = np.asarray(profiles.sum(axis=1), dtype=int).ravel()
        budgets = np.minimum(totals[:n], totals[n:])
        changing = [(slice(profiles.indptr[i], profiles.indptr[i + 1]), budgets[i % n])
                    for i, total in enumerate(totals) if total > budgets[i % n]]
        for simulation in range(repetitions):
            sampled = profiles.data.copy()
            for section, length in changing:
                sampled[section] = rng.multivariate_hypergeometric(profiles.data[section].astype(int), length)
            sampled *= idf[profiles.indices]
            unit = normalize(csr_matrix((sampled, profiles.indices, profiles.indptr), shape=profiles.shape))
            for j, data in enumerate((book, film)):
                vectors = unit[j * n:(j + 1) * n]
                cosine = (vectors @ vectors.T).toarray()[first, second].round(12)
                estimated[story_index, simulation, j] = (
                    data['edge_ranks'] @ unit_residuals(cosine, first, second, n)
                )
    return estimated


def write_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def by_medium(rows, field):
    return {medium: float(np.mean([row[field] for row in rows if row['medium'] == medium]))
            for medium in MEDIUMS}


def save_results(out, primary, draws, sensitivity, counts, audit, axes):
    pairs = []
    stories = []
    for book, film in zip(primary[::2], primary[1::2]):
        names = book['names']
        stories.append({
            'story': book['story'], 'title': SHORT_TITLES[book['story'] - 1],
            'characters': names, 'pairs': len(book['edges']),
            'score': {row['medium']: row['score'] for row in (book, film)},
            'raw_spearman': {row['medium']: row['raw'] for row in (book, film)},
            'observed_nonzero_edges': {row['medium']: int(np.count_nonzero(row['edges']))
                                       for row in (book, film)},
        })
        for k, (i, j) in enumerate(zip(book['first'], book['second'])):
            info = {'story': book['story'], 'character_a': names[i], 'character_b': names[j]}
            for row in (book, film):
                prefix = row['medium'].lower()
                info[f'{prefix}_switches'] = int(row['edges'][k])
                info[f'{prefix}_expected_switches'] = round(row['expected'][k], 8)
                info[f'{prefix}_cosine'] = row['cosine'][k]
            pairs.append(info)
    write_rows(out / 'matched_pairs.csv', pairs)
    write_rows(out / 'equal_length_draws.csv',
               [{'draw': n, 'book_score': b, 'film_score': f, 'book_minus_film': b - f}
                for n, (b, f) in enumerate(draws, 1)])

    full_scores = by_medium(primary, 'score')
    gap = full_scores['Books'] - full_scores['Films']
    gap_draws = draws[:, 0] - draws[:, 1]
    median_gap = float(np.median(gap_draws))
    paired_content_by_story = [np.asarray(row['profiles'].sum(axis=1), dtype=int).reshape(2, row['n'])
                               for row in primary[::2]]
    paired_content = np.concatenate(paired_content_by_story, axis=1)
    raw_totals = {
        medium: sum(counts[row['story'], medium][name].total()
                    for row in primary[::2] for name in row['names'])
        for medium in MEDIUMS
    }
    block_null = {
        medium: np.mean([row['null'] for row in primary if row['medium'] == medium], axis=0)
        for medium in MEDIUMS
    }
    summary = {
        'sources': audit, 'minimum_raw_words': MIN_WORDS, 'seed': SEED,
        'matched_pairs': len(pairs), 'matched_speaker_story_profiles': paired_content.shape[1],
        'unique_speakers': len({name for story in stories for name in story['characters']}),
        'matched_raw_word_totals': raw_totals,
        'matched_filtered_word_totals': dict(zip(MEDIUMS, paired_content.sum(axis=1).tolist())),
        'language_axes': axes,
        'median_book_to_film_filtered_word_ratio': float(np.median(paired_content[0] / paired_content[1])),
        'stories': stories, 'full_scores': full_scores, 'raw_scores': by_medium(primary, 'raw'),
        'full_gap': gap, 'equal_length_draws': len(draws), 'equal_length_median_gap': median_gap,
        'equal_length_middle_95_gap': np.quantile(gap_draws, [.025, .975]).tolist(),
        'equal_length_median_scores': dict(zip(MEDIUMS, np.median(draws, axis=0).tolist())),
        'fraction_gap_reduced': 1 - median_gap / gap,
        'block_shuffle_draws': SHUFFLES,
        'block_shuffle_reference': {
            label: np.quantile(draws, [.025, .5, .975]).tolist()
            for label, draws in {**block_null, 'Books minus films': block_null['Books'] - block_null['Films']}.items()
        },
        'relative_to_expected_switches_scores': by_medium(primary, 'opportunity_adjusted'),
        'sensitivity_scores': sensitivity,
    }
    (out / 'statistics.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo', type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    blocks, counts, name_words, audit = read_project(args.repo.resolve())
    rng = np.random.default_rng(SEED)
    primary = settings_result(blocks, counts, name_words, shuffles=SHUFFLES, rng=rng)
    draws_by_story = equal_length_benchmark(primary, THINNINGS, np.random.default_rng(SEED + 1))
    draws = draws_by_story.mean(axis=0)
    sensitivity = {
        label: by_medium(settings_result(blocks, counts, mask, **options), 'score')
        for label, mask, options in (
            ('60-word minimum', name_words, {'minimum': 60}),
            ('150-word minimum', name_words, {'minimum': 150}),
            ('Two-row window', name_words, {'window': 2}),
            ('Five-row window', name_words, {'window': 5}),
            ('Unigrams from names retained', set(), {}),
            ('Harry omitted from the matched cast', name_words, {'exclude': ('Harry Potter',)}),
            ('Only switches between rows with at least four words', name_words, {'min_turn_words': 4}),
        )
    }
    # Two book files have substantially fewer recorded sections than chapters.
    keep = np.array([True, True, True, False, False, True, True])
    reduced_draws = draws_by_story[keep].mean(axis=0)
    sensitivity['Omitting Goblet of Fire and Order of the Phoenix'] = {
        **by_medium([row for row in primary if keep[row['story'] - 1]], 'score'),
        'median_equal_length_gap': float(np.median(reduced_draws[:, 0] - reduced_draws[:, 1])),
    }
    repeated_blocks, repeated_counts, removed = omit_ambiguous_repeats(blocks, counts)
    sensitivity['Omitting adjacent identical lines credited to different speakers'] = {
        **by_medium(settings_result(repeated_blocks, repeated_counts, name_words), 'score'),
        'removed_rows': removed,
    }
    from language_axes import character_axes
    axes = character_axes(blocks, primary)
    summary = save_results(args.out, primary, draws, sensitivity, counts, audit, axes)
    from figures import render
    render(args.out, draws, summary)
    print(f'Wrote three figures and analysis results to {args.out.resolve()}')


if __name__ == '__main__':
    main()
