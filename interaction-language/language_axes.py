"""Measure visible dialogue cues for interpersonal tone and speech acts."""

import re
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr


CUE_GROUPS = {
    'hostile': (
        "how dare you", "don't you dare", "you fool", "you idiot", "you stupid", "stupid boy",
        "shut up", "get lost", "you'll pay", "you will pay", "you'll regret", "you will regret",
        "i'll kill you", "i will kill you", "i'm going to kill you", "i hate you",
        "pathetic", "coward", "filthy mudblood", "filthy little", "i'll hurt you", "i will hurt you",
    ),
    'supportive': (
        "don't worry", "do not worry", "don't be afraid", "are you all right", "are you alright",
        "you all right", "you're safe", "you are safe", "you'll be fine", "you'll be all right",
        "it'll be all right", "it's all right", "it's okay", "i'll help", "let me help", "thank you",
        "thanks", "well done", "i'm proud of you", "i love you", "you can trust me", "i'm here for you",
        "i'm sorry",
    ),
}
PRIOR_WORDS = 300
CUE_EXCEPTIONS = {
    'thanks': r'thanks(?! to(?: |$))',
    'pathetic': r'(?<!not )pathetic',
    'shut up': r'(?<!to )shut up',
}
DIRECTIVE_OPENERS = (
    "do not", "don't", "you must", "you need to", "you have to", "be careful", "make sure",
    "stop", "go", "come", "get", "give", "tell", "leave", "stay", "keep", "take", "put",
    "move", "stand", "sit", "run", "follow", "remember", "watch", "hold", "turn",
)
ATTENTION_OPENERS = ("listen", "look", "wait", "let")
SEGMENTS = re.compile(r"[^.!?]+[.!?]*")


def cue_pattern(cues):
    """Match whole tokens, preferring longer overlapping cues."""
    alternatives = '|'.join(CUE_EXCEPTIONS.get(cue, re.escape(cue))
                            for cue in sorted(cues, key=len, reverse=True))
    return re.compile(r'(?<!\S)(?:' + alternatives + r')(?!\S)')


def tone_scores(counts, words, baseline, prior=PRIOR_WORDS):
    rates = 1000 * (counts + prior * baseline) / (words + prior)
    return rates[..., 1] - rates[..., 0]


def paired_indices(keys):
    index = {key: i for i, key in enumerate(keys)}
    names = sorted({name for name, _ in keys})
    return tuple(np.array([index[name, medium] for name in names]) for medium in ('Books', 'Films'))


def tone_extremity(keys, counts, lengths, baseline, coordinates):
    """Compare absolute tone scores before and after equalizing paired word counts."""
    books, films = paired_indices(keys)
    observed = np.abs(coordinates[films]) - np.abs(coordinates[books])
    matched_words = np.empty_like(lengths)
    matched_words[books] = matched_words[films] = np.minimum(lengths[books], lengths[films])
    rng = np.random.default_rng(2810)
    repetitions = 5000
    resampled = np.empty((len(keys), repetitions))
    for i, (total, keep, row) in enumerate(zip(lengths, matched_words, counts)):
        # Treat each cue as a mark on its starting token. Sample token slots
        # without replacement while leaving the other version and prior fixed.
        if keep < total:
            sampled = rng.multivariate_hypergeometric(
                np.r_[row, total - row.sum()], keep, size=repetitions)[:, :len(CUE_GROUPS)]
        else:
            sampled = row
        resampled[i] = tone_scores(sampled, keep, baseline)
    gap = np.abs(resampled[films]) - np.abs(resampled[books])
    return {
        'characters_with_larger_film_magnitude': (observed > 0).sum(axis=0).tolist(),
        'mean_absolute_score': {'Books': np.abs(coordinates[books]).mean(axis=0).tolist(),
                                'Films': np.abs(coordinates[films]).mean(axis=0).tolist()},
        'mean_magnitude_gap': observed.mean(axis=0).tolist(),
        'equal_length_draws': repetitions,
        'equal_length_mean_gap_quantiles': np.quantile(gap.mean(axis=0), [.025, .5, .975], axis=0).tolist(),
        'equal_length_median_characters_with_larger_film_magnitude': np.median(
            (gap > 0).sum(axis=0), axis=0).tolist(),
    }


def directive_pattern(openers):
    alternatives = '|'.join(map(re.escape, sorted(openers, key=len, reverse=True)))
    return re.compile(r'^(?:(?:please|just|now) )?(?:' + alternatives + r')(?: |$)')


def directive_share(counts, baseline, prior=PRIOR_WORDS):
    adjusted = counts + prior * baseline
    return 100 * adjusted[..., 1] / adjusted.sum(axis=-1)


def directive_comparison(keys, acts, lengths, broad, plotted):
    books, films = paired_indices(keys)
    shift = plotted[films] - plotted[books]
    baseline = acts.sum(axis=0) / lengths.sum()
    alternative = directive_share(broad, broad.sum(axis=0) / lengths.sum())
    longer_prior = directive_share(acts, baseline, prior=600)
    return {
        'total_questions': int(acts[:, 0].sum()),
        'total_directives': int(acts[:, 1].sum()),
        'corpus_share': float(100 * baseline[1] / baseline.sum()),
        'more_directive_in_films': int((shift > 0).sum()),
        'mean_percentage_point_shift': float(shift.mean()),
        'median_percentage_point_shift': float(np.median(shift)),
        'broader_opener_rank_agreement': float(spearmanr(plotted, alternative).statistic),
        'broader_opener_more_directive_in_films': int((alternative[films] > alternative[books]).sum()),
        'larger_prior_rank_agreement': float(spearmanr(plotted, longer_prior).statistic),
        'larger_prior_more_directive_in_films': int((longer_prior[films] > longer_prior[books]).sum()),
    }


def character_axes(blocks, primary):
    from word_counts import tokenize

    eligible = {row['story']: set(row['names']) for row in primary[::2]}
    patterns = [cue_pattern(cues) for cues in CUE_GROUPS.values()]
    sizes = Counter()
    hits = defaultdict(lambda: [Counter() for pattern in patterns])
    evidence = defaultdict(list)
    questions, directives, broad = Counter(), Counter(), Counter()
    opener = directive_pattern(DIRECTIVE_OPENERS)
    wider_opener = directive_pattern(ATTENTION_OPENERS)
    for (story, medium), source in blocks.items():
        for block in source:
            for line in block.lines:
                if line.speaker not in eligible[story]:
                    continue
                key = (line.speaker, medium)
                sizes[key] += len(line.words)
                text = ' '.join(line.words)
                for group, pattern, counter in zip(CUE_GROUPS, patterns, hits[key]):
                    matches = pattern.findall(text)
                    counter.update(matches)
                    if matches:
                        evidence[group, medium].append({
                            'group': group, 'medium': medium, 'character': line.speaker,
                            'source': block.source, 'record': line.row, 'matches': matches,
                        })
                for segment in SEGMENTS.findall(line.text):
                    words = ' '.join(tokenize(segment))
                    if not words:
                        continue
                    match = opener.match(words)
                    if '?' in segment:
                        questions[key] += 1
                        group, cue = 'questions', '?'
                    elif match:
                        directives[key] += 1
                        broad[key] += 1
                        group, cue = 'directives', match[0].strip()
                    else:
                        broad[key] += bool(wider_opener.match(words))
                        continue
                    evidence[group, medium].append({
                        'group': group, 'medium': medium, 'character': line.speaker,
                        'source': block.source, 'record': line.row, 'matches': [cue],
                    })

    keys = sorted(sizes)
    lengths = np.array([sizes[key] for key in keys])
    counts = np.array([[counter.total() for counter in hits[key]] for key in keys])
    baseline = counts.sum(axis=0) / lengths.sum()

    acts = np.array([[questions[key], directives[key]] for key in keys])
    broader_acts = np.array([[questions[key], broad[key]] for key in keys])
    x = tone_scores(counts, lengths[:, None], baseline)
    y = directive_share(acts, acts.sum(axis=0) / lengths.sum())
    points = [{
        'character': name, 'medium': medium,
        'x': float(tone), 'y': float(directing), 'words': sizes[name, medium],
        'questions': questions[name, medium], 'directives': directives[name, medium],
        'hits': {group: dict(counter.most_common()) for group, counter in
                 zip(CUE_GROUPS, hits[name, medium])},
    } for (name, medium), tone, directing in zip(keys, x, y)]
    checks = {}
    for prior in (0, 150, 600):
        other = tone_scores(counts, lengths[:, None], baseline, prior)
        checks[str(prior)] = float(spearmanr(x, other).statistic)

    rng = np.random.default_rng(2805)
    audit = [rows[i] for _, rows in sorted(evidence.items())
             for i in rng.choice(len(rows), size=min(3, len(rows)), replace=False)]
    return {
        'cues': CUE_GROUPS, 'cue_exceptions': CUE_EXCEPTIONS, 'prior_words': PRIOR_WORDS,
        'cue_totals': dict(zip(CUE_GROUPS, counts.sum(axis=0).tolist())),
        'directive_openers': DIRECTIVE_OPENERS,
        'attention_openers_for_sensitivity': ATTENTION_OPENERS,
        'prior_sensitivity_rank_agreement': checks,
        'tone_extremity': tone_extremity(keys, counts, lengths, baseline, x),
        'directive_comparison': directive_comparison(keys, acts, lengths, broader_acts, y),
        'coordinates': points, 'audit_records': audit,
    }
