"""Render three alternatives for the second presentation slide from saved results."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch
from scipy.stats import rankdata

from figures import MAP_GRID, MAP_INK, MAP_MUTED, MAP_PAPER, MEDIA_STYLE, SERIF, axis, plt, save, text


MEDIA = ('Books', 'Films')
SHUFFLES = 1999
CARD = '#ffffff'
CARD_BORDER = '#e5e5df'
SHUFFLE_FILL = '#ced6e0'
SHUFFLE_INK = '#758399'


def read_results(root):
    stats = json.loads((root / 'statistics.json').read_text(encoding='utf-8'))
    grouped = defaultdict(list)
    with (root / 'matched_pairs.csv').open(encoding='utf-8', newline='') as source:
        for row in csv.DictReader(source):
            grouped[int(row['story'])].append(row)

    stories = []
    for record in stats['stories']:
        lookup = {name: i for i, name in enumerate(record['characters'])}
        matrices = {m: np.zeros((2, len(lookup), len(lookup))) for m in MEDIA}
        rows = grouped[record['story']]
        assert len(rows) == len(lookup) * (len(lookup) - 1) // 2
        for row in rows:
            a, b = lookup[row['character_a']], lookup[row['character_b']]
            for m in MEDIA:
                prefix = m.lower()
                matrices[m][:, a, b] = matrices[m][:, b, a] = (
                    float(row[f'{prefix}_switches']), float(row[f'{prefix}_cosine']))
        stories.append(matrices)
    return stats, stories


def frequency_bands(stories):
    """Rank frequency among linked pairs and retain a separate zero-switch group."""
    by_medium = {m: [[] for _ in range(5)] for m in MEDIA}
    for story in stories:
        for m, (counts, similarity) in story.items():
            pair = np.triu_indices(len(counts), k=1)
            counts = counts[pair]
            percentiles = 100 * (rankdata(similarity[pair]) - .5) / len(counts)
            linked = counts > 0
            bins = np.zeros(len(counts), dtype=int)
            bins[linked] = (4 * (rankdata(counts[linked]) - .5) / linked.sum()).astype(int) + 1
            for i, values in enumerate(by_medium[m]):
                values.extend(percentiles[bins == i])
    return {m: [{'pairs': len(values), 'quartiles': np.quantile(values, [.25, .5, .75]).tolist()}
                for values in bands] for m, bands in by_medium.items()}


def top_membership(matrix, k=5, only_positive=False):
    """Give tied candidates fractional credit at the top-k cutoff."""
    result = np.zeros_like(matrix)
    for i, row in enumerate(matrix):
        valid = np.arange(len(row)) != i
        if only_positive:
            valid &= row > 0
        length = min(k, int(valid.sum()))
        if not length:
            continue
        cutoff = np.partition(row[valid], -length)[-length]
        above = valid & (row > cutoff)
        tied = valid & (row == cutoff)
        result[i, above] = 1
        result[i, tied] = (length - above.sum()) / tied.sum()
    return result


def neighborhood_overlap(stories):
    """Shuffle whole vocabulary profiles among the cast within each story."""
    rng = np.random.default_rng(2805)
    observed = {m: 0. for m in MEDIA}
    null = {m: np.zeros(SHUFFLES) for m in MEDIA}
    profiles = 0
    for story in stories:
        length = len(story['Books'][0])
        permutations = np.stack([rng.permutation(length) for _ in range(SHUFFLES)])
        for m, (counts, similarities) in story.items():
            partners = top_membership(counts, only_positive=True)
            vocabulary = top_membership(similarities)
            observed[m] += float((partners * vocabulary).sum())
            relabeled = vocabulary[permutations[:, :, None], permutations[:, None, :]]
            null[m] += (partners[None, :, :] * relabeled).sum(axis=(1, 2))
        profiles += length
    comparison = {
        'character_story_profiles_per_medium': profiles,
        'shuffles': SHUFFLES,
        'by_medium': {m: {'observed': observed[m] / profiles,
                          'shuffle_mean': float(null[m].mean() / profiles),
                          'shuffle_interval': np.quantile(null[m] / profiles, [.025, .975]).tolist()}
                      for m in MEDIA},
    }
    return comparison, {m: null[m] / profiles for m in MEDIA}


def slide(title, subtitle):
    fig = plt.figure(figsize=(13.3333, 7.5), facecolor=MAP_PAPER)
    text(fig, .071, .920, title, 32, MAP_INK, fontfamily=SERIF, weight='bold')
    text(fig, .073, .852, subtitle, 14, MAP_MUTED)
    return fig


def chart(fig, bounds):
    ax = axis(fig, bounds)
    ax.tick_params(colors=MAP_MUTED)
    return ax


def card(fig, bounds, fill=CARD):
    x, y, width, height = bounds
    fig.add_artist(FancyBboxPatch(
        (x, y), width, height, transform=fig.transFigure,
        boxstyle='round,pad=0.005,rounding_size=0.014',
        facecolor=fill, edgecolor=CARD_BORDER, linewidth=.85, zorder=-1,
    ))


def footer(fig, left, right=None):
    fig.add_artist(plt.Line2D([.073, .927], [.102, .102], transform=fig.transFigure,
                            color=MAP_GRID, linewidth=1))
    text(fig, .073, .06, left, 11, MAP_MUTED)
    if right:
        text(fig, .927, .06, right, 11, MAP_MUTED, ha='right')


def medium_comparison(out, stats):
    fig = slide('The link is stronger in every book',
                'Do characters who speak together more often use more similar words?')
    card(fig, [.073, .188, .656, .594])
    card(fig, [.752, .188, .175, .594], fill='#f2f3ee')
    text(fig, .095, .733, 'Association by story', 14, MAP_INK, weight='bold')
    ax = chart(fig, [.26, .281, .432, .407])
    ax.set(xlim=(-.003, .63), ylim=(6.55, -.55), yticks=[])
    ax.set_xticks([0, .2, .4, .6], ['0', '0.2', '0.4', '0.6'])
    ax.xaxis.grid(color=MAP_GRID, linewidth=.9)
    ax.axvline(0, color='#a2adba', lw=1.15)
    for i, story in enumerate(stats['stories']):
        book, film = (story['score'][m] for m in MEDIA)
        if i % 2 == 0:
            ax.axhspan(i - .39, i + .39, color='#f3f5f6', lw=0, zorder=0)
        ax.plot([film, book], [i, i], color='#aab6c6', lw=2.5, solid_capstyle='round')
        for m, value in (('Books', book), ('Films', film)):
            color, marker = MEDIA_STYLE[m]
            ax.scatter(value, i, s=86, color=color, marker=marker,
                       edgecolor=CARD, linewidth=.75, zorder=3)
        ax.text(-.04, i, story['title'], transform=ax.get_yaxis_transform(), va='center',
                ha='right', color=MAP_INK, fontsize=13)
    text(fig, .476, .224, 'Speaker-adjusted rank correlation', 12.5, MAP_INK, ha='center')

    text(fig, .771, .730, 'Mean across stories', 13.5, MAP_INK, weight='bold')
    for m, y, glyph in (('Books', .592, '●'), ('Films', .426, '◆')):
        color, _ = MEDIA_STYLE[m]
        text(fig, .772, y + .06, f'{glyph}  {m}', 12.5, color, weight='bold')
        text(fig, .769, y - .012, f'{stats["full_scores"][m]:.2f}', 40,
             color, fontfamily=SERIF, weight='bold')
    fig.add_artist(plt.Line2D([.771, .908], [.326, .326], transform=fig.transFigure,
                             color='#d9dfd6', linewidth=1))
    text(fig, .771, .263, f'{stats["full_gap"]:+.2f}', 19, MAP_INK, weight='bold')
    text(fig, .843, .263, 'gap', 13, MAP_MUTED)
    footer(fig, 'Each row compares the same characters and pairs.',
           f'{stats["matched_pairs"]} pair–story comparisons per medium')
    save(fig, out, '02a_book_film_comparison')


def frequency_plot(out, bands):
    fig = slide('Frequent partners use more similar words',
                'The most frequent pairs typically rank near the 93rd percentile in both books and films.')
    card(fig, [.073, .192, .854, .595])
    text(fig, .095, .737, 'Dialogue frequency', 13.5, MAP_INK, weight='bold')
    text(fig, .268, .737, 'Vocabulary similarity percentile', 13.5, MAP_INK, weight='bold')
    for m, symbol, left in (('Books', '●', .753), ('Films', '◆', .834)):
        text(fig, left, .737, f'{symbol}  {m}', 12.5, MEDIA_STYLE[m][0], weight='bold')
    ax = chart(fig, [.266, .279, .63, .405])
    groups = ('No exchanges', 'Least frequent', 'Less frequent', 'More frequent', 'Most frequent')
    ax.set(xlim=(0, 100), ylim=(4.55, -.55), yticks=[])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(color=MAP_GRID, linewidth=.8)
    ax.axhspan(-.44, .44, color='#f2f4f5', lw=0, zorder=0)
    ax.axhspan(3.56, 4.44, color='#f2f4f5', lw=0, zorder=0)
    ax.axhline(.5, color='#c5ced9', linewidth=.85)
    for i, group in enumerate(groups):
        ax.text(-.271, i, group, transform=ax.get_yaxis_transform(),
                va='center', color=MAP_INK, fontsize=12.5,
                weight='bold' if i in (0, 4) else 'normal')
    for m, offset in (('Books', -.165), ('Films', .165)):
        color, marker = MEDIA_STYLE[m]
        low, med, high = np.array([row['quartiles'] for row in bands[m]]).T
        rows = np.arange(len(groups)) + offset
        ax.hlines(rows, low, high, linewidths=6, color=to_rgba(color, .24),
                  capstyle='round', zorder=2)
        ax.scatter(med, rows, color=color, marker=marker, s=64,
                   linewidth=.85, edgecolor=CARD, zorder=4)
        for j in (0, 4):
            ax.annotate(str(int(np.floor(med[j] + .5))), (med[j], rows[j]), (7, 0),
                        textcoords='offset points', color=color, weight='bold',
                        va='center', fontsize=10.5, zorder=5,
                        bbox={'facecolor': '#f2f4f5', 'edgecolor': 'none', 'pad': 1})
    text(fig, .266, .227, 'Less similar', 10.5, MAP_MUTED)
    text(fig, .896, .227, 'More similar', 10.5, MAP_MUTED, ha='right')
    fig.add_artist(plt.Line2D([.073, .927], [.125, .125], transform=fig.transFigure,
                             color=MAP_GRID, linewidth=1))
    text(fig, .073, .094, 'We count an exchange each time one character speaks right after the other in the same chapter or scene.',
         10.5, MAP_MUTED)
    text(fig, .073, .064, 'We rank vocabulary similarity and group pairs with exchanges by frequency within each story and medium.',
         10.5, MAP_MUTED)
    text(fig, .073, .034, 'Markers show medians. Pale bars span the middle half of pairs in each group.',
         10.5, MAP_MUTED)
    save(fig, out, '02b_frequency_and_vocabulary')


def neighbors_plot(out, overlap, null_distributions):
    fig = slide('Dialogue partners share vocabulary',
                'Compare each character’s five closest vocabulary neighbors with up to five frequent dialogue partners.')
    card(fig, [.073, .198, .854, .580])
    text(fig, .097, .729, 'Observed and shuffled overlap', 14, MAP_INK, weight='bold')
    ax = chart(fig, [.196, .278, .69, .381])
    ax.set(xlim=(0, 5), ylim=(-.34, 1.70), yticks=[])
    ax.set_xticks(range(6))
    ax.xaxis.grid(color=MAP_GRID, linewidth=.85)
    bins = np.arange(0, 5.05, .065)
    histograms = {m: np.histogram(null_distributions[m], bins=bins)[0] for m in MEDIA}
    peak = max(h.max() for h in histograms.values())
    for base, m in ((1, 'Books'), (0, 'Films')):
        color, marker = MEDIA_STYLE[m]
        score = overlap['by_medium'][m]
        value, average = score['observed'], score['shuffle_mean']
        ax.hlines(base, 0, 5, color='#d7dde6', linewidth=1, zorder=1)
        ax.stairs(base + .54 * histograms[m] / peak, bins, baseline=base,
                  fill=True, color=SHUFFLE_FILL, alpha=.93, linewidth=0, zorder=2)
        ax.vlines(average, base, base + .55, color=SHUFFLE_INK, linestyle=(0, (2, 2)), lw=1.2, zorder=3)
        ax.vlines(value, base, base + .43, color=color, linewidth=2.2, zorder=3)
        ax.scatter(value, base + .43, color=color, marker=marker, s=103,
                   edgecolor=CARD, linewidth=.8, zorder=5)
        ax.text(-.043, base + .27, m, color=color, transform=ax.get_yaxis_transform(),
                va='center', ha='right', fontsize=16, weight='bold')
        ax.text(value + .14, base + .43, f'{value:.2f} observed', color=color,
                va='center', fontsize=21, weight='bold')
        ax.text(average, base - .16, f'{average:.2f} shuffled mean', ha='center', va='center',
                color=MAP_MUTED, fontsize=10.5)
    text(fig, .541, .229, 'Average number of shared neighbors out of five', 13, MAP_INK, ha='center')
    footer(fig, 'Gray distributions come from 1,999 within-story vocabulary-profile shuffles.',
           f'{overlap["character_story_profiles_per_medium"]} character–story profiles per medium')
    save(fig, out, '02c_dialogue_and_vocabulary_neighbors')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'results' / 'interaction_language',
                        help='Directory containing the results written by analyse.py.')
    args = parser.parse_args()
    stats, stories = read_results(args.results_dir)

    output = args.results_dir / 'slide_2_options'
    output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'svg.fonttype': 'path',
                         'svg.hashsalt': 'hp-slide-two-options'})
    bands = frequency_bands(stories)
    overlap, null_distributions = neighborhood_overlap(stories)
    medium_comparison(output, stats)
    frequency_plot(output, bands)
    neighbors_plot(output, overlap, null_distributions)
    (output / 'figure_data.json').write_text(json.dumps({
        'story_comparison': stats['stories'],
        'frequency_bands': bands,
        'neighborhood_overlap': overlap,
    }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Saved the three slide alternatives to {output}')


if __name__ == '__main__':
    main()
