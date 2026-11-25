"""Compare each character's book dialogue with the cast of its film adaptation."""

import argparse
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import rankdata
from sklearn.preprocessing import normalize

from analyse import SHORT_TITLES, read_project, settings_result, write_rows
from figures import MAP_GRID, MAP_INK, MAP_MUTED, MEDIA_STYLE, axis, plt, save, text
from slide_2_options import card, slide


FEATURED = {
    'Harry Potter': 'Harry',
    'Hermione Granger': 'Hermione',
    'Ron Weasley': 'Ron',
    'Draco Malfoy': 'Draco',
    'Severus Snape': 'Snape',
    'Albus Dumbledore': 'Dumbledore',
    'Voldemort': 'Voldemort',
}
MATCH = MEDIA_STYLE['Books'][0]
NEAR = '#e1eeeb'
LOWER = '#f7e9d8'
MISSING = '#f1f3f5'
FIGURE_NAME = '04_vocabulary_across_adaptations'
SIMILARITY_FIGURE_NAME = '04_adaptation_vocabulary_similarity'
SIMILARITY_CHARACTERS = {
    'Harry Potter': ('#08786f', 'o'),
    'Ron Weasley': ('#af551f', 's'),
    'Hermione Granger': ('#6946a4', 'D'),
    'Dobby': ('#28598d', 'P'),
    'Voldemort': ('#a63d56', '^'),
    'Severus Snape': ('#46566a', 'X'),
    'Bellatrix Lestrange': ('#56722a', 'v'),
}
STORY_TICKS = ('Philosopher’s\nStone', 'Chamber of\nSecrets', 'Prisoner of\nAzkaban',
               'Goblet of\nFire', 'Order of the\nPhoenix', 'Half-Blood\nPrince', 'Deathly\nHallows')


def adaptation_ranks(repo):
    blocks, counts, name_words, _ = read_project(repo)
    matched = settings_result(blocks, counts, name_words)
    rows = []
    for book in matched[::2]:
        story, n, names = book['story'], book['n'], book['names']
        # The same story-level vocabulary and IDF weight both media.
        vectors = normalize(book['profiles'].multiply(book['idf'])).tocsr()
        scores = (vectors[:n] @ vectors[n:].T).toarray().round(12)
        ranks = rankdata(-scores, axis=1, method='average').diagonal()
        for i, name in enumerate(names):
            best = scores[i].max()
            rows.append({
                'story': story,
                'title': SHORT_TITLES[story - 1],
                'character': name,
                'book_words': counts[story, 'Books'][name].total(),
                'film_words': counts[story, 'Films'][name].total(),
                'film_candidates': n,
                'own_film_rank': float(ranks[i]),
                'own_film_cosine': float(scores[i, i]),
                'closest_film_characters': ' | '.join(names[j] for j in np.flatnonzero(scores[i] == best)),
                'closest_film_cosine': float(best),
            })
    return rows


def rounded_box(fig, left, bottom, width, height, color, radius=.006):
    fig.add_artist(FancyBboxPatch(
        (left, bottom), width, height, transform=fig.transFigure,
        boxstyle=f'round,pad=0,rounding_size={radius}', facecolor=color, linewidth=0,
    ))


def rank_cell(fig, x, y, rank):
    if rank is None:
        fill, ink, label = MISSING, MAP_MUTED, '–'
    elif rank == 1:
        fill, ink, label = MATCH, 'white', '1'
    elif rank <= 3:
        fill, ink, label = NEAR, '#245a52', f'{rank:g}'
    else:
        fill, ink, label = LOWER, '#784e28', f'{rank:g}'
    rounded_box(fig, x - .028, y - .024, .056, .048, fill)
    text(fig, x, y, label, 13, ink, ha='center', weight='bold')


def render(out, rows):
    closest = sum(row['own_film_rank'] == 1 for row in rows)
    fig = slide('Whose vocabulary carries over to film?',
                f'In {closest} of {len(rows)} comparisons, the closest vocabulary match is the same character on screen.')
    card(fig, [.073, .184, .856, .608])
    rounded_box(fig, .766, .249, .158, .533, '#f3f6f5', radius=.011)

    text(fig, .094, .749, 'Book and adaptation', 12.5, MAP_INK, weight='bold')
    text(fig, .270, .749, 'Rank of the character’s own film version', 12.5, MAP_INK, weight='bold')
    text(fig, .779, .749, 'Ranked first', 12.5, MAP_INK, weight='bold')
    text(fig, .779, .702, 'Across the full matched cast', 9.5, MAP_MUTED)
    columns = {name: .292 + i * .067 for i, name in enumerate(FEATURED)}
    for name, x in columns.items():
        text(fig, x, .699, FEATURED[name], 10.5, MAP_INK, ha='center', weight='bold')

    by_story = {story: {row['character']: row['own_film_rank'] for row in rows if row['story'] == story}
                for story in range(1, len(SHORT_TITLES) + 1)}
    for story, title in enumerate(SHORT_TITLES, 1):
        y = .650 - (story - 1) * .062
        line = plt.Line2D([.093, .910], [y + .031] * 2, transform=fig.transFigure,
                          color=MAP_GRID, linewidth=.58)
        fig.add_artist(line)
        text(fig, .094, y, title, 12, MAP_INK)
        ranks = by_story[story]
        for name, x in columns.items():
            rank_cell(fig, x, y, ranks.get(name))
        count = sum(rank == 1 for rank in ranks.values())
        rounded_box(fig, .780, y - .007, .076, .014, '#dce6e4', radius=.004)
        if count:
            rounded_box(fig, .780, y - .007, .076 * count / len(ranks), .014, MATCH, radius=.004)
        text(fig, .911, y, f'{count} / {len(ranks)}', 11, MAP_INK, ha='right', weight='bold')

    for x, fill, label in ((.095, MATCH, '#1  Closest match'),
                           (.251, NEAR, '#2–3  Near the top'),
                           (.404, LOWER, '#4+  Lower rank'),
                           (.545, MISSING, '–  Too little dialogue in one version')):
        rounded_box(fig, x, .204, .015, .022, fill, radius=.003)
        text(fig, x + .022, .215, label, 9.5, MAP_MUTED)

    fig.add_artist(plt.Line2D([.073, .929], [.127, .127], transform=fig.transFigure,
                             color=MAP_GRID, linewidth=1))
    text(fig, .073, .093, 'We rank each book character’s film candidates by TF–IDF cosine similarity.', 10.5, MAP_MUTED)
    text(fig, .073, .050, 'Characters need 100+ words in each version. Deathly Hallows combines Parts 1 and 2.', 10.5, MAP_MUTED)
    save(fig, out, FIGURE_NAME)


def spread_points(ax, story_rows):
    """Separate close dots horizontally, keeping their exact cosine values."""
    gap = 7.5 * ax.figure.dpi / 72
    positions = {}
    placed = []
    offsets = [0] + [offset for step in range(1, len(story_rows)) for offset in (step, -step)]
    for row in sorted(story_rows, key=lambda r: (r['own_film_cosine'], r['character'])):
        story, cosine = row['story'], row['own_film_cosine']
        center, y = ax.transData.transform((story, cosine))
        x = next(center + gap * offset for offset in offsets
                 if all((center + gap * offset - px)**2 + (y - py)**2 >= gap**2 - 1e-6
                        for px, py in placed))
        placed.append((x, y))
        positions[row['character']] = (ax.transData.inverted().transform((x, y))[0], cosine)
    return positions


def render_similarity(out, rows, *, figure_name=SIMILARITY_FIGURE_NAME,
                      ylabel='Book-to-film vocabulary similarity', limits=None):
    rows = [row for row in rows if row['character'] in SIMILARITY_CHARACTERS]
    paper, ink, muted = '#fcfaf5', MAP_INK, MAP_MUTED
    stories = range(1, len(SHORT_TITLES) + 1)
    fig = plt.figure(figsize=(13.3333, 5.6), facecolor=paper)
    ax = axis(fig, [.090, .170, .715, .785])
    if limits is None:
        limits = (0, max(.62, max(row['own_film_cosine'] for row in rows) + .045))
    ax.set(xlim=(.54, 7.46), ylim=limits, yticks=np.arange(limits[0], limits[1] + 1e-9, .1))
    ax.set_xticks(list(stories), STORY_TICKS)
    ax.set_ylabel(ylabel, fontsize=12, color=ink, labelpad=12)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.tick_params(axis='y', labelsize=11.5, colors=muted, pad=7)
    ax.tick_params(axis='x', labelsize=11.5, colors=ink, pad=11)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for story in stories:
        ax.axvspan(story - .5, story + .5, color='#f5f3ed' if story % 2 else '#fffdf9', lw=0, zorder=-1)
    ax.grid(axis='y', color=MAP_GRID, linewidth=.7)

    by_character = defaultdict(dict)
    for story in stories:
        story_rows = [row for row in rows if row['story'] == story]
        for name, point in spread_points(ax, story_rows).items():
            by_character[name][story] = point

    for name, (color, marker) in SIMILARITY_CHARACTERS.items():
        locations = by_character.get(name)
        if not locations:
            continue
        points = np.array([locations.get(story, (np.nan, np.nan)) for story in stories])
        ax.plot(*points.T, color=color, marker=marker, linewidth=1.8, markersize=7,
                linestyle='-' if len(locations) > 1 else 'none', markeredgecolor=paper,
                markeredgewidth=.75, label=name, zorder=3)
        for start, end in pairwise(locations):
            if end - start > 1:
                # A dashed segment skips installments without a comparison.
                ax.plot(*zip(locations[start], locations[end]), color=color,
                        linestyle='--', linewidth=1.1, alpha=.75, zorder=2)

    ax.legend(loc='center left', bbox_to_anchor=(1.02, .5), frameon=False, borderaxespad=0,
              fontsize=11, labelcolor=ink, handlelength=2.5, labelspacing=.55)

    save(fig, out, figure_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo', type=Path, help='Repository containing the dialogue CSVs.')
    parser.add_argument('--out', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'results' / 'interaction_language',
                        help='Directory for the figure and per-character ranks.')
    args = parser.parse_args()
    rows = adaptation_ranks(args.repo.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'svg.fonttype': 'path',
                         'svg.hashsalt': 'hp-adaptation-vocabulary'})
    write_rows(args.out / 'adaptation_vocabulary_ranks.csv', rows)
    render(args.out, rows)
    render_similarity(args.out, rows)
    print(f'Saved the adaptation vocabulary figures and ranks to {args.out.resolve()}')


if __name__ == '__main__':
    main()
