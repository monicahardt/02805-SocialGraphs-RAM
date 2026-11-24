"""Compare each character's book dialogue with the cast of its film adaptation."""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import rankdata
from sklearn.preprocessing import normalize

from analyse import MIN_WORDS, SHORT_TITLES, read_project, settings_result, write_rows
from figures import MAP_GRID, MAP_INK, MAP_MUTED, MEDIA_STYLE, SERIF, axis, plt, save, text
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
TRACKED_CHARACTERS = {
    'Harry Potter': ('Harry', '#08786f', 'o', -.13),
    'Hermione Granger': ('Hermione', '#6946a4', 'D', 0),
    'Ron Weasley': ('Ron', '#af551f', 's', .13),
    'Voldemort': ('Voldemort', '#a63d56', '^', -.13),
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
    positions = {row['character']: (row['story'] + TRACKED_CHARACTERS[row['character']][3],
                                    row['own_film_cosine'])
                 for row in story_rows if row['character'] in TRACKED_CHARACTERS}
    placed = [ax.transData.transform(position) for position in positions.values()]
    offsets = [0] + [offset for step in range(1, len(story_rows)) for offset in (step, -step)]
    for row in sorted(story_rows, key=lambda r: r['own_film_cosine']):
        if row['character'] in TRACKED_CHARACTERS:
            continue
        story, cosine = row['story'], row['own_film_cosine']
        center, y = ax.transData.transform((story, cosine))
        x = next(center + gap * offset for offset in offsets
                 if all((center + gap * offset - px)**2 + (y - py)**2 >= gap**2 - 1e-6
                        for px, py in placed))
        placed.append((x, y))
        positions[row['character']] = (ax.transData.inverted().transform((x, y))[0], cosine)
    return positions


def render_similarity(out, rows):
    paper, ink, muted = '#fcfaf5', MAP_INK, MAP_MUTED
    gray = '#7890a7'
    by_character = {(row['story'], row['character']): row for row in rows}
    stories = range(1, len(SHORT_TITLES) + 1)
    first = sum(row['own_film_rank'] == 1 for row in rows)
    fig = plt.figure(figsize=(13.3333, 7.5), facecolor=paper)
    text(fig, .071, .923, 'Whose words carry over to film?', 34, ink, fontfamily=SERIF, weight='bold')
    text(fig, .073, .851, 'Each dot compares the same character’s dialogue in a book and its film adaptation.', 13.5, muted)
    text(fig, .939, .920, f'{first} of {len(rows)}', 30, ink, ha='right', weight='bold')
    text(fig, .939, .855, 'closest match is the same character', 11, muted, ha='right')

    text(fig, .073, .760, 'Book-to-film vocabulary similarity', 13.5, ink, weight='bold')
    fig.add_artist(plt.Line2D([.818], [.760], marker='o', markersize=5.5, transform=fig.transFigure,
                             linestyle='none', markeredgecolor='none', markerfacecolor=gray, alpha=.43))
    text(fig, .831, .760, 'Other characters', 11, muted)

    ax = axis(fig, [.102, .225, .809, .497])
    top = max(.62, max(row['own_film_cosine'] for row in rows) + .045)
    ax.set(xlim=(.54, 7.46), ylim=(0, top), yticks=np.arange(0, top, .1))
    ax.set_xticks(list(stories), STORY_TICKS)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.tick_params(axis='y', labelsize=11.5, colors=muted, pad=7)
    ax.tick_params(axis='x', labelsize=11.5, colors=ink, pad=11)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for story in stories:
        ax.axvspan(story - .5, story + .5, color='#f5f3ed' if story % 2 else '#fffdf9', lw=0, zorder=-1)
    ax.grid(axis='y', color=MAP_GRID, linewidth=.7)

    locations = {}
    for story in stories:
        story_rows = [row for row in rows if row['story'] == story]
        positions = spread_points(ax, story_rows)
        locations.update({(story, name): location for name, location in positions.items()})

    background = np.array([point for (_, name), point in locations.items() if name not in TRACKED_CHARACTERS])
    if background.size:
        ax.scatter(*background.T, s=29, c=gray, alpha=.38, linewidths=0, zorder=2)

    label_offsets = {'Harry Potter': .038, 'Ron Weasley': -.002, 'Hermione Granger': -.026, 'Voldemort': 0}
    for name, (label, color, marker, _) in TRACKED_CHARACTERS.items():
        points = np.array([locations.get((story, name), (np.nan, np.nan)) for story in stories])
        valid = points[np.isfinite(points[:, 0])]
        if not len(valid):
            continue
        if name != 'Voldemort':
            ax.plot(*points.T, color=color, linewidth=2.3, zorder=3,
                    path_effects=[withStroke(linewidth=4.6, foreground=paper)])
        ax.scatter(*points.T, c=color, s=83, marker=marker, edgecolors=paper, linewidths=1.15, zorder=4)
        x, y = valid[-1]
        ax.annotate(label, xy=(x, y), xytext=(7.49, y + label_offsets[name]), color=color,
                    va='center', fontsize=12, weight='bold', zorder=5,
                    arrowprops={'arrowstyle': '-', 'color': color, 'lw': .95, 'shrinkA': 5, 'shrinkB': 5})

    voldemort = by_character.get((6, 'Voldemort'))
    if voldemort:
        x, y = locations[6, 'Voldemort']
        ax.annotate(f'Rank {voldemort["own_film_rank"]:g} of {voldemort["film_candidates"]}',
                    xy=(x, y), xytext=(x + .16, y + .036), color=TRACKED_CHARACTERS['Voldemort'][1],
                    va='center', fontsize=10.5, weight='bold', zorder=5,
                    arrowprops={'arrowstyle': '-', 'color': TRACKED_CHARACTERS['Voldemort'][1],
                                'lw': .8, 'shrinkA': 4, 'shrinkB': 5})

    fig.add_artist(plt.Line2D([.073, .939], [.115, .115], transform=fig.transFigure, color=MAP_GRID, linewidth=1))
    text(fig, .073, .070, 'Similarity uses TF–IDF and cosine. Deathly Hallows combines both films.', 11, muted)
    save(fig, out, SIMILARITY_FIGURE_NAME)


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
