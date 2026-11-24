"""Show dialogue ties on the language map, with a slide and an interactive view."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import StrMethodFormatter

from analyse import MEDIUMS, write_rows
from figures import HIGHLIGHTS, MAP_GRID, MAP_INK, MAP_MUTED, MAP_PAPER, MEDIA_STYLE, SERIF, draw_profile, plt, save, text, voice_axes


FIGURE = '05_dialogue_network_by_language'
FEATURED_COUNT = 5
LABEL_OFFSETS = {
    ('Ron Weasley', 'Books'): (-13, -13),
    ('Hermione Granger', 'Books'): (0, -26),
    ('Harry Potter', 'Books'): (14, -12),
    ('Albus Dumbledore', 'Books'): (-23, 3),
    ('Rubeus Hagrid', 'Books'): (-40, 24),
    ('Ron Weasley', 'Films'): (-9, 8),
    ('Hermione Granger', 'Films'): (3, 14),
    ('Harry Potter', 'Films'): (14, 0),
    ('Albus Dumbledore', 'Films'): (12, 5),
    ('Rubeus Hagrid', 'Films'): (13, 6),
}


def networks(results):
    stats = json.loads((results / 'statistics.json').read_text(encoding='utf-8'))
    positions = {(row['character'], row['medium']): (row['x'], row['y'])
                 for row in stats['language_axes']['coordinates']}
    counts = {medium: Counter() for medium in MEDIUMS}
    with (results / 'matched_pairs.csv').open(encoding='utf-8', newline='') as source:
        for row in csv.DictReader(source):
            pair = tuple(sorted((row['character_a'], row['character_b'])))
            for medium in MEDIUMS:
                exchanges = int(row[f'{medium.lower()}_switches'])
                if exchanges:
                    counts[medium][pair] += exchanges
    return positions, counts


def short_name(name):
    return HIGHLIGHTS[name][0] if name in HIGHLIGHTS else name


def network_rows(counts):
    rows = []
    for medium, weights in counts.items():
        total = sum(weights.values())
        ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
        strongest = {}
        for (left, right), count in ranked:
            strongest.setdefault(left, count)
            strongest.setdefault(right, count)
        for rank, ((left, right), count) in enumerate(ranked, 1):
            featured = rank <= FEATURED_COUNT
            rows.append(dict(medium=medium, character_a=left, character_b=right, exchanges=count,
                             share_of_medium_exchanges=count / total, featured=featured,
                             shown=featured or count in (strongest[left], strongest[right])))
    return rows


def line_width(share):
    return .45 + 23 * share


def render(out, positions, edges):
    featured = [e for e in edges if e['featured']]
    context = [e for e in edges if e['shown'] and not e['featured']]
    highlighted = {(e[end], e['medium']) for e in featured for end in ('character_a', 'character_b')}
    top_pairs = [{(e['character_a'], e['character_b']) for e in featured if e['medium'] == medium}
                 for medium in MEDIUMS]
    same_top = top_pairs[0] == top_pairs[1] and len(top_pairs[0]) == FEATURED_COUNT
    title = 'The same five ties lead in books and films' if same_top else 'The strongest ties in each dialogue network'
    subtitle = 'The language map connects characters who exchange turns.'
    fig = plt.figure(figsize=(13.3333, 7.5), facecolor=MAP_PAPER)
    text(fig, .071, .923, title, 33, MAP_INK, fontfamily=SERIF, weight='bold')
    text(fig, .073, .853, subtitle, 13.5, MAP_MUTED)

    text(fig, .101, .778, 'Directions per 100 questions and directions ↑', 13.5, MAP_INK, weight='bold')
    ax = voice_axes(fig, [.101, .238, .824, .503], positions)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.grid(color='#e8ecf0', linewidth=.72)
    for x, medium in zip((.734, .832), MEDIUMS):
        color, marker = MEDIA_STYLE[medium]
        text(fig, x, .853, f'{"●" if marker == "o" else "◆"}  {medium}', 13, color, weight='bold')

    text(fig, .496, .778, 'Line width · share of exchanges', 10.7, MAP_INK, weight='bold')
    for x, share in zip((.711, .786, .863), (.01, .05, .15)):
        fig.add_artist(plt.Line2D([x, x + .026], [.778, .778], transform=fig.transFigure,
                                 color=MAP_MUTED, linewidth=line_width(share), solid_capstyle='round'))
        text(fig, x + .034, .778, f'{share:.0%}', 10, MAP_MUTED)

    for edge in reversed(context):
        medium = edge['medium']
        (x1, y1), (x2, y2) = (positions[edge[key], medium] for key in ('character_a', 'character_b'))
        ax.plot([x1, x2], [y1, y2], color=MEDIA_STYLE[medium][0], alpha=.21,
                linewidth=line_width(edge['share_of_medium_exchanges']), zorder=2, solid_capstyle='round')

    for (name, medium), position in positions.items():
        if (name, medium) not in highlighted:
            draw_profile(ax, position, medium, size=29, opacity=.48, order=3)
    for edge in reversed(featured):
        medium = edge['medium']
        pair = edge['character_a'], edge['character_b']
        left, right = sorted(positions[name, medium] for name in pair)
        width = line_width(edge['share_of_medium_exchanges'])
        # Separate the longest book edge from the two nearly collinear book edges.
        bend = .14 if medium == 'Books' and pair == ('Harry Potter', 'Ron Weasley') else 0
        arrow = FancyArrowPatch(left, right, arrowstyle='-', shrinkA=0, shrinkB=0, capstyle='round',
                                connectionstyle=f'arc3,rad={bend}', color=MEDIA_STYLE[medium][0],
                                linewidth=width, alpha=.88, zorder=4)
        ax.add_patch(arrow)
    for name, medium in sorted(highlighted):
        point = positions[name, medium]
        color, _ = MEDIA_STYLE[medium]
        draw_profile(ax, point, medium, size=85, order=5)
        offset = LABEL_OFFSETS.get((name, medium), (9, 10))
        ax.annotate(short_name(name), point, xytext=offset, textcoords='offset points',
                    ha='right' if offset[0] < 0 else 'left', va='center', color=color,
                    fontsize=11.2, weight='bold', zorder=6,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.6, 'alpha': .96},
                    arrowprops={'arrowstyle': '-', 'color': color, 'lw': .75, 'shrinkB': 5})

    film_context = sorted((key for key in positions if key[1] == 'Films' and key not in highlighted),
                          key=lambda key: positions[key][0])
    for key, dy in zip(film_context[:2], (-11, 10)):
        ax.annotate(short_name(key[0]), positions[key], xytext=(9, dy), textcoords='offset points',
                    color=MEDIA_STYLE['Films'][0], fontsize=10.4, ha='left', va='center', zorder=6,
                    arrowprops={'arrowstyle': '-', 'color': MEDIA_STYLE['Films'][0], 'lw': .6, 'shrinkB': 3})

    box = FancyBboxPatch((.023, .680), .278, .278, boxstyle='round,pad=.009,rounding_size=.016',
                         transform=ax.transAxes, facecolor='#f7f9f6', edgecolor='#e8ece5', linewidth=.85, zorder=6)
    ax.add_patch(box)
    summary_label = 'Top five pairs’ share of exchanges'
    ax.text(.041, .903, summary_label, transform=ax.transAxes, color=MAP_INK,
            fontsize=11, weight='bold', va='center', zorder=7)
    for x, medium in zip((.041, .176), MEDIUMS):
        share = sum(e['share_of_medium_exchanges'] for e in featured if e['medium'] == medium)
        color, _ = MEDIA_STYLE[medium]
        ax.text(x, .812, f'{share:.1%}', transform=ax.transAxes, color=color,
                fontsize=22, weight='bold', va='center', zorder=7)
        ax.text(x, .725, medium, transform=ax.transAxes, color=MAP_MUTED,
                fontsize=10.5, va='center', zorder=7)

    text(fig, .101, .169, '← More hostile', 13, MAP_INK, weight='bold')
    text(fig, .514, .169, 'Supportive minus hostile phrases per 1,000 words', 10.7, MAP_MUTED, ha='center')
    text(fig, .925, .169, 'More supportive →', 13, MAP_INK, weight='bold', ha='right')
    fig.add_artist(plt.Line2D([.073, .932], [.124, .124], transform=fig.transFigure, color=MAP_GRID))
    text(fig, .073, .092, 'Bold lines mark the top five pairs per medium. Faint lines show each character’s strongest tie.',
         10.5, MAP_MUTED)
    cast = len({name for name, _ in positions})
    text(fig, .932, .092, f'{cast} characters · {len(context) + len(featured)} of {len(edges)} links shown',
         10.5, MAP_MUTED, ha='right')
    text(fig, .073, .049, 'A vertical score of 20 means about 20 directions and 80 questions.', 10.5, MAP_MUTED)
    text(fig, .932, .049, 'A 5% line means 5 of every 100 exchanges in that medium.', 10.5, MAP_MUTED, ha='right')
    save(fig, out, FIGURE)


def render_interactive(out, positions, edges):
    nodes = [dict(name=name, label=short_name(name), medium=medium, x=x, y=y)
             for (name, medium), (x, y) in positions.items()]
    data = json.dumps(dict(nodes=nodes, edges=edges), separators=(',', ':')).replace('<', r'\u003c')
    template = Path(__file__).with_name('dialogue_network_view.html').read_text(encoding='utf-8')
    (out / f'{FIGURE}.html').write_text(template.replace('NETWORK_PAYLOAD', data), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'results' / 'interaction_language',
                        help='Directory containing the outputs from analyse.py.')
    parser.add_argument('--out', type=Path, help='Output directory. Defaults to --results-dir.')
    args = parser.parse_args()
    positions, counts = networks(args.results_dir)
    edges = network_rows(counts)
    out = args.out or args.results_dir
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'svg.fonttype': 'path',
                         'svg.hashsalt': 'hp-dialogue-language-network'})
    render(out, positions, edges)
    render_interactive(out, positions, edges)
    write_rows(out / 'dialogue_network_edges.csv', edges)
    print(f'Saved the dialogue network views and edge list to {out.resolve()}')


if __name__ == '__main__':
    main()
