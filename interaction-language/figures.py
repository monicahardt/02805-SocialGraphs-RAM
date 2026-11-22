"""Render the character networks and two matched comparisons on 16:9 canvases."""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MultipleLocator, PercentFormatter


BG = '#181723'
PANEL = '#242134'
INK = '#f9f3e9'
MUTED = '#c1b8cc'
GRID = '#423d52'
BOOK = '#80dfc9'
FILM = '#caaeef'
GOLD = '#ffd093'
EDGE = LinearSegmentedColormap.from_list('similarity', ['#484967', '#8982af', GOLD])
SERIF = ['Georgia', 'DejaVu Serif']
MAP_PAPER = '#fbfaf6'
MAP_INK = '#17233b'
MAP_MUTED = '#526177'
MAP_GRID = '#e1e5ea'
MEDIA_STYLE = {'Books': ('#18796f', 'o'), 'Films': ('#7251aa', 'D')}
HIGHLIGHTS = {
    'Harry Potter': ('Harry', (12, 2)),
    'Ron Weasley': ('Ron', (-13, -17)),
    'Hermione Granger': ('Hermione', (12, 8)),
    'Draco Malfoy': ('Draco', (11, -11)),
    'Bellatrix Lestrange': ('Bellatrix', (-12, 7)),
    'Rubeus Hagrid': ('Hagrid', (12, 8)),
    'Voldemort': ('Voldemort', (-14, 13)),
    'Severus Snape': ('Snape', (-14, 7)),
    'Albus Dumbledore': ('Dumbledore', (14, 10)),
}
COMPACT_POINTS = {('Ron Weasley', 'Films'), ('Rubeus Hagrid', 'Books')}
CONNECTOR_BENDS = {'Voldemort': .70, 'Albus Dumbledore': .55}


def text(fig, x, y, value, size, color=INK, **kwargs):
    return fig.text(x, y, value, fontsize=size, color=color, va='center', **kwargs)


def canvas(title, subtitle):
    fig = plt.figure(figsize=(13.3333, 7.5), facecolor=BG)
    text(fig, .064, .922, title, 34, fontfamily=SERIF, weight='bold')
    text(fig, .066, .855, subtitle, 14.5, MUTED)
    return fig


def panel(fig, left, bottom, width, height):
    fig.add_artist(FancyBboxPatch(
        (left, bottom), width, height, boxstyle='round,pad=0.008,rounding_size=0.014',
        transform=fig.transFigure, facecolor=PANEL, edgecolor=GRID, linewidth=1,
        zorder=-1,
    ))


def axis(fig, bounds):
    ax = fig.add_axes(bounds, facecolor='none')
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.tick_params(axis='both', colors=MUTED, labelsize=12, length=0, pad=8)
    ax.set_axisbelow(True)
    return ax


def footer(fig, value):
    paper_line = plt.Line2D([.066, .934], [.086, .086], color=GRID, lw=1, transform=fig.transFigure)
    fig.add_artist(paper_line)
    text(fig, .066, .047, value, 11, MUTED)


def save(fig, out, name):
    fig.savefig(out / f'{name}.png', dpi=210)
    fig.savefig(out / f'{name}.svg', metadata={'Date': None})
    plt.close(fig)


def mark_versions(ax, positions, name, emphasized):
    points = np.array([positions[name, medium] for medium in MEDIA_STYLE])
    if name in CONNECTOR_BENDS:
        # Route the connector around nearby markers while keeping both scores
        # at their recorded positions.
        steps = np.linspace(0, 1, 32)
        points = (1 - steps[:, None]) * points[0] + steps[:, None] * points[1]
        points[:, 0] += 2 * CONNECTOR_BENDS[name] * steps * (1 - steps)
    ax.plot(*points.T, color=MAP_INK if emphasized else '#8392a8',
            lw=1.3 if emphasized else .72, alpha=.94 if emphasized else .42,
            linestyle=(0, (1, 2.2)) if emphasized else (0, (.7, 2.4)),
            dash_capstyle='round', zorder=3 if emphasized else 2)


def draw_profile(ax, position, medium, size, opacity=1., order=4):
    color, marker = MEDIA_STYLE[medium]
    ax.scatter(*position, s=size, color=color, marker=marker,
               edgecolor=MAP_PAPER, linewidth=1, alpha=opacity, zorder=order)


def voice_axes(fig, bounds, positions):
    ax = axis(fig, bounds)
    ax.set_facecolor('#ffffff')
    ax.tick_params(colors=MAP_MUTED)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis='y', color=MAP_GRID, linewidth=.82)
    ax.grid(axis='x', color=MAP_GRID, linewidth=.57, alpha=.65)
    ax.axvline(0, color='#9ba7b8', lw=1, zorder=1)
    x, y = np.array(list(positions.values())).T
    ax.set(xlim=(x.min() - .85, x.max() + .95), ylim=(max(0, y.min() - 3), min(100, y.max() + 3)))
    return ax


def character_voice_map(out, axes):
    positions = {(p['character'], p['medium']): (p['x'], p['y'])
                 for p in axes['coordinates']}
    fig = plt.figure(figsize=(13.3333, 7.5), facecolor=MAP_PAPER)
    text(fig, .071, .923, 'How the characters speak', 35, MAP_INK, fontfamily=SERIF, weight='bold')
    text(fig, .073, .855, f'Book and film dialogue for {len(positions) // 2} characters',
         14.5, MAP_MUTED)
    ax = voice_axes(fig, [.105, .230, .823, .505], positions)
    paired = [
        name
        for name, medium in positions if medium == 'Books' and (name, 'Films') in positions
    ]
    for name in paired:
        mark_versions(ax, positions, name, emphasized=name in HIGHLIGHTS)
    for (name, medium), point in positions.items():
        highlighted = name in HIGHLIGHTS
        size, order = (91, 5) if highlighted else (30, 4)
        if (name, medium) in COMPACT_POINTS:
            size = 47
            order = 5.5 if medium == 'Films' else 5
        draw_profile(ax, point, medium, size,
                     opacity=1 if highlighted else .49, order=order)
    for name, (label, offset) in HIGHLIGHTS.items():
        if name not in paired:
            continue
        align = 'right' if offset[0] < 0 else 'left'
        label_medium = 'Books' if name == 'Ron Weasley' else 'Films'
        ax.annotate(label, positions[name, label_medium], xytext=offset, textcoords='offset points',
                    color=MAP_INK, fontsize=12.2,
                    weight='bold', ha=align, va='center', zorder=6,
                    bbox={'facecolor': '#ffffff', 'edgecolor': 'none', 'pad': 1.8, 'alpha': .92},
                    arrowprops={'arrowstyle': '-', 'color': '#8e98a7', 'lw': .7, 'shrinkB': 4})
    for medium, x_legend in (('Books', .759), ('Films', .848)):
        color, marker = MEDIA_STYLE[medium]
        text(fig, x_legend, .855, f'{"●" if marker == "o" else "◆"}  {medium}', 13, color, weight='bold')
    text(fig, .105, .779, 'Share of directions ↑', 15, MAP_INK, weight='bold')
    text(fig, .305, .779, 'among questions and directions', 11.8, MAP_MUTED)
    count = axes['directive_comparison']['more_directive_in_films']
    text(fig, .928, .779, f'{count} of {len(paired)} characters score higher in films',
         11.8, MAP_MUTED, ha='right')
    text(fig, .105, .155, '←  Hostile', 15, MAP_INK, weight='bold')
    text(fig, .928, .155, 'Supportive  →', 15, MAP_INK, weight='bold', ha='right')
    text(fig, .105, .117, '“how dare you” · “shut up”', 11, MAP_MUTED)
    text(fig, .928, .117, '“don’t worry” · “thank you”', 11, MAP_MUTED, ha='right')
    fig.add_artist(plt.Line2D([.073, .928], [.087, .087], color=MAP_GRID, transform=fig.transFigure))
    text(fig, .073, .062, 'Tone is net supportive minus hostile phrases per 1,000 words.', 10.4, MAP_MUTED)
    text(fig, .073, .032, 'Dotted lines connect the book and film points for each character.', 10.4, MAP_MUTED)
    save(fig, out, '01_switches_and_vocabulary')


def adaptation_comparison(out, stats):
    fig = canvas('The link is stronger in every book',
                 'Do character pairs with more speaker switches tend to share more vocabulary?')
    ax = axis(fig, [.255, .235, .43, .48])
    ax.set_ylim(6.6, -.6)
    ax.set_xlim(-.002, .61)
    ax.set_xticks([0, .2, .4, .6], ['0', '0.2', '0.4', '0.6'])
    ax.set_yticks([])
    ax.xaxis.grid(True, color=GRID, lw=1)
    for y, story in enumerate(stats['stories']):
        book, film = story['score']['Books'], story['score']['Films']
        ax.text(-.052, y, story['title'], transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=14, color=INK)
        ax.plot([film, book], [y, y], color='#8a809d', lw=2.0, zorder=1)
        ax.scatter(book, y, color=BOOK, s=72, zorder=2)
        ax.scatter(film, y, color=FILM, marker='D', s=55, zorder=2)
    text(fig, .47, .149, 'Rank correlation, controlling for each speaker', 13.5, MUTED, ha='center')

    panel(fig, .716, .259, .218, .448)
    text(fig, .737, .659, 'MEAN ACROSS STORIES', 10.5, MUTED, weight='bold')
    text(fig, .739, .56, f'{stats["full_scores"]["Books"]:.2f}', 36, BOOK, weight='bold')
    text(fig, .85, .56, 'Books', 13, BOOK, weight='bold')
    text(fig, .739, .442, f'{stats["full_scores"]["Films"]:.2f}', 36, FILM, weight='bold')
    text(fig, .85, .442, 'Films', 13, FILM, weight='bold')
    text(fig, .74, .327, f'{stats["full_gap"]:+.2f} book–film gap', 13, INK)
    footer(fig, f'{stats["unique_speakers"]} characters across seven stories. The same '
                'cast is compared in each book and its adaptation.')
    save(fig, out, '02_matched_books_and_films')


def compression_benchmark(out, draws, stats):
    pct = round(100 * stats['fraction_gap_reduced'])
    fig = canvas('Equal word counts shrink the gap',
                 'We randomly trim the longer dialogue profile for each character to match the shorter one.')
    ax = axis(fig, [.111, .228, .573, .485])
    gap_draws = draws[:, 0] - draws[:, 1]
    median = stats['equal_length_median_gap']
    actual = stats['full_gap']
    low, high = stats['equal_length_middle_95_gap']
    start = min(-.068, .01 * np.floor(gap_draws.min() / .01) - .01)
    stop = max(.211, actual + .038, gap_draws.max() + .03)
    bins = np.arange(start, stop + .01, .01)
    frequencies, _, bars = ax.hist(gap_draws, bins=bins, edgecolor=BG, linewidth=1.4)
    top = max(frequencies)
    for bar, mass in zip(bars, frequencies):
        bar.set_facecolor(EDGE(.38 + .49 * mass / top))
    ax.set(xlim=(start, stop), ylim=(0, top * 1.29))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_major_locator(MultipleLocator(.05))
    ax.xaxis.set_major_formatter(lambda x, _: '0' if abs(x) < .001 else f'{x:+.2f}'.replace('-', '−'))
    ax.yaxis.grid(True, color=GRID, lw=.8)
    ax.axvline(0, color=MUTED, alpha=.62, lw=1, linestyle='--')
    for x, label, color in ((median, 'Matched length', GOLD), (actual, 'Full dialogue', BOOK)):
        ax.vlines(x, 0, top * 1.01, color=color, lw=2.2)
        ax.text(x, top * 1.077, f'{x:+.2f}  {label}', ha='center', va='bottom',
                fontsize=11.5, color=color, weight='bold')
    text(fig, .112, .744, 'Number of samples', 12, MUTED)
    text(fig, .398, .149, 'Difference in rank correlation (books minus films)', 13, MUTED, ha='center')

    panel(fig, .716, .257, .218, .451)
    text(fig, .739, .576, f'{pct}%', 57, GOLD, fontfamily=SERIF, weight='bold')
    text(fig, .74, .459, 'smaller gap', 20, INK, weight='bold')
    text(fig, .74, .383, 'at the median of', 13, MUTED)
    text(fig, .74, .340, f'{len(draws):,} matched samples', 13, MUTED)
    footer(fig, 'Interaction counts and IDF weights stay fixed. '
                f'The middle 95% of sampled gaps runs from {low:+.2f} to {high:+.2f}.'.replace('-', '−'))
    save(fig, out, '03_equal_dialogue_length')


def render(out, draws, stats):
    plt.rcParams.update({
        'font.family': ['Segoe UI', 'DejaVu Sans'],
        'svg.fonttype': 'path',
        'svg.hashsalt': 'harry-potter-dialogue',
        'text.color': INK,
    })
    character_voice_map(out, stats['language_axes'])
    adaptation_comparison(out, stats)
    compression_benchmark(out, draws, stats)
