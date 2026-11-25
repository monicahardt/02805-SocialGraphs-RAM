"""Plot the matched MiniLM dialogue profiles in a shared PaCMAP or t-SNE projection."""

import argparse
from pathlib import Path

import numpy as np
from adjustText import adjust_text
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MaxNLocator
from sklearn.manifold import TSNE

from adaptation_semantics import mean_cast_profiles, selected_cast_profiles, top_matched_profiles
from analyse import read_project, top_book_characters
from figures import SERIF, axis, plt, save, text
from final_slide import BASELINE, BOOK, FILM, GRID, INK, PAPER, SECONDARY
from slide_2_options import MEDIA, read_results


ROOT = Path(__file__).resolve().parents[1]
FIGURES = {
    'pacmap': ('PaCMAP', '08_bert_pacmap_books_and_films'),
    'tsne': ('t-SNE', '09_bert_tsne_books_and_films'),
}
MAX_EDGE_WIDTH = 5.2


def project_profiles(names, profiles, projection):
    """Average matched stories for each character, then fit one map for both media."""
    means = mean_cast_profiles(names, profiles)
    vectors = np.ascontiguousarray(means.reshape(-1, means.shape[-1]), dtype=np.float32)

    if projection == 'tsne':
        reducer = TSNE(n_components=2, perplexity=min(10, len(vectors) - 1), init='pca',
                       learning_rate='auto', method='exact', random_state=42)
        points = reducer.fit_transform(vectors)
    else:
        import pacmap

        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=5, MN_ratio=.5, FP_ratio=1,
                                knn_backend='faiss', random_state=42)
        points = reducer.fit_transform(vectors, init='pca')
    return points.reshape(len(MEDIA), len(names), 2)


def sum_exchanges(names, stories, matrices):
    """Sum each selected pair's observed exchanges over its matched stories."""
    index = {name: i for i, name in enumerate(names)}
    totals = {medium: np.zeros((len(names), len(names))) for medium in MEDIA}
    for story, data in zip(stories, matrices, strict=True):
        cast = story['characters']
        source = [i for i, name in enumerate(cast) if name in index]
        target = [index[cast[i]] for i in source]
        for medium in MEDIA:
            totals[medium][np.ix_(target, target)] += data[medium][0][np.ix_(source, source)]
    return totals


def draw_exchanges(ax, points, exchanges, max_count, color):
    """Draw one edge per active pair on the shared square-root width scale."""
    first, second = np.triu_indices_from(exchanges, k=1)
    counts = exchanges[first, second]
    active = counts > 0
    segments = points[np.column_stack((first[active], second[active]))]
    widths = MAX_EDGE_WIDTH * np.sqrt(counts[active] / max_count)
    ax.add_collection(LineCollection(segments, colors=color, linewidths=widths,
                                     alpha=.30, capstyle='round', zorder=1))


def render(out, names, points, story_count, projection, *, exchanges=None, by_book_words=False):
    """Draw both media on identical axes and label the character positions."""
    label, filename = FIGURES[projection]
    if by_book_words:
        filename += '_top_20'
        subtitle = f'MiniLM embeddings · Top {len(names)} matched characters by book dialogue word count'
    else:
        subtitle = f'MiniLM embeddings · Top {len(names)} by dialogue lines across {story_count} matched book and film stories'
    if exchanges is not None:
        filename += '_with_exchanges'
        max_count = max(1, max(matrix.max() for matrix in exchanges.values()))
    fig = plt.figure(figsize=(16, 9), facecolor=PAPER)
    text(fig, .055, .936, f'Character dialogue in a shared {label} projection', 29, INK, fontfamily=SERIF)
    text(fig, .055, .881, subtitle, 13, SECONDARY)

    lower = points.min(axis=(0, 1))
    upper = points.max(axis=(0, 1))
    padding = .14 * (upper - lower)
    x_limits, y_limits = zip(lower - padding, upper + padding)
    for medium, coords, color, left in zip(MEDIA, points, (BOOK, FILM), (.055, .525), strict=True):
        width = .420
        text(fig, left + width / 2, .802, medium, 16, color, ha='center', weight='bold')
        ax = axis(fig, [left, .165, width, .600])
        ax.set(xlim=x_limits, ylim=y_limits, xlabel=f'{label} 1', aspect='equal')
        if medium == 'Books':
            ax.set_ylabel(f'{label} 2', labelpad=9)
        ax.xaxis.label.set_color(SECONDARY)
        ax.yaxis.label.set_color(SECONDARY)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(colors=SECONDARY, labelsize=9, pad=4)
        ax.grid(color=GRID, linewidth=.65)
        x, y = coords.T
        if exchanges is not None:
            draw_exchanges(ax, coords, exchanges[medium], max_count, color)
        ax.scatter(x, y, s=26, color=color, edgecolor=PAPER, linewidth=.55, zorder=2)
        labels = [ax.text(a, b, name, fontsize=8.5, color=INK, ha='center', va='center', zorder=3)
                  for name, (a, b) in zip(names, coords, strict=True)]
        if exchanges is not None:
            for name_label in labels:
                name_label.set_path_effects([withStroke(linewidth=2, foreground=PAPER)])
        adjust_text(labels, ax=ax, target_x=x, target_y=y, iter_lim=350,
                    expand=(1.06, 1.18), min_arrow_len=2,
                    arrowprops={'arrowstyle': '-', 'color': color, 'alpha': .40, 'linewidth': .50})

    fig.add_artist(plt.Line2D([.055, .945], [.112, .112], transform=fig.transFigure,
                             color=BASELINE, linewidth=.7))
    if exchanges is None:
        text(fig, .055, .072,
             'Each point averages a character’s embeddings across their matched stories. '
             'Nearby points tend to have similar dialogue embeddings.', 10.5, SECONDARY)
    else:
        text(fig, .055, .077, 'Positions reflect similarity in MiniLM dialogue embeddings.', 10.5, SECONDARY)
        text(fig, .055, .050, 'Line thickness represents back-to-back dialogue exchanges across matched stories.',
             10.5, SECONDARY)
        keys = [count for count in (10, 100, 1_000) if count <= max_count] or [1]
        samples = [Line2D([], [], color=SECONDARY, alpha=.55, solid_capstyle='round',
                          linewidth=MAX_EDGE_WIDTH * np.sqrt(count / max_count), label=f'{count:,}')
                   for count in keys]
        fig.legend(handles=samples, title='Dialogue exchanges', loc='center right',
                   bbox_to_anchor=(.945, .065), ncol=len(samples), frameon=False, labelcolor=INK)
    save(fig, out, filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results' / 'interaction_language')
    parser.add_argument('--model-dir', type=Path, help='Local copy of the pinned MiniLM checkpoint.')
    parser.add_argument('--projection', choices=FIGURES, default='pacmap')
    parser.add_argument('--show-exchanges', action='store_true', help='Draw lines scaled by observed exchanges.')
    parser.add_argument('--top-20-book-words', action='store_true', help='Select the top 20 by book dialogue word count.')
    args = parser.parse_args()

    stats, matrices = read_results(args.results_dir)
    stories = stats['stories']
    blocks, counts, _, _ = read_project(ROOT)
    if args.top_20_book_words:
        names = top_book_characters(stories, counts)
        profiles = selected_cast_profiles(blocks, stories, names, args.model_dir)
    else:
        names, profiles = top_matched_profiles(blocks, stories, args.model_dir)
    points = project_profiles(names, profiles, args.projection)
    exchanges = sum_exchanges(names, stories, matrices) if args.show_exchanges else None
    with plt.rc_context({'font.family': ['Segoe UI', 'DejaVu Sans'], 'svg.fonttype': 'path',
                         'svg.hashsalt': FIGURES[args.projection][1]}):
        render(args.results_dir, names, points, len(stories), args.projection,
               exchanges=exchanges, by_book_words=args.top_20_book_words)


if __name__ == '__main__':
    main()
