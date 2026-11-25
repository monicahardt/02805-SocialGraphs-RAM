"""Render closing-slide comparisons from saved book and film results."""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, PercentFormatter

from analyse import read_project, top_book_characters, unit_residuals
from figures import SERIF, axis, plt, save, text
from slide_2_options import MEDIA, read_results


FIGURE_NAME = '06_interaction_and_vocabulary_final_slide'
PERCENTAGE_FIGURE_NAME = '06_interaction_and_vocabulary_percentage_drop'
FIGURE_NAMES = {
    'correlation': FIGURE_NAME,
    'percentage': PERCENTAGE_FIGURE_NAME,
    'percentage_labels': '06_interaction_and_vocabulary_percentage_overlay',
    'tfidf_matrix': '07_all_character_tfidf_similarity_matrices',
    'bert_matrix': '07_all_character_bert_similarity_matrices',
    'bert_top_30_matrix': '07_top_30_bert_similarity_matrices',
    'tfidf_top_20_matrix': '07_top_20_tfidf_similarity_matrices',
    'bert_top_20_matrix': '07_top_20_bert_similarity_matrices',
}
PAPER = '#faf8f2'
INK = '#202c2c'
SECONDARY = '#4b5653'
GRID = '#deded4'
BASELINE = '#86918d'
BOOK = '#20665e'
BOOK_BAR = '#91bcb1'
FILM = '#754566'
MARKER = '#8b4a34'
PAIR_LABELS = {
    ('Harry Potter', 'Ron Weasley'): 'Harry–Ron',
    ('Harry Potter', 'Hermione Granger'): 'Harry–Hermione',
    ('Hermione Granger', 'Ron Weasley'): 'Ron–Hermione',
}
PAIR_POSITIONS = (.145, .479, .835)


def pair_contributions(story, matrices):
    """Extract the three pairs' contributions to the book-minus-film correlation."""
    names = story['characters']
    n = len(names)
    first, second = np.triu_indices(n, k=1)
    indices = {tuple(sorted((names[a], names[b]))): k for k, (a, b) in enumerate(zip(first, second))}
    contributions = {}
    for medium, (exchanges, similarity) in matrices.items():
        exchange_ranks = unit_residuals(exchanges[first, second], first, second, n)
        vocabulary_ranks = unit_residuals(similarity[first, second], first, second, n)
        contributions[medium] = exchange_ranks * vocabulary_ranks
    change = contributions['Books'] - contributions['Films']
    return [float(change[indices[pair]]) if pair in indices else 0 for pair in PAIR_LABELS]


def pair_cosines(story, matrices):
    """Return each pair's raw TF–IDF cosine similarity in books and films."""
    indices = {name: i for i, name in enumerate(story['characters'])}
    return [
        [matrices[medium][1, indices[a], indices[b]] for medium in MEDIA]
        if a in indices and b in indices else [np.nan] * len(MEDIA)
        for a, b in PAIR_LABELS
    ]


def mean_pair_cosines(cosines):
    """Average each pair and medium across the stories where it is available."""
    return np.ma.masked_invalid(cosines).mean(axis=0).filled(np.nan)


def cast_mean_cosines(stories, matrices, *, cast=None):
    """Average matched cosine scores and order characters by total exchanges."""
    names = sorted({name for story in stories for name in story['characters']})
    index = {name: i for i, name in enumerate(names)}
    shape = (len(names), len(names))
    counts = np.zeros(shape, dtype=int)
    totals = np.zeros((len(MEDIA), *shape))
    exchange_counts = np.zeros(len(names))
    for story, data in zip(stories, matrices, strict=True):
        positions = [index[name] for name in story['characters']]
        cells = np.ix_(positions, positions)
        counts[cells] += 1
        for i, medium in enumerate(MEDIA):
            exchanges, similarities = data[medium]
            totals[i][cells] += similarities
            exchange_counts[positions] += exchanges.sum(axis=1)
    scores = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
    for matrix in scores:
        np.fill_diagonal(matrix, 1)
    order = np.argsort(-exchange_counts, kind='stable') if cast is None else [index[name] for name in cast]
    return [names[i] for i in order], scores[:, order][:, :, order]


def render_similarity_matrices(out, names, mean_scores, story_count, *, mode):
    """Draw the book and film cosine matrices on a common scale."""
    bert = mode.startswith('bert')
    book_ranked = mode in ('tfidf_top_20_matrix', 'bert_top_20_matrix')
    compact_cast = book_ranked or mode == 'bert_top_30_matrix'
    title = 'BERT cosine similarity' if bert else 'TF–IDF cosine similarity'
    lower_limit = -1 if np.nanmin(mean_scores) < 0 else 0
    fig = plt.figure(figsize=(30, 19.5), facecolor=PAPER)
    text(fig, .040, .957, title, 38, INK, fontfamily=SERIF)
    if book_ranked:
        subtitle = f'Top {len(names)} matched characters by total book dialogue word count'
    elif compact_cast:
        subtitle = f'Top {len(names)} by dialogue lines across {story_count} matched book and film stories'
    else:
        subtitle = f'{len(names)} matched characters across {story_count} stories · Ordered by total book and film exchanges'
    text(fig, .040, .921, subtitle, 17, SECONDARY)

    width = .373
    height = width * fig.get_figwidth() / fig.get_figheight()
    for medium, scores, left, color, light_color in zip(
        MEDIA, mean_scores, (.100, .587), (BOOK, FILM), ('#edf5f1', '#f6eff4'), strict=True
    ):
        center = left + width / 2
        text(fig, center, .855, medium, 24, color, ha='center', weight='bold')
        ax = axis(fig, [left, .249, width, height])
        palette = LinearSegmentedColormap.from_list(medium, (light_color, color))
        palette.set_bad('#dbdcd8')
        heatmap = ax.imshow(scores, cmap=palette, vmin=lower_limit, vmax=1, interpolation='nearest')
        ax.set_xticks(range(len(names)), names, rotation=90)
        ax.set_yticks(range(len(names)), names)
        ax.tick_params(axis='both', which='both', length=0, labelsize=10 if compact_cast else 7.5, colors=INK, pad=4)
        boundaries = np.arange(len(names) - 1) + .5
        ax.set_xticks(boundaries, minor=True)
        ax.set_yticks(boundaries, minor=True)
        ax.grid(which='minor', color=PAPER, linewidth=.35)
        for (i, j), score in np.ndenumerate(scores):
            if np.isfinite(score):
                ax.text(j, i, f'{score:.3f}', va='center', ha='center', fontsize=10 if book_ranked else 8 if compact_cast else 4.7,
                        color=PAPER if heatmap.norm(score) >= .65 else INK)

        scale = axis(fig, [center - .105, .133, .210, .011])
        ticks = [lower_limit, (lower_limit + 1) / 2, 1]
        legend = fig.colorbar(heatmap, cax=scale, orientation='horizontal', ticks=ticks)
        legend.ax.tick_params(length=0, labelsize=11, colors=SECONDARY, pad=5)
        legend.outline.set_visible(False)

    fig.add_artist(plt.Line2D([.040, .960], [.091, .091], transform=fig.transFigure,
                             color=BASELINE, linewidth=.7))
    if bert and compact_cast:
        note = 'Each cell compares two character profiles averaged over their own matched stories. All pairs are included.'
    else:
        note = 'Scores average the stories where both characters are matched. Gray cells have no shared matched story.'
    text(fig, .040, .063 if bert else .060, note, 13, SECONDARY)
    if bert:
        text(fig, .040, .038, 'MiniLM embeddings retain stop words and character names in dialogue.', 13, SECONDARY)
    save(fig, out, FIGURE_NAMES[mode])


def draw_pair_columns(fig, changes, bounds, y_limits, *, percentage=False):
    """Plot the endpoints of each pair's signed contribution on a common scale."""
    x, bottom, width, height = bounds
    extent = max(12 if percentage else .03, np.max(np.abs(changes)) * 1.12)
    rows = np.arange(len(changes))
    column_width = .27 * width
    heading = ('Pair contributions (percentage points)' if percentage
               else 'Pair contributions to the book–film gap')
    text(fig, x + width / 2, bottom + height + .069, heading, 11.5, INK, ha='center', weight='bold')
    text(fig, x + width / 2, bottom - .049, '← Narrows the gap     Widens the gap →', 9.5, SECONDARY, ha='center')
    for col, (position, label) in enumerate(zip(PAIR_POSITIONS, PAIR_LABELS.values(), strict=True)):
        center = x + position * width
        text(fig, center, bottom + height + .030, label, 10.5, MARKER, ha='center', weight='bold')
        ax = axis(fig, [center - column_width / 2, bottom, column_width, height])
        ax.set(xlim=(-extent, extent), ylim=y_limits, yticks=[])
        if percentage:
            ax.set_xticks([-10, 0, 10], ['−10', '0', '+10'])
        else:
            ax.set_xticks([-.02, 0, .02], ['−0.02', '0', '+0.02'])
        ax.tick_params(axis='x', colors=SECONDARY, labelsize=8.5, pad=6)
        ax.grid(axis='x', color=GRID, linewidth=.55)
        ax.axvline(0, color=BASELINE, linewidth=.85, zorder=1)
        ax.hlines(rows, 0, changes[:, col], color=MARKER, linewidth=1.6, zorder=2)
        ax.scatter(changes[:, col], rows, s=26, color=MARKER, edgecolor=PAPER, linewidth=.55, zorder=3)


def render(out, stats, matrices, *, mode='correlation'):
    percentage = mode == 'percentage'
    percentage_labels = mode == 'percentage_labels'
    stories = sorted(stats['stories'], key=lambda row: row['story'])
    row_labels = [row['title'] for row in stories]
    scores = np.array([(row['score']['Books'], row['score']['Films']) for row in stories])
    story_matrices = {row['story']: data for row, data in zip(stats['stories'], matrices, strict=True)}
    if percentage_labels:
        row_labels.append(f'Mean across {len(stories)} stories')
        scores = np.vstack((scores, scores.mean(axis=0)))
        cosines = np.array([pair_cosines(row, story_matrices[row['story']]) for row in stories])
        mean_cosines = mean_pair_cosines(cosines)
        cosines = np.concatenate((cosines, mean_cosines[None, :, :]))
    else:
        pair_changes = np.array([pair_contributions(row, story_matrices[row['story']]) for row in stories])
    if percentage or percentage_labels:
        if np.any(scores[:, 0] <= 0):
            raise ValueError('Percentage drops require positive book correlation scores.')
        scale = 100 / scores[:, 0]
        percentage_drops = (scores[:, 0] - scores[:, 1]) * scale
    if percentage:
        bar_values = percentage_drops
        pair_changes *= scale[:, None]
    else:
        bar_values = scores
    y_limits = (len(row_labels) - .48, -.52)
    plot_bottom = .210
    plot_top = .860 if percentage_labels else .684
    plot_height = plot_top - plot_bottom
    header_y = plot_top + .030
    section_y = plot_top + .069

    fig = plt.figure(figsize=(40 / 3, 7.5), facecolor=PAPER)
    if not percentage_labels:
        text(fig, .048, .923, 'Dialogue exchanges and vocabulary', 28, INK, fontfamily=SERIF)
        text(fig, .048, .857, 'are less closely linked in films', 28, INK, fontfamily=SERIF)
        text(fig, .831, .937, f'Mean correlation across {len(stories)} stories', 10.5, SECONDARY, ha='center')
        text(fig, .774, .881, f'{stats["full_scores"]["Books"]:.2f}', 27, BOOK, ha='center', fontfamily=SERIF)
        text(fig, .831, .880, '→', 19, SECONDARY, ha='center')
        text(fig, .888, .881, f'{stats["full_scores"]["Films"]:.2f}', 27, FILM, ha='center', fontfamily=SERIF)
        text(fig, .774, .834, 'Book', 10.5, BOOK, ha='center', weight='bold')
        text(fig, .888, .834, 'Film', 10.5, FILM, ha='center', weight='bold')
        fig.add_artist(plt.Line2D([.048, .952], [.785, .785], transform=fig.transFigure,
                                 color=BASELINE, linewidth=.85))

    text(fig, .048, header_y, 'Story', 11.5, SECONDARY, weight='bold')
    if percentage:
        text(fig, .251, section_y, 'Percentage drop', 11.5, INK, weight='bold')
        text(fig, .251, header_y, 'Relative to each book’s score', 10.5, SECONDARY)
    else:
        text(fig, .251, section_y, 'Correlation by version', 11.5, INK, weight='bold')
        fig.legend(handles=[Patch(facecolor=BOOK_BAR, edgecolor='none', label='Book (full length)'),
                            Patch(facecolor=FILM, edgecolor='none', label='Film')],
                   loc='center left', bbox_to_anchor=(.251, header_y), ncols=2, frameon=False,
                   borderaxespad=0, handlelength=1.2, handletextpad=.45, columnspacing=1.25,
                   fontsize=10.5, labelcolor=INK)
        gap_label = 'Percentage lost' if percentage_labels else 'Gap = book − film'
        text(fig, .560, header_y, gap_label, 10, SECONDARY, ha='right')
    text(fig, .900, header_y, 'Book', 11.5, BOOK, ha='right', weight='bold')
    text(fig, .952, header_y, 'Film', 11.5, FILM, ha='right', weight='bold')

    pair_left, pair_width = .575, .274
    if percentage_labels:
        pair_centers = [pair_left + p * pair_width for p in PAIR_POSITIONS]
        text(fig, pair_left + pair_width / 2, section_y, 'TF–IDF cosine similarity by pair',
             11.5, INK, ha='center', weight='bold')
        for center, label in zip(pair_centers, PAIR_LABELS.values(), strict=True):
            text(fig, center, header_y, label, 10.5, INK, ha='center', weight='bold')
    else:
        draw_pair_columns(fig, pair_changes, [pair_left, plot_bottom, pair_width, plot_height],
                          y_limits, percentage=percentage)
    ax = axis(fig, [.251, plot_bottom, .309, plot_height])
    left = min(0, bar_values.min())
    if percentage:
        right = max(70, bar_values.max() + 3)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        x_label = 'Percentage drop in rank correlation'
    else:
        right = max(.6, scores.max() + .085)
        ax.xaxis.set_major_locator(MultipleLocator(.2))
        ax.xaxis.set_major_formatter(lambda x, _: '0' if abs(x) < 1e-8 else f'{x:.1f}'.replace('-', '−'))
        x_label = 'Speaker-adjusted rank correlation'
    ax.set(xlim=(left, right), ylim=y_limits, yticks=[])
    ax.set_xlabel(x_label, color=SECONDARY, fontsize=10.5, labelpad=8)
    ax.tick_params(axis='x', colors=SECONDARY, labelsize=10, pad=6)
    ax.grid(axis='x', color=GRID, linewidth=.65)
    ax.axvline(0, color=BASELINE, linewidth=.95, zorder=1)

    row_height = plot_height / (y_limits[0] - y_limits[1])
    for i, (title, (book, film)) in enumerate(zip(row_labels, scores, strict=True)):
        is_mean = percentage_labels and i == len(stories)
        weight = 'bold' if is_mean else 'normal'
        _, y = fig.transFigure.inverted().transform(ax.transData.transform((0, i)))
        if i < len(row_labels) - 1:
            boundary = y - .5 * row_height
            above_mean = percentage_labels and i == len(stories) - 1
            fig.add_artist(plt.Line2D([.048, .952], [boundary, boundary], transform=fig.transFigure,
                                     color=BASELINE if above_mean else GRID,
                                     linewidth=1 if above_mean else .5, zorder=-1))
        if percentage:
            drop = bar_values[i]
            ax.barh(i, drop, height=.47, color=BOOK, linewidth=0, zorder=2)
            ax.annotate(f'{drop:.1f}%'.replace('-', '−'), (0, i),
                        (3.5 if drop >= 0 else -3.5, 0), textcoords='offset points',
                        color=PAPER if drop else INK, fontsize=11, fontweight='bold',
                        ha='left' if drop >= 0 else 'right', va='center', zorder=3)
        else:
            ax.barh(i, book, height=.47, color=BOOK_BAR, linewidth=0, zorder=2)
            ax.barh(i, film, height=.47, color=FILM, linewidth=0, zorder=3)
            gap_text = f'{percentage_drops[i]:.1f}%' if percentage_labels else f'{book - film:.2f}'
            ax.annotate(gap_text.replace('-', '−'), (max(0, book, film), i),
                        (5, 0), textcoords='offset points', color=INK, fontsize=11,
                        fontweight='bold', va='center', zorder=4)

        text(fig, .048, y, title, 14.5, INK, weight=weight)
        text(fig, .900, y, f'{book:.3f}', 13, BOOK, ha='right', weight=weight)
        text(fig, .952, y, f'{film:.3f}', 13, FILM, ha='right', weight=weight)
        if percentage_labels:
            for center, pair_scores in zip(pair_centers, cosines[i], strict=True):
                for score, offset, color in zip(pair_scores, (.16, -.16), (BOOK, FILM), strict=True):
                    label = f'{score:.3f}' if np.isfinite(score) else '–'
                    text(fig, center, y + offset * row_height, label, 12.5, color, ha='center', weight=weight)

    fig.add_artist(plt.Line2D([.048, .952], [.120, .120], transform=fig.transFigure, color=BASELINE, linewidth=.7))
    if percentage:
        text(fig, .048, .091, 'Percentage drop = 100 × (book − film) ÷ book, using the unrounded scores.',
             9.5, SECONDARY)
        text(fig, .048, .062,
             'Pair markers show changes in contributions to the rank correlation relative to the book score. '
             'Across all matched pairs, they sum to the percentage drop.', 9.5, SECONDARY)
    elif percentage_labels:
        text(fig, .048, .091,
             'Percentage lost = 100 × (book − film) ÷ book. The bottom row uses mean book and film scores.',
             9.5, SECONDARY)
        text(fig, .048, .062,
             'Pair scores show book above film. Higher cosine similarity means more similar vocabulary.',
             9.5, SECONDARY)
    else:
        text(fig, .048, .091,
             'Each marker is the pair’s book contribution minus its film contribution. '
             'Across the matched cast, those differences sum to the gap between the two bars.',
             9.5, SECONDARY)
        text(fig, .048, .062,
             'A pair’s contribution is the product of its normalized, speaker-adjusted ranks '
             'for adjacent exchanges and TF–IDF cosine similarity.',
             9.5, SECONDARY)
    text(fig, .048, .033,
         'TF–IDF gives more weight to words a character uses often that are rare across profiles.',
         9.5, SECONDARY)

    save(fig, out, FIGURE_NAMES[mode])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'results' / 'interaction_language',
                        help='Directory containing statistics.json and matched_pairs.csv.')
    views = parser.add_mutually_exclusive_group()
    views.add_argument('--percentage', dest='mode', action='store_const', const='percentage',
                       help='Export bars showing percentage drops.')
    views.add_argument('--percentage-labels', dest='mode', action='store_const', const='percentage_labels',
                       help='Export overlaid correlation bars with percentage-loss labels.')
    views.add_argument('--tf-idf-matrix', dest='mode', action='store_const', const='tfidf_matrix',
                       help='Export mean book and film TF–IDF cosine similarity matrices for all matched characters.')
    views.add_argument('--bert-matrix', dest='mode', action='store_const', const='bert_matrix',
                       help='Export mean book and film MiniLM cosine similarity matrices for all matched characters.')
    views.add_argument('--bert-top-30-matrix', dest='mode', action='store_const', const='bert_top_30_matrix',
                       help='Export MiniLM cosine matrices for every pair among the 30 characters with most dialogue.')
    views.add_argument('--tf-idf-top-20-matrix', dest='mode', action='store_const', const='tfidf_top_20_matrix',
                       help='Export TF–IDF matrices for the top 20 by book dialogue word count.')
    views.add_argument('--bert-top-20-matrix', dest='mode', action='store_const', const='bert_top_20_matrix',
                       help='Export MiniLM matrices for the top 20 by book dialogue word count.')
    parser.add_argument('--model-dir', type=Path, help='Local MiniLM checkpoint for BERT matrices.')
    parser.set_defaults(mode='correlation')
    args = parser.parse_args()
    stats, matrices = read_results(args.results_dir)
    stories = stats['stories']
    repo = Path(__file__).resolve().parents[1]

    with plt.rc_context({'font.family': ['Segoe UI', 'DejaVu Sans'], 'svg.fonttype': 'path',
                         'svg.hashsalt': FIGURE_NAMES[args.mode]}):
        if args.mode in ('bert_top_30_matrix', 'bert_top_20_matrix'):
            from adaptation_semantics import top_cast_cosines

            names, scores = top_cast_cosines(repo, stories, args.model_dir,
                                             by_book_words=args.mode == 'bert_top_20_matrix')
            render_similarity_matrices(args.results_dir, names, scores, len(stories), mode=args.mode)
        elif args.mode == 'tfidf_top_20_matrix':
            _, counts, _, _ = read_project(repo)
            cast = top_book_characters(stories, counts)
            names, scores = cast_mean_cosines(stories, matrices, cast=cast)
            render_similarity_matrices(args.results_dir, names, scores, len(stories), mode=args.mode)
        elif args.mode in ('tfidf_matrix', 'bert_matrix'):
            if args.mode == 'bert_matrix':
                from adaptation_semantics import matched_cast_matrices

                matrices = matched_cast_matrices(repo, stories, matrices, args.model_dir)
            names, scores = cast_mean_cosines(stories, matrices)
            render_similarity_matrices(args.results_dir, names, scores, len(stories), mode=args.mode)
        else:
            render(args.results_dir, stats, matrices, mode=args.mode)


if __name__ == '__main__':
    main()
