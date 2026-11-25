"""Render the presentation figures for the book–film conversation comparison."""

import math
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter


PAPER = "#f8f7f3"
INK = "#17263b"
MUTED = "#4b5c6e"
GRID = "#dde2e6"
SERIF = ["Georgia", "DejaVu Serif"]
MEDIA = {"books": ("#17695e", "o", "-"), "films": ("#6d47a0", "D", "--")}
HARRY_HERMIONE = ("Harry Potter", "Hermione Granger")
HARRY_RON = ("Harry Potter", "Ron Weasley")
FOCAL_PAIRS = {HARRY_HERMIONE: "Harry & Hermione", HARRY_RON: "Harry & Ron"}
SHORT_NAMES = {
    "Harry Potter": "Harry", "Hermione Granger": "Hermione", "Ron Weasley": "Ron",
    "Albus Dumbledore": "Dumbledore", "Severus Snape": "Snape", "Remus Lupin": "Lupin",
}
# Shared coordinates, ordered by a character's total share of spell-tagged turns.
LAYOUT = [(4.75, 2.05, 0, 0), (7.02, 3.19, 11, 0), (7.02, 0.93, 11, 0),
          (1.53, 2.05, -11, 0), (4.50, 3.61, 0, 12), (2.04, 3.20, -9, 8),
          (2.14, 0.87, -10, -7), (4.64, 0.52, 0, -13)]


def caption(fig, x, y, value, size, color=INK, **kwargs):
    return fig.text(x, y, value, fontsize=size, color=color, va="center", **kwargs)


def canvas(title, subtitle):
    fig = plt.figure(figsize=(13 + 1 / 3, 7.5), facecolor=PAPER)
    caption(fig, .048, .925, title, 32, fontfamily=SERIF, weight="bold")
    caption(fig, .05, .856, subtitle, 15, MUTED)
    return fig


def save(fig, out, filename, title):
    with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": filename}):
        fig.savefig(out / f"{filename}.svg", metadata={"Date": None, "Title": title})
        fig.savefig(out / f"{filename}.png", dpi=180)
    plt.close(fig)


def pooled_counts(stories):
    totals = Counter()
    counts = {medium: defaultdict(Counter) for medium in MEDIA}
    for story in stories.values():
        for medium, pairs in story.items():
            for pair, result in pairs.items():
                totals[medium] += result.turns
                for topic, hits in result.hits.items():
                    counts[medium][topic][pair] += hits
    return totals, counts


def pair_shares(counts, topic):
    result = {}
    for medium, topics in counts.items():
        pairs = topics[topic]
        total = pairs.total()
        result[medium] = Counter({pair: 100 * hits / total for pair, hits in pairs.items()})
    return result


def media_legend(fig, show_lines=True):
    legend = [Line2D([], [], color=color, marker=marker, linestyle=linestyle if show_lines else "None",
                     lw=2.15, label=medium.title())
              for medium, (color, marker, linestyle) in MEDIA.items()]
    fig.legend(handles=legend, loc="center right", bbox_to_anchor=(.950, .856), frameon=False,
               labelcolor=INK, fontsize=13, ncols=2)


def plot_spell_networks(stories, out):
    totals, counts = pooled_counts(stories)
    shares = pair_shares(counts, "spells")
    combined = shares["books"] + shares["films"]
    importance = Counter()
    for pair, share in combined.items():
        for name in pair:
            importance[name] += share
    cast = sorted(importance, key=lambda name: (-importance[name], name))[:len(LAYOUT)]
    positions = dict(zip(cast, LAYOUT))
    edges = sorted((pair for pair in combined if all(name in positions for name in pair)),
                   key=lambda pair: (-combined[pair], pair))[:9]

    title = "Who talks about spells with whom?"
    fig = canvas(title, "Harry and Ron account for a smaller share of spell-tagged turns in the films.")
    for (medium, (color, _, _)), left in zip(MEDIA.items(), (.050, .513)):
        fig.add_artist(FancyBboxPatch((left, .233), .437, .558,
                                     boxstyle="round,pad=0.003,rounding_size=.012",
                                     transform=fig.transFigure, facecolor="white", edgecolor=GRID,
                                     linewidth=1, zorder=-1))
        caption(fig, left + .016, .746, medium.title(), 21, color, fontfamily=SERIF, weight="bold")
        caption(fig, left + .421, .746, f"{counts[medium]['spells'].total():,} of {totals[medium]:,} turns tagged",
                11, MUTED, ha="right")
        ax = fig.add_axes([left + .012, .252, .413, .442], xlim=(.1, 9.0), ylim=(.12, 4.08))
        ax.set_axis_off()
        draw_network(ax, positions, edges, shares[medium], color)

    for (pair, label), left in zip(FOCAL_PAIRS.items(), (.050, .513)):
        caption(fig, left, .148, label, 15, weight="bold")
        caption(fig, left + .25, .148, f"{shares['books'][pair]:.1f}%", 19, MEDIA["books"][0], ha="right", weight="bold")
        caption(fig, left + .263, .148, "→", 17, MUTED, ha="center")
        caption(fig, left + .277, .148, f"{shares['films'][pair]:.1f}%", 19, MEDIA["films"][0], weight="bold")

    caption(fig, .05, .059,
            "Line width shows the pair’s share of all spell-tagged turns. The nine strongest links among eight characters are shown.",
            11, MUTED)
    save(fig, out, "01_spell_conversations", title)


def draw_network(ax, positions, edges, shares, color):
    active = set()
    for pair in sorted(edges, key=lambda pair: pair in FOCAL_PAIRS):
        share = shares[pair]
        if not share:
            continue
        active.update(pair)
        a, b = (positions[name] for name in pair)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=max(.75, .40 * share),
                alpha=.96 if pair in FOCAL_PAIRS else .37, solid_capstyle="round")

    focal_names = set().union(*FOCAL_PAIRS)
    for name, (x, y, dx, dy) in positions.items():
        label = SHORT_NAMES.get(name, name)
        if name == "Harry Potter":
            ax.annotate(label, (x, y), ha="center", va="center", fontsize=14, color="white", weight="bold",
                        bbox={"boxstyle": "round,pad=.25", "facecolor": color, "edgecolor": "white", "linewidth": 1},
                        zorder=4)
            continue
        emphasis = name in focal_names
        ax.scatter(x, y, s=79 if emphasis else 52, color=color if name in active else GRID,
                   edgecolors="white", linewidths=.9, zorder=3)
        alignment = "right" if dx < 0 else "left" if dx > 0 else "center"
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=alignment, va="center", fontsize=14, color=INK, weight="bold" if emphasis else "normal",
                    zorder=4)


def plot_topic_trends(topics, stories, out):
    rates = {topic["id"]: {medium: [] for medium in MEDIA} for topic in topics}
    for story in stories.values():
        for medium, pairs in story.items():
            total = sum(result.turns for result in pairs.values())
            counts = sum((result.hits for result in pairs.values()), Counter())
            for topic_id in rates:
                rates[topic_id][medium].append(100 * counts[topic_id] / total if total else math.nan)

    title = "What changes from book to film?"
    fig = canvas(title, "Share of eligible conversation turns tagged with each topic")
    top = max(value for topic_rates in rates.values() for series in topic_rates.values() for value in series if math.isfinite(value))
    top = max(2, 2 * math.ceil(top / 2))
    grid = fig.add_gridspec(2, 3, left=.086, right=.945, top=.744, bottom=.156, wspace=.32, hspace=.52)
    for index, topic in enumerate(topics):
        ax = fig.add_subplot(grid[index], facecolor="white")
        ax.set_title(topic["label"], loc="left", fontsize=16, color=INK, pad=10, weight="bold")
        ax.set(xlim=(.7, len(stories) + .3), xticks=list(stories), ylim=(0, top * 1.025), yticks=[0, top / 2, top])
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.tick_params(labelcolor=MUTED, labelsize=11, length=0, pad=5)
        ax.spines[:].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=.9)
        for medium, (color, marker, linestyle) in MEDIA.items():
            ax.plot(list(stories), rates[topic["id"]][medium], color=color, lw=2.15, linestyle=linestyle,
                    marker=marker, markersize=6.3, markeredgecolor="white", markeredgewidth=.7, clip_on=False)

    media_legend(fig)
    caption(fig, .05, .065, "Stories appear in order from 1 to 7. Both Deathly Hallows films are grouped under story 7.", 11, MUTED)
    save(fig, out, "02_topics_by_story", title)


def plot_topic_pairs(topics, stories, out):
    _, counts = pooled_counts(stories)
    comparisons = []
    for topic in topics:
        shares = pair_shares(counts, topic["id"])
        combined = shares["books"] + shares["films"]
        pairs = sorted(combined, key=lambda pair: (-combined[pair], pair))[:2]
        comparisons.append((topic["label"], shares, pairs))

    largest = max((shares[medium][pair] for _, shares, pairs in comparisons for pair in pairs for medium in MEDIA), default=0)
    limit = max(10, 10 * math.ceil(largest / 10))
    title = "Who leads each conversation topic?"
    fig = canvas(title, "The two pairs with the largest combined book and film shares for each topic")
    media_legend(fig, show_lines=False)

    for index, (label, shares, pairs) in enumerate(comparisons):
        left = .05 + .463 * (index % 2)
        top = .745 - .205 * (index // 2)
        caption(fig, left, top, label, 16, weight="bold")
        ax = fig.add_axes([left + .185, top - .142, .245, .108], facecolor="white")
        ax.set(xlim=(-1, limit), xticks=range(0, limit + 1, 10), ylim=(-.55, 1.55), yticks=[])
        ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.tick_params(labelcolor=MUTED, labelsize=10.5, length=0, pad=4)
        ax.spines[:].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=.9)

        for y, pair in zip((1, 0), pairs):
            names = " & ".join(SHORT_NAMES.get(name, name) for name in pair)
            ax.text(-.036, y, names, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=12, color=INK)
            book, film = (shares[medium][pair] for medium in MEDIA)
            ax.plot([book, film], [y + .10, y - .10], color="#9daab5", lw=1.4, zorder=2)
            for (medium, (color, marker, _)), offset in zip(MEDIA.items(), (.10, -.10)):
                ax.plot(shares[medium][pair], y + offset, color=color, marker=marker, markersize=7.4,
                        markeredgecolor="white", markeredgewidth=.7, zorder=3)

    caption(fig, .05, .060, "Each percentage is the pair’s share of all turns tagged with that topic in its medium.", 11, MUTED)
    save(fig, out, "03_leading_pairs_by_topic", title)
