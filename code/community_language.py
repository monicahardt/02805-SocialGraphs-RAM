"""Compare language similarity against dialogue communities and direct ties.

Run from the repository root with python -m code.community_language.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .dialogue_csv_reader import read_dialogue_records
from .interaction_graph import NON_SPECIFIC_MOVIE_LABELS
from .word_counts import tokenize

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "results" / "vocabulary_map"
OUTPUT = ROOT / "results" / "analysis" / "community_language"
SEED = 42
REPLAYS = 120
SAMPLE_WORDS = 100
SHUFFLES = 10_000
TITLES = (
    "Philosopher's Stone", "Chamber of Secrets", "Prisoner of Azkaban",
    "Goblet of Fire", "Order of the Phoenix", "Half-Blood Prince", "Deathly Hallows",
)
SOURCES = {
    "Books": ROOT / "book_data" / "all_books.csv",
    "Films": ROOT / "movie_data" / "all_movies.csv",
}
ROLE_LABELS = {"hero": "Heroes", "villain": "Villains"}
BLUE, ORANGE, INK, MUTED = "#24648a", "#b44f30", "#202c3a", "#525e70"


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def collect_inputs(profiles, vocabulary, aliases):
    word_ids = {word: i for i, word in enumerate(vocabulary)}
    excluded = NON_SPECIFIC_MOVIE_LABELS | {"unknown", "wakanda"}
    graphs = {}
    corpus = defaultdict(list)
    for medium, path in SOURCES.items():
        graph = nx.Graph()
        previous = previous_segment = None
        profile_medium = "books" if medium == "Books" else "movies"
        lookup = {
            (int(installment), row["character"]): int(row["id"])
            for row in profiles
            if row["medium"] == profile_medium
            for installment in row["installments"].split()
        }
        for record in read_dialogue_records(path):
            speaker = aliases.get(record.character, record.character)
            tokens = tokenize(record.line)
            if (
                speaker.casefold() in excluded
                or record.confidence.casefold() in {"low", "unattributed"}
                or not tokens
            ):
                previous = previous_segment = None
                continue
            segment = record.book, record.chapter
            installment = next(i for i, title in enumerate(TITLES, 1) if title in record.book)
            profile_id = lookup.get((installment, speaker))
            if profile_id is not None:
                corpus[profile_id].extend(word_ids.get(word, -1) for word in tokens)
            graph.add_node(speaker)
            if segment == previous_segment and speaker != previous:
                weight = graph.get_edge_data(previous, speaker, {}).get("weight", 0)
                graph.add_edge(previous, speaker, weight=weight + 1)
            previous, previous_segment = speaker, segment
        graphs[medium] = graph

    ordered = [np.array(corpus[int(row["id"])], dtype=int) for row in profiles]
    if [len(words) for words in ordered] != [int(row["words"]) for row in profiles]:
        raise ValueError("The saved vocabulary profiles and dialogue rows disagree.")
    return graphs, ordered


def cosine_matrix(sequences, idf):
    counts = np.stack([
        np.bincount(sequence[sequence >= 0], minlength=len(idf))
        for sequence in sequences
    ])
    matrix = counts * idf
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    return unit @ unit.T


def controlled_cosines(sequences, masked_idf, rng):
    total = np.zeros((len(sequences), len(sequences)))
    for _ in range(REPLAYS):
        sample = [rng.choice(words, SAMPLE_WORDS, replace=False) for words in sequences]
        total += cosine_matrix(sample, masked_idf)
    return total / REPLAYS


def groups_for(graph, weight):
    return {
        name: i
        for i, group in enumerate(nx.community.louvain_communities(graph, weight=weight, seed=SEED))
        for name in group
    }


def measure(similarity, in_group, permutations):
    a, b = np.triu_indices(len(similarity), 1)
    values = similarity[a, b]
    first, second = values[in_group], values[~in_group]
    null = similarity[permutations[:, a], permutations[:, b]]
    gaps = null[:, in_group].mean(axis=1) - null[:, ~in_group].mean(axis=1)
    observed = float(first.mean() - second.mean())
    lower, upper = np.percentile(gaps, [2.5, 97.5])
    return {
        "first_pairs": len(first), "second_pairs": len(second),
        "first_mean": float(first.mean()), "second_mean": float(second.mean()),
        "gap": observed, "null_lower": float(lower), "null_upper": float(upper),
        "one_sided_p": float((1 + np.count_nonzero(gaps >= observed)) / (1 + len(gaps))),
    }


def summarize(profiles, graphs, sequences, vocabulary):
    df = Counter(word for sequence in sequences for word in set(sequence) if word >= 0)
    idf = np.log(len(profiles) / np.array([df[i] for i in range(len(vocabulary))]))
    full_sims = cosine_matrix(sequences, idf)
    saved = np.load(MAP / "cosine_similarity.npy")
    np.testing.assert_allclose(full_sims, saved, atol=1e-12, rtol=0)

    names = {
        part.casefold() for row in profiles
        for part in row["character"].split() if len(part) >= 3
    }
    masked_idf = idf * np.array([word.removesuffix("'s") not in names for word in vocabulary])
    records = []
    rng = np.random.default_rng(SEED)
    for role, role_name in ROLE_LABELS.items():
        for medium, graph in graphs.items():
            profile_medium = "books" if medium == "Books" else "movies"
            ids = [
                int(row["id"]) for row in profiles
                if row["role"] == role and row["medium"] == profile_medium
            ]
            people = [profiles[i]["character"] for i in ids]
            n = len(people)
            a, b = np.triu_indices(n, 1)
            paired_names = [(people[i], people[j]) for i, j in zip(a, b)]
            permutations = np.stack([rng.permutation(n) for _ in range(SHUFFLES)])
            adjusted = controlled_cosines([sequences[i] for i in ids], masked_idf, rng)
            full = full_sims[np.ix_(ids, ids)]

            weighted, unweighted = (groups_for(graph, weight) for weight in ("weight", None))
            comparisons = {
                "Weighted community": np.array([weighted[x] == weighted[y] for x, y in paired_names]),
                "Unweighted community": np.array([unweighted[x] == unweighted[y] for x, y in paired_names]),
                "Direct tie": np.array([graph.has_edge(x, y) for x, y in paired_names]),
            }
            for kind, in_group in comparisons.items():
                for condition, similarity in (("Full profile", full), ("Equal length, names masked", adjusted)):
                    row = measure(similarity, in_group, permutations)
                    records.append({
                        "medium": medium, "role": role_name, "characters": n,
                        "comparison": kind, "dialogue": condition, **row,
                    })
    return records


def chart_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK, "axes.titleweight": "bold",
        "axes.spines.right": False, "axes.spines.top": False, "axes.spines.left": False,
        "axes.edgecolor": "#b9c1cb", "ytick.color": INK, "xtick.color": MUTED,
        "svg.fonttype": "none",
    })


def layout(title, subtitle, labels):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.3))
    fig.subplots_adjust(left=.145, right=.965, bottom=.31, top=.71, wspace=.31)
    fig.text(.145, .935, title, fontsize=19, fontweight="bold")
    fig.text(.145, .875, subtitle, fontsize=10.3, color=MUTED)
    for ax in axes:
        ax.set_ylim(3.62, -.62)
        ax.set_yticks(range(4), labels)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.axhline(1.5, color="#d9dee6", linewidth=.8)
        ax.grid(axis="x", color="#ecf0f3")
        ax.set_axisbelow(True)
    axes[1].tick_params(labelleft=False)
    return fig, axes


def draw_null(ax, y, row, color):
    ax.hlines(y, row["null_lower"], row["null_upper"], color="#aab5c1", lw=4, zorder=2)
    ax.scatter(row["gap"], y, color=color, edgecolor="white", lw=.8, s=71, zorder=3)


def save(fig, filename, footer):
    fig.text(.145, .047, footer, color=MUTED, fontsize=8.1, linespacing=1.4)
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT / f"{filename}.{extension}", dpi=220)
    plt.close(fig)


def community_figure(records):
    rows = [
        r for r in records if r["comparison"] == "Weighted community" and r["dialogue"] == "Full profile"
    ]
    alt = [
        r for r in records if r["comparison"] == "Unweighted community" and r["dialogue"] == "Full profile"
    ]
    labels = [r["role"] + " · " + r["medium"].lower() for r in rows]
    fig, axes = layout(
        "Same-role vocabulary in dialogue communities",
        "Communities come from consecutive speaker turns. Vocabulary similarity is measured separately.",
        labels,
    )
    left, right = axes
    left.set_title("Mean similarity by pair type", loc="left", pad=11)
    right.set_title("Within-community gap and shuffle benchmark", loc="left", pad=11)
    for i, (row, other) in enumerate(zip(rows, alt)):
        lo, hi = row["second_mean"], row["first_mean"]
        left.hlines(i, lo, hi, color="#aab5c1", lw=1.8)
        left.scatter(lo, i, color=ORANGE, edgecolor="white", s=62, zorder=3)
        left.scatter(hi, i, color=BLUE, marker="D", edgecolor="white", s=56, zorder=4)
        left.text(.287, i, f'{row["first_pairs"]}/{row["second_pairs"]}', va="center", color=MUTED, fontsize=8)
        draw_null(right, i, row, BLUE)
        right.scatter(other["gap"], i+.19, edgecolor=ORANGE, facecolor="white", marker="D", s=35, zorder=4)
    left.set_xlim(0, .329)
    left.set_xticks([0,.05,.10,.15,.20,.25])
    left.text(.286, -.46, "Pairs¹", fontsize=8, color=MUTED, va="center")
    left.set_xlabel("Full-vector TF-IDF cosine similarity")
    right.set_xlim(-.081, .108)
    right.set_xlabel("Within-community minus between-community similarity")
    right.axvline(0, color="#7d8795", lw=1, linestyle="--")
    left.scatter([], [], color=BLUE, marker="D", label="Same community")
    left.scatter([], [], color=ORANGE, label="Different communities")
    right.scatter([], [], color=BLUE, label="Weighted")
    right.scatter([], [], edgecolor=ORANGE, facecolor="white", marker="D", label="Unweighted")
    right.plot([], [], color="#aab5c1", lw=4, label="Weighted shuffle range")
    for ax in axes:
        ax.legend(loc="upper left", bbox_to_anchor=(-.02, -.27), ncol=3, frameon=False, fontsize=8.3, handletextpad=.4, columnspacing=.85)
    save(fig, "01_community_similarity",
         "¹ Pair counts are same-community / different-community. Weighted Louvain communities count repeated adjacent turns.\n"
         "Gray lines show the middle 95% of 10,000 within-role character shuffles for the weighted partition.\n"
         "The figure compares 20 heroes and nine villains with profiles in both media. Pair scores share characters.")


def direct_figure(records):
    conditions = ("Full profile", "Equal length, names masked")
    rows = {
        view: [r for r in records if r["comparison"] == "Direct tie" and r["dialogue"] == view]
        for view in conditions
    }
    labels = [
        f'{r["role"]} · {r["medium"].lower()}\n{r["first_pairs"]} tied / {r["second_pairs"]} untied'
        for r in rows[conditions[0]]
    ]
    fig, axes = layout(
        "Same-role characters who speak alongside each other",
        "Pairs with an observed direct tie versus pairs without one. The horizontal scales differ.",
        labels,
    )
    for ax, view, color in zip(axes, conditions, (BLUE, ORANGE)):
        ax.set_title(view, loc="left", pad=11)
        for i, row in enumerate(rows[view]):
            draw_null(ax, i, row, color)
        ax.axvline(0, color="#7d8795", lw=1, linestyle="--")
        ax.set_xlabel("Mean TF-IDF cosine gap (tied minus untied)")
        ax.scatter([], [], color=color, label="Observed gap")
        ax.plot([], [], color="#aab5c1", lw=4, label="Shuffle range")
        ax.legend(loc="upper left", bbox_to_anchor=(-.02, -.27), ncol=2, frameon=False, fontsize=8.3)
    save(fig, "02_direct_dialogue_ties",
         "A tie means at least one transition on adjacent attributed lines in the same source segment.\n"
         "Gray lines show the middle 95% of 10,000 within-role character shuffles.\n"
         "The right panel averages 120 rounds of sampling 100 raw words per profile and masking character-name tokens.")


def main():
    profiles = sorted(read_table(MAP / "nodes.csv"), key=lambda row: int(row["id"]))
    vocabulary = [row["word"] for row in read_table(MAP / "vocabulary.csv")]
    aliases = json.loads((MAP / "metrics.json").read_text(encoding="utf-8"))["aliases"]
    graphs, sequences = collect_inputs(profiles, vocabulary, aliases)
    records = summarize(profiles, graphs, sequences, vocabulary)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "plot_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    chart_style()
    community_figure(records)
    direct_figure(records)
    print(f"Wrote both figures and their data to {OUTPUT}")


if __name__ == "__main__":
    main()
