"""A matched book/film vocabulary comparison for Umbridge in Phoenix.

Run from the repository root: python scripts/compare_umbridge.py
Reads the existing verified counts and shared TF-IDF vocabulary. No model is
refitted and no source data or previous analysis outputs are modified.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/comparison/umbridge_phoenix"
SOURCE = "Harry Potter and the Order of the Phoenix"
CHARACTER = "Dolores Umbridge"
BOOK_COLOR, FILM_COLOR = "#217a9b", "#b45535"


def read(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def compression_benchmark(rows):
    """Compare observed word presence with random deletion from the book bag.

    Coverage is the fraction of book TF-IDF weight attached to word types
    present in the shorter text. The benchmark samples token positions without
    replacement; it represents random removal, not an editing process.
    """
    book = np.array([row["book_count"] for row in rows], dtype=np.int64)
    film = np.array([row["film_count"] for row in rows], dtype=np.int64)
    weights = np.array([row["book_tfidf"] for row in rows])
    sample_length = int(film.sum())
    if sample_length > book.sum() or weights.sum() <= 0:
        raise ValueError("Benchmark requires a shorter film and positive book TF-IDF weights.")
    observed = float(weights[film > 0].sum() / weights.sum())
    rng = np.random.default_rng(42)
    draws = rng.multivariate_hypergeometric(book, sample_length, size=1000)
    assert np.all(draws.sum(axis=1) == sample_length)
    assert np.all(draws <= book)
    simulated = (draws > 0) @ weights / weights.sum()
    # Independently compute the exact expectation using absence probabilities.
    n = int(book.sum())
    probabilities = []
    for count in book:
        if n - count < sample_length:
            probabilities.append(1.0)
        else:
            # Product over the removed occurrences is stable and short: this
            # is C(n-count,m)/C(n,m), written using count rather than m factors.
            absence = 1.0
            for k in range(int(count)):
                absence *= (n - sample_length - k) / (n - k)
            probabilities.append(1 - absence)
    exact_mean = float(np.dot(weights, probabilities) / weights.sum())
    low, high = np.quantile(simulated, [.025, .975])
    settings = {"definition": "fraction of book TF-IDF weight whose word types occur in the comparison text",
                "observed_film_coverage": observed, "simulated_mean": float(simulated.mean()),
                "exact_expected_coverage": exact_mean,
                "simulated_2_5_percentile": float(low), "simulated_97_5_percentile": float(high),
                "simulations": 1000, "seed": 42, "book_words": n, "sample_words": sample_length,
                "sampling": "word tokens drawn without replacement; no order or sentence preservation"}
    (OUT / "compression_benchmark.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    with (OUT / "compression_simulations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["simulation", "weighted_coverage"])
        writer.writerows(enumerate(simulated, 1))
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    ax.hist(100 * simulated, bins=28, color="#b8c6ce", label="1,000 random cuts of the book dialogue")
    ax.axvline(100 * observed, color=FILM_COLOR, lw=2.5,
               label=f"Film: {observed:.1%}")
    ax.axvline(100 * exact_mean, color=BOOK_COLOR, lw=1.5, linestyle="--",
               label=f"Exact random-cut expectation: {exact_mean:.1%}")
    ax.set(xlim=(25, 60), xlabel="Book TF-IDF weight attached to word types retained in the shorter text (%)",
           ylabel="Number of simulated cuts", title="How much vocabulary would random shortening retain?\n"
           f"Umbridge in Order of the Phoenix: {n:,} words reduced to {sample_length:,}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=.95,
              fontsize=9, loc="upper left")
    fig.savefig(OUT / "compression_benchmark.png", dpi=160)
    plt.close(fig)
    return settings


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for medium in ("books", "movies"):
        counts[medium] = {r["word"]: int(r["count"]) for r in
                          read(f"results/word_counts/{medium}/character_word_counts.csv")
                          if r["character"] == CHARACTER and r["source"] == SOURCE}
    totals = {medium: sum(values.values()) for medium, values in counts.items()}
    idf = {r["word"]: float(r["idf"]) for r in read("results/tfidf/vocabulary.csv")}
    n_documents = len(read("results/tfidf/documents.csv"))
    rows = []
    for word in sorted(counts["books"].keys() | counts["movies"].keys()):
        book, film = counts["books"].get(word, 0), counts["movies"].get(word, 0)
        rows.append({"word": word, "book_count": book, "film_count": film,
                     "book_per_1000": 1000 * book / totals["books"],
                     "film_per_1000": 1000 * film / totals["movies"],
                     "idf": idf[word], "book_tfidf": book / totals["books"] * idf[word],
                     "film_tfidf": film / totals["movies"] * idf[word],
                     "presence": "shared" if book and film else "book only" if book else "film only",
                     "display_eligible": word not in ENGLISH_STOP_WORDS and len(word) >= 3 and book + film >= 2})
    with (OUT / "word_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(17, 6.8), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1.7, 1))
    left, center, right = [fig.add_subplot(grid[0, i]) for i in range(3)]
    side_limit = max(max(r["book_tfidf"], r["film_tfidf"]) for r in rows
                     if r["presence"] != "shared" and r["display_eligible"]) * 1.05
    for ax, presence, medium, color, label in (
        (left, "book only", "book", BOOK_COLOR, "Book dialogue only"),
        (right, "film only", "film", FILM_COLOR, "Film dialogue only"),
    ):
        selected = sorted((r for r in rows if r["presence"] == presence and r["display_eligible"]),
                          key=lambda r: (-r[f"{medium}_tfidf"], r["word"]))[:8][::-1]
        labels = [f"{r['word']} (n={r[f'{medium}_count']})" for r in selected]
        ax.barh(labels, [r[f"{medium}_tfidf"] for r in selected], color=color)
        ax.set_title(label + "\nHighest TF-IDF among eligible terms", fontsize=11)
        ax.set_xlabel("TF-IDF weight")
        ax.set_xlim(0, side_limit)
        ax.spines[["top", "right"]].set_visible(False)

    # Identical count pairs have identical positions. Group them so labels and
    # marks do not imply that coincident terms occupy distinct coordinates.
    groups = defaultdict(list)
    for row in rows:
        if row["presence"] == "shared" and row["display_eligible"]:
            groups[row["book_count"], row["film_count"]].append(row)
    for group in groups.values():
        row = group[0]
        x, y = row["book_per_1000"], row["film_per_1000"]
        center.scatter(x, y, s=25 + 12 * math.sqrt(len(group)),
                       color=FILM_COLOR if y > x else BOOK_COLOR, alpha=.75)
    labels_to_show = {"educational", "decree", "punishments", "hem", "professor", "mr", "inquisitor", "potter"}
    offsets = {"educational": (10, 12), "decree": (10, -22), "hem": (10, -18),
               "professor": (8, -20), "mr": (-40, 12), "inquisitor": (8, 12), "potter": (8, 12)}
    for group in groups.values():
        names = sorted(r["word"] for r in group if r["word"] in labels_to_show)
        if not names:
            continue
        row = group[0]
        center.annotate(" / ".join(names), (row["book_per_1000"], row["film_per_1000"]),
                        xytext=offsets.get(names[0], (10, 10)), textcoords="offset points", fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8},
                        arrowprops={"arrowstyle": "-", "lw": .6, "color": "#888"})
    shared = [r for r in rows if r["presence"] == "shared" and r["display_eligible"]]
    lower = min(min(r["book_per_1000"], r["film_per_1000"]) for r in shared) * .65
    upper = max(max(r["book_per_1000"], r["film_per_1000"]) for r in shared) * 1.8
    center.plot([lower, upper], [lower, upper], color="#a9adb1", lw=1, zorder=0)
    center.set(xscale="log", yscale="log", xlim=(lower, upper), ylim=(lower, upper),
               xlabel="Book occurrences per 1,000 Umbridge words",
               ylabel="Film occurrences per 1,000 Umbridge words")
    center.set_title("Shared words\nAbove the diagonal: more concentrated in the film", fontsize=11)
    center.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Which parts of Umbridge's vocabulary survive adaptation?\n"
                 f"Order of the Phoenix: {totals['books']:,} book words and {totals['movies']:,} film words", fontsize=14)
    fig.savefig(OUT / "vocabulary_comparison.png", dpi=160)
    plt.close(fig)
    benchmark = compression_benchmark(rows)

    notes = ["# A matched comparison: Umbridge in Order of the Phoenix", "",
        "[Open the comparison](vocabulary_comparison.png)", "",
        "The figure compares the same character in the matching book and film. It uses raw word counts, rates per 1,000 words and the existing jointly fitted TF-IDF weights.", "",
        "- Center: words present in both dialogue sets. Coordinates are their normalized book and film frequencies. The diagonal means equal rates. Orange marks are more concentrated in the film; blue marks are more concentrated in the book. A mark can represent several words with identical counts. Labels are selected examples, not an exhaustive list.",
        "- Left and right: terms that occur in only one of these two dialogue sets, ordered by TF-IDF. n is the raw count. These terms may occur elsewhere in the series or in another character's dialogue.",
        "- To keep the figure legible, only terms with at least two total occurrences, at least three letters and no membership in sklearn's English stopword list are displayed. Counts, TF denominators and IDF remain unchanged. Names remain eligible. The full unfiltered table is in `word_comparison.csv`.", "",
        "Only three film-only terms meet the display threshold; many film-only word types occur once. Both side panels use the same TF-IDF scale. Word-type absence is sensitive to text length, so it should be considered alongside the shortening benchmark below.", "",
        f"The book denominator is {totals['books']:,} words and the film denominator is {totals['movies']:,} words. IDF still uses the same {n_documents} reference documents as step 3.",
        "The film has approximately one quarter of the attributed word tokens, while several institutional terms occur at higher normalized rates. This is consistent with concentrated characterization in a shorter script, pending attribution checks and a broader predefined vocabulary. Words shared across media may still occur in different scenes.", "",
        "A useful interactive extension would let the reader choose a character/installment, select a word and read its book and film quotations with source references. Missing terms belong in the side panels, rather than being hidden by logarithmic axes.", "",
        "## A random-shortening benchmark", "",
        "[Open the benchmark](compression_benchmark.png)", "",
        "The measure here is the fraction of the book's total TF-IDF weight attached to word types that also occur in the film dialogue. It measures exact word presence, not semantic or personality preservation. It includes all words and the existing shared IDF, without the display filters used above.",
        f"The observed film covers {benchmark['observed_film_coverage']:.1%} of that book weight. Sampling {totals['movies']:,} word tokens without replacement from the {totals['books']:,}-word book dialogue yields an exact expected coverage of {benchmark['exact_expected_coverage']:.1%}.",
        f"Across 1,000 seeded random cuts, the middle 95% range is {benchmark['simulated_2_5_percentile']:.1%} to {benchmark['simulated_97_5_percentile']:.1%}. This is variation under the chosen random-cut model, not uncertainty about the extracted dialogue or a general adaptation effect.",
        "This makes a specific comparison: how observed word preservation differs from uniformly removing book word tokens to the same length. Adaptation also selects scenes, rephrases speech and introduces vocabulary. The benchmark preserves no sentence structure and is not a model of a screenwriter. Synonyms, names, transcription errors and attribution errors can affect the observed gap. Equal-length sampling by dialogue line or scene, corrected names/spelling, and additional matched cases are useful sensitivity checks.",
        "Parameters, exact expectation and simulated range are in `compression_benchmark.json`; individual draws are in `compression_simulations.csv`. The exact expectation is calculated independently using the probability that each book word type survives a sample without replacement.", "",
        "This visual is a first comparison case, using existing data and models. Speaker assignments and transcription artifacts are preserved. Novel presentation should expose these details and support interpretation.", ""]
    (OUT / "README.md").write_text("\n".join(notes), encoding="utf-8")
    print(f"Compared {len(rows)} words; denominators {totals}; results: {OUT}")


if __name__ == "__main__":
    main()
