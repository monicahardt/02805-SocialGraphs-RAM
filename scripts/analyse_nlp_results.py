"""Inspect existing NLP outputs and produce three diagnostic scientific plots.

Run from the repository root: python scripts/analyse_nlp_results.py
The source data and previous method outputs are read only.
"""

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from dialogue_csv_reader import read_dialogue_records
from word_counts import tokenize

OUT = ROOT / "results/analysis"
BOOK_COLOR, FILM_COLOR = "#217a9b", "#b45535"
SUFFIXES = ["Philosopher's Stone", "Chamber of Secrets", "Prisoner of Azkaban",
            "Goblet of Fire", "Order of the Phoenix", "Half-Blood Prince", "Deathly Hallows"]
SHORT = ["1 Stone", "2 Chamber", "3 Azkaban", "4 Goblet", "5 Phoenix", "6 Half-Blood Prince", "7 Hallows"]
MARKERS = {"yeh", "ter", "yer", "fer"}


def read(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def installment(source):
    for index, suffix in enumerate(SUFFIXES, 1):
        if suffix in source:
            return index
    raise ValueError(source)


def word_profiles():
    profiles = defaultdict(Counter)
    for medium in ("books", "movies"):
        for row in read(f"results/word_counts/{medium}/character_word_counts.csv"):
            profiles[medium, installment(row["source"]), row["character"]][row["word"]] += int(row["count"])
    return profiles


def language_plot(profiles):
    rows, summaries = [], {}
    for medium in ("books", "movies"):
        pooled = Counter()
        for number in range(1, 8):
            counts = profiles[medium, number, "Rubeus Hagrid"]
            pooled.update(counts)
            n = sum(counts[word] for word in MARKERS)
            rows.append({"character": "Rubeus Hagrid", "medium": medium, "installment": number,
                         "feature": "yeh + ter + yer + fer", "count": n,
                         "words": counts.total(), "per_1000": 1000 * n / counts.total()})
        summaries[medium] = {"marker_count": sum(pooled[word] for word in MARKERS), "words": pooled.total()}
        summaries[medium]["per_1000"] = 1000 * summaries[medium]["marker_count"] / pooled.total()
        for word in ("educational", "decree", "punishments", "hem", "mr", "professor", "minerva"):
            counts = profiles[medium, 5, "Dolores Umbridge"]
            rows.append({"character": "Dolores Umbridge", "medium": medium, "installment": 5,
                         "feature": word, "count": counts[word], "words": counts.total(),
                         "per_1000": 1000 * counts[word] / counts.total()})
    save("character_word_rates.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), layout="constrained")
    for ax, character in zip(axes, ("Rubeus Hagrid", "Dolores Umbridge")):
        book = [r for r in rows if r["character"] == character and r["medium"] == "books"]
        film = [r for r in rows if r["character"] == character and r["medium"] == "movies"]
        y = np.arange(len(book))
        b, f = [r["per_1000"] for r in book], [r["per_1000"] for r in film]
        ax.hlines(y, np.minimum(b, f), np.maximum(b, f), color="#b9bdc2", lw=1.4)
        ax.scatter(b, y, color=BOOK_COLOR, marker="o", label="Books", s=55, zorder=3)
        ax.scatter(f, y, color=FILM_COLOR, marker="^", label="Films", s=60, zorder=3)
        labels = SHORT if character == "Rubeus Hagrid" else [r["feature"] for r in book]
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(left=-1)
        ax.set_xlabel("Occurrences per 1,000 of that character's words")
        ax.set_title("Hagrid: four accent spellings by installment" if character == "Rubeus Hagrid"
                     else "Umbridge: selected words in Order of the Phoenix", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", frameon=False)
    fig.suptitle("Character language after normalizing for dialogue length", fontsize=14)
    fig.savefig(OUT / "character_word_rates.png", dpi=160)
    plt.close(fig)
    return summaries, rows


def context_support():
    ppmi = read("results/ppmi/top_contexts.csv")
    settings = json.loads((ROOT / "results/ppmi/settings.json").read_text(encoding="utf-8"))
    window = settings["parameters"]["window"]
    requested = {(row["word"], row["context"]): row for row in ppmi}
    eligibility = {(r["medium"], r["source"], r["character"]) for r in read("results/ppmi/documents.csv")}
    wanted = defaultdict(set)
    for target, context in requested:
        wanted[target].add(context)
    evidence = defaultdict(list)
    for medium, path in (("books", "book_data/all_books.csv"), ("movies", "movie_data/all_movies.csv")):
        for index, record in enumerate(read_dialogue_records(ROOT / path), 1):
            if (medium, record.book, record.character) not in eligibility:
                continue
            tokens, local = tokenize(record.line), Counter()
            for i, target in enumerate(tokens):
                if target in wanted:
                    for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                        if i != j and tokens[j] in wanted[target]:
                            local[target, tokens[j]] += 1
            for pair, count in local.items():
                evidence[pair].append({"word": pair[0], "context": pair[1], "medium": medium,
                                       "source": record.book, "chapter": record.chapter,
                                       "character": record.character, "record_index": index,
                                       "pair_count": count, "text": record.line})
    support, excerpts = [], []
    for pair, row in requested.items():
        lines = evidence[pair]
        assert sum(r["pair_count"] for r in lines) == int(row["pair_count"])
        support.append({"word": pair[0], "context": pair[1], "ppmi": float(row["ppmi"]),
                        "pair_count": int(row["pair_count"]), "dialogue_lines": len(lines),
                        "sources": len({(r["medium"], r["source"]) for r in lines}),
                        "speaker_labels": len({r["character"] for r in lines}),
                        "book_lines": sum(r["medium"] == "books" for r in lines),
                        "film_lines": sum(r["medium"] == "movies" for r in lines)})
        # Evidence includes all lines for the reported associations, not just a chosen sample.
        excerpts.extend(lines)
    save("ppmi_support.csv", support)
    save("ppmi_evidence.csv", excerpts)
    fig, ax = plt.subplots(figsize=(10.5, 6.8), layout="constrained")
    colors = {"blood": "#217a9b", "power": "#b45535", "death": "#7662a3", "love": "#497b50"}
    shapes = {"blood": "o", "power": "^", "death": "s", "love": "D"}
    for target in colors:
        rows = [row for row in support if row["word"] == target]
        ax.scatter([r["dialogue_lines"] for r in rows], [r["ppmi"] for r in rows],
                   label=target, color=colors[target], marker=shapes[target], s=48, alpha=.8)
    offsets = {("blood", "pure"): (8, 10), ("death", "eaters"): (-85, 12),
               ("power", "punishments"): (8, 12), ("love", "impossible"): (8, -21),
               ("love", "potion"): (8, -24)}
    for row in support:
        pair = (row["word"], row["context"])
        if pair in offsets:
            ax.annotate(" / ".join(pair), (row["dialogue_lines"], row["ppmi"]),
                        xytext=offsets[pair], textcoords="offset points", fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8},
                        arrowprops={"arrowstyle": "-", "lw": .7, "color": "#777"})
    ax.set(xscale="log", xlim=(.85, 230), xlabel="Dialogue lines containing the pair (log scale)",
           ylabel="PPMI: association strength", title="Strong association can come from very few lines")
    ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Target word", frameon=False, loc="lower right")
    fig.savefig(OUT / "ppmi_support.png", dpi=160)
    plt.close(fig)
    return support


def topic_profiles():
    rows = read("results/lda/document_topics.csv")
    characters = ["Harry Potter", "Albus Dumbledore", "Rubeus Hagrid",
                  "Voldemort", "Dolores Umbridge", "Bellatrix Lestrange"]
    profiles = []
    for character in characters:
        for medium in ("books", "movies"):
            docs = [r for r in rows if r["character"] == character and r["medium"] == medium]
            total = sum(int(r["modeled_words"]) for r in docs)
            profile = {"character": character, "medium": medium, "documents": len(docs), "modeled_words": total}
            for topic in range(1, 9):
                profile[f"topic_{topic}"] = sum(int(r["modeled_words"]) * float(r[f"topic_{topic}"]) for r in docs) / total
            assert math.isclose(sum(profile[f"topic_{topic}"] for topic in range(1, 9)), 1)
            profiles.append(profile)
    save("topic_profiles.csv", profiles)
    values = np.array([[r[f"topic_{topic}"] for topic in range(1, 9)] for r in profiles])
    fig, ax = plt.subplots(figsize=(11.5, 7), layout="constrained")
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(8), [f"Topic {i}" for i in range(1, 9)])
    ax.set_yticks(range(len(profiles)), [r["character"] + " / " + r["medium"] for r in profiles])
    for i in range(len(profiles)):
        for j in range(8):
            if values[i, j] >= .025:
                ax.text(j, i, f"{values[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if values[i, j] > .55 else "#182330")
    for y in np.arange(1.5, len(profiles) - 1, 2):
        ax.axhline(y, color="white", lw=2)
    ax.set_title("LDA profiles: token-weighted averages over each character's eligible sources", fontsize=12)
    fig.colorbar(image, ax=ax, label="Mean topic proportion", shrink=.85)
    fig.savefig(OUT / "topic_profiles.png", dpi=160)
    plt.close(fig)
    shares = {}
    for topic in (2, 6, 8):
        mass = sum(int(r["modeled_words"]) * float(r[f"topic_{topic}"]) for r in rows)
        by_character, by_source = Counter(), Counter()
        for row in rows:
            value = int(row["modeled_words"]) * float(row[f"topic_{topic}"])
            by_character[row["character"]] += value
            by_source[installment(row["source"])] += value
        shares[topic] = {"character_shares": {key: value / mass for key, value in by_character.most_common(6)},
                         "installment_shares": {str(key): value / mass for key, value in sorted(by_source.items())}}
    return shares


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    profiles = word_profiles()
    hagrid, rates = language_plot(profiles)
    support = context_support()
    topics = topic_profiles()
    umbridge = {}
    for medium in ("books", "movies"):
        counts = profiles[medium, 5, "Dolores Umbridge"]
        selected = sum(counts[word] for word in ("educational", "decree", "punishments"))
        umbridge[medium] = {"word_occurrences": selected, "words": counts.total(), "per_1000": 1000 * selected / counts.total()}
    summary = {"hagrid_dialect": hagrid,
               "hagrid_rate_ratio_books_over_films": hagrid["books"]["per_1000"] / hagrid["movies"]["per_1000"],
               "umbridge_institutional_terms_phoenix": umbridge,
               "umbridge_rate_ratio_films_over_books": umbridge["movies"]["per_1000"] / umbridge["books"]["per_1000"],
               "topic_support": topics,
               "focus_ppmi_pairs": [r for r in support if (r["word"], r["context"]) in
                                    {("blood", "pure"), ("death", "eaters"), ("power", "punishments"), ("love", "impossible")}]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote diagnostic plots and evidence to {OUT}")


if __name__ == "__main__":
    main()
