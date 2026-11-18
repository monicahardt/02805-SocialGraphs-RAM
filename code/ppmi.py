"""Measure nearby word associations using positive pointwise mutual information.

Run from the repository root: python -m code.ppmi
Uses original dialogue lines, the word-count tokenizer and the same minimum
character/source word count as LDA and TF-IDF. Both media contribute to one
shared word-context matrix. All dependencies are in requirements-lda.txt.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

if __package__:
    from .dialogue_csv_reader import read_dialogue_records
    from .word_counts import tokenize
else:
    from dialogue_csv_reader import read_dialogue_records
    from word_counts import tokenize


def load_dialogue(paths: dict[str, Path], min_words: int):
    """Keep eligible character/source documents, preserving each utterance."""
    utterances, totals = [], Counter()
    for medium, path in paths.items():
        for record in read_dialogue_records(path):
            key = (medium, record.book, record.character)
            tokens = tokenize(record.line)
            totals[key] += len(tokens)
            utterances.append((key, tokens))
    eligible = {key for key, count in totals.items()
                if key[2].casefold() != "unknown" and count >= min_words}
    sequences = [tokens for key, tokens in utterances if key in eligible]
    documents = [{"medium": key[0], "source": key[1], "character": key[2], "words": totals[key]}
                 for key in sorted(eligible)]
    return sequences, documents


def build_vocabulary(sequences, min_count: int, max_words: int):
    """Select frequent words; retain names, stopwords and short tokens."""
    frequencies = Counter(word for sequence in sequences for word in sequence)
    candidates = sorted((word for word in frequencies if frequencies[word] >= min_count),
                        key=lambda word: (-frequencies[word], word))[:max_words]
    words = sorted(candidates)
    return {word: index for index, word in enumerate(words)}, frequencies


def cooccurrence(sequences, vocabulary: dict[str, int], window: int):
    """Count directed target/context pairs in a symmetric original-token window.

    Each positional pair contributes once in each direction. A token is never
    paired with itself, but two occurrences of the same word may form a pair.
    Vocabulary filtering leaves original token positions intact. Windows cannot
    cross an utterance boundary, even for the same speaker.
    """
    if window < 1:
        raise ValueError("Window must be positive.")
    pairs = Counter()
    for tokens in sequences:
        for position, word in enumerate(tokens):
            if word not in vocabulary:
                continue
            first = vocabulary[word]
            for other in tokens[position + 1:position + window + 1]:
                if other in vocabulary:
                    second = vocabulary[other]
                    pairs[first, second] += 1
                    pairs[second, first] += 1
    shape = (len(vocabulary), len(vocabulary))
    if not pairs:
        return sparse.csr_matrix(shape, dtype=np.int64)
    indices = list(pairs)
    return sparse.csr_matrix((
        [pairs[pair] for pair in indices],
        ([pair[0] for pair in indices], [pair[1] for pair in indices]),
    ), shape=shape, dtype=np.int64)


def positive_pmi(counts):
    """PPMI=max(log2(C_ij * total_pairs / (row_i * column_j)), 0).

    Unobserved pairs receive zero. Marginals use the complete supplied count
    matrix. No smoothing or rare-pair pruning is applied to that matrix.
    Copy every sparse buffer so the input counts remain unchanged.
    """
    counts = sparse.coo_matrix(counts, dtype=float, copy=True)
    if not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError("Pair counts must be finite and nonnegative.")
    counts.sum_duplicates()
    counts.eliminate_zeros()
    if not counts.nnz:
        return counts.tocsr()
    row_totals = np.asarray(counts.sum(axis=1)).ravel()
    column_totals = np.asarray(counts.sum(axis=0)).ravel()
    values = np.maximum(np.log2(counts.data * counts.sum() /
                               (row_totals[counts.row] * column_totals[counts.col])), 0)
    ppmi = sparse.csr_matrix((values, (counts.row, counts.col)), shape=counts.shape)
    ppmi.eliminate_zeros()
    ppmi.sort_indices()
    return ppmi


def strongest_contexts(counts, ppmi, vocabulary, targets, min_pair_count, top):
    """Rank direct word associations; omit self-pairs and weakly supported pairs.

    The pair-count threshold affects this report only, never PMI marginals.
    """
    words = {index: word for word, index in vocabulary.items()}
    row_totals = np.asarray(counts.sum(axis=1)).ravel()
    column_totals = np.asarray(counts.sum(axis=0)).ravel()
    total = float(counts.sum())
    result = []
    for target in dict.fromkeys(targets):
        if target not in vocabulary:
            continue
        row = vocabulary[target]
        values = ppmi.getrow(row)
        candidates = [int(column) for column in values.indices
                      if column != row and counts[row, column] >= min_pair_count]
        candidates.sort(key=lambda column: (-ppmi[row, column], words[column]))
        for rank, column in enumerate(candidates[:top], 1):
            expected = row_totals[row] * column_totals[column] / total
            result.append({"word": target, "rank": rank, "context": words[column],
                           "pair_count": int(counts[row, column]), "expected_count": expected,
                           "row_total": int(row_totals[row]),
                           "column_total": int(column_totals[column]),
                           "total_pairs": int(total), "ppmi": float(ppmi[row, column])})
    return result


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_contexts(rows, targets, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 2
    height = (len(targets) + 1) // 2
    fig, axes = plt.subplots(height, columns, figsize=(12, 3.7 * height), sharex=True, squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index >= len(targets):
            ax.set_visible(False)
            continue
        target = targets[index]
        selected = [row for row in rows if row["word"] == target][:8][::-1]
        labels = [f"{row['context']} (n={row['pair_count']})" for row in selected]
        ax.barh(labels, [row["ppmi"] for row in selected], color="#327e9a")
        ax.set_title(f"Nearby words associated with '{target}'", fontsize=11)
        ax.set_xlabel("PPMI = log2(observed / expected)", fontsize=9)
        ax.tick_params(labelsize=9, labelbottom=True)
        ax.spines[["top", "right"]].set_visible(False)
        if not selected:
            ax.text(.5, .5, "No contexts pass the reporting thresholds", ha="center", transform=ax.transAxes)
    fig.suptitle("PPMI word associations in pooled book and film dialogue\n"
                 "n = observed target/context pair count; all panels share one scale", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books", type=Path, default=Path("book_data/all_books.csv"))
    parser.add_argument("--movies", type=Path, default=Path("movie_data/all_movies.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ppmi"))
    parser.add_argument("--min-words", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--max-words", type=int, default=5000)
    parser.add_argument("--min-pair-count", type=int, default=3)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--words", nargs="+", default=["blood", "power", "death", "love"])
    args = parser.parse_args()
    if min(args.min_words, args.window, args.min_count, args.max_words, args.min_pair_count, args.top) < 1:
        parser.error("All numeric settings must be positive.")
    targets = list(dict.fromkeys(word.lower() for word in args.words))
    paths = {"books": args.books, "movies": args.movies}
    sequences, documents = load_dialogue(paths, args.min_words)
    vocabulary, frequencies = build_vocabulary(sequences, args.min_count, args.max_words)
    if not vocabulary:
        parser.error("No vocabulary words meet the minimum occurrence threshold.")
    counts = cooccurrence(sequences, vocabulary, args.window)
    if not counts.nnz:
        parser.error("No word pairs fall within the chosen windows.")
    ppmi = positive_pmi(counts)
    rows = strongest_contexts(counts, ppmi, vocabulary, targets, args.min_pair_count, args.top)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(args.output_dir / "counts.npz", counts)
    sparse.save_npz(args.output_dir / "ppmi.npz", ppmi)
    row_totals = np.asarray(counts.sum(axis=1)).ravel()
    column_totals = np.asarray(counts.sum(axis=0)).ravel()
    write_csv(args.output_dir / "vocabulary.csv", ["index", "word", "occurrences", "row_total", "column_total"], (
        {"index": index, "word": word, "occurrences": frequencies[word],
         "row_total": int(row_totals[index]), "column_total": int(column_totals[index])}
        for word, index in vocabulary.items()
    ))
    write_csv(args.output_dir / "documents.csv", ["medium", "source", "character", "words"], documents)
    write_csv(args.output_dir / "top_contexts.csv",
              ["word", "rank", "context", "pair_count", "expected_count", "row_total", "column_total", "total_pairs", "ppmi"], rows)
    plot_contexts(rows, targets, args.output_dir / "top_contexts.png")
    settings = {
        "documents": len(documents), "dialogue_lines": len(sequences),
        "word_occurrences": sum(map(len, sequences)), "vocabulary_words": len(vocabulary),
        "directed_context_pairs": int(counts.sum()), "observed_pair_cells": counts.nnz,
        "positive_ppmi_cells": ppmi.nnz,
        "formula": "max(log2(pair_count * total_pairs / (row_total * column_total)), 0)",
        "corpus": "eligible dialogue from both media pooled into one matrix",
        "boundaries": "original token positions; each dialogue line is a separate sequence",
        "parameters": {key: getattr(args, key) for key in
                       ("min_words", "window", "min_count", "max_words", "min_pair_count", "top")},
        "target_words": targets, "missing_target_words": [word for word in targets if word not in vocabulary],
        "inputs": {medium: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for medium, path in paths.items()},
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")},
    }
    (args.output_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    notes = ["# Step 4: PPMI", "",
        "PPMI measures how strongly nearby word pairs are associated relative to a baseline where target and context words are independent. It uses words and their local contexts, rather than the document/word weights used by TF-IDF.", "",
        "PPMI = max(log2(observed pair count / expected pair count), 0).",
        "Expected count = target row total * context column total / total counted pairs.",
        "For example, a PPMI of 3 corresponds to eight times the independent expected count. Zero means no positive association under this measurement; it does not prove a lack of relationship.", "",
        f"This run pools {len(documents)} eligible character/source documents containing {len(sequences):,} dialogue lines from books and films. Character/source groups require {args.min_words} raw words, and Unknown speakers are excluded, matching the previous steps' document eligibility. All original lines remain separate.",
        f"The context window extends {args.window} original word-token positions to either side within a line. Each positional pair contributes both directions. Punctuation does not end a window, but a new dialogue line always does.",
        f"Vocabulary words require at least {args.min_count} occurrences, with at most {args.max_words:,} words selected by frequency ({len(vocabulary):,} retained). Names, stopwords and short tokens remain eligible. Words outside the vocabulary still occupy their original positions. Marginals and the total count refer only to pairs with both words in the retained vocabulary.", "",
        "## Files", "",
        "- `top_contexts.csv`: direct nearby-word associations for each requested word, with observed and expected counts, marginals and PPMI. These contexts are not cosine-similarity neighbors.",
        f"- `top_contexts.png`: up to eight high-scoring contexts per target; n is the observed pair count. Reported pairs need at least {args.min_pair_count} observations, positive PPMI and a context different from the target. These reporting filters are applied after calculating the full matrices.",
        "- `counts.npz` and `ppmi.npz`: complete sparse word/context matrices. Rows are target words; columns are context words. Both use the indices in `vocabulary.csv`. Load with `scipy.sparse.load_npz`.",
        "- `vocabulary.csv`: word indices, raw occurrences and context marginals. Raw occurrences differ from row totals because one occurrence can contribute several pairs.",
        "- `documents.csv` and `settings.json`: eligibility, input hashes, software versions and calculation settings.", "",
        "PPMI scores are weights, not probabilities or significance tests. Rare pairs can receive high scores, so the report includes support counts. These pooled results describe associations in the combined dialogue, without identifying which characters use them or estimating a book/film adaptation effect. Existing speaker assignments and spelling are preserved.", "",
        "Rerun with a different window or requested words, for example:", "",
        "```powershell", ".venv\\Scripts\\python.exe -m code.ppmi --window 2 --output-dir results/ppmi_window2", "```", "",
        "[Course method: week 6, word contexts and PPMI](https://sunelehmann.com/socialgraphs2026-web/weeks/week6.html)", ""]
    (args.output_dir / "README.md").write_text("\n".join(notes), encoding="utf-8")
    print(f"Counted {int(counts.sum()):,} directed context pairs across {len(vocabulary):,} words and {len(documents)} documents.")
    for target in targets:
        selected = [row for row in rows if row["word"] == target][:5]
        print(target + ": " + ", ".join(f"{row['context']} ({row['ppmi']:.2f}; n={row['pair_count']})" for row in selected))
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
