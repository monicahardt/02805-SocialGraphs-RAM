"""Weight the verified dialogue word counts using the course TF-IDF formula.

Run from the repository root: python -m code.tfidf
One document is a character's dialogue within one book or film, with at least
100 raw words by default. Unknown-speaker documents are excluded. Names,
stopwords and short tokens are retained. Both media share one vocabulary/IDF.
Dependencies are already listed in requirements-lda.txt.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer


def load_documents(path: Path, medium: str, min_words: int) -> list[dict]:
    """Read all word counts, retaining the original source and speaker labels."""
    grouped = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["source"], row["character"])
            counts = grouped.setdefault(key, Counter())
            word, count = row["word"], int(row["count"])
            if count < 1 or word in counts:
                raise ValueError(f"Invalid or duplicate count for {key}: {word!r}")
            counts[word] = count
    return [
        {"medium": medium, "source": source, "character": character,
         "words": counts.total(), "counts": counts}
        for (source, character), counts in sorted(grouped.items())
        if character.casefold() != "unknown" and counts.total() >= min_words
    ]


def course_tfidf(counts):
    """TF=count/document length; IDF=ln(N/df); weights=TF*IDF.

    No IDF smoothing, additive constant, or final vector normalization is used.
    A word present in every document has weight zero. Empty rows are rejected.
    Return weights, document frequencies, IDFs and document lengths.
    """
    # Copy all sparse buffers so weighting cannot alter the original counts.
    counts = sparse.csr_matrix(counts, dtype=float, copy=True)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    counts.sort_indices()
    if counts.shape[0] < 1 or counts.shape[1] < 1:
        raise ValueError("TF-IDF needs documents and vocabulary words.")
    if not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError("Word counts must be finite and nonnegative.")
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    if (lengths == 0).any():
        raise ValueError("Documents must contain at least one word.")
    frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.zeros(counts.shape[1], dtype=float)
    present = frequency > 0
    idf[present] = np.log(counts.shape[0] / frequency[present])
    weights = (sparse.diags(1 / lengths) @ counts @ sparse.diags(idf)).tocsr()
    weights.eliminate_zeros()
    weights.sort_indices()
    return weights, frequency, idf, lengths


def ranked_terms(weights, row: int, limit: int):
    """Rank positive weights, breaking ties by the alphabetic vocabulary index."""
    values = weights.getrow(row)
    order = np.lexsort((values.indices, -values.data))[:limit]
    return values.indices[order]


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_results(documents, counts, vocabulary, weights, frequency, idf, output, top_words):
    output.mkdir(parents=True, exist_ok=True)
    index_rows, top_rows = [], []
    for row, document in enumerate(documents):
        metadata = {key: document[key] for key in ("medium", "source", "character", "words")}
        index_rows.append({"row": row, **metadata,
                           "unique_words": counts[row].nnz,
                           "positive_tfidf_words": weights[row].nnz})
        for rank, column in enumerate(ranked_terms(weights, row, top_words), 1):
            count = int(counts[row, column])
            top_rows.append({"row": row, **metadata, "rank": rank,
                             "word": vocabulary[column], "count": count,
                             "tf": count / document["words"],
                             "document_frequency": int(frequency[column]),
                             "idf": idf[column], "tfidf": float(weights[row, column])})
    write_csv(output / "documents.csv",
              ["row", "medium", "source", "character", "words", "unique_words", "positive_tfidf_words"], index_rows)
    write_csv(output / "top_words.csv",
              ["row", "medium", "source", "character", "words", "rank", "word", "count", "tf", "document_frequency", "idf", "tfidf"], top_rows)
    totals = np.asarray(counts.sum(axis=0)).ravel()
    write_csv(output / "vocabulary.csv", ["column", "word", "count", "document_frequency", "idf"], (
        {"column": column, "word": word, "count": int(totals[column]),
         "document_frequency": int(frequency[column]), "idf": idf[column]}
        for column, word in enumerate(vocabulary)
    ))
    sparse.save_npz(output / "weights.npz", weights)


def plot_examples(documents, vocabulary, weights, output, characters):
    """Illustrate individual documents; source titles show their story coverage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [name for name in dict.fromkeys(characters)
                 if any(doc["character"] == name for doc in documents)]
    if not available:
        return []
    fig, axes = plt.subplots(len(available), 2, figsize=(13, 3.5 * len(available)),
                             sharex=True, squeeze=False)
    selected = []
    for panel_row, character in enumerate(available):
        for panel_column, medium in enumerate(("books", "movies")):
            ax = axes[panel_row, panel_column]
            candidates = [i for i, doc in enumerate(documents)
                          if doc["character"] == character and doc["medium"] == medium]
            if not candidates:
                ax.set_visible(False)
                continue
            row = max(candidates, key=lambda i: documents[i]["words"])
            doc = documents[row]
            best = ranked_terms(weights, row, 8)[::-1]
            values = [float(weights[row, column]) for column in best]
            ax.barh(vocabulary[best], values, color="#327e9a" if medium == "books" else "#a34c4c")
            source = doc["source"].removeprefix("Harry Potter and the ")
            ax.set_title(character + "\n" + textwrap.fill(f"{medium.capitalize()}: {source}", 44), fontsize=10)
            ax.set_xlabel("TF-IDF weight (all panels share the same scale)", fontsize=9)
            ax.tick_params(labelsize=9, labelbottom=True)
            ax.spines[["top", "right"]].set_visible(False)
            selected.append({"row": row, "medium": medium, "character": character, "source": doc["source"]})
    fig.suptitle("TF-IDF: highest-weight words in selected dialogue documents\n"
                 "For each character and medium, the document with the most words is shown", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output / "top_words.png", dpi=150)
    plt.close(fig)
    write_csv(output / "plotted_documents.csv", ["row", "medium", "character", "source"], selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books", type=Path, default=Path("results/word_counts/books/character_word_counts.csv"))
    parser.add_argument("--movies", type=Path, default=Path("results/word_counts/movies/character_word_counts.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tfidf"))
    parser.add_argument("--min-words", type=int, default=100)
    parser.add_argument("--top-words", type=int, default=15)
    parser.add_argument("--characters", nargs="+", default=["Voldemort", "Dolores Umbridge", "Bellatrix Lestrange"])
    args = parser.parse_args()
    if args.min_words < 1 or args.top_words < 1:
        parser.error("--min-words and --top-words must be positive.")
    documents = load_documents(args.books, "books", args.min_words)
    documents += load_documents(args.movies, "movies", args.min_words)
    if len(documents) < 2:
        parser.error("Need at least two eligible documents to compare word rarity.")
    vectorizer = DictVectorizer(dtype=np.int64, sort=True)
    counts = vectorizer.fit_transform(doc["counts"] for doc in documents).tocsr()
    counts.sort_indices()
    vocabulary = vectorizer.get_feature_names_out()
    weights, frequency, idf, lengths = course_tfidf(counts)
    np.testing.assert_array_equal(lengths, [doc["words"] for doc in documents])
    save_results(documents, counts, vocabulary, weights, frequency, idf, args.output_dir, args.top_words)
    plotted = plot_examples(documents, vocabulary, weights, args.output_dir, args.characters)
    settings = {
        "document_definition": "character within a single book or film",
        "documents": len(documents), "vocabulary_words": len(vocabulary),
        "word_occurrences": int(counts.sum()),
        "formula": "(count / total document word tokens) * ln(N / document frequency)",
        "idf_scope": "all eligible book and film documents jointly",
        "stopwords_and_names": "retained", "idf_smoothing": False, "vector_normalization": False,
        "parameters": {"min_words": args.min_words, "top_words": args.top_words},
        "inputs": {medium: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for medium, path in (("books", args.books), ("movies", args.movies))},
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "matplotlib")},
    }
    (args.output_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    notes = ["# Step 3: TF-IDF", "",
        "TF-IDF gives a word a high weight when it forms a large share of one document and occurs in relatively few documents across the corpus.", "",
        "- TF = word count / total word tokens in the document.",
        "- IDF = ln(number of documents / number of documents containing the word).",
        "- TF-IDF = TF * IDF. A word found in every document has zero IDF.", "",
        f"This run uses {len(documents)} documents, {len(vocabulary):,} vocabulary words and {int(counts.sum()):,} word occurrences. A document is one character's dialogue within one book or film, requiring {args.min_words} raw words. Unknown speakers are excluded. The final two films remain separate.",
        "The document definition and length threshold match the LDA step. For TF-IDF, all counted words remain, including names, English stopwords and short words. There is no document-frequency cutoff. The TF denominator is the full raw document length. Both media share the same vocabulary and IDF.", "",
        "## Reading the outputs", "",
        "- `top_words.csv`: the highest-weight words in each document. Columns include the raw count, TF, document frequency, IDF and final TF-IDF weight, so each score can be checked.",
        "- `documents.csv`: metadata and matrix row indices for every eligible document.",
        "- `vocabulary.csv`: word statistics and matrix column indices. Document frequency counts documents containing a word, not total occurrences.",
        "- `weights.npz`: the complete sparse document-by-word weight matrix; load with `scipy.sparse.load_npz`.",
        "- `top_words.png`: examples for the requested characters. Each panel is one character/source document, and all panels use the same horizontal scale. These weights are not probabilities and do not sum to one.",
        "- `plotted_documents.csv`: the exact documents selected for the plot. Selection uses the greatest raw word count for that character in each medium. Source titles can differ, so the figure is illustrative rather than a controlled adaptation comparison.",
        "- `settings.json`: parameters, formulas, input hashes and package versions.", "",
        "TF-IDF describes words in individual documents. It does not create topics, identify villains, measure sentiment or prove a stable speaking style. Names, rare spellings and transcription artifacts may receive high weights. Existing speaker assignments are preserved.", "",
        "This uses the course formula directly, with no IDF smoothing or additive constant. It therefore differs from scikit-learn's default TfidfVectorizer. LDA outputs remain unchanged.", "",
        "[Course formula: week 6](https://sunelehmann.com/socialgraphs2026-web/weeks/week6.html)", ""]
    (args.output_dir / "README.md").write_text("\n".join(notes), encoding="utf-8")
    print(f"Weighted {len(documents)} documents and {len(vocabulary):,} vocabulary words. Results: {args.output_dir}")
    for example in plotted:
        terms = vocabulary[ranked_terms(weights, example["row"], 5)]
        print(f"{example['character']} ({example['medium']}, {example['source']}): " + ", ".join(terms))


if __name__ == "__main__":
    main()
