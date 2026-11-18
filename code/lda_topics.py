"""Find LDA topics using the verified character word-count tables.

Run from the repository root: python -m code.lda_topics
One document is a character's dialogue in one book or film. Both media share
one model. Raw counts feed LDA; English stopwords and words shorter than three
letters are removed for this step. Character names remain in the vocabulary.
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
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def load_documents(path: Path, medium: str, min_words: int) -> list[dict]:
    """Read step one's counts, keeping each source and character separate."""
    grouped = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["source"], row["character"])
            counts = grouped.setdefault(key, Counter())
            word, count = row["word"], int(row["count"])
            if count < 1 or word in counts:
                raise ValueError(f"Invalid or duplicate word count for {key}: {word!r}")
            counts[word] = count
    documents = []
    for (source, character), counts in sorted(grouped.items()):
        if character.casefold() == "unknown" or counts.total() < min_words:
            continue
        content = {word: count for word, count in counts.items()
                   if len(word) >= 3 and word not in ENGLISH_STOP_WORDS}
        documents.append({"medium": medium, "source": source, "character": character,
                          "raw_words": counts.total(), "counts": content})
    return documents


def document_matrix(documents: list[dict], min_df: int, max_df: float):
    """Build an integer document-term matrix and filter document frequency.

    Thresholds apply to all length-eligible documents. Documents with no words
    left after vocabulary filtering are then excluded from model fitting.
    """
    if not documents:
        raise ValueError("No documents meet the minimum word count.")
    vectorizer = DictVectorizer(dtype=np.int64, sort=True)
    counts = vectorizer.fit_transform(doc["counts"] for doc in documents).tocsr()
    frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
    keep = (frequency >= min_df) & (frequency <= max_df * len(documents))
    if not keep.any():
        raise ValueError("No vocabulary survives; lower --min-df or raise --max-df.")
    vocabulary = vectorizer.get_feature_names_out()[keep]
    counts = counts[:, keep].tocsr(copy=True)
    active = np.asarray(counts.sum(axis=1)).ravel() > 0
    documents = [doc for doc, present in zip(documents, active) if present]
    counts = counts[active].tocsr(copy=True)
    counts.sort_indices()
    if len(documents) < 2:
        raise ValueError("LDA needs at least two documents with retained words.")
    return documents, counts, vocabulary


def fit_lda(counts, topics: int = 8, seed: int = 0, iterations: int = 30):
    """Fit counts without changing the caller's sparse values or index buffers."""
    counts = sparse.csr_matrix(counts, dtype=float, copy=True)
    counts.sort_indices()
    if counts.nnz == 0 or not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError("LDA requires finite, nonnegative word counts.")
    model = LatentDirichletAllocation(
        n_components=topics, learning_method="batch", max_iter=iterations,
        random_state=seed, n_jobs=1,
    )
    mixtures = model.fit_transform(counts)
    probabilities = model.components_ / model.components_.sum(axis=1, keepdims=True)
    return mixtures, probabilities


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_results(documents, counts, vocabulary, mixtures, probabilities, output):
    output.mkdir(parents=True, exist_ok=True)
    topic_columns = [f"topic_{i + 1}" for i in range(probabilities.shape[0])]
    totals = np.asarray(counts.sum(axis=1)).ravel()
    rows = []
    for index, document in enumerate(documents):
        row = {key: document[key] for key in ("medium", "source", "character", "raw_words")}
        row["modeled_words"] = int(totals[index])
        row["dominant_topic"] = int(mixtures[index].argmax()) + 1
        row.update(dict(zip(topic_columns, mixtures[index])))
        rows.append(row)
    write_csv(output / "document_topics.csv",
              ["medium", "source", "character", "raw_words", "modeled_words", "dominant_topic", *topic_columns], rows)

    words, examples, summaries = [], [], []
    for index, topic in enumerate(probabilities):
        top = np.argsort(-topic, kind="stable")[:15]
        summaries.append(f"{index + 1}. " + ", ".join(vocabulary[top[:10]]))
        for rank, word_index in enumerate(top, 1):
            words.append({"topic": index + 1, "rank": rank,
                          "word": vocabulary[word_index], "probability": topic[word_index]})
        for rank, doc_index in enumerate(np.argsort(-mixtures[:, index], kind="stable")[:3], 1):
            doc = documents[doc_index]
            examples.append({"topic": index + 1, "rank": rank, "medium": doc["medium"],
                             "source": doc["source"], "character": doc["character"],
                             "topic_proportion": mixtures[doc_index, index]})
    write_csv(output / "topic_words.csv", ["topic", "rank", "word", "probability"], words)
    write_csv(output / "example_documents.csv",
              ["topic", "rank", "medium", "source", "character", "topic_proportion"], examples)
    write_csv(output / "vocabulary.csv", ["word"], ({"word": word} for word in vocabulary))
    print("\n".join(summaries))
    return summaries


def plot_topics(probabilities, vocabulary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 2
    rows = (len(probabilities) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, 2.7 * rows), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index >= len(probabilities):
            ax.set_visible(False)
            continue
        topic = probabilities[index]
        best = np.argsort(-topic, kind="stable")[:8][::-1]
        ax.barh(vocabulary[best], topic[best], color="#327e9a")
        ax.set_title(f"Topic {index + 1}", fontsize=11)
        ax.set_xlabel("Word probability within topic", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("LDA: shared topics across book and film dialogue", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books", type=Path, default=Path("results/word_counts/books/character_word_counts.csv"))
    parser.add_argument("--movies", type=Path, default=Path("results/word_counts/movies/character_word_counts.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/lda"))
    parser.add_argument("--topics", type=int, default=8)
    parser.add_argument("--min-words", type=int, default=100)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if min(args.min_words, args.min_df, args.iterations) < 1 or args.topics < 2 or not 0 < args.max_df <= 1:
        parser.error("Thresholds and iterations must be positive; topics >= 2; 0 < max-df <= 1.")
    documents = load_documents(args.books, "books", args.min_words)
    documents += load_documents(args.movies, "movies", args.min_words)
    candidate_count = len(documents)
    documents, counts, vocabulary = document_matrix(documents, args.min_df, args.max_df)
    mixtures, probabilities = fit_lda(counts, args.topics, args.seed, args.iterations)
    if not np.allclose(mixtures.sum(axis=1), 1) or not np.allclose(probabilities.sum(axis=1), 1):
        raise RuntimeError("Topic probability normalization failed.")
    summaries = save_results(documents, counts, vocabulary, mixtures, probabilities, args.output_dir)
    plot_topics(probabilities, vocabulary, args.output_dir / "topic_words.png")
    metadata = {
        "document_definition": "character within a single book or film",
        "model": "one shared batch LDA for both media",
        "parameters": {key: getattr(args, key) for key in ("topics", "min_words", "min_df", "max_df", "seed", "iterations")},
        "candidate_documents": candidate_count, "modeled_documents": len(documents),
        "vocabulary_words": len(vocabulary), "modeled_word_occurrences": int(counts.sum()),
        "inputs": {medium: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for medium, path in (("books", args.books), ("movies", args.movies))},
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "matplotlib")},
        "removed_english_stopwords": sorted(ENGLISH_STOP_WORDS),
    }
    (args.output_dir / "settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    description = ["# Step 2: LDA topics", "",
        "LDA groups words that tend to occur in the same documents. Each topic is a distribution over words; each document has a mixture of topics.", "",
        f"This run fits {args.topics} shared topics to {len(documents)} documents using {len(vocabulary):,} vocabulary words and {int(counts.sum()):,} word occurrences.",
        "One document is a character's dialogue within one book or film. The final two films stay separate, as in the word-count tables. Both media use the same model, making topic IDs comparable.", "",
        f"Documents need {args.min_words} raw words. Unknown-speaker documents are excluded. English stopwords and tokens shorter than three letters are removed; names and contractions remain.",
        f"Vocabulary words must occur in at least {args.min_df} and at most {args.max_df:.0%} of the length-eligible documents. Any empty documents after filtering are dropped ({candidate_count - len(documents)} in this run). LDA uses counts. Seed={args.seed}; iterations={args.iterations}.", "",
        "## Topic words", "", *summaries, "",
        "## Files", "",
        "- `topic_words.csv` and `topic_words.png`: most probable words in each topic. Probabilities use the full topic vocabulary, not just the displayed words.",
        "- `document_topics.csv`: every modeled document and its topic proportions, which sum to one.",
        "- `example_documents.csv`: three documents with the highest proportion of each topic. Check their dialogue in the source CSV before interpreting a topic.",
        "- `vocabulary.csv` and `settings.json`: modeled words, filtering settings, input hashes and software versions.", "",
        "These are exploratory topics. Names can dominate; dialect and scene vocabulary can also form topics. Topic names are human interpretations, and a topic is not a villain label. The existing speaker assignments remain as supplied. Eight topics is a starting setting, not a validated optimum.", "",
        "[Course method: week 6, topic modeling](https://sunelehmann.com/socialgraphs2026-web/weeks/week6.html)", ""]
    (args.output_dir / "README.md").write_text("\n".join(description), encoding="utf-8")
    print(f"\nModeled {len(documents)} documents and {len(vocabulary):,} words. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
