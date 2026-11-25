"""Build the offline comparison of TF-IDF, Word2Vec and BERT dialogue profiles.

Run python -m code.embedding_comparison with NumPy, SciPy, NetworkX, PyTorch
and Sentence Transformers installed. Use --offline and --bert-cache-dir with a cached
checkpoint. The resulting HTML includes all data needed to open it offline.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from scipy.sparse import load_npz
from sentence_transformers import SentenceTransformer
from torch.nn import Embedding
from torch.nn import functional as F

from .dialogue_csv_reader import read_dialogue_records
from .interaction_graph import NON_SPECIFIC_MOVIE_LABELS, _js_json
from .word_counts import tokenize

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "vocabulary_map"
SOURCE_PATHS = {
    "books": ROOT / "book_data" / "all_books.csv",
    "movies": ROOT / "movie_data" / "all_movies.csv",
}
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SEED = 42
WORD2VEC_DIMENSIONS = 64
WORD2VEC_EPOCHS = 5


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_profiles():
    """Reproduce the paired vocabulary map's speaker and installment coverage."""
    nodes = sorted(read_table(REFERENCE / "nodes.csv"), key=lambda row: int(row["id"]))
    aliases = json.loads((REFERENCE / "metrics.json").read_text(encoding="utf-8"))["aliases"]
    titles = (
        "Philosopher's Stone", "Chamber of Secrets", "Prisoner of Azkaban",
        "Goblet of Fire", "Order of the Phoenix", "Half-Blood Prince", "Deathly Hallows",
    )
    excluded = NON_SPECIFIC_MOVIE_LABELS | {"unknown", "wakanda"}
    passages = defaultdict(list)
    training_turns = []
    centrality = {}
    graph_sizes = {}
    for medium, path in SOURCE_PATHS.items():
        graph = nx.Graph()
        previous_speaker = previous_segment = None
        for record in read_dialogue_records(path):
            speaker = aliases.get(record.character, record.character)
            tokens = tokenize(record.line)
            if (
                speaker.casefold() in excluded
                or record.confidence.casefold() in {"low", "unattributed"}
                or not tokens
            ):
                previous_speaker = previous_segment = None
                continue
            installment = next(i for i, title in enumerate(titles, 1) if title in record.book)
            segment = (record.book, record.chapter)
            passages[medium, installment, speaker].append(record.line)
            training_turns.append(tokens)
            graph.add_node(speaker)
            if previous_segment == segment and previous_speaker != speaker:
                graph.add_edge(previous_speaker, speaker)
            previous_speaker, previous_segment = speaker, segment
        centrality[medium] = nx.betweenness_centrality(graph, normalized=True)
        graph_sizes[medium] = [len(graph), graph.number_of_edges()]

    dialogues = []
    for node in nodes:
        medium, character = node["medium"], node["character"]
        installments = [int(number) for number in node["installments"].split()]
        lines = [
            line
            for number in installments
            for line in passages[medium, number, character]
        ]
        if sum(len(tokenize(line)) for line in lines) != int(node["words"]):
            raise ValueError(f"Dialogue and vocabulary map disagree for {character} ({medium}).")
        dialogues.append(lines)
    return nodes, dialogues, training_turns, centrality, graph_sizes


def skipgram_pairs(turn_ids, tokens, retain, rng):
    """Sample Word2Vec context pairs without crossing an utterance boundary."""
    keep = rng.random(len(tokens)) < retain[tokens]
    words, utterances = tokens[keep], turn_ids[keep]
    windows = rng.integers(1, 6, size=len(words))
    centers, contexts = [], []
    for offset in range(1, 6):
        same_turn = utterances[:-offset] == utterances[offset:]
        forward = same_turn & (windows[:-offset] >= offset)
        backward = same_turn & (windows[offset:] >= offset)
        centers.extend((words[:-offset][forward], words[offset:][backward]))
        contexts.extend((words[offset:][forward], words[:-offset][backward]))
    return np.concatenate(centers), np.concatenate(contexts)


def word2vec(turns, tfidf, features):
    """Train shared skip-gram with negative sampling and use TF-IDF pooling."""
    counts = Counter(token for turn in turns for token in turn)
    vocab = {word: i for i, word in enumerate(sorted(w for w in counts if counts[w] >= 3))}
    encoded = [[vocab[word] for word in turn if word in vocab] for turn in turns]
    tokens = np.fromiter((word for turn in encoded for word in turn), dtype=np.int64)
    turn_ids = np.repeat(np.arange(len(encoded)), [len(turn) for turn in encoded])
    frequencies = np.array([counts[w] for w in vocab], dtype=float)
    frequencies /= frequencies.sum()
    retain = np.minimum(1, (np.sqrt(frequencies / 0.001) + 1) * 0.001 / frequencies)
    noise = frequencies**0.75
    noise /= noise.sum()

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    center = Embedding(len(vocab), WORD2VEC_DIMENSIONS, sparse=True)
    context = Embedding(len(vocab), WORD2VEC_DIMENSIONS, sparse=True)
    with torch.no_grad():
        center.weight.uniform_(-0.5 / WORD2VEC_DIMENSIONS, 0.5 / WORD2VEC_DIMENSIONS)
        context.weight.zero_()
    optimizer = torch.optim.SGD([center.weight, context.weight], lr=0.025)
    batch_size = 512
    for epoch in range(WORD2VEC_EPOCHS):
        left, right = skipgram_pairs(turn_ids, tokens, retain, rng)
        order = rng.permutation(len(left))
        neg = rng.choice(len(vocab), size=(len(left), 5), p=noise)
        optimizer.param_groups[0]["lr"] = np.interp(epoch, [0, WORD2VEC_EPOCHS - 1], [0.025, 0.005])
        mean_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            x = center(torch.from_numpy(left[batch]))
            y = context(torch.from_numpy(right[batch]))
            negatives = context(torch.from_numpy(neg[batch]))
            positive_score = (x * y).sum(dim=1)
            negative_score = torch.bmm(negatives, x.unsqueeze(-1)).squeeze(-1)
            loss = -(F.logsigmoid(positive_score).sum() + F.logsigmoid(-negative_score).sum())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()
        print(f"Word2Vec epoch {epoch + 1}: {len(left):,} pairs, loss {mean_loss / len(left):.3f}", flush=True)

    lookup = np.array([vocab.get(word, -1) for word in features])
    covered = lookup >= 0
    vectors = tfidf[:, covered] @ center.weight.detach().numpy()[lookup[covered]]
    return np.asarray(vectors)


def split_long_utterance(text, tokenizer, max_length):
    """Apply the BERT notebooks' recursive split at word boundaries."""
    token_count = len(tokenizer.encode(text, add_special_tokens=True, truncation=False, verbose=False))
    words = text.split()
    if token_count <= max_length or len(words) <= 1:
        return [text]
    midpoint = len(words) // 2
    return (
        split_long_utterance(" ".join(words[:midpoint]), tokenizer, max_length)
        + split_long_utterance(" ".join(words[midpoint:]), tokenizer, max_length)
    )


def bert(dialogues, model_name, cache_dir, offline):
    """Use the token-weighted passage pooling from hp_embeddings_bert.ipynb."""
    model = SentenceTransformer(
        model_name,
        cache_folder=str(cache_dir) if cache_dir is not None else None,
        local_files_only=offline,
        revision=DEFAULT_MODEL_REVISION if model_name == DEFAULT_MODEL else None,
        device="cpu",
    )
    passages_by_profile = [
        [
            passage
            for line in lines
            for passage in split_long_utterance(line, model.tokenizer, model.max_seq_length)
        ]
        for lines in dialogues
    ]
    passages = list(dict.fromkeys(passage for profile in passages_by_profile for passage in profile))
    lookup = {passage: i for i, passage in enumerate(passages)}
    print(f"BERT: encoding {len(passages):,} distinct passages", flush=True)
    embeddings = model.encode(
        passages, batch_size=32, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    tokenized = model.tokenizer(passages, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
    weights = np.array([max(1, len(ids)) for ids in tokenized])
    profiles = []
    for profile_passages in passages_by_profile:
        indices = [lookup[passage] for passage in profile_passages]
        profile = np.average(embeddings[indices], axis=0, weights=weights[indices])
        profiles.append(profile / np.linalg.norm(profile))
    return np.stack(profiles)


def project(vectors, dimensions=2):
    """Fit a joint PCA to normalized book and film vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (norms == 0).any():
        raise ValueError("Every embedding profile must have a finite, nonzero vector.")
    normalized = vectors / norms
    centered = normalized - normalized.mean(axis=0)
    values, components = np.linalg.eigh(centered @ centered.T)
    if len(values) < dimensions or values[-dimensions] <= 0:
        raise ValueError(f"At least {dimensions} independent dimensions are needed for the projection.")
    leading = values[-dimensions:][::-1]
    scores = components[:, -dimensions:][:, ::-1] * np.sqrt(leading)
    signs = np.sign(scores[np.abs(scores).argmax(axis=0), np.arange(dimensions)])
    return {
        "positions": np.round(scores * signs, 6).tolist(),
        "variance": np.round(100 * leading / values.sum(), 1).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bert-model", default=DEFAULT_MODEL)
    parser.add_argument("--bert-cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "embedding_comparison" / "index.html")
    args = parser.parse_args()

    template = (ROOT / "code" / "templates" / "embedding_comparison.html").read_text(encoding="utf-8")
    torch.set_num_threads(4)
    nodes, dialogues, turns, centrality, graph_sizes = read_profiles()
    tfidf = load_npz(REFERENCE / "vectors.npz").tocsr()
    features = [row["word"] for row in read_table(REFERENCE / "vocabulary.csv")]
    if tfidf.shape != (len(nodes), len(features)):
        raise ValueError("Vocabulary map vectors do not align with its character and word tables.")
    tfidf_dense = tfidf.toarray()
    bert_dense = bert(dialogues, args.bert_model, args.bert_cache_dir, args.offline)
    projections = {
        "tfidf": project(tfidf_dense),
        "word2vec": project(word2vec(turns, tfidf, features)),
        "bert": project(bert_dense),
    }
    profiles = [
        {
            "name": row["character"],
            "medium": row["medium"],
            "words": int(row["words"]),
            "installments": [int(number) for number in row["installments"].split()],
            "centrality": round(centrality[row["medium"]][row["character"]], 9),
        }
        for row in nodes
    ]
    payload = {
        "profiles": profiles,
        "projections": projections,
        "projections3d": {"tfidf": project(tfidf_dense, 3), "bert": project(bert_dense, 3)},
        "graphSizes": graph_sizes,
        "bertModel": args.bert_model if not Path(args.bert_model).is_dir() else Path(args.bert_model).name,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace("__EMBEDDING_DATA__", _js_json(payload)), encoding="utf-8")
    print(f"Wrote {len(profiles)} profiles and three projections to {args.output}", flush=True)


if __name__ == "__main__":
    main()
