"""Create a shared PaCMAP map of book/movie BERT character embeddings.

Install requirements-bert-map.txt, then run:
    python3 code/build_bert_character_map.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from big_graphs import ROOT, _canonical, _line_data, _read_inputs
from interaction_graph import NON_SPECIFIC_MOVIE_LABELS


OUTPUT = ROOT / "results" / "bert_character_map" / "pacmap_coordinates.json"


def _split_long(text, tokenizer, max_length):
    token_count = len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
    if token_count <= max_length:
        return [text]
    words = text.split()
    if len(words) <= 1:
        return [text]
    midpoint = len(words) // 2
    return _split_long(" ".join(words[:midpoint]), tokenizer, max_length) + _split_long(
        " ".join(words[midpoint:]), tokenizer, max_length
    )


def main():
    try:
        import pacmap
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "BERT map dependencies are missing. Install them with: "
            "python3 -m pip install -r requirements-bert-map.txt"
        ) from exc

    corpora = _read_inputs()
    _lines, profile_counts, _installment_counts, _raw_words, utterances, _sequences, _rows = _line_data(corpora)
    dialogues = defaultdict(list)
    for medium, records in corpora.items():
        for row in records:
            name = row.character.strip()
            if name.casefold() == "unknown":
                continue
            name = _canonical(name)
            if medium == "movies" and name.casefold() in NON_SPECIFIC_MOVIE_LABELS:
                continue
            if sum(profile_counts.get((medium, name), {}).values()) >= 100 and utterances.get((medium, name), 0) >= 5:
                if row.line.strip():
                    dialogues[(medium, name)].append(row.line.strip())

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer, max_length = model.tokenizer, model.max_seq_length
    vectors, keys = [], []
    for key in sorted(dialogues, key=lambda item: (item[0], item[1].casefold())):
        passages = []
        for dialogue in dialogues[key]:
            passages.extend(_split_long(dialogue, tokenizer, max_length))
        if not passages:
            continue
        passage_vectors = model.encode(
            passages, batch_size=32, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        weights = np.array([
            max(1, len(tokenizer.encode(text, add_special_tokens=False))) for text in passages
        ], dtype=np.float32)
        vector = np.average(passage_vectors, axis=0, weights=weights)
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        vectors.append(vector)
        keys.append(key)

    matrix = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
    reducer = pacmap.PaCMAP(
        n_components=2, n_neighbors=5, MN_ratio=0.5, FP_ratio=1.0,
        knn_backend="faiss", verbose=True, random_state=42,
    )
    coordinates = reducer.fit_transform(matrix, init="pca")
    result = {"books": {}, "movies": {}}
    for (medium, name), (x, y) in zip(keys, coordinates):
        result[medium][name] = [round(float(x), 6), round(float(y), 6)]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote a shared PaCMAP map for {len(keys)} character-medium profiles to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
