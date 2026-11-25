"""Compare dialogue meaning across adaptations using sentence embeddings.

Requires requirements-map.txt plus PyTorch and Transformers from the BERT setup.
Run from the project root with: python interaction-language/adaptation_semantics.py .
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from adaptation_vocabulary import SIMILARITY_CHARACTERS, render_similarity
from analyse import MEDIUMS, MIN_WORDS, SHORT_TITLES, read_project, write_rows
from figures import plt


MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
FIGURE_NAME = '04_adaptation_semantic_similarity'
PASSAGE_TOKENS = 126  # Leave room for [CLS] and [SEP] within 128 positions.
BATCH_SIZE = 32
CAST_SIZE = 30


def load_encoder(model_dir):
    source = str(model_dir) if model_dir is not None else MODEL_ID
    options = {'local_files_only': True} if model_dir is not None else {'revision': MODEL_REVISION}
    tokenizer = AutoTokenizer.from_pretrained(source, **options)
    model = AutoModel.from_pretrained(source, **options).eval()
    return model, tokenizer


def dialogue_passages(blocks, story, names, tokenizer):
    """Group the matched characters' dialogue into passages within each source block."""
    names = set(names)
    passages = defaultdict(list)
    for medium in MEDIUMS:
        for block in blocks[story, medium]:
            turns = defaultdict(list)
            for line in block.lines:
                if line.speaker in names:
                    turns[line.speaker].append(line.text)
            for name, dialogue in turns.items():
                tokens = tokenizer.encode(' '.join(dialogue), add_special_tokens=False, verbose=False)
                passages[story, medium, name].extend(
                    tokens[start:start + PASSAGE_TOKENS] for start in range(0, len(tokens), PASSAGE_TOKENS)
                )
    return passages


def matched_passages(blocks, counts, tokenizer):
    """Keep the vocabulary graph's matching rule and group speech within source blocks."""
    rows, passages = [], defaultdict(list)
    for story, title in enumerate(SHORT_TITLES, 1):
        names = [name for name in SIMILARITY_CHARACTERS if all(
            name in counts[story, medium] and counts[story, medium][name].total() >= MIN_WORDS
            for medium in MEDIUMS
        )]
        passages.update(dialogue_passages(blocks, story, names, tokenizer))
        for name in names:
            rows.append({
                'story': story, 'title': title, 'character': name,
                'book_words': counts[story, 'Books'][name].total(),
                'film_words': counts[story, 'Films'][name].total(),
                'book_passages': len(passages[story, 'Books', name]),
                'film_passages': len(passages[story, 'Films', name]),
            })
    return rows, passages


@torch.inference_mode()
def encode_profiles(model, tokenizer, passages):
    """Mean-pool each passage, then combine passages weighted by WordPiece count."""
    examples = sorted(((key, tokens) for key, profile in passages.items() for tokens in profile),
                      key=lambda item: len(item[1]))
    totals = defaultdict(lambda: np.zeros(model.config.hidden_size))
    print(f'Encoding {len(examples):,} dialogue passages with {MODEL_ID}', flush=True)
    for start in range(0, len(examples), BATCH_SIZE):
        batch = examples[start:start + BATCH_SIZE]
        features = [{'input_ids': tokenizer.build_inputs_with_special_tokens(tokens)} for _, tokens in batch]
        inputs = tokenizer.pad(features, return_tensors='pt')
        output = model(**inputs).last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1)
        embeddings = torch.nn.functional.normalize(pooled, dim=1).numpy()
        for (key, tokens), vector in zip(batch, embeddings):
            totals[key] += vector * len(tokens)
    return {key: vector / np.linalg.norm(vector) for key, vector in totals.items()}


def matched_cast_profiles(blocks, stories, model_dir):
    """Encode the characters used in the matched book and film analysis."""
    model, tokenizer = load_encoder(model_dir)
    passages = {}
    for record in stories:
        passages.update(dialogue_passages(blocks, record['story'], record['characters'], tokenizer))
    return encode_profiles(model, tokenizer, passages)


def top_matched_profiles(blocks, stories, model_dir):
    """Select and encode the matched characters with the most dialogue lines."""
    counts = Counter()
    for record in stories:
        matched = set(record['characters'])
        for medium in MEDIUMS:
            for block in blocks[record['story'], medium]:
                counts.update(line.speaker for line in block.lines if line.speaker in matched)
    names = sorted(counts, key=lambda name: (-counts[name], name))[:CAST_SIZE]
    selected = set(names)
    selected_stories = [{**record, 'characters': [name for name in record['characters'] if name in selected]}
                        for record in stories]
    return names, matched_cast_profiles(blocks, selected_stories, model_dir)


def mean_cast_profiles(names, profiles):
    """Average each character's matched stories separately for books and films."""
    by_character = defaultdict(list)
    for (_, medium, name), vector in profiles.items():
        by_character[medium, name].append(vector)
    means = np.array([[np.mean(by_character[medium, name], axis=0) for name in names]
                      for medium in MEDIUMS])
    return means / np.linalg.norm(means, axis=-1, keepdims=True)


def top_cast_cosines(repo, stories, model_dir):
    """Compare the averaged embeddings of every pair in the selected cast."""
    blocks, _, _, _ = read_project(repo)
    names, profiles = top_matched_profiles(blocks, stories, model_dir)
    vectors = mean_cast_profiles(names, profiles)
    return names, np.clip(vectors @ vectors.swapaxes(-1, -2), -1, 1)


def matched_cast_matrices(repo, stories, matrices, model_dir):
    """Replace within-medium TF–IDF similarities with embedding cosine scores."""
    blocks, _, _, _ = read_project(repo)
    profiles = matched_cast_profiles(blocks, stories, model_dir)
    results = []
    for record, data in zip(stories, matrices, strict=True):
        story = record['story']
        story_matrices = {}
        for medium in MEDIUMS:
            vectors = np.stack([profiles[story, medium, name] for name in record['characters']])
            similarities = np.clip(vectors @ vectors.T, -1, 1)
            story_matrices[medium] = (data[medium][0], similarities)
        results.append(story_matrices)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo', type=Path, help='Repository containing the dialogue CSVs.')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'results' / 'interaction_language')
    parser.add_argument('--model-dir', type=Path, help='Local copy of the pinned MiniLM checkpoint.')
    args = parser.parse_args()

    model, tokenizer = load_encoder(args.model_dir)
    blocks, counts, _, _ = read_project(args.repo.resolve())
    rows, passages = matched_passages(blocks, counts, tokenizer)
    profiles = encode_profiles(model, tokenizer, passages)
    for row in rows:
        story, name = row['story'], row['character']
        cosine = profiles[story, 'Books', name] @ profiles[story, 'Films', name]
        row['own_film_cosine'] = float(np.clip(cosine, -1, 1))

    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'svg.fonttype': 'path',
                         'svg.hashsalt': 'hp-adaptation-semantic-similarity'})
    write_rows(args.out / f'{FIGURE_NAME}.csv', rows)
    limits = (-1, 1) if min(row['own_film_cosine'] for row in rows) < 0 else (0, 1)
    render_similarity(args.out, rows, figure_name=FIGURE_NAME,
                      ylabel='Book-to-film dialogue similarity (MiniLM cosine)', limits=limits)
    print(f'Saved {len(rows)} dialogue comparisons to {args.out.resolve()}')


if __name__ == '__main__':
    main()
