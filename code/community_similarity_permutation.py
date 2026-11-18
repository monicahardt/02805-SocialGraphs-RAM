"""Permutation test: are same-community dialogue profiles more TF-IDF-similar?

For each medium and edge rule, hold the interaction-derived communities and
their sizes fixed, randomly reassign eligible character profiles to community
labels, and compare the within-minus-between mean cosine gap. This tests the
community/language association beyond what community sizes alone produce.

Run: python3 code/community_similarity_permutation.py [permutations]
"""

import csv
import sys
from pathlib import Path

import numpy as np

from big_graphs import RULES, _cosine, _line_data, _profile_vectors, _read_inputs, build_data


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "analysis" / "community_similarity_permutation.csv"
SEED = 20261118


def holm_adjust(p_values):
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def run(permutations=10_000):
    graph1, graph2, _graph3, _counts = build_data()
    corpora = _read_inputs()
    _lines, profile_counts, _installment_counts, _raw_words, _utterances, _sequences, _rows = _line_data(corpora)
    vectors = _profile_vectors(profile_counts)
    rng = np.random.default_rng(SEED)
    results = []

    for medium in ("books", "movies"):
        for rule in RULES:
            eligible_names = [point["name"] for point in graph1[medium][rule]]
            community_by_name = {node["id"]: node["community"] for node in graph2[medium][rule]["nodes"]}
            names = [name for name in eligible_names if name in community_by_name and (medium, name) in vectors]
            labels = np.array([community_by_name[name] for name in names])
            left, right = np.triu_indices(len(names), k=1)
            similarities = np.array([
                _cosine(vectors[(medium, names[i])], vectors[(medium, names[j])])
                for i, j in zip(left, right)
            ], dtype=float)
            same = labels[left] == labels[right]
            observed_within = float(similarities[same].mean())
            observed_between = float(similarities[~same].mean())
            observed_gap = observed_within - observed_between

            null_gaps = np.empty(permutations, dtype=float)
            for iteration in range(permutations):
                shuffled = rng.permutation(labels)
                permuted_same = shuffled[left] == shuffled[right]
                null_gaps[iteration] = similarities[permuted_same].mean() - similarities[~permuted_same].mean()

            p_value = (1 + int(np.count_nonzero(null_gaps >= observed_gap))) / (permutations + 1)
            results.append({
                "medium": medium,
                "edge_rule": rule,
                "eligible_characters": len(names),
                "communities": len(set(labels.tolist())),
                "within_pairs": int(same.sum()),
                "between_pairs": int((~same).sum()),
                "observed_within_mean": observed_within,
                "observed_between_mean": observed_between,
                "observed_gap": observed_gap,
                "null_mean_gap": float(null_gaps.mean()),
                "null_95th_percentile": float(np.quantile(null_gaps, 0.95)),
                "permutations": permutations,
                "p_one_sided": p_value,
            })

    adjusted = holm_adjust([result["p_one_sided"] for result in results])
    for result, p_adjusted in zip(results, adjusted):
        result["p_holm_8_tests"] = p_adjusted

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return results


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    output = run(count)
    for row in output:
        print(
            f"{row['medium']:6s} {row['edge_rule']:7s} n={row['eligible_characters']:3d} "
            f"within={row['observed_within_mean']:.4f} between={row['observed_between_mean']:.4f} "
            f"gap={row['observed_gap']:.4f} null95={row['null_95th_percentile']:.4f} "
            f"p={row['p_one_sided']:.5f} Holm={row['p_holm_8_tests']:.5f}"
        )
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
