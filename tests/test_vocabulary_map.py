"""Checks for source matching, cosine edges and character-level label testing."""

import csv
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np

from code.vocabulary_map import matched_profiles, vocabulary_vectors, nearest_graph, label_check, connection_payload


class VocabularyMapTests(unittest.TestCase):
    def test_profiles_match_installments_and_merge_final_films_and_riddle_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = {medium: Path(temporary) / (medium + ".csv") for medium in ("books", "movies")}
            rows = {
                "books": [("Philosopher's Stone", "Harry Potter", 5),
                          ("Chamber of Secrets", "Harry Potter", 100),
                          ("Deathly Hallows", "Voldemort", 8)],
                "movies": [("Philosopher's Stone", "Harry Potter", 4),
                           ("Deathly Hallows Part 1", "Tom Riddle", 3),
                           ("Deathly Hallows Part 2", "Voldemort", 5)],
            }
            for medium, path in paths.items():
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["source", "character", "word", "count"])
                    writer.writerows(("Harry Potter and the " + title, name, "wand", count) for title, name, count in rows[medium])
            profiles, excluded = matched_profiles(paths, min_words=3)
        self.assertEqual(len(profiles), 4)
        self.assertEqual([p["words"] for p in profiles], [5, 4, 8, 8])
        self.assertEqual([p["installments"] for p in profiles], ["1", "1", "7", "7"])
        self.assertEqual(profiles[-1]["character"], "Voldemort")
        self.assertFalse(excluded)

    def test_cosine_matches_hand_calculation_and_preserves_counts(self):
        profiles = [
            {"character": "A", "counts": {"power": 3, "wand": 1}},
            {"character": "A", "counts": {"power": 2, "wand": 2}},
            {"character": "B", "counts": {"love": 3, "spell": 1}},
            {"character": "B", "counts": {"love": 2, "spell": 2}},
        ]
        before = [dict(p["counts"]) for p in profiles]
        vectors, similarity, words, lengths = vocabulary_vectors(profiles)
        self.assertAlmostEqual(similarity[0, 1], 8 / np.sqrt(10 * 8))
        self.assertEqual(similarity[0, 2], 0)
        np.testing.assert_allclose(similarity, similarity.T)
        np.testing.assert_allclose(np.asarray(vectors.multiply(vectors).sum(axis=1)).ravel(), 1)
        self.assertEqual(before, [dict(p["counts"]) for p in profiles])

    def test_knn_edges_use_high_dimensional_similarity_and_exclude_self(self):
        similarities = np.array([[1, .8, .1], [.8, 1, .6], [.1, .6, 1]])
        graph = nearest_graph(similarities, k=1)
        self.assertEqual({tuple(sorted(edge)) for edge in graph.edges}, {(0, 1), (1, 2)})
        self.assertAlmostEqual(graph[1][2]["weight"], .6)
        self.assertEqual(list(nx.selfloop_edges(graph)), [])

    def test_label_check_excludes_identity_edges_and_weights_remaining_edges(self):
        profiles = [{"character": name} for name in ("A", "A", "B", "B", "C", "C", "D", "D")]
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, 100), (2, 3, 100), (0, 2, 1),
                                       (1, 3, 1), (4, 6, 1), (5, 7, 1), (0, 4, .1)])
        roles = {"A": "hero", "B": "hero", "C": "villain", "D": "villain"}
        result = label_check(graph, profiles, roles, permutations=100)
        self.assertEqual(result["eligible_edges"], 5)
        self.assertEqual(result["eligible_characters"], 4)
        self.assertAlmostEqual(result["same_role_weight_fraction"], 4 / 4.1)
        self.assertAlmostEqual(result["same_role_neighbor_share"]["hero"], 4 / 4.1)
        self.assertAlmostEqual(result["same_role_neighbor_share"]["villain"], 4 / 4.1)
        self.assertEqual(result, label_check(graph, profiles, roles, permutations=100))

    def test_connection_details_match_graph_edges_and_keep_counterpart_separate(self):
        profiles = [{"character": name} for name in ("A", "A", "B", "C", "D")]
        graph = nx.Graph()
        graph.add_nodes_from(range(5))
        graph.add_weighted_edges_from([(0, 1, .9), (0, 2, .4), (0, 3, .2), (1, 4, .6)])
        details = [{"source": a, "target": b, "cosine": weight,
                    "shared_terms": "wand, power" if b == 2 else "spell"}
                   for a, b, weight in [(0, 1, .9), (0, 2, .4), (0, 3, .2), (1, 4, .6)]]
        connections = connection_payload(graph, profiles, details)
        self.assertEqual([r["id"] for r in connections["0"]], [2, 3])
        self.assertEqual(connections["0"][0]["shared_terms"], ["wand", "power"])
        self.assertEqual([r["id"] for r in connections["1"]], [4])
        self.assertEqual(connections["0"][0]["cosine"], .4)
        self.assertEqual([r["id"] for r in connections["2"]], [0])


if __name__ == "__main__":
    unittest.main()
