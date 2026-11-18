"""Independent PPMI arithmetic and original-position context-window checks."""

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from code.ppmi import build_vocabulary, cooccurrence, load_dialogue, positive_pmi, strongest_contexts


class PPMITests(unittest.TestCase):
    def test_ppmi_by_hand_with_negative_and_unobserved_cells(self):
        # Total 10, row totals [5,5], column totals [4,6].
        counts = sparse.csr_matrix([[4, 1], [0, 5]])
        expected = [[math.log2(4 * 10 / (5 * 4)), 0],
                    [0, math.log2(5 * 10 / (5 * 6))]]
        np.testing.assert_allclose(positive_pmi(counts).toarray(), expected)
        self.assertEqual(positive_pmi(sparse.csr_matrix((2, 2))).nnz, 0)

    def test_independent_pairs_have_zero_ppmi(self):
        self.assertEqual(positive_pmi(sparse.csr_matrix([[2, 2], [3, 3]])).nnz, 0)

    def test_windows_keep_original_distances_and_line_boundaries(self):
        vocabulary = {"power": 0, "wand": 1}
        sequences = [["power", "rareword", "wand"], ["power"], ["wand"]]
        np.testing.assert_array_equal(cooccurrence(sequences, vocabulary, 1).toarray(), [[0, 0], [0, 0]])
        np.testing.assert_array_equal(cooccurrence(sequences, vocabulary, 2).toarray(), [[0, 1], [1, 0]])

    def test_repeated_words_use_distinct_positions_and_both_directions(self):
        counts = cooccurrence([["a", "a", "b"]], {"a": 0, "b": 1}, window=1)
        np.testing.assert_array_equal(counts.toarray(), [[2, 1], [1, 0]])
        np.testing.assert_array_equal(cooccurrence([["a"]], {"a": 0}, 5).toarray(), [[0]])

    def test_input_counts_and_sparse_storage_remain_unchanged(self):
        counts = sparse.csr_matrix((np.array([2, 3, 1, 4]), np.array([1, 0, 1, 0]), np.array([0, 2, 4])), shape=(2, 2))
        indices, values = counts.indices.copy(), counts.data.copy()
        original = counts.toarray()
        positive_pmi(counts)
        np.testing.assert_array_equal(counts.indices, indices)
        np.testing.assert_array_equal(counts.data, values)
        np.testing.assert_array_equal(counts.toarray(), original)

    def test_invalid_counts_and_windows_are_rejected(self):
        for matrix in ([[-1, 2]], [[float('nan'), 1]]):
            with self.assertRaises(ValueError):
                positive_pmi(sparse.csr_matrix(matrix))
        with self.assertRaises(ValueError):
            cooccurrence([["a", "b"]], {"a": 0, "b": 1}, 0)

    def test_vocabulary_frequency_and_alphabetical_tie_breaks(self):
        vocabulary, counts = build_vocabulary([["b", "a", "c", "b", "a", "rare"]], min_count=2, max_words=1)
        self.assertEqual(vocabulary, {"a": 0})
        self.assertEqual(counts["b"], 2)

    def test_pair_support_filter_only_affects_report(self):
        counts = sparse.csr_matrix([[0, 4, 1], [4, 0, 1], [1, 1, 10]])
        ppmi = positive_pmi(counts)
        original = ppmi.copy()
        rows = strongest_contexts(counts, ppmi, {"a": 0, "b": 1, "c": 2}, ["a"], min_pair_count=3, top=5)
        self.assertEqual([row["context"] for row in rows], ["b"])
        self.assertAlmostEqual(rows[0]["expected_count"], 25 / 22)
        np.testing.assert_array_equal(ppmi.toarray(), original.toarray())

    def test_eligibility_keeps_sources_separate_and_preserves_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dialogue.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["book", "chapter", "character", "dialog"])
                writer.writerows([
                    ["Book 1", "1", "Harry", "Power power"],
                    ["Book 1", "1", "Harry", "Wand wand"],
                    ["Book 2", "1", "Harry", "Sparse"],
                    ["Book 1", "1", "Unknown", "Power power wand wand"],
                ])
            sequences, documents = load_dialogue({"books": path}, min_words=4)
        self.assertEqual(sequences, [["power", "power"], ["wand", "wand"]])
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["words"], 4)


if __name__ == "__main__":
    unittest.main()
