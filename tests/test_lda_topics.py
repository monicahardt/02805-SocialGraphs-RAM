"""Check LDA inputs, probability distributions and a known two-theme example."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from code.lda_topics import document_matrix, fit_lda, load_documents


class LDATests(unittest.TestCase):
    def test_load_keeps_source_boundaries_and_excludes_unknown_and_short_documents(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "counts.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["source", "character", "word", "count"])
                writer.writerows([
                    ["Book 1", "Harry", "the", 10], ["Book 1", "Harry", "wand", 5],
                    ["Book 2", "Harry", "spell", 12], ["Book 1", "Unknown", "wand", 100],
                    ["Book 1", "Ron", "spell", 1],
                ])
            documents = load_documents(path, "books", min_words=10)
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0]["raw_words"], 15)
        self.assertEqual(documents[0]["counts"], {"wand": 5})
        self.assertEqual(documents[1]["source"], "Book 2")

    def test_matrix_has_exact_integer_counts_and_correct_document_frequencies(self):
        documents = [{"counts": {"shared": 2, "alpha": 3, "rare": 9}},
                     {"counts": {"shared": 8, "alpha": 1, "beta": 5}},
                     {"counts": {"shared": 1, "beta": 4}}]
        kept, matrix, words = document_matrix(documents, min_df=2, max_df=.8)
        self.assertEqual(words.tolist(), ["alpha", "beta"])
        np.testing.assert_array_equal(matrix.toarray(), [[3, 0], [1, 5], [0, 4]])
        self.assertEqual(len(kept), 3)
        self.assertTrue(np.issubdtype(matrix.dtype, np.integer))

    def test_two_separate_themes_and_probability_sums(self):
        # Three fruit documents and three transport documents, with no shared terms.
        counts = sparse.csr_matrix([[40, 30, 0, 0], [30, 40, 0, 0], [35, 35, 0, 0],
                                    [0, 0, 40, 30], [0, 0, 30, 40], [0, 0, 35, 35]])
        original = counts.copy()
        mixtures, probabilities = fit_lda(counts, topics=2)
        np.testing.assert_allclose(mixtures.sum(axis=1), 1)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.assertTrue(np.isfinite(mixtures).all())
        first, second = mixtures[:3].mean(axis=0).argmax(), mixtures[3:].mean(axis=0).argmax()
        self.assertNotEqual(first, second)
        self.assertGreater(mixtures[:3, first].min(), .9)
        self.assertGreater(mixtures[3:, second].min(), .9)
        np.testing.assert_array_equal(counts.toarray(), original.toarray())

    def test_sparse_storage_order_does_not_change_results_or_input(self):
        counts = sparse.csr_matrix([[9, 8, 1, 0], [8, 9, 0, 1], [1, 0, 8, 9], [0, 1, 9, 8]])
        unsorted = counts.copy()
        for start, end in zip(unsorted.indptr[:-1], unsorted.indptr[1:]):
            unsorted.indices[start:end] = unsorted.indices[start:end][::-1]
            unsorted.data[start:end] = unsorted.data[start:end][::-1]
        unsorted.has_sorted_indices = False
        original_indices = unsorted.indices.copy()
        a, p = fit_lda(counts, topics=2)
        b, q = fit_lda(unsorted, topics=2)
        np.testing.assert_allclose(a, b)
        np.testing.assert_allclose(p, q)
        np.testing.assert_array_equal(unsorted.indices, original_indices)


if __name__ == "__main__":
    unittest.main()
