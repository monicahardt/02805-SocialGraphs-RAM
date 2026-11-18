"""Independent TF-IDF arithmetic, document boundaries and input preservation."""

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from code.tfidf import course_tfidf, load_documents, ranked_terms


class TFIDFTests(unittest.TestCase):
    def test_course_example_with_universal_word(self):
        # Columns: the, superpower, mutant, other; each document has 7 words.
        counts = sparse.csr_matrix([[1, 1, 1, 4], [1, 1, 0, 5], [2, 0, 0, 5]])
        weights, frequency, idf, lengths = course_tfidf(counts)
        np.testing.assert_array_equal(lengths, [7, 7, 7])
        np.testing.assert_array_equal(frequency, [3, 2, 1, 3])
        np.testing.assert_allclose(weights[:, 0].toarray(), 0)
        self.assertAlmostEqual(weights[0, 1], math.log(3 / 2) / 7)
        self.assertAlmostEqual(weights[0, 2], math.log(3) / 7)
        self.assertEqual(idf[0], 0)

    def test_repeating_document_words_does_not_change_its_tf(self):
        original = sparse.csr_matrix([[2, 1, 0], [0, 2, 1], [1, 0, 2]])
        repeated = sparse.csr_matrix([[20, 10, 0], [0, 2, 1], [1, 0, 2]])
        first, *_ = course_tfidf(original)
        second, *_ = course_tfidf(repeated)
        np.testing.assert_allclose(first.toarray(), second.toarray())

    def test_weighting_preserves_unsorted_integer_count_buffers(self):
        counts = sparse.csr_matrix((
            np.array([2, 3, 1, 4, 2, 5]),
            np.array([2, 0, 1, 1, 2, 0]),
            np.array([0, 3, 6]),
        ), shape=(2, 3))
        original_values = counts.data.copy()
        original_indices = counts.indices.copy()
        original = counts.toarray()
        course_tfidf(counts)
        np.testing.assert_array_equal(counts.data, original_values)
        np.testing.assert_array_equal(counts.indices, original_indices)
        np.testing.assert_array_equal(counts.toarray(), original)

    def test_invalid_counts_are_rejected(self):
        for counts in ([[0, 0], [1, 0]], [[-1, 2]], [[float('nan'), 1]]):
            with self.assertRaises(ValueError):
                course_tfidf(sparse.csr_matrix(counts))

    def test_loading_retains_stopwords_and_names_and_keeps_sources_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "counts.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["source", "character", "word", "count"])
                writer.writerows([
                    ["Book 1", "Harry", "the", 10], ["Book 1", "Harry", "ron", 5],
                    ["Book 2", "Harry", "wand", 12], ["Book 1", "Unknown", "wand", 100],
                    ["Book 1", "Ron", "wand", 1],
                ])
            documents = load_documents(path, "books", min_words=10)
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0]["counts"], {"the": 10, "ron": 5})
        self.assertEqual(documents[0]["words"], 15)
        self.assertEqual(documents[1]["source"], "Book 2")

    def test_ranking_excludes_zero_weights_and_breaks_ties_alphabetically(self):
        weights = sparse.csr_matrix([[.2, 0, .4, .2]])
        self.assertEqual(ranked_terms(weights, 0, 10).tolist(), [2, 0, 3])


if __name__ == "__main__":
    unittest.main()
