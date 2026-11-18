"""Source matching and measurement checks for the series-wide comparison."""
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from code.series_comparison import load_lexicon, polarity_counts, paired_topics, split_contexts
from code.ppmi import cooccurrence


class SeriesComparisonTests(unittest.TestCase):
    def test_polarity_uses_all_tokens_and_preserves_counts(self):
        counts=Counter({'good':2,'evil':1,'wand':7})
        before=counts.copy()
        result=polarity_counts(counts,{'good':'positive','evil':'negative'})
        self.assertEqual(result['positive_per_1000'],200)
        self.assertEqual(result['negative_per_1000'],100)
        self.assertEqual(result['balance_per_1000'],100)
        self.assertAlmostEqual(result['lexicon_coverage'],.3)
        self.assertEqual(counts,before)
        with self.assertRaises(ValueError):
            polarity_counts({}, {})
        # Negated language remains an explicit limitation of token matching.
        self.assertEqual(polarity_counts(Counter({'not':1,'good':1}),{'good':'positive'})['positive'],1)

    def test_conflicting_dictionary_entries_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'positive-words.txt').write_text('; source comment\ngood\nenvious\n')
            (path/'negative-words.txt').write_text('; source comment\nevil\nenvious\n')
            self.assertEqual(load_lexicon(path),{'good':'positive','evil':'negative'})

    def test_topic_matching_weights_final_parts_and_canonical_aliases(self):
        def document(name,medium,source,tokens,topic):
            return {'character':name,'medium':medium,'source':source,'modeled_words':str(tokens),
                    **{f'topic_{i}':str(int(i==topic)) for i in range(1,9)}}
        rows=[document('Voldemort','books','Deathly Hallows',100,1),
              document('Tom Riddle','movies','Deathly Hallows Part 1',30,1),
              document('Voldemort','movies','Deathly Hallows Part 2',90,2),
              document('Harry Potter','books','Chamber of Secrets',500,3)]
        result=paired_topics(rows)
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]['character'],'Voldemort')
        self.assertEqual(result[1]['modeled_words'],120)
        self.assertEqual(result[1]['original_documents'],2)
        self.assertAlmostEqual(result[1]['topic_1'],.25)
        self.assertAlmostEqual(result[1]['topic_2'],.75)
        for row in result:
            self.assertAlmostEqual(sum(row[f'topic_{i}'] for i in range(1,9)),1)

    def test_split_contexts_reconstruct_pool_but_use_separate_marginals(self):
        sequences={'books':[['good','bad','good']],'movies':[['good','good','neutral']]}
        vocabulary={'good':0,'bad':1,'neutral':2}
        counts,scores=split_contexts(sequences,vocabulary,window=1)
        pooled=cooccurrence(sequences['books']+sequences['movies'],vocabulary,1)
        np.testing.assert_array_equal((counts['books']+counts['movies']).toarray(),pooled.toarray())
        self.assertAlmostEqual(scores['books'][0,1],1)
        self.assertAlmostEqual(scores['movies'][0,2],np.log2(4/3))
        self.assertEqual(scores['movies'][0,0],0)
        self.assertEqual(counts['books'][0,1],2)


if __name__=='__main__':
    unittest.main()
