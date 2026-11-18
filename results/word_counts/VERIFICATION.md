# Step 1 verification

Both input CSVs were independently reread using `csv.DictReader`, bypassing the project's dialogue reader. A character-by-character tokenizer, independent of the regular expression in `word_counts.py`, reproduced every row's tokens.

Every exported corpus word/count pair, every source/character word count, every per-character utterance/word/distinct-word total, and every per-1,000 rate was checked against that independent calculation. Rates agree to their stored six-decimal rounding precision. Counts contain no duplicate or missing source/character/word entries. Hand-calculated examples covered punctuation, curly apostrophes, contractions, hyphens, numbers and empty text.

| Corpus | Dialogue rows | Word occurrences | Distinct words |
| --- | ---: | ---: | ---: |
| Books | 35,550 | 343,976 | 12,385 |
| Films | 7,445 | 76,253 | 5,905 |

All checks passed. These verify the stated English-letter tokenization and aggregation; they do not validate the source speaker assignments. Names, common words, stage-direction words and Unknown-speaker dialogue are intentionally included in these raw counts.
