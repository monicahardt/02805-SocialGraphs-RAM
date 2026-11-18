# Step 4: PPMI

PPMI measures how strongly nearby word pairs are associated relative to a baseline where target and context words are independent. It uses words and their local contexts, rather than the document/word weights used by TF-IDF.

PPMI = max(log2(observed pair count / expected pair count), 0).
Expected count = target row total * context column total / total counted pairs.
For example, a PPMI of 3 corresponds to eight times the independent expected count. Zero means no positive association under this measurement; it does not prove a lack of relationship.

This run pools 377 eligible character/source documents containing 37,907 dialogue lines from books and films. Character/source groups require 100 raw words, and Unknown speakers are excluded, matching the previous steps' document eligibility. All original lines remain separate.
The context window extends 5 original word-token positions to either side within a line. Each positional pair contributes both directions. Punctuation does not end a window, but a new dialogue line always does.
Vocabulary words require at least 10 occurrences, with at most 5,000 words selected by frequency (2,506 retained). Names, stopwords and short tokens remain eligible. Words outside the vocabulary still occupy their original positions. Marginals and the total count refer only to pairs with both words in the retained vocabulary.

## Files

- `top_contexts.csv`: direct nearby-word associations for each requested word, with observed and expected counts, marginals and PPMI. These contexts are not cosine-similarity neighbors.
- `top_contexts.png`: up to eight high-scoring contexts per target; n is the observed pair count. Reported pairs need at least 3 observations, positive PPMI and a context different from the target. These reporting filters are applied after calculating the full matrices.
- `counts.npz` and `ppmi.npz`: complete sparse word/context matrices. Rows are target words; columns are context words. Both use the indices in `vocabulary.csv`. Load with `scipy.sparse.load_npz`.
- `vocabulary.csv`: word indices, raw occurrences and context marginals. Raw occurrences differ from row totals because one occurrence can contribute several pairs.
- `documents.csv` and `settings.json`: eligibility, input hashes, software versions and calculation settings.

PPMI scores are weights, not probabilities or significance tests. Rare pairs can receive high scores, so the report includes support counts. These pooled results describe associations in the combined dialogue, without identifying which characters use them or estimating a book/film adaptation effect. Existing speaker assignments and spelling are preserved.

Rerun with a different window or requested words, for example:

```powershell
.venv\Scripts\python.exe -m code.ppmi --window 2 --output-dir results/ppmi_window2
```

[Course method: week 6, word contexts and PPMI](https://sunelehmann.com/socialgraphs2026-web/weeks/week6.html)
