# Step 3: TF-IDF

TF-IDF gives a word a high weight when it forms a large share of one document and occurs in relatively few documents across the corpus.

- TF = word count / total word tokens in the document.
- IDF = ln(number of documents / number of documents containing the word).
- TF-IDF = TF * IDF. A word found in every document has zero IDF.

This run uses 377 documents, 12,642 vocabulary words and 371,191 word occurrences. A document is one character's dialogue within one book or film, requiring 100 raw words. Unknown speakers are excluded. The final two films remain separate.
The document definition and length threshold match the LDA step. For TF-IDF, all counted words remain, including names, English stopwords and short words. There is no document-frequency cutoff. The TF denominator is the full raw document length. Both media share the same vocabulary and IDF.

## Reading the outputs

- `top_words.csv`: the highest-weight words in each document. Columns include the raw count, TF, document frequency, IDF and final TF-IDF weight, so each score can be checked.
- `documents.csv`: metadata and matrix row indices for every eligible document.
- `vocabulary.csv`: word statistics and matrix column indices. Document frequency counts documents containing a word, not total occurrences.
- `weights.npz`: the complete sparse document-by-word weight matrix; load with `scipy.sparse.load_npz`.
- `top_words.png`: examples for the requested characters. Each panel is one character/source document, and all panels use the same horizontal scale. These weights are not probabilities and do not sum to one.
- `plotted_documents.csv`: the exact documents selected for the plot. Selection uses the greatest raw word count for that character in each medium. Source titles can differ, so the figure is illustrative rather than a controlled adaptation comparison.
- `settings.json`: parameters, formulas, input hashes and package versions.

TF-IDF describes words in individual documents. It does not create topics, identify villains, measure sentiment or prove a stable speaking style. Names, rare spellings and transcription artifacts may receive high weights. Existing speaker assignments are preserved.

This uses the course formula directly, with no IDF smoothing or additive constant. It therefore differs from scikit-learn's default TfidfVectorizer. LDA outputs remain unchanged.

[Course formula: week 6](https://sunelehmann.com/socialgraphs2026-web/weeks/week6.html)
