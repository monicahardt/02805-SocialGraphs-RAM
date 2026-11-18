# TF-IDF verification

Six TF-IDF tests passed, covering the course's hand-calculated example, zero IDF for universal words, length normalization, preservation of unsorted integer count buffers, invalid inputs, document boundaries and deterministic rankings. The four existing LDA tests also passed.

An independent scalar calculation reread the verified word-count CSVs, rebuilt document frequencies using dictionaries and counters, and used Python's `math.log` to calculate `count / document length * ln(N / df)`.

All 122,953 observed word/document pairs and all remaining zero matrix cells matched the saved sparse matrix. Every vocabulary count, document frequency, IDF, document total, exported top-word rank and normalized term frequency was checked. All 377 document identities exactly match the previous LDA run. Input hashes matched the files used to calculate the results.

The example plot was visually checked. Its panels share one horizontal scale and identify the character and source selected. Scores are weights, not probabilities. These checks validate the arithmetic and exports; they do not validate the source speaker labels or spelling.
