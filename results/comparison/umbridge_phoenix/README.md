# A matched comparison: Umbridge in Order of the Phoenix

[Open the comparison](vocabulary_comparison.png)

The figure compares the same character in the matching book and film. It uses raw word counts, rates per 1,000 words and the existing jointly fitted TF-IDF weights.

- Center: words present in both dialogue sets. Coordinates are their normalized book and film frequencies. The diagonal means equal rates. Orange marks are more concentrated in the film; blue marks are more concentrated in the book. A mark can represent several words with identical counts. Labels are selected examples, not an exhaustive list.
- Left and right: terms that occur in only one of these two dialogue sets, ordered by TF-IDF. n is the raw count. These terms may occur elsewhere in the series or in another character's dialogue.
- To keep the figure legible, only terms with at least two total occurrences, at least three letters and no membership in sklearn's English stopword list are displayed. Counts, TF denominators and IDF remain unchanged. Names remain eligible. The full unfiltered table is in `word_comparison.csv`.

Only three film-only terms meet the display threshold; many film-only word types occur once. Both side panels use the same TF-IDF scale. Word-type absence is sensitive to text length, so it should be considered alongside the shortening benchmark below.

The book denominator is 3,546 words and the film denominator is 884 words. IDF still uses the same 377 reference documents as step 3.
The film has approximately one quarter of the attributed word tokens, while several institutional terms occur at higher normalized rates. This is consistent with concentrated characterization in a shorter script, pending attribution checks and a broader predefined vocabulary. Words shared across media may still occur in different scenes.

A useful interactive extension would let the reader choose a character/installment, select a word and read its book and film quotations with source references. Missing terms belong in the side panels, rather than being hidden by logarithmic axes.

## A random-shortening benchmark

[Open the benchmark](compression_benchmark.png)

The measure here is the fraction of the book's total TF-IDF weight attached to word types that also occur in the film dialogue. It measures exact word presence, not semantic or personality preservation. It includes all words and the existing shared IDF, without the display filters used above.
The observed film covers 34.6% of that book weight. Sampling 884 word tokens without replacement from the 3,546-word book dialogue yields an exact expected coverage of 50.0%.
Across 1,000 seeded random cuts, the middle 95% range is 47.3% to 52.4%. This is variation under the chosen random-cut model, not uncertainty about the extracted dialogue or a general adaptation effect.
This makes a specific comparison: how observed word preservation differs from uniformly removing book word tokens to the same length. Adaptation also selects scenes, rephrases speech and introduces vocabulary. The benchmark preserves no sentence structure and is not a model of a screenwriter. Synonyms, names, transcription errors and attribution errors can affect the observed gap. Equal-length sampling by dialogue line or scene, corrected names/spelling, and additional matched cases are useful sensitivity checks.
Parameters, exact expectation and simulated range are in `compression_benchmark.json`; individual draws are in `compression_simulations.csv`. The exact expectation is calculated independently using the probability that each book word type survives a sample without replacement.

This visual is a first comparison case, using existing data and models. Speaker assignments and transcription artifacts are preserved. Novel presentation should expose these details and support interpretation.
