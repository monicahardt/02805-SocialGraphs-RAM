# From page to screen: the series-wide figures

[Open the figure collection](index.html) · [Interactive character polarity map](character_word_map.html) · [Existing vocabulary network](../vocabulary_map/vocabulary_map.html)

These figures replace the earlier selected examples as the main project overview. All seven book installments and all eight films contribute. Umbridge remains an optional case study in ../comparison/umbridge_phoenix.

## What each figure covers

| Figure | Scope | Underlying data |
|---|---|---|
| 01_dialogue_volume | Every dialogue row, including unknown speakers; final films grouped into installment 7 | series_word_counts.csv |
| 02_topic_journeys | All 7 stories, all 8 original shared LDA topics, matched eligible speakers within each story | series_topics.csv; matched_topic_documents.csv |
| 02b_all_character_topics | All 51 characters with common eligible LDA installments | character_topics.csv |
| 03_vocabulary_retention | All 59 matched character pairs, including every eligible villain | character_vocabulary.csv |
| 04_word_contexts | Series-wide book/film PPMI for blood, power, death and love | context_comparison.csv; context_vocabulary.csv; context_counts_*.npz; ppmi_*.npz |
| 05_character_word_map | All 59 matched character pairs, positive/negative word densities | character_polarity.csv; polarity_changes.csv; word_contributions.csv |

Every static figure is saved as PNG and editable SVG. The HTML map embeds Plotly and works locally without a server. Read the figures as exploratory comparisons of the supplied dialogue, with source attribution and transcript limitations.

## Concrete results

- Across the supplied dialogue, books contain 343,976 word tokens and films contain 76,253. The volume plot exposes this compression for every story.
- Among characters with at least 300 words in each medium, the largest vocabulary differences include Bartemius Crouch (0.058), Argus Filch (0.102), Lucius Malfoy (0.209). The closest vocabularies include Dobby (0.881), Harry Potter (0.734), Hermione Granger (0.674). These are full TF-IDF cosine scores, not morality or personality similarity.
- The largest matched story-level LDA shift in this fitted model is topic 8 in installment 1: 45.1% in books and 18.7% in films. Names and scenes contribute to topics. This is one existing eight-topic fit, without a topic-stability claim.
- Topic 2 (lord, severus, dark, draco, professor) has a higher film proportion in 7 of 7 matched story comparisons. It includes names and references to authority; this is not a direct measure of villains being more negative.
- Positive/negative word counts do not establish a clean hero/villain boundary. For the 20 heroes and 9 villains, equal-character mean positive-minus-negative densities are: books 7.43 vs 7.40; films 10.31 vs 6.92 per 1,000 words. These are descriptive group means for this selected set, not an accuracy score or significance test.

Largest polarity-density changes among profiles with at least 300 words in both media:

- Argus Filch: 5.6 → -26.5 matches per 1,000 (change -32.0; 540/340 book/film words).
- George Weasley: 7.5 → 35.1 matches per 1,000 (change +27.5; 3,848/485 book/film words).
- Petunia Dursley: 0.0 → 22.8 matches per 1,000 (change +22.8; 939/394 book/film words).
- Remus Lupin: 4.3 → 23.6 matches per 1,000 (change +19.3; 7,670/1,907 book/film words).

Positive balance means more positive than negative dictionary matches under this particular instrument. The unequal list sizes and domain limitations prevent treating zero as a calibrated neutral point or the balance as a morality score.

## Source matching and exclusions

Character TF-IDF and polarity use the existing vocabulary-network matching rule: pool each character only over installments with dialogue in both media; require >=100 total raw words in each medium. The two final films form installment seven. The map includes 59 characters; excluded characters and word budgets are in excluded_characters.csv. Small markers flag fewer than 300 words; robust examples above require at least 300 each. Coverage is all eligible characters across the series, not every named person or every scene.

The existing alias table is reused locally, including Tom Riddle → Voldemort. Original speaker labels remain unchanged. Moody impersonation and Crouch identity ambiguity remain provisional. Matching installments does not imply matching scenes or equal word budgets. Longer installments contribute more to pooled character profiles.

LDA reuses original document-topic probabilities rather than training a new model. Each original document needed >=100 raw words. Canonical character/installment groups must be present in both media. Topic proportions are aggregated with modeled-token weights, and each medium retains its own word budget. Matched speaker counts and all contributing documents are exported. Sparse voices can have a pooled vocabulary profile but no matched eligible LDA document, explaining 59 versus 51 characters.

TF-IDF uses the same pooled-profile vectors as the current network: English stopwords and short tokens removed; min_df=2; TF=count/retained document length, IDF=ln(N/df); cosine normalizes vector length. Word-count and polarity rates instead use full raw dialogue word totals.

## PPMI: now a real medium comparison

The original PPMI eligibility and shared 2,506-word vocabulary are retained. Book and film co-occurrence counts are computed separately, with separate marginal totals and PPMI calculations. Their raw matrices add exactly to the original pooled matrix; recomputed pooled PPMI agrees to numerical precision. The previous pooled files are preserved.

The four target words are the previous thematic probes, not the entire vocabulary. Full matrices for every retained word are saved. The figure selects up to 8 contexts per target by pooled PPMI, requiring >=3 overall pair observations. Only alphabetic words of at least three letters outside the English stopword list are displayed. These display filters never change the matrices. Pairs with <3 observations in one medium are pale, including zeros. Scores use complete within-medium marginals, with no smoothing or support pruning. Zero PPMI means no positive association under this instrument; it is not proof of no semantic relationship. Media differ in text amount, characters and scenes, so score differences do not isolate a causal adaptation effect.

## Positive/negative word categorization

The [Hu–Liu opinion lexicon](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html), distributed by [NLTK](https://www.nltk.org/howto/corpus.html), supplies word categories. Cite: Minqing Hu and Bing Liu, *Mining and Summarizing Customer Reviews*, KDD 2004. Original files, headers, CC BY 4.0 notice and provenance are in config/opinion_lexicon. The entry envious appears in both lists and is omitted; the resulting dictionary has 6,783 unique entries. No Harry Potter role labels enter the word categories, coordinates or TF-IDF.

Coordinates are exact token counts per 1,000 full raw dialogue words. Matching uses the project tokenizer, with lowercase/apostrophe normalization; no stemming or negation reversal. Unmatched words have no polarity label. The whole dictionary and every observed word contribution are exported. A term such as well can be a discourse filler; a hero can describe a murder; a villain can say good. The method does not measure intent or morality. It does not resolve negation, sarcasm, mixed sentiment or fantasy senses. The interactive view exposes up to three original lines for each displayed term; sentiment_evidence.csv contains all matching dialogue lines, and sentiment_audit_sample.csv is a reproducible 20-line audit sample, not a held-out accuracy evaluation.

The map is a transparent descriptive baseline inspired by [course week 7: sentiment as a measurement](https://sunelehmann.com/socialgraphs2026-web/weeks/week7.html). The original four methods follow [course week 6](https://sunelehmann.com/socialgraphs2026-web/weeks/week6.html). No sentiment-classification performance claim is made for this domain.

Read [the qualitative measurement audit](MEASUREMENT_AUDIT.md) for observed dictionary/context problems and a confirmed speaker-attribution error. These limit interpretation even when the arithmetic is correct.

## Reproduce

Run `.venv\Scripts\python.exe -m code.series_comparison`. Uses requirements-map.txt and the saved original method outputs. summary.json records results, thresholds and input hashes. Original datasets, earlier method outputs and the existing interactive network are preserved.
