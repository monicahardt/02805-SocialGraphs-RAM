# PPMI verification

Nine PPMI tests and the ten existing LDA/TF-IDF tests passed. The new tests cover a hand-calculated matrix, negative PMI clipping, unobserved and independent pairs, original token distances, utterance boundaries, repeated-word positions, sparse input preservation, invalid inputs, vocabulary thresholds and report-only support filtering.

An independent recount read the original CSVs directly and scanned every target's preceding and following window positions. This differs from the implementation, which scans following positions and mirrors each pair.

All 2,381,142 directed pair occurrences matched the saved count matrix, including all 514,364 observed cells. Independent scalar calculations using `math.log2` matched all 400,669 positive PPMI cells. All other cells correctly receive zero. Vocabulary occurrence counts, row and column marginals, observed/expected counts, report rankings and input hashes also matched.

The 377 eligible document identities exactly match the TF-IDF step. Source word totals and dialogue-line counts were verified. The example plot was inspected for layout and consistent scales.

For example, `death` / `eaters` has 141 observed pairs, expected count 1.0440435723698964 and PPMI 7.077369429446734. The expected count uses the marginals of the complete retained-vocabulary pair matrix.

These checks validate counting, weighting and exports. They do not establish statistical significance, semantic similarity or correct source speaker attribution.
