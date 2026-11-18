# What the polarity map measures

The map counts dictionary matches. It is an inspectable descriptive baseline, not a validated sentiment classifier or a measure of good and evil. Story roles are never inputs to its coordinates.

The 20 reproducible examples in sentiment_audit_sample.csv were read to inspect the measurement. These examples were selected with NumPy seed 42 from all supporting dialogue lines, without using hero/villain labels. They do not form a held-out accuracy test. Observed issues include:

| Matched word | Observed use | Implication |
|---|---|---|
| cave | Hagrid describes being in a cave | A negative dictionary entry can be an ordinary place noun. |
| right | Ron describes the light moving to his chest | A positive entry can indicate direction. |
| great | Dumbledore refers to the Great Hall | Lowercasing makes a proper place name a positive-word match. |
| work | Dumbledore describes Arthur's work for the Order | A positive entry can refer to a neutral activity. |
| work | George says something did not work | Exact token matching does not handle negation. |
| fault | Lupin says it is not the other person's fault | A negative entry can occur in reassurance. |
| magic / magical | Ginny and Hermione discuss magic and a magical historian | A general opinion lexicon assigns polarity to fantasy-domain words. |
| well | Lockhart starts an utterance with a discourse filler | This usage does not necessarily express a favorable opinion. |

These examples are why axes are labeled positive/negative **word matches**, with contributing words and original lines accessible. The lexicon has not been edited to make the expected heroes/villains separate. The conflicting entry `envious` is excluded from both categories; the original lexicon files are retained with attribution and license notice.

## A confirmed source-attribution problem

In the supplied book dialogue, parsed record 14077 in all_books.csv is assigned to Voldemort. It contains the thanks for a beautiful new hand. The surrounding original text in Book4.txt, around lines 33186–33200, explicitly describes Wormtail receiving and admiring that hand, thanking his master and kissing Voldemort's robes. The speaker is Wormtail/Peter Pettigrew.

The line can be reached in the map under Voldemort → Book → `master`. It demonstrates that even the correct count of a positive term may be attached to the wrong character by the source extraction. The original speaker labels and text have been preserved. A systematic attribution review is needed before treating small character differences as reliable psychological or adaptation claims.

## Interpretation

- Positions and movements are literal vocabulary-density comparisons under a fixed published dictionary.
- Names, scene selection, source coverage and text length can affect these results; profiles under 300 words are flagged.
- Negative language can describe danger faced by a hero; positive words can express a villain's praise, demands or loyalty. These do not imply moral equivalence.
- Separate book and film PPMI uses complete within-medium marginals. Its small pair counts are visible, and no significance or causal-adaptation claim is attached to a difference in those weights.
- LDA reuses one eight-topic fit. Human-readable topic names are interpretations; topics can capture dialect and references to people as well as themes.
