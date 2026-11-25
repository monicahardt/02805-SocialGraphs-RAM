# Who gets the next word with Harry?

**Research question. How do the Harry Potter films change the balance between Harry’s dialogue with Ron and his dialogue with Hermione?**

Harry–Ron accounts for a smaller share of Harry’s adjacent speaker changes in every film adaptation. Among Harry’s exchanges with either friend, Hermione’s share rises in all seven installments under the main counting rule. The films contain fewer exchanges for both pairings. The change is in how the remaining exchanges are distributed. The two largest film departures from the speaker-matched benchmark are in *Prisoner of Azkaban* and *Deathly Hallows*.

The corpus supports a report about dialogue structure. It also supports several narrower questions about which pairs survive adaptation, which incantations remain audible in the transcript, and which lines can be traced to a different speaker.

## Data audit

The book CSV contains 35,550 dialogue records and 343,976 words. The film CSV contains 7,445 records and 76,253 words. Both combined files reproduce their individual source CSVs in order. Each record includes a title, a chapter or segment label, a speaker label and dialogue text. These files cover seven books and eight films. The two *Deathly Hallows* films are one story installment for comparison, with no transition counted across the boundary between the films.

Film transcripts contain 22.2% as many words as the book corpus. The ratio varies from 11.7% in *Order of the Phoenix* to 42.3% in *Philosopher’s Stone*. The book CSV has 3,416 unknown-speaker rows containing 35,839 words, or 10.4% of its words. The film CSV has no rows labeled Unknown and includes some generic speaker labels. The named-speaker analysis excludes Unknown, the non-specific label list used in the repository’s interaction graph, and the label Wakanda, following the vocabulary map. It uses the repository tokenizer and seven existing name aliases. The corpus contains 308,137 attributed words under this filter for books and 75,121 for films.

Book speakers were inferred from OCR text. The extractor sometimes guesses from nearby names, pronouns and alternating speakers. The combined CSVs have no confidence field. Chapter labels can also cover much more than a single scene. In the fifth book, one label covers 6,126 of 8,794 rows. The original text and source CSVs are available for checking examples. Film speaker labels in this audit come from the supplied transcripts.

Sources and methods. [Combined book data](../../book_data/all_books.csv) · [Combined film data](../../movie_data/all_movies.csv) · [Book dialogue extractor](../../scripts/extract_book_dialogue.py) · [Interaction graph rules](../../code/interaction_graph.py) · [Tokenizer](../../code/word_counts.py)

## Main result

An exchange is a change between two different named speakers on consecutive nonempty dialogue records inside the same source segment. An unknown or generic speaker breaks the sequence. Both endpoints must speak in both versions of that installment. Other named speakers still occupy their original positions in the sequence. An exchange is undirected and gives evidence of dialogue proximity. A record can contain several sentences, and the intended addressee is sometimes ambiguous.

For each installment, Hermione’s share is Harry–Hermione divided by the sum of Harry–Hermione and Harry–Ron. Two benchmarks each simulate 1,000 cuts of the book dialogue, sampling without replacement. The first samples down to the total named-speaker row count in the film. The second samples each book speaker separately up to their film row count, capped at the number of available book rows. It matches the film counts exactly for the three main characters in every installment. Across all speakers it covers 91.3–97.6% of the film’s named-speaker rows because film-only speakers and a few film counts exceed what the book supplies. Both benchmarks preserve sampled row order and the original breaks at segment boundaries, unknown speakers and generic speakers. They rebuild the adjacency counts, so retained rows that become neighbors after a cut can form a pair.

| Installment | Harry–Ron, book → film | Harry–Hermione, book → film | Hermione’s share, book → film | Uniform cut, middle 95% | Speaker-matched cut, middle 95% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Philosopher’s Stone | 285 → 130 | 86 → 68 | 23.2% → 34.3% | 17.9–31.1% | 26.4–39.6% |
| Chamber of Secrets | 328 → 128 | 56 → 44 | 14.6% → 25.6% | 10.4–22.5% | 15.3–28.7% |
| Prisoner of Azkaban | 228 → 41 | 237 → 86 | 51.0% → 67.7% | 40.0–58.7% | 49.5–66.4% |
| Goblet of Fire | 383 → 46 | 241 → 39 | 38.6% → 45.9% | 31.0–53.5% | 35.0–57.4% |
| Order of the Phoenix | 444 → 38 | 545 → 71 | 55.1% → 65.1% | 46.5–64.4% | 53.0–71.1% |
| Half-Blood Prince | 387 → 85 | 334 → 97 | 46.3% → 53.3% | 37.7–55.9% | 42.1–59.8% |
| Deathly Hallows | 532 → 159 | 536 → 246 | 50.2% → 60.7% | 44.1–56.0% | 48.5–58.5% |
| Pooled counts | 2,587 → 627 | 2,035 → 651 | 44.0% → 50.9% | | |

The film point exceeds the uniform-cut interval in five installments and the speaker-matched interval in two. In five installments, the observed balance is compatible with random cuts that reproduce the trio’s individual dialogue budgets. *Prisoner of Azkaban* and *Deathly Hallows* exceed both intervals. Those cases give the report a place to examine which scenes or exchanges drive the remaining gap. The intervals describe each simulation. Book attribution errors and transcript omissions remain sources of uncertainty.

Three checks sharpen the interpretation.

1. **The shift mainly reflects a weaker Harry–Ron edge.** Harry–Ron falls from 24.5% to 19.9% of Harry’s exchanges with all matching named characters across the series, and falls in each installment. Harry–Hermione rises from 19.3% to 20.7% overall and rises in five installments. In *Goblet of Fire* and *Order of the Phoenix*, both pairs lose share among Harry’s exchanges with the whole cast, and Harry–Ron loses more.
2. **Ron and Hermione’s dialogue budgets also change.** Hermione accounts for 48.9% of the pair’s combined attributed words in the books and 52.4% in the films. The balance of their interactions with the rest of the cast moves much less, from 47.9% to 48.6% of non-Harry edge endpoints belonging to either friend. The Harry-specific split moves from 44.0% to 50.9%. In *Deathly Hallows* the Harry-specific share rises from 50.2% to 60.7% while the non-Harry share falls from 48.5% to 47.4%. These are descriptive comparisons to speaker availability and network prominence.
3. **The pattern survives several changes to the graph, with one source-sensitive comparison.** The Hermione share rises in all seven installments for windows of one, two and five nearby rows. It also rises in all seven if excluded labels are skipped instead of breaking the adjacency sequence, or if consecutive lines by one speaker are collapsed before using the five-row window. In a stricter book-side check, only adjacent pairs where both speakers have immediate named speech tags in the original OCR text are counted. The result holds in six of seven comparisons. *Chamber of Secrets* reverses under that check, from 31.3% in the restricted book data to 25.6% in the film transcript. The strict rule recognized 12,558 book rows and agreed with the recorded speaker for each of those rows. The subset favors explicitly tagged speech and supplies a sensitivity check on the full estimate.

*Prisoner of Azkaban* is a good close reading. It has the largest shift in Hermione’s share, from 51.0% to 67.7%. The book segment “Hermione’s Secret” contains 101 of the book’s 237 Harry–Hermione exchanges. The film segments “Truly Seen, Truly Free” and “Saving Lives” contain 38 of the film’s 86. The transcripts give concrete places to examine how the time-travel sequence contributes to the pattern. For a second short example, the *Deathly Hallows* film segments “Forest Fugitives” and “Bathilda’s Secret” account for 86 of 246 Harry–Hermione exchanges across the two films. [Book 3 dialogue](../../book_data/book3_dialogue.csv) · [Film 3 transcript](../../movie_data/hp3.csv) · [Deathly Hallows Part 1 transcript](../../movie_data/hp7.csv)

Another result provides useful opening context. The trio has 49.7% of attributed book words and 37.9% of attributed film words, and its share falls in all seven installments. It also falls in all seven when the denominator includes all words or is limited to speakers present in both versions of the installment. The pattern within Harry’s inner circle therefore occurs alongside a broader decline in the trio’s share of dialogue.

## Six pages and four figures

| Page | Content | Figure |
| --- | --- | --- |
| 1 | Introduce the question and the seven comparable stories. Establish how strongly the films compress dialogue and how the trio’s share changes. | 1. Paired dots for the trio’s share of attributed words in each installment. Annotate the raw word totals or add a narrow panel for film-to-book word ratios. |
| 2 | Describe the sources, named-speaker filter, graph construction, adjacency denominator and the book attribution limitation. Explain why the two final films are grouped. | |
| 3 | Show which edges change. Explain that the balance moves primarily because the Harry–Ron edge loses more weight. | 2. A heatmap of film minus book share of all matched-speaker adjacency events. Use seven installment columns and rows for Harry–Ron, Harry–Hermione and Ron–Hermione. |
| 4 | Compare uniform cuts with cuts calibrated to each speaker’s film dialogue. Discuss the two films that exceed both simulated ranges. | 3. For each installment, plot the film estimate and both simulation intervals for Hermione’s share. |
| 5 | Check whether the shift looks specific to Harry. Read one or two supporting transcript segments and report the strict-tag sensitivity result. | 4. Compare the change in Hermione’s share around Harry with the change in her share among Ron and Hermione’s edge endpoints elsewhere. Annotate *Prisoner of Azkaban* and *Deathly Hallows*. |
| 6 | Answer the question, discuss the source and measurement limitations, and explain how much follows from the benchmark. Include the short methods references and source list. | |

[Figure data](adaptation_dialogue_audit.csv) contains one row per medium and installment. For Figure 2, use each pair count over the corresponding all-pairs count. For Figure 4, an edge between Ron and Hermione contributes one endpoint to each friend. Use the two-row and five-row columns for a sensitivity check.

## Other questions the data can answer

### Which book dialogue pairs survive the cut?

Take each pair in each installment as one observation, and start with speakers who appear in both versions. Of 959 book adjacency pairs, 549 occur in the films. Repeated book pairs have substantially higher raw retention. The random-shortening benchmark also produces that gradient and provides a second result. The films preserve the strongest pairs less often than the benchmark predicts.

| Book exchanges per pair | Pairs that reappear in the film | Observed | Random shortening, mean and middle 95% |
| --- | ---: | ---: | ---: |
| 1 | 58 / 194 | 29.9% | 29.6% (23.2–35.6%) |
| 2–4 | 116 / 274 | 42.3% | 46.5% (41.2–51.8%) |
| 5–9 | 105 / 180 | 58.3% | 70.2% (64.4–76.1%) |
| 10 or more | 270 / 311 | 86.8% | 94.4% (92.0–96.5%) |

This works as a full network report. Start with installment word budgets, show retention by book weight against the benchmark, and use a paired network view or a ranked plot of changed edge weights to show which dialogue pairs drive the aggregate. The eligible roster is fixed within each installment, so this comparison concerns dialogue pairs among retained speakers. Pairs with ten or more book exchanges still reappear much more often than one-off pairs under two- and five-row windows.

### Do the films cut dialogue faster than they cut incantations?

The spell dictionary finds 360 mentions in the books and 135 in the films after documented spelling variants. Those are 10.47 and 17.70 mentions per 10,000 dialogue words. The film rate is 1.69 times the book rate, or 2.48 times under strict matching. The pooled result stays above one when any single installment is removed. The variant-inclusive leave-one-out ratios range from 1.53 to 1.88.

That pattern varies by story. The film rate is higher in five of seven installments with variants and six of seven under strict matching. *Goblet of Fire* has zero matches to the incantation dictionary in the supplied film dialogue, even though the transcript includes a duel and an explicit reference to *Priori Incantatem*. This makes a clear case study in the difference between detected incantations and film action. The strongest figures would show counts alongside rates, the seven installment comparisons with spelling sensitivity, and which spell families drive the rate gap. The mention totals are independent of which speaker got credit for a line. [Spell data and dictionary methodology](../spells/README.md) · [Installment counts](../spells/spell_by_installment.csv) · [Goblet of Fire transcript](../../movie_data/hp4.csv)

### Whose lines change hands?

An exact-match scan provides a feasible case study. Match complete tokenized rows, require at least seven words, and keep rows that occur exactly once in each version of an installment and have named speakers in both. There are 43 eligible matched lines, of which 22 have different supplied labels. A source review supports at least six changes when the immediate book narration and supplied film dialogue are read together.

| Installment | Source-supported book speaker → film transcript speaker | Scene |
| --- | --- | --- |
| 1 | Seamus Finnigan → Hermione Granger | The question about Nearly Headless Nick. [Book](../../book_data/Book1.txt#L6057) · [Film](../../movie_data/hp1.csv#L283) |
| 2 | Pomona Sprout → Severus Snape | Asking Lockhart about the Chamber’s entrance. [Book](../../book_data/Book2.txt#L14393) · [Film](../../movie_data/hp2.csv#L808) |
| 2 | Harry Potter → Hermione Granger | Asking Madam Pomfrey about restoring Harry’s arm bones. [Book](../../book_data/Book2.txt#L8515) · [Film](../../movie_data/hp2.csv#L453) |
| 3 | Harry Potter → George Weasley | Speaking the Marauder’s Map activation phrase. [Book](../../book_data/Book3.txt#L9397) · [Film](../../movie_data/hp3.csv#L379) |
| 6 | Narcissa Malfoy → Bellatrix Lestrange | Asking Snape to protect Draco during the Unbreakable Vow. [Book](../../book_data/Book6.txt#L1809) · [Film](../../movie_data/hp6.csv#L31) |
| 7 | Fenrir Greyback → Scabior | Taking credit for capturing Harry. [Book](../../book_data/Book7.txt#L23042) · [Film](../../movie_data/hp7.csv#L958) |

Examples that collapse under review include Rufus Scrimgeour reading Dumbledore’s will in both versions, and Flitwick warning about the limits of Hogwarts’ defenses in both versions. The book CSV assigns those lines to Dumbledore and Sprout respectively. A Trelawney line labeled Umbridge in the film transcript is ambiguous because the surrounding exchange makes the transcript attribution questionable. Exact matching also misses paraphrases and lines split differently between files. This question can support a hand-verified report about selected scenes and types of transfers. A series-wide ranking of who inherits the most would require alignment with more complete attribution review and a way to estimate missed cases. [Will in book](../../book_data/Book7.txt#L6319) · [Will in film transcript](../../movie_data/hp7.csv#L220) · [Flitwick in book](../../book_data/Book7.txt#L30144) · [Flitwick in film transcript](../../movie_data/hp8.csv#L315)

### How much of the vocabulary clustering comes from heroes talking alike?

The existing TF-IDF similarity network provides an exploratory answer. Heroes send 91.9% of role-labeled incident edge weight to heroes versus 65.7% under label shuffling. Villains send 28.0% to villains versus 23.5% under the same shuffle. The labeled set contains 20 heroes and nine villains. A different question about whether films change villain vocabulary more than hero vocabulary is much less settled. At the 100-word profile cutoff, the hero group has higher mean book–film cosine similarity, 0.382 versus 0.294, whereas the villain group has higher median similarity, 0.365 versus 0.284. The ranking by medians reverses at a 300-word cutoff. Role labels are provisional and similarities depend on scene selection and word budgets. [Vocabulary network findings](../vocabulary_map/FINDINGS.md) · [Character cosine scores](../series_comparison/character_vocabulary.csv)

## Scope of the report’s conclusion

A defensible conclusion is that the supplied film dialogue consistently shifts the balance of Harry’s nearby exchanges toward Hermione within the two pairings. The speaker-matched cuts bring five of seven film estimates inside the simulated range. *Prisoner of Azkaban* and *Deathly Hallows* still stand out, and particular transcript segments help explain where their extra weight appears. The attribution check makes *Chamber of Secrets* the exception to the broad sensitivity pattern. It remains an open question how much of the residual shift comes from scene choice, different dialogue within shared scenes, or errors and gaps in the sources.
