# What the current NLP results suggest

The strongest leads concern individual character voices, story context and adaptation compression. The current results support exploratory case studies. They do not yet establish a general difference between villains and heroes.

This analysis reads the existing word counts, LDA, TF-IDF and PPMI outputs. It leaves those results and the source CSVs unchanged. Reproduce the diagnostic figures and evidence with:

```powershell
.venv\Scripts\python.exe scripts/analyse_nlp_results.py
```

## 1. Umbridge: concentrated institutional language in the film

In the matched Order of the Phoenix sources, Umbridge has 3,546 book word tokens and 884 film tokens. The film contains approximately one quarter as many attributed words.

| Selected term | Book occurrences | Film occurrences | Book per 1,000 words | Film per 1,000 words |
| --- | ---: | ---: | ---: | ---: |
| educational | 3 | 3 | 0.85 | 3.39 |
| decree | 2 | 3 | 0.56 | 3.39 |
| punishments | 3 | 3 | 0.85 | 3.39 |
| hem | 12 | 2 | 3.38 | 2.26 |

The three selected institutional terms together rise from 2.26 to 10.18 occurrences per 1,000 words, a **4.51-fold density increase**. Their raw counts are 8 and 9, so this is a small, selected vocabulary rather than a validated register measure. A plausible interpretation is that the adaptation concentrates recognizable bureaucratic language while shortening her dialogue. Differences in scenes and extraction coverage are alternative explanations.

The TF-IDF rankings make these words visible. The raw counts explain the difference. `hem` also appears in the film table; absence from its top TF-IDF list does not mean absence from the transcript.

This is the most direct starting case for the villain-focused research question. Extend it by predefining a broader authority/control vocabulary and reading the supporting lines before evaluating its behavior in other characters.

## 2. Hagrid: a strong writing-style signal, with installment exceptions

For the exploratory marker set `yeh`, `ter`, `yer`, `fer`:

| Corpus | Marker occurrences | Hagrid word tokens | Markers per 1,000 words |
| --- | ---: | ---: | ---: |
| Books | 539 | 11,647 | 46.28 |
| Films | 37 | 3,487 | 10.61 |

The pooled book rate is **4.36 times** the pooled film rate. The rate is lower in films for five of seven matched installments, but higher in Half-Blood Prince and Deathly Hallows. The latter film samples have only 233 and 158 Hagrid word tokens. The two final films are grouped into installment seven for these rate comparisons; Hagrid's supplied dialogue is in Part 1.

The LDA profile is consistent with a dialect-related signal in this fitted model: Topic 8 accounts for approximately 94% of Hagrid's token-weighted book mixture and 60% of his film mixture. These percentages use the LDA-filtered word counts, while the marker rates use full raw dialogue lengths.

This measures written transcript spelling. Studying pronunciation would require audio. It provides a useful comparison case for separating distinctive character language from villain-related themes.

**Figure:** [Normalized character word rates](character_word_rates.png). Each connector joins a book rate and a film rate. The left panel uses matching installments; the right panel uses Umbridge in Order of the Phoenix. Counts and denominators are in `character_word_rates.csv`.

## 3. LDA reflects story content as well as speaker identity

Topic 6 contains `wand`, `sword`, `hallows`, `elder`, and `vault`. Approximately **72.9% of its token-weighted contribution** comes from Deathly Hallows, combining the book and both films. Those sources contain only **17.8% of the LDA-modeled tokens**. This topic is strongly associated with the late-story material in this fit.

Topic 2 contains `lord`, `severus`, `dark`, `draco`, and `professor`. Its highest-proportion example document is Dumbledore in the Half-Blood Prince film. Its vocabulary therefore crosses character alignments and describes references or scenes rather than directly identifying villain speech.

Voldemort's aggregate profiles look similar in books and films: Topic 2 contributes about 48% in each. However, the summaries combine three eligible book documents and five film documents with different story coverage. They also omit film dialogue labeled Tom Riddle. This pooled similarity is a lead to investigate with matched sources, not a conclusion about an unchanged voice.

**Figure:** [Character/topic heatmap](topic_profiles.png). Each row is the mean topic mixture over a character's eligible documents, weighted by the number of LDA-modeled tokens. Values below 2.5% are left unlabeled. Book/film rows can cover different source sets. The figure uses one fitted seed and topic count; stability across settings has not been evaluated. Exact values, document counts and token totals are in `topic_profiles.csv`.

## 4. PPMI: some apparent themes reduce to a few scene-specific lines

| Pair | PPMI | Word-pair occurrences | Dialogue lines | Distinct book/film sources |
| --- | ---: | ---: | ---: | ---: |
| blood / pure | 7.82 | 22 | 20 | 6 |
| death / eaters | 7.08 | 141 | 137 | 8 |
| power / punishments | 6.99 | 3 | 3 | 2 |
| love / impossible | 6.29 | 3 | 2 | 2 |

`death / eaters` recurs across many more lines and sources than the other examples. `blood / pure` is also recurrent. The broader blood contexts include `traitor`, `unicorn`, `prince`, and `half`, mixing social status, magical substances and names. One word's context vector can combine several senses.

The `power / punishments` evidence consists of one Draco prefect statement, one Umbridge book passage and one Umbridge film decree. This is a promising small qualitative case about authority at school. Three lines are insufficient evidence for a broad speaking-style claim.

The `love / impossible` result comes from two corresponding explanations of Amortentia, one in the book and one in the film. The book line contributes two nearby word pairs. This is a repeated scene-specific statement about manufacturing love, rather than broad evidence for a pessimistic treatment of love. Book/film copies of the same scene are not independent observations.

**Figure:** [PPMI strength against dialogue-line support](ppmi_support.png). Each point is one of the existing reported associations. Strong PPMI can appear on the left with very little text behind it. Shapes and colors identify the four target words. `ppmi_support.csv` gives support for all 60 reported pairs; `ppmi_evidence.csv` contains all supporting lines and one-based parsed-record indices in the combined source CSVs. These are parsed records, not physical CSV line numbers.

## Visualization priorities

| Priority | Visualization | What it helps answer | Suggested design |
| --- | --- | --- | --- |
| 1 | Matched character word-rate plot | Which aspects of a voice are retained or concentrated in an adaptation? | One row per word or predefined word family; book and film rates per 1,000 connected by a line; counts and source titles visible. Prototype provided above. |
| 2 | Book-versus-film term-frequency scatter | Which words remain common in both, and which shift? | Select a character and matching installment. Each point is a term; x is the book rate and y the film rate, with a diagonal reference. Handle zero frequencies explicitly. Clicking a word should show the original lines. |
| 3 | Character/topic heatmap with an installment selector | Do topics follow a character, a particular plot segment, or both? | Reuse the shared LDA topics; show paired book/film rows, document sizes and top topic words. Prototype provided above. Validate topic stability before treating small changes as findings. |
| 4 | PPMI strength/support scatter and a small context graph | Which associations recur, and how many meanings does a word carry? | Keep pair counts and distinct-line counts visible. In a graph around blood, label associations and expose quotes on selection. Add a support threshold control; do not interpret it as a significance test. Scatter prototype provided above. |
| 5 | Spell-mention heatmap by character and installment | Which characters mention which spells, and does that change in the films? | Separate panels for books/films, normalized rates and visible counts. The next context-analysis step should distinguish mentions, quotations and actual casting. This analysis has not yet been implemented. |
| 6 | Dialogue-network view linked to topic profiles | Do interaction communities share language? | Keep network positions fixed; selecting a character reveals its topic mixture and distinctive terms. Compare network communities with language patterns rather than assuming agreement. |

The term-frequency scatter follows the [course's two-corpus comparison](https://sunelehmann.com/socialgraphs2026-web/weeks/week6.html): rates make corpora comparable, and reading the lines explains outlying terms. For the report, one well-supported matched case study and two complementary figures would be a strong initial structure.

## Interpretation limits that matter here

- Raw book dialogue contains 343,976 word tokens versus 76,253 film tokens, a 4.51-fold difference. Raw frequency rankings across media therefore require normalization. Of the book tokens, 35,839 (10.4%) have an Unknown speaker; the character-based NLP steps exclude these.
- Character names are retained in LDA and TF-IDF. A high-weight name can identify an addressee, subject or scene. Compare a names-retained run with a documented names-filtered sensitivity run before interpreting semantic clusters. Broad name masks can also remove ordinary words such as black.
- Film tables contain 1,031 word tokens under Tom Riddle, while the book tables have no separate Tom Riddle speaker. Alias handling must be resolved before making full-series Voldemort comparisons.
- The high-weight film token `shail` occurs twice in Voldemort's transcript, in wording that appears to intend `shall`. This illustrates TF-IDF's sensitivity to transcription errors.
- A targeted source check confirmed an attribution error in the love-potion example. Book CSV parsed record 25109 is labeled Draco Malfoy. The surrounding passage in `book_data/Book6.txt` explicitly places Slughorn as the speaker, with Malfoy and Nott listening. Film CSV parsed record 4914 labels its corresponding explanation Horace Slughorn. The source data have been preserved.
- Word sets, focus characters and examples in this analysis were selected after inspecting the results. They are exploratory. There are no hypothesis tests, uncertainty intervals, held-out attribution accuracy estimates or topic-stability claims here.

## Suggested research direction

**Which recognizable features of a villain's dialogue are preserved or concentrated when the books become films?**

Begin with Umbridge and an explicitly defined authority/register vocabulary, compare matching installments, and read the supporting lines. Use Hagrid as a style-oriented comparison and PPMI to explore the contexts of key terms. This builds a more testable project argument than interpreting an LDA topic or a rare TF-IDF word as a moral label.
