# Spell analysis: books versus films

[Open the spell explorer and figures](index.html)

## What is covered

All supplied dialogue from seven books and eight films. No character word threshold is imposed: a one-line spell still counts. Unknown speakers remain included. The two final films form installment seven. The dictionary contains 109 curated incantations, explicit spell-family groupings, separate named references and spelling variants. This is a study of spell language in the supplied dialogue, not an exhaustive count of magic performed in the stories.

## Results you can present

- Book dialogue contains 360 full incantation mentions after documented OCR variants, versus 135 in films. Normalizing by all dialogue words gives **10.47 versus 17.70 per 10,000 words**: films are **1.69 times as dense** under this dictionary.
- Strict spellings give 244 book mentions and 134 film mentions. The corresponding rate ratio is 2.48; the direction of the density result survives including the documented spelling variants. The precise magnitude depends strongly on spelling treatment. Films also contain the documented Avada Kadavra transcription variant, included in the expanded count.
- The observed inventory contains 78 book incantation types and 50 film types. Do not infer lower intrinsic diversity from these raw type counts: the supplied book dialogue is much larger.
- Harry and McGonagall use Unforgivable Curses in checked book passages. For example, the book identifies Harry casting Imperio on the Gringotts goblin and McGonagall casting Imperio on Amycus. Both lines are incorrectly labeled Ted Tonks in the supplied dialogue. Bellatrix casting Crucio on Neville is labeled Neville. These cases establish that a speaker-to-spell count is not automatically a caster-to-spell count.
- A spell name can be a recollection: Ron describing Moody's Avada Kedavra is not Ron casting it. Manual reviews distinguish selected discussion/reported-casting cases from attempts indicated by surrounding text. Attempt does not assert success.
- Spell/word PPMI is sparse because many incantations occupy whole dialogue lines. With a ±5-token window confined to a line and at least three distinct supporting lines, the reported pairs are: Accio / wand: PPMI 3.09, 5 pairs across 4 lines. No unsupported context graph or topic model is inferred from the sparse spell subset.

## Figures

- 01_spell_frequencies.png/.svg: the 20 most frequent spell families across the whole corpus, normalized by dialogue length; strict book counts are shown as a sensitivity check.
- 02_spell_timeline.png/.svg: 18 frequent families across all seven story installments; marker area encodes normalized frequency. Genuine zero cells are retained in the CSV.
- 03_spell_context_cases.png/.svg: four checked interpretation examples, including a recollection and misattributed casting attempts. These are illustrative cases, not a casting census.

## Matching and context

Dictionary matching uses the existing English-letter tokenizer, Unicode/apostrophe normalization and longest non-overlapping token sequences. Lumos Maxima is not counted a second time as Lumos; it is later grouped into the Lumos family for visualization. Finite Incantatem, Protego variants, Oculus Reparo and Bombarda Maxima are handled similarly. Names such as Killing Curse, Cruciatus and Imperius are exported as named references and excluded from primary incantation counts.

Explicit OCR variants such as Stupefyl and ExpelliarmusV contribute only to the variant-inclusive book count. The observed film transcription Avada Kadavra is explicitly mapped to Avada Kedavra in the expanded film count. Their original forms, lines and match status are retained. These are exploratory text repairs; they do not fix speaker attribution. No global fuzzy matching or automatic correction of the ordinary word crucial is used. Deliberate movie pronunciation variants, partial Expecto/Avada starts and other unmatched candidates are separately exported and excluded from primary counts. Pack and Point Me are omitted because ordinary language can collide with them. Unspoken spells, stage action absent from the dialogue, truncations and names outside this curated inventory can be missed. Absence here never proves a spell is absent from a book or film.

The shape label incantation_only means that the line consists entirely of recognized spell forms; it is not an automatic casting label. embedded_in_dialogue also does not mean discussion: a character can utter a spell and continue speaking. Review labels apply only to selected records explicitly listed in config/spell_reviews.csv. Book reviews reference surrounding original narration. Film reviews are transcript-based inferences; video was not independently reviewed. All remaining mentions are unreviewed. Context/actor reviews are annotations for this analysis, and do not modify original inputs or other project outputs.

KWIC left/right snippets use up to eight neighboring word tokens in the same line. The explorer additionally shows two preceding and two following dialogue records within the same source/chapter, with supplied speaker labels. These are reading aids, not confirmed casting targets or interaction edges.

Spell/context PPMI uses a rectangular incantation-by-context-word matrix pooled across the two media. Whole spell spans and partial repetitions of the target's own components are excluded from context. A window extends five original token positions on either side of the whole phrase, without crossing a dialogue line. A repeated context token at a different position counts again. Context-word eligibility requires at least three occurrences across incantation-bearing records (each record counted once). PPMI uses complete row/column marginals; the three-distinct-line and English-stopword filters apply only to the displayed report. This spell-conditioned matrix is a separate representation from the earlier ordinary word/word PPMI. It is not used to claim a medium-specific context difference.

## Networks relevance

character_spell_network.graphml is a bipartite graph: one node set is canonical supplied speaker labels and the other is spell families. An edge means that a speaker's attributed dialogue references an incantation. Edge attributes retain book and film counts; weight is their sum. This is not a confirmed casting network and has no inferred targets. Unknown and misattributed labels remain visible. network_edges.csv and character_spell_mentions.csv expose these data; per-character rates use all their dialogue and are descriptive, without matched-installment control.

## Files and reproducibility

spell_mentions.csv preserves every match and its original text, source, parsed-record index, speaker label, form, detection status, contexts and any review. spell_totals.csv and spell_by_installment.csv include counts, strict counts, denominators and normalized rates. source_coverage.csv exposes all seven stories' word budgets. spelling_variant_evidence.csv and partial_or_pronunciation_candidates.csv expose detection choices. manual_reviews.csv contains the reviewed matches. spell_context_counts.npz and spell_context_ppmi.npz use context_spell_index.csv and context_word_index.csv; spell_contexts.csv contains the supported report. summary.json records input hashes, settings and findings.

Run `.venv\Scripts\python.exe -m code.spell_analysis` with requirements-map.txt. Earlier source files, analysis outputs and interactive maps are preserved.

The spell vocabulary is curated against the supplied text and the [official spell guide](https://www.harrypotter.com/features/your-guide-to-the-best-spells-in-harry-potter). The three Unforgivable Curses follow the [official encyclopedia](https://www.harrypotter.com/fact-file/spells/the-unforgivable-curses). Exact incantations and source references are in config/spells.json.
