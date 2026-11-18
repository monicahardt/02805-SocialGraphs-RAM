# Step 2: LDA topics

LDA groups words that tend to occur in the same documents. Each topic is a distribution over words; each document has a mixture of topics.

This run fits 8 shared topics to 377 documents using 3,374 vocabulary words and 124,238 word occurrences.
One document is a character's dialogue within one book or film. The final two films stay separate, as in the word-count tables. Both media use the same model, making topic IDs comparable.

Documents need 100 raw words. Unknown-speaker documents are excluded. English stopwords and tokens shorter than three letters are removed; names and contractions remain.
Vocabulary words must occur in at least 5 and at most 50% of the length-eligible documents. Any empty documents after filtering are dropped (0 in this run). LDA uses counts. Seed=0; iterations=30.

## Topic words

1. yeah, didn't, hermione, dumbledore, ron, can't, i've, we're, really, tell
2. lord, severus, dark, draco, professor, shall, boy, hogwarts, school, tom
3. really, look, can't, i've, malfoy, bit, we've, ron, i'll, you've
4. black, ron, professor, sirius, dementors, hermione, hagrid, didn't, i'll, what's
5. yeah, sir, dobby, didn't, crouch, i've, really, what's, told, look
6. wand, sword, hallows, gryffindor, elder, death, vault, wizard, seek, help
7. dumbledore, professor, voldemort, sirius, course, ministry, tell, need, snape, magic
8. yeh, ter, hagrid, dumbledore, don, i've, yer, tell, fer, look

## Files

- `topic_words.csv` and `topic_words.png`: most probable words in each topic. Probabilities use the full topic vocabulary, not just the displayed words.
- `document_topics.csv`: every modeled document and its topic proportions, which sum to one.
- `example_documents.csv`: three documents with the highest proportion of each topic. Check their dialogue in the source CSV before interpreting a topic.
- `vocabulary.csv` and `settings.json`: modeled words, filtering settings, input hashes and software versions.

These are exploratory topics. Names can dominate; dialect and scene vocabulary can also form topics. Topic names are human interpretations, and a topic is not a villain label. The existing speaker assignments remain as supplied. Eight topics is a starting setting, not a validated optimum.

[Course method: week 6, topic modeling](https://sunelehmann.com/socialgraphs2026-web/weeks/week6.html)
