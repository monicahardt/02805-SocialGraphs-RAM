# Harry Potter dialogue networks

Social Graphs project comparing character interactions and language across the Harry Potter books and films.

**Research question**

Are characters who interact more frequently linguistically more similar, and does this relationship differ between the books and film adaptations?

## Top 20 characters

The top-20 figures rank characters by their total recorded dialogue words across all seven books. To qualify, a character needs at least 100 dialogue words in both the book and film versions of at least one story.

| Rank | Character | Book dialogue words |
| ---: | --- | ---: |
| 1 | Harry Potter | 74,983 |
| 2 | Ron Weasley | 39,941 |
| 3 | Hermione Granger | 38,280 |
| 4 | Albus Dumbledore | 23,585 |
| 5 | Rubeus Hagrid | 11,647 |
| 6 | Severus Snape | 8,795 |
| 7 | Remus Lupin | 7,867 |
| 8 | Minerva McGonagall | 6,048 |
| 9 | Fred Weasley | 5,919 |
| 10 | Molly Weasley | 5,351 |
| 11 | Sirius Black | 5,255 |
| 12 | Draco Malfoy | 4,558 |
| 13 | Cornelius Fudge | 4,236 |
| 14 | George Weasley | 3,848 |
| 15 | Dolores Umbridge | 3,670 |
| 16 | Alastor Moody | 3,082 |
| 17 | Ginny Weasley | 2,889 |
| 18 | Voldemort | 2,752 |
| 19 | Horace Slughorn | 2,734 |
| 20 | Vernon Dursley | 2,687 |

Ted Tonks has 3,253 recorded dialogue words in the books and zero in the film dataset, so the matched cast includes Vernon Dursley at rank 20.

## Reproduce the presentation graphs

Run these commands from the repository root in PowerShell. The first two set up the environment.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-map.txt
.\.venv\Scripts\python.exe interaction-language/analyse.py .
.\.venv\Scripts\python.exe interaction-language/slide_2_options.py
```

On macOS or Linux, use `python3` to create the environment and `.venv/bin/python` for the other commands.

`analyse.py` reads the dialogue CSVs, counts back-to-back lines by different speakers, and measures vocabulary similarity with TF–IDF and cosine similarity. It saves the analysis tables and three main figures in [`results/interaction_language/`](results/interaction_language/).

`slide_2_options.py` reads those saved tables and generates the three [slide alternatives](results/interaction_language/slide_2_options/). Run it on its own to redraw the alternatives from existing results. If you set a custom `--out` directory for `analyse.py`, pass the same path to `slide_2_options.py --results-dir`.

| Presentation figure | What it shows |
| --- | --- |
| [How the characters speak](results/interaction_language/01_switches_and_vocabulary.svg) | Book and film dialogue profiles by tone and share of directions |
| [Matched book–film comparison](results/interaction_language/02_matched_books_and_films.svg) | The association between exchange frequency and vocabulary similarity by story |
| [Equal dialogue length](results/interaction_language/03_equal_dialogue_length.svg) | How the book–film difference changes when the two versions use equal amounts of dialogue |
| [Book–film comparison slide](results/interaction_language/slide_2_options/02a_book_film_comparison.svg) | Story-by-story comparison formatted as a presentation slide |
| [Vocabulary similarity by dialogue frequency](results/interaction_language/slide_2_options/02b_frequency_and_vocabulary.svg) | How similarity ranks change across frequency groups |
| [Dialogue partners and vocabulary matches](results/interaction_language/slide_2_options/02c_dialogue_and_vocabulary_neighbors.svg) | Overlap between the two top-five lists compared with a shuffled baseline |

## Conversation topics in the books and films

| Static figure | What it shows | Files |
| --- | --- | --- |
| Who talks about spells with whom? | Side-by-side book and film networks with the same character positions and edge scale | [PNG](results/topic_conversations/01_spell_conversations.png) · [SVG](results/topic_conversations/01_spell_conversations.svg) |
| What changes from book to film? | The share of conversation turns tagged with each of six topics in all seven stories | [PNG](results/topic_conversations/02_topics_by_story.png) · [SVG](results/topic_conversations/02_topics_by_story.svg) |
| Who leads each conversation topic? | The two highest-ranked character pairs per topic and their shares in each medium | [PNG](results/topic_conversations/03_leading_pairs_by_topic.png) · [SVG](results/topic_conversations/03_leading_pairs_by_topic.svg) |

Harry and Hermione account for 16.7% of spell-tagged turns in the books and 19.0% in the films. Harry and Ron account for 14.7% and 6.9%, respectively. Hogwarts and Quidditch tags each account for a smaller share of eligible turns in all seven film adaptations. Story numbers follow series order from *Philosopher’s Stone* through *Deathly Hallows*, with both final films grouped as story 7.

For an A–B–A sequence of consecutive turns, the analysis counts B’s turn and infers A as the partner. Both characters must have attributed dialogue in both versions of that story. Topic tags come from explicit words and phrases and can overlap. Speaker attribution in the books and partner inference in both media are approximate. The pair figures divide each pair’s tagged turns by all turns tagged with that topic in the medium. The story figure divides tagged turns by all eligible conversation turns for that story and medium. [Topic cue lists](config/conversation_topics.json) · [Spell inventory](config/spells.json)

Run `python -m code.topic_conversations` from the repository root with Matplotlib installed to rebuild the 16:9 figures and their [pair-level counts](results/topic_conversations/topic_conversation_pairs.csv).

## Repository layout

| Location | Contents |
| --- | --- |
| [`book_data/`](book_data/) and [`movie_data/`](movie_data/) | Source dialogue CSVs |
| [`interaction-language/`](interaction-language/) | Presentation analysis and plotting scripts |
| [`code/`](code/) and [`scripts/`](scripts/) | Shared dialogue processing, network analysis, and NLP tools |
| [`results/`](results/) and [`graphs/`](graphs/) | Generated analysis outputs and networks |
| [`notebooks/`](notebooks/) | Exploratory notebooks and visualizations |
| [`config/`](config/) and [`tests/`](tests/) | Analysis configuration and tests |

The [original planning notes](docs/project-planning.md) contain the group's earlier research ideas and task list.
