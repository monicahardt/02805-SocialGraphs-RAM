# Harry Potter dialogue networks

Social Graphs project comparing character interactions and language across the Harry Potter books and films.

**Research question**

Are characters who interact more frequently linguistically more similar, and does this relationship differ between the books and film adaptations?

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
