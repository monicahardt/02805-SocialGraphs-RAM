# What this map currently shows

The map contains 59 matched characters, each with a book node and a film node: 118 nodes and 298 nearest-vocabulary edges. Its layout uses TF-IDF/cosine similarities. Story labels are added afterward.

## The main grouping result is a cohesive hero core

Among edges connecting different characters with explicit hero/villain labels:

| Edge type | Number of edges | Sum of cosine weights |
| --- | ---: | ---: |
| Hero / hero | 79 | 28.9113 |
| Hero / villain | 21 | 5.0904 |
| Villain / villain | 4 | 0.9878 |

Overall, 85.5% of eligible edge weight joins the same alignment, versus 55.9% on average in 1,000 character-label shuffles. Identity edges between two versions of the same character are excluded, and the two medium versions retain a common label in each shuffle. The exploratory one-sided permutation value is 1/1001.

The breakdown matters: approximately **91.9% of the heroes' eligible incident edge weight connects to heroes**, while **28.0% of the villains' eligible incident edge weight connects to villains**. The corresponding shuffled-label averages are 65.7% and 23.5%. Each endpoint contributes separately to these within-group neighbor shares.

The overall signal is therefore largely associated with the hero core. The figure does not establish a single homogeneous villain vocabulary. Several antagonists lie outside the main villain neighborhood. Only 20 heroes and 9 villains enter this check; ambiguous and unassigned labels are omitted from its edges. Source selection, narrative roles, sample sizes and speaker-attribution errors remain relevant explanations.

## The book/film dimension

Each pair uses the same set of installments, combines the two final films, and shares one fitted vocabulary/IDF. The map connects the book and film profiles of a character so the reader can inspect changes in vocabulary and neighbors. Arrow length is a property of the network drawing; the cosine shown on selection is the numerical comparison.

Examples of full-vector book/film cosine: Dobby 0.881, Harry 0.734, Umbridge 0.400, Voldemort 0.390, and Bellatrix 0.365. Scores compare exact weighted vocabulary and are influenced by scene coverage, name use and the amount of dialogue. They do not measure preserved personality.

Separate 3-nearest-neighbor networks built within each medium have same-alignment edge-weight shares of **83.6% for books** and **79.0% for films**. Their corresponding shuffled-label means are approximately 55.8% and 55.7%. These graphs use only within-medium candidate neighbors; they are separate checks, not the induced layers obtained by hiding nodes in the interactive joint map. The difference between media has not been tested or controlled for dialogue length. Details are in `within_medium_checks.json`.

## Initial sensitivity checks

Removing character-name components gives an overall same-role edge-weight share of 84.7%. Requiring 300 matched word tokens in each medium gives 90.7%, with fewer characters. The complete checks are saved in `sensitivity.json`. These checks support further investigation of the observed structure, without establishing a stable classifier or eliminating attribution and scene-coverage concerns.

The role table is editable in `config/map_roles.csv`; its labels are provisional whole-series interpretations. Graph positions and edge construction never use those labels.
