# Character vocabulary map

[Open the interactive map](vocabulary_map.html) · [Static map](vocabulary_map.png)

This is a 118-node, 298-edge vocabulary-similarity network: 59 characters, each represented once for books and once for films.
A node contains the pooled dialogue of one character in one medium, restricted to installments where that character has speech in both media. The two final films form installment seven. Each medium must supply at least the configured word threshold. Matched installments do not guarantee identical scenes or word budgets.
Explicit Tom Riddle labels are merged with Voldemort locally, and several cross-media name variants are normalized. Source files remain unchanged. Crouch father/son and Moody impersonation cases remain unresolved and are marked ambiguous in the role table.

## Representation and edges

TF-IDF is fitted jointly to these pooled profiles, using the same course formula as step 3. This is a new document unit; it does not reuse the previous per-installment IDF unchanged. English stopwords, tokens shorter than three letters and words occurring in fewer than two profiles are excluded. TF uses the retained word-count matrix. Vectors are then normalized for cosine similarity. Names are retained unless --remove-names is supplied.
Each profile chooses its three nearest positive-cosine neighbors by default. The graph is the undirected union of those choices. Edge weight is full-vector cosine similarity, rather than distance in the picture. Links express vocabulary similarity, rather than dialogue interaction. Identity links drawn between the two versions of a character are a separate visual guide; they are not added to the graph unless the nearest-neighbor rule already creates that edge.
Positions use a weighted spring layout with seed 42. Book circles and film triangles are drawn together in one coordinate system. All other labels, including hero/villain colors, are added after vectors, graph and coordinates are calculated. Distances, apparent clusters and arrow lengths in the drawing are approximate; use the displayed cosine scores for numerical comparisons.

## What to inspect

Select a point or character in the interactive map. Selection focuses one book or film profile, labels up to three of its strongest actual graph connections, and fits the view to those nodes. The adjacent panel names those exact connections, reports their cosine scores and shared contributing words, and lets you follow a connection by clicking its name. The character's own book/film counterpart is shown separately as a dotted link and similarity score. List all reveals additional incoming graph connections while the map keeps the three strongest labeled. Switch profile tabs or filter books/films/both. Underlying graph positions remain fixed. Network-community IDs come from weighted Louvain; these communities are not trained moral categories.
Story labels are provisional and editable in config/map_roles.csv. Unlisted characters are unassigned. The graph representation and positions do not use these labels. Labels describe broad series roles; contested and changing allegiances should be examined separately.

Louvain finds 12 communities with weighted modularity 0.443 in this sparsified graph. These values depend on the nearest-neighbor rule and preprocessing.

## Hero/villain grouping check

The check uses only graph edges whose endpoints have hero/villain labels and refer to different characters. Same-character links are excluded. It compares the cosine-weighted share of edges connecting the same role with 1,000 permutations of character labels. Both media keep the same permuted label for a character. This controls class imbalance and avoids treating the two versions as independent label assignments.
```json
{
  "eligible_characters": 29,
  "villain_characters": 9,
  "hero_characters": 20,
  "eligible_edges": 104,
  "same_role_weight_fraction": 0.8545166781389485,
  "permutation_mean": 0.5586139788432266,
  "permutation_95_range": [
    0.3866637123664051,
    0.7637845820249676
  ],
  "one_sided_permutation_p": 0.000999000999000999,
  "same_role_neighbor_share": {
    "hero": 0.9190885509702276,
    "villain": 0.2795855756327432
  },
  "permuted_neighbor_share_mean": {
    "hero": 0.6567969313677907,
    "villain": 0.23457856741714245
  },
  "permutations": 1000,
  "seed": 42,
  "unit": "character; both medium versions keep the same shuffled label",
  "identity_edges": "excluded"
}
```

Read the per-role neighbor shares alongside the overall score. The overall same-role fraction can be driven mainly by one cohesive group. A high overall fraction alone does not establish that all villains form a cluster. Neighbor shares use only the hero/villain-labeled, different-character edges described above; ambiguous and unassigned endpoints are omitted.

This is an exploratory check for this representation, graph and chosen labels. It is not held-out classification accuracy or evidence that vocabulary determines morality. No cross-preprocessing or graph-threshold stability has been established.

## Files and rerunning

nodes.csv and edges.csv expose every profile, edge weight, source coverage and important shared terms. network.graphml opens in network-analysis software. vectors.npz and cosine_similarity.npy retain the full representation and distances. metrics.json records parameters, graph statistics, input hashes and alias choices.
Run `.venv\Scripts\python.exe -m code.vocabulary_map` from the repository root. A sensitivity run can use `--min-words 300` or `--remove-names --output-dir results/vocabulary_map_without_names`. Name-component removal can also remove ordinary words such as black or wood.
Speaker-attribution and transcription errors in the supplied data remain. Sparse profiles have less stable vocabulary estimates. Use this map to select cases for checking against original passages, and compare with the existing interaction network to ask where linguistic and structural neighbors agree.

[Course: TF-IDF and cosine](https://sunelehmann.com/socialgraphs2026-web/weeks/week6.html) · [Course: corpus maps and their limits](https://sunelehmann.com/socialgraphs2026-web/weeks/week7.html)
