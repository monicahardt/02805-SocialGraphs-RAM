# Map validation

- All 24 tests pass, including tests for matched installment coverage and aliases, independent cosine arithmetic, nearest-neighbor edge construction, character-level label checks excluding identity edges, and connection details that exactly match graph edges while keeping book/film counterparts separate.
- Every exported cosine was checked against the dot product of the saved unit-normalized vectors. Diagonals, symmetry, finite values and score bounds were verified.
- All 298 edges were independently reconstructed from the three-nearest-neighbor rule. Exported weights match the original-space similarities.
- Each of the 59 book/film pairs has identical installment coverage. Full-vector counterpart scores, original input hashes and the 118-node/298-edge GraphML export were checked.
- Overall and per-role edge-weight fractions were checked against the edge table. Role colors are applied after vector construction, graph construction and the deterministic network layout.
- A local headless Chrome check confirmed that the map renders without JavaScript errors. Actual pointer clicks on nodes, sidebar navigation, book/film profile tabs, books/films/both filters, rapid selections, list expansion, clear selection and the empty-neighborhood state worked. The named sidebar entries, highlighted edges and labels refer to the same graph neighbors. Harry's 55 other-character connections remain accessible while the map labels the three strongest.
- Desktop (1440px) and narrow (390px) layouts were inspected. The narrow layout has no horizontal page overflow. Selection fits the view to its labeled neighborhood, and the book/film counterpart has its own dotted link and score.
- The readability update leaves the complete exported node table (including coordinates), edge table and metrics exactly equal to their saved pre-update versions. It changes presentation and interaction only.
- Names-removed and higher-word-threshold sensitivity checks were run. These are exploratory checks of this dataset, rather than held-out prediction or causal adaptation tests.

The source CSVs and the earlier LDA, TF-IDF and PPMI outputs were preserved.
