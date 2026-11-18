"""A paired character vocabulary map and TF-IDF similarity network.

Run: python -m code.vocabulary_map
Reads verified word-count tables. Produces a self-contained interactive map,
a static figure, graph exports and a character-level label-permutation check.
"""

import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

if __package__:
    from .interaction_graph import NON_SPECIFIC_MOVIE_LABELS
    from .tfidf import course_tfidf
else:
    from interaction_graph import NON_SPECIFIC_MOVIE_LABELS
    from tfidf import course_tfidf

ROOT = Path(__file__).resolve().parents[1]
TITLES = ("Philosopher's Stone", "Chamber of Secrets", "Prisoner of Azkaban",
          "Goblet of Fire", "Order of the Phoenix", "Half-Blood Prince", "Deathly Hallows")
ALIASES = {"Tom Riddle": "Voldemort", "Garrick Ollivander": "Ollivander",
           "Poppy Pomfrey": "Madam Pomfrey", "Rolanda Hooch": "Madam Hooch",
           "Stanley Shunpike": "Stan Shunpike", "Collin Creevey": "Colin Creevey",
           "Ernest Macmillan": "Ernie Macmillan"}
COLORS = {"hero": "#277b82", "villain": "#b64a3f", "ambiguous": "#8b7294", "unassigned": "#a6a29b"}
ROLE_NAMES = {"hero": "Hero", "villain": "Villain", "ambiguous": "Ambiguous", "unassigned": "Unassigned"}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def installment(title):
    for number, suffix in enumerate(TITLES, 1):
        if suffix in title:
            return number
    raise ValueError(f"Unknown source title: {title}")


def matched_profiles(paths, min_words):
    """Pool only installments in which this speaker has dialogue in BOTH media.

    The final two films form installment seven. Thresholds apply to each pooled
    character/medium profile. Alias changes are local to this analysis.
    """
    groups = defaultdict(Counter)
    for medium, path in paths.items():
        for row in read_csv(path):
            name = ALIASES.get(row["character"], row["character"])
            if name.casefold() in NON_SPECIFIC_MOVIE_LABELS | {"unknown", "wakanda"}:
                continue
            key = (name, installment(row["source"]), medium)
            groups[key][row["word"]] += int(row["count"])
    profiles, excluded = [], []
    for name in sorted({key[0] for key in groups}):
        common = [i for i in range(1, 8) if groups[name, i, "books"] and groups[name, i, "movies"]]
        counts = {medium: sum((groups[name, i, medium] for i in common), Counter())
                  for medium in ("books", "movies")}
        if not common or any(value.total() < min_words for value in counts.values()):
            excluded.append({"character": name, "shared_installments": " ".join(map(str, common)),
                             "book_words": counts["books"].total(), "film_words": counts["movies"].total()})
            continue
        for medium in ("books", "movies"):
            profiles.append({"character": name, "medium": medium,
                             "installments": " ".join(map(str, common)),
                             "words": counts[medium].total(), "counts": counts[medium]})
    return profiles, excluded


def vocabulary_vectors(profiles, remove_names=False):
    """Fit the course TF-IDF formula jointly to the new character profiles."""
    name_words = {part.lower() for p in profiles for part in p["character"].split() if len(part) >= 3}
    stop = set(ENGLISH_STOP_WORDS) | (name_words if remove_names else set())
    content = [{word: count for word, count in p["counts"].items()
                if len(word) >= 3 and word not in stop and word.removesuffix("'s") not in stop}
               for p in profiles]
    vectorizer = DictVectorizer(dtype=np.int64, sort=True)
    full = vectorizer.fit_transform(content).tocsr()
    frequency = np.asarray((full > 0).sum(axis=0)).ravel()
    keep = frequency >= 2
    counts = full[:, keep].tocsr(copy=True)
    vocabulary = vectorizer.get_feature_names_out()[keep]
    weights, _, _, lengths = course_tfidf(counts)
    norms = np.sqrt(np.asarray(weights.multiply(weights).sum(axis=1)).ravel())
    if (norms == 0).any():
        raise ValueError("A profile has no weighted vocabulary. Raise the word threshold or adjust preprocessing.")
    unit = (sparse.diags(1 / norms) @ weights).tocsr()
    similarities = cosine_similarity(unit)
    similarities = np.clip((similarities + similarities.T) / 2, 0, 1)
    np.fill_diagonal(similarities, 1)
    return unit, similarities, vocabulary, lengths


def nearest_graph(similarities, k=3):
    """Undirected union of each profile's k highest positive cosine neighbors."""
    n = len(similarities)
    if n < 2 or k < 1:
        raise ValueError("Need two profiles and a positive k.")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        neighbors = sorted((j for j in range(n) if j != i and similarities[i, j] > 0),
                           key=lambda j: (-similarities[i, j], j))[:k]
        for j in neighbors:
            graph.add_edge(i, j, weight=float(similarities[i, j]))
    return graph


def label_check(graph, profiles, roles, seed=42, permutations=1000):
    """Test weighted same-role edge share; shuffle at CHARACTER level.

    Exclude edges joining two versions of the same character. That avoids
    counting identities as evidence that vocabulary separates story roles.
    """
    names = sorted({p["character"] for p in profiles if roles.get(p["character"]) in {"hero", "villain"}})
    index = {name: i for i, name in enumerate(names)}
    edges = [(index[profiles[i]["character"]], index[profiles[j]["character"]], data["weight"])
             for i, j, data in graph.edges(data=True)
             if profiles[i]["character"] in index and profiles[j]["character"] in index
             and profiles[i]["character"] != profiles[j]["character"]]
    if not edges:
        return {"eligible_edges": 0}
    a, b, weights = (np.asarray(values) for values in zip(*edges))
    a, b = a.astype(int), b.astype(int)
    labels = np.array([roles[name] == "villain" for name in names])
    if not labels.any() or labels.all():
        return {"eligible_edges": len(edges), "reason": "Both hero and villain labels are required."}

    def neighbor_share(values, category):
        first, second = values[a] == category, values[b] == category
        denominator = float(weights[first].sum() + weights[second].sum())
        return float(2 * weights[first & second].sum() / denominator) if denominator else 0.0

    observed = float(weights[labels[a] == labels[b]].sum() / weights.sum())
    rng = np.random.default_rng(seed)
    null, hero_null, villain_null = [], [], []
    for _ in range(permutations):
        shuffled = rng.permutation(labels)
        null.append(float(weights[shuffled[a] == shuffled[b]].sum() / weights.sum()))
        hero_null.append(neighbor_share(shuffled, False))
        villain_null.append(neighbor_share(shuffled, True))
    return {"eligible_characters": len(names), "villain_characters": int(labels.sum()),
            "hero_characters": int((~labels).sum()), "eligible_edges": len(edges),
            "same_role_weight_fraction": observed, "permutation_mean": float(np.mean(null)),
            "permutation_95_range": np.quantile(null, [.025, .975]).tolist(),
            "one_sided_permutation_p": float((1 + np.sum(np.array(null) >= observed)) / (permutations + 1)),
            "same_role_neighbor_share": {"hero": neighbor_share(labels, False), "villain": neighbor_share(labels, True)},
            "permuted_neighbor_share_mean": {"hero": float(np.mean(hero_null)), "villain": float(np.mean(villain_null))},
            "permutations": permutations, "seed": seed,
            "unit": "character; both medium versions keep the same shuffled label",
            "identity_edges": "excluded"}


def top_terms(unit, vocabulary, row, n=8):
    values = unit.getrow(row)
    order = np.lexsort((values.indices, -values.data))[:n]
    return [str(vocabulary[j]) for j in values.indices[order]]


def static_map(graph, profiles, positions, nodes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    background = "#f6f3ec"
    fig, ax = plt.subplots(figsize=(15, 10.5), facecolor=background)
    ax.set_facecolor(background)
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#8e9795", alpha=.13, width=.65)
    for i in range(0, len(profiles), 2):
        first, second = positions[i], positions[i + 1]
        role = nodes[i]["role"]
        ax.annotate("", second, first,
                    arrowprops={"arrowstyle": "->", "color": COLORS[role],
                                "alpha": .38 if role == "villain" else .13, "lw": .8,
                                "shrinkA": 4, "shrinkB": 5}, zorder=1)
    for medium, marker in (("books", "o"), ("movies", "^")):
        for role, color in COLORS.items():
            indices = [i for i, p in enumerate(profiles) if p["medium"] == medium and nodes[i]["role"] == role]
            if indices:
                ax.scatter([positions[i][0] for i in indices], [positions[i][1] for i in indices],
                           marker=marker, s=42, facecolors=background if medium == "books" else color,
                           edgecolors=color, linewidths=1.3, zorder=3)
    label_names = {"Harry Potter", "Ron Weasley", "Hermione Granger", "Albus Dumbledore",
                   "Rubeus Hagrid", "Severus Snape", "Draco Malfoy", "Voldemort",
                   "Dolores Umbridge", "Bellatrix Lestrange", "Lucius Malfoy", "Peter Pettigrew",
                   "Quirinus Quirrell", "Gilderoy Lockhart", "Scabior", "Corban Yaxley"}
    # Put one label near the midpoint of each pair; interactive view labels every node on hover.
    points = np.array([positions[i] for i in range(len(profiles))])
    lower, upper = points.min(axis=0), points.max(axis=0)
    padding = .08 * (upper - lower)
    ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    ax.axis("off")
    fig.subplots_adjust(left=.025, right=.98, top=.86, bottom=.09)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = []
    offsets = [(7, 8), (7, -15), (-7, 8), (-7, -15), (12, 22), (-12, 22), (12, -27), (-12, -27), (25, 0), (-25, 0)]
    for i in range(0, len(profiles), 2):
        name = profiles[i]["character"]
        if name not in label_names:
            continue
        point = positions[i]
        best = None
        for dx, dy in offsets:
            annotation = ax.annotate(name, point, xytext=(dx, dy), textcoords="offset points",
                                     ha="left" if dx > 0 else "right", fontsize=9,
                                     color="#263c3d", fontfamily="DejaVu Sans")
            box = annotation.get_window_extent(renderer).expanded(1.05, 1.25)
            collisions = sum(box.overlaps(old) for old in occupied)
            annotation.remove()
            if best is None or collisions < best[0]:
                best = (collisions, dx, dy, box)
            if not collisions:
                break
        _, dx, dy, box = best
        occupied.append(box)
        ax.annotate(name, point, xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", fontsize=9, color="#263c3d",
                    bbox={"facecolor": background, "edgecolor": "none", "alpha": .85, "pad": 1.5},
                    arrowprops={"arrowstyle": "-", "color": "#92958e", "lw": .4}, zorder=4)
    fig.text(.045, .955, "FROM PAGE TO SCREEN", fontsize=10, color="#6a7169", fontfamily="DejaVu Sans")
    fig.text(.045, .912, "A network of character vocabularies", fontsize=26,
             fontfamily="DejaVu Serif", color="#203b3b")
    fig.text(.045, .877, "Open circles: books     Triangles: films     Faint edges: nearest vocabularies     Arrows: the same character", fontsize=10, color="#59655e")
    legend = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=color,
                     markeredgecolor=color, label=ROLE_NAMES[role]) for role, color in COLORS.items()]
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(.04, .038), ncol=4, frameon=False, fontsize=10)
    fig.text(.045, .02, "Positions use TF-IDF similarities and a network layout. Story labels are added afterward; map distance is approximate.", fontsize=9, color="#69736b")
    fig.savefig(path, dpi=180, facecolor=background)
    plt.close(fig)


def connection_payload(graph, profiles, edge_details):
    """Expose the actual graph neighbors and the words behind each edge.

    Book/film counterparts are a separate comparison, so they are omitted from
    the other-character connection list. Incoming kNN edges remain inspectable.
    """
    lookup = {tuple(sorted((int(edge["source"]), int(edge["target"])))): edge
              for edge in edge_details}
    result = {}
    for i in graph:
        rows = []
        for j in graph.neighbors(i):
            if profiles[i]["character"] == profiles[j]["character"]:
                continue
            edge = lookup[tuple(sorted((i, j)))]
            rows.append({"id": j, "cosine": float(edge["cosine"]),
                         "shared_terms": edge["shared_terms"].split(", ") if edge["shared_terms"] else []})
        result[str(i)] = sorted(rows, key=lambda row: (-row["cosine"], row["id"]))
    return result


def interactive_map(graph, profiles, positions, nodes, similarities, path, edge_details):
    import plotly.graph_objects as go
    import plotly.io as pio

    traces = []
    xs, ys = [], []
    for i, j in graph.edges:
        xs += [positions[i][0], positions[j][0], None]
        ys += [positions[i][1], positions[j][1], None]
    traces.append(go.Scatter(x=xs, y=ys, mode="lines", line={"color": "#79918a", "width": .8},
                             opacity=.24, hoverinfo="skip", showlegend=False))
    # The counterpart link appears only for a selected character, instead of
    # adding 59 unlabelled identity connections to the overview.
    traces.append(go.Scatter(x=[], y=[], mode="lines", line={"color": "#55777d", "width": 2, "dash": "dot"},
                             hoverinfo="skip", showlegend=False))
    trace_nodes = []
    for role, color in COLORS.items():
        for medium, symbol in (("books", "circle-open"), ("movies", "triangle-up")):
            indices = [i for i, p in enumerate(profiles) if p["medium"] == medium and nodes[i]["role"] == role]
            trace_nodes.append({"trace": len(traces), "medium": medium, "ids": indices})
            traces.append(go.Scatter(x=[positions[i][0] for i in indices], y=[positions[i][1] for i in indices],
                customdata=indices, mode="markers",
                marker={"symbol": symbol, "size": 13, "color": color, "line": {"width": 1.7}},
                name=ROLE_NAMES[role], legendgroup=role, showlegend=False,
                hovertext=[f"<b>{html.escape(profiles[i]['character'])}</b> · {medium}<br>"
                           f"{nodes[i]['words']:,} words · installments {nodes[i]['installments']}<br>"
                           f"{ROLE_NAMES[role]} · community {nodes[i]['community']}<br>"
                           f"Book–film cosine: {nodes[i]['counterpart_cosine']:.3f}<br>"
                           f"Distinctive terms: {html.escape(nodes[i]['top_words'])}" for i in indices],
                hoverinfo="text"))
    focus_traces = []
    for _ in range(3):
        focus_traces.append(len(traces))
        traces.append(go.Scatter(x=[], y=[], mode="lines", line={"color": "#284e49", "width": 2.6},
                                 hoverinfo="skip", showlegend=False))
    hover_trace = len(traces)
    traces.append(go.Scatter(x=[], y=[], mode="lines", line={"color": "#182f2d", "width": 4},
                             hoverinfo="skip", showlegend=False))
    fig = go.Figure(traces)
    fig.update_layout(paper_bgcolor="#f7f5ef", plot_bgcolor="#f7f5ef", height=650,
                      margin={"l": 28, "r": 28, "t": 28, "b": 28},
                      font={"family": "Arial, sans-serif", "size": 14, "color": "#263e37"},
                      hoverlabel={"font": {"size": 14}}, hovermode="closest", dragmode="pan",
                      xaxis={"visible": False}, yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1})
    plot = pio.to_html(fig, full_html=False, include_plotlyjs=True, div_id="map",
                       config={"responsive": True, "displaylogo": False, "scrollZoom": True,
                               "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]})
    data = {"nodes": nodes, "positions": {str(i): positions[i].tolist() for i in graph},
            "edges": list(graph.edges), "traceNodes": trace_nodes, "focusTraces": focus_traces,
            "hoverTrace": hover_trace, "connections": connection_payload(graph, profiles, edge_details)}
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    options = "".join(f'<option value="{i}">{html.escape(p["character"])}</option>' for i, p in enumerate(profiles) if i % 2 == 0)
    template_dir = Path(__file__).resolve().parent / "templates"
    document = (template_dir / "vocabulary_map.html").read_text(encoding="utf-8")
    script = (template_dir / "vocabulary_map.js").read_text(encoding="utf-8")
    path.write_text(document.replace("__MAP_DATA__", encoded)
                    .replace("__CHARACTER_OPTIONS__", options)
                    .replace("__MAP_SCRIPT__", script)
                    .replace("__MAP_PLOT__", plot), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books", type=Path, default=ROOT / "results/word_counts/books/character_word_counts.csv")
    parser.add_argument("--movies", type=Path, default=ROOT / "results/word_counts/movies/character_word_counts.csv")
    parser.add_argument("--roles", type=Path, default=ROOT / "config/map_roles.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/vocabulary_map")
    parser.add_argument("--min-words", type=int, default=100)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--remove-names", action="store_true")
    args = parser.parse_args()
    if args.min_words < 1 or args.neighbors < 1:
        parser.error("Word thresholds and neighbor counts must be positive.")
    paths = {"books": args.books, "movies": args.movies}
    profiles, excluded = matched_profiles(paths, args.min_words)
    if len(profiles) < 4:
        parser.error("Need at least two characters with sufficient matched dialogue.")
    unit, similarities, vocabulary, lengths = vocabulary_vectors(profiles, args.remove_names)
    graph = nearest_graph(similarities, args.neighbors)
    # Roles are loaded only after the representation, edges and positions are fixed.
    positions = nx.spring_layout(graph, weight="weight", seed=42, iterations=600, k=1.6 / math.sqrt(len(graph)))
    communities = list(nx.community.louvain_communities(graph, weight="weight", seed=42))
    membership = {node: number for number, group in enumerate(communities, 1) for node in group}
    role_rows = read_csv(args.roles)
    if len({r["character"] for r in role_rows}) != len(role_rows) or any(r["role"] not in COLORS for r in role_rows):
        raise ValueError("Role table contains duplicate characters or unsupported roles.")
    roles = {r["character"]: r["role"] for r in role_rows}
    nodes = []
    for i, p in enumerate(profiles):
        other = i + 1 if i % 2 == 0 else i - 1
        row = {"id": i, "character": p["character"], "medium": p["medium"],
               "role": roles.get(p["character"], "unassigned"), "words": p["words"],
               "installments": p["installments"], "retained_vocabulary_words": int(lengths[i]),
               "community": membership[i], "degree": graph.degree(i),
               "counterpart_cosine": float(similarities[i, other]),
               "top_words": ", ".join(top_terms(unit, vocabulary, i)),
               "x": float(positions[i][0]), "y": float(positions[i][1])}
        nodes.append(row)
        graph.nodes[i].update({key: value for key, value in row.items() if key != "id"})
    edges = []
    for i, j, data in graph.edges(data=True):
        product = unit[i].multiply(unit[j]).toarray().ravel()
        shared = [str(vocabulary[column]) for column in np.argsort(-product, kind="stable") if product[column] > 0][:5]
        edges.append({"source": i, "target": j, "source_character": profiles[i]["character"],
                      "source_medium": profiles[i]["medium"], "target_character": profiles[j]["character"],
                      "target_medium": profiles[j]["medium"], "cosine": data["weight"],
                      "shared_terms": ", ".join(shared)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "nodes.csv", nodes)
    write_csv(args.output_dir / "edges.csv", edges)
    write_csv(args.output_dir / "excluded_characters.csv", excluded)
    write_csv(args.output_dir / "vocabulary.csv", [{"column": i, "word": str(word)} for i, word in enumerate(vocabulary)])
    sparse.save_npz(args.output_dir / "vectors.npz", unit)
    np.save(args.output_dir / "cosine_similarity.npy", similarities)
    nx.write_graphml(graph, args.output_dir / "network.graphml")
    static_map(graph, profiles, positions, nodes, args.output_dir / "vocabulary_map.png")
    interactive_map(graph, profiles, positions, nodes, similarities, args.output_dir / "vocabulary_map.html", edges)
    check = label_check(graph, profiles, roles)
    metrics = {"characters": len(profiles) // 2, "nodes": len(graph), "edges": graph.number_of_edges(),
               "components": nx.number_connected_components(graph), "communities": len(communities),
               "weighted_modularity": float(nx.community.modularity(graph, communities, weight="weight")),
               "role_characters": dict(Counter(nodes[i]["role"] for i in range(0, len(nodes), 2))),
               "role_permutation_check": check,
               "parameters": {"min_words_per_medium": args.min_words, "nearest_neighbors": args.neighbors,
                              "min_document_frequency": 2, "remove_names": args.remove_names, "seed": 42},
               "aliases": ALIASES,
               "inputs": {medium: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for medium, path in paths.items()},
               "roles_sha256": hashlib.sha256(args.roles.read_bytes()).hexdigest()}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = ["# Character vocabulary map", "",
        "[Open the interactive map](vocabulary_map.html) · [Static map](vocabulary_map.png)", "",
        f"This is a {len(graph)}-node, {graph.number_of_edges()}-edge vocabulary-similarity network: {len(profiles)//2} characters, each represented once for books and once for films.",
        "A node contains the pooled dialogue of one character in one medium, restricted to installments where that character has speech in both media. The two final films form installment seven. Each medium must supply at least the configured word threshold. Matched installments do not guarantee identical scenes or word budgets.",
        "Explicit Tom Riddle labels are merged with Voldemort locally, and several cross-media name variants are normalized. Source files remain unchanged. Crouch father/son and Moody impersonation cases remain unresolved and are marked ambiguous in the role table.", "",
        "## Representation and edges", "",
        "TF-IDF is fitted jointly to these pooled profiles, using the same course formula as step 3. This is a new document unit; it does not reuse the previous per-installment IDF unchanged. English stopwords, tokens shorter than three letters and words occurring in fewer than two profiles are excluded. TF uses the retained word-count matrix. Vectors are then normalized for cosine similarity. Names are retained unless --remove-names is supplied.",
        "Each profile chooses its three nearest positive-cosine neighbors by default. The graph is the undirected union of those choices. Edge weight is full-vector cosine similarity, rather than distance in the picture. Links express vocabulary similarity, rather than dialogue interaction. Identity links drawn between the two versions of a character are a separate visual guide; they are not added to the graph unless the nearest-neighbor rule already creates that edge.",
        "Positions use a weighted spring layout with seed 42. Book circles and film triangles are drawn together in one coordinate system. All other labels, including hero/villain colors, are added after vectors, graph and coordinates are calculated. Distances, apparent clusters and arrow lengths in the drawing are approximate; use the displayed cosine scores for numerical comparisons.", "",
        "## What to inspect", "",
        "Select a point or character in the interactive map. Selection focuses one book or film profile, labels up to three of its strongest actual graph connections, and fits the view to those nodes. The adjacent panel names those exact connections, reports their cosine scores and shared contributing words, and lets you follow a connection by clicking its name. The character's own book/film counterpart is shown separately as a dotted link and similarity score. List all reveals additional incoming graph connections while the map keeps the three strongest labeled. Switch profile tabs or filter books/films/both. Underlying graph positions remain fixed. Network-community IDs come from weighted Louvain; these communities are not trained moral categories.",
        "Story labels are provisional and editable in config/map_roles.csv. Unlisted characters are unassigned. The graph representation and positions do not use these labels. Labels describe broad series roles; contested and changing allegiances should be examined separately.", "",
        f"Louvain finds {len(communities)} communities with weighted modularity {metrics['weighted_modularity']:.3f} in this sparsified graph. These values depend on the nearest-neighbor rule and preprocessing.", "",
        "## Hero/villain grouping check", "",
        "The check uses only graph edges whose endpoints have hero/villain labels and refer to different characters. Same-character links are excluded. It compares the cosine-weighted share of edges connecting the same role with 1,000 permutations of character labels. Both media keep the same permuted label for a character. This controls class imbalance and avoids treating the two versions as independent label assignments.",
        "```json", json.dumps(check, indent=2), "```", "",
        "Read the per-role neighbor shares alongside the overall score. The overall same-role fraction can be driven mainly by one cohesive group. A high overall fraction alone does not establish that all villains form a cluster. Neighbor shares use only the hero/villain-labeled, different-character edges described above; ambiguous and unassigned endpoints are omitted.", "",
        "This is an exploratory check for this representation, graph and chosen labels. It is not held-out classification accuracy or evidence that vocabulary determines morality. No cross-preprocessing or graph-threshold stability has been established.", "",
        "## Files and rerunning", "",
        "nodes.csv and edges.csv expose every profile, edge weight, source coverage and important shared terms. network.graphml opens in network-analysis software. vectors.npz and cosine_similarity.npy retain the full representation and distances. metrics.json records parameters, graph statistics, input hashes and alias choices.",
        "Run `.venv\\Scripts\\python.exe -m code.vocabulary_map` from the repository root. A sensitivity run can use `--min-words 300` or `--remove-names --output-dir results/vocabulary_map_without_names`. Name-component removal can also remove ordinary words such as black or wood.",
        "Speaker-attribution and transcription errors in the supplied data remain. Sparse profiles have less stable vocabulary estimates. Use this map to select cases for checking against original passages, and compare with the existing interaction network to ask where linguistic and structural neighbors agree.", "",
        "[Course: TF-IDF and cosine](https://sunelehmann.com/socialgraphs2026-web/weeks/week6.html) · [Course: corpus maps and their limits](https://sunelehmann.com/socialgraphs2026-web/weeks/week7.html)", ""]
    (args.output_dir / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Map: {args.output_dir / 'vocabulary_map.html'}")


if __name__ == "__main__":
    main()
