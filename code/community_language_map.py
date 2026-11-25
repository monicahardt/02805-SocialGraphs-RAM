"""Render the large interactive map of dialogue ties and vocabulary similarity.

Run python -m code.community_language_map after code.community_language.
"""

import json
from itertools import combinations

import networkx as nx
import numpy as np

from .community_language import MAP, OUTPUT, ROOT, SEED, collect_inputs, groups_for, read_table
from .interaction_graph import _js_json

TEMPLATE = ROOT / "code" / "templates" / "community_language_map.html"
# Horizontal and vertical spread are controlled separately for each community.
REGIONS = (
    (293, 236, 208, 145),
    (875, 236, 208, 145),
    (432, 569, 258, 94),
    (972, 570, 113, 77),
)


def layout_community(graph, names, region):
    cx, cy, rx, ry = region
    if len(names) == 1:
        return {names[0]: (cx, cy)}
    subnet = nx.Graph(graph.subgraph(names))
    for _, _, edge in subnet.edges(data=True):
        edge["layout_weight"] = float(np.log1p(edge["weight"]))
    coordinates = nx.spring_layout(subnet, weight="layout_weight", seed=SEED, iterations=160)
    radius = max(np.linalg.norm(position) for position in coordinates.values())
    points = np.stack([coordinates[name] for name in names]) * np.array([rx, ry]) / radius
    for _ in range(120):
        for first, second in combinations(range(len(points)), 2):
            direction = points[second] - points[first]
            distance = np.linalg.norm(direction)
            if 0 < distance < 65:
                adjustment = direction * (65 - distance) / (2 * distance)
                points[first] -= adjustment
                points[second] += adjustment
        radii = np.linalg.norm(points / [rx, ry], axis=1, keepdims=True)
        points /= np.maximum(radii, 1)
    return {
        name: (round(cx + x, 2), round(cy + y, 2))
        for name, (x, y) in zip(names, points)
    }


def build_view(medium, graph, profiles, similarity, statistics):
    source = "books" if medium == "Books" else "movies"
    rows = {
        row["character"]: row
        for row in profiles
        if row["medium"] == source and row["role"] in {"hero", "villain"}
    }
    groups = groups_for(graph, "weight")
    visible_groups = {}
    for name in rows:
        visible_groups.setdefault(groups[name], []).append(name)
    clusters = sorted(visible_groups.values(), key=lambda names: (-len(names), min(names)))
    if len(clusters) > len(REGIONS):
        raise ValueError("More visible communities than there are positions on the map.")

    nodes = []
    communities = []
    lookup = {}
    for i, (names, region) in enumerate(zip(clusters, REGIONS)):
        points = layout_community(graph, names, region)
        for name in names:
            x, y = points[name]
            lookup[name] = len(nodes)
            nodes.append({
                "name": name, "role": rows[name]["role"], "community": i, "x": x, "y": y,
            })
        cx, cy, rx, ry = region
        communities.append({
            "label": chr(ord("A") + i), "x": cx, "y": cy,
            "rx": rx + 43 if len(names) > 1 else 95,
            "ry": ry + 38 if len(names) > 1 else 66,
            "count": len(names),
        })

    edges = []
    for first, second, metadata in graph.subgraph(rows).edges(data=True):
        i, j = (int(rows[name]["id"]) for name in (first, second))
        edges.append({
            "a": lookup[first], "b": lookup[second], "turns": metadata["weight"],
            "cosine": round(float(similarity[i, j]), 6),
        })

    summary = [
        {
            "role": row["role"],
            "inside": float(row["first_mean"]),
            "across": float(row["second_mean"]),
            "insidePairs": int(row["first_pairs"]),
            "acrossPairs": int(row["second_pairs"]),
        }
        for row in statistics if (
            row["medium"] == medium
            and row["comparison"] == "Weighted community"
            and row["dialogue"] == "Full profile"
        )
    ]
    return {"nodes": nodes, "edges": edges, "communities": communities, "summary": summary}


def main():
    profiles = sorted(read_table(MAP / "nodes.csv"), key=lambda row: int(row["id"]))
    vocabulary = [row["word"] for row in read_table(MAP / "vocabulary.csv")]
    aliases = json.loads((MAP / "metrics.json").read_text(encoding="utf-8"))["aliases"]
    similarity = np.load(MAP / "cosine_similarity.npy")
    statistics = read_table(OUTPUT / "plot_data.csv")
    graphs, _ = collect_inputs(profiles, vocabulary, aliases)
    payload = {
        medium: build_view(medium, graph, profiles, similarity, statistics)
        for medium, graph in graphs.items()
    }
    destination = OUTPUT / "network.html"
    destination.write_text(
        TEMPLATE.read_text(encoding="utf-8").replace("__COMMUNITY_DATA__", _js_json(payload)),
        encoding="utf-8",
    )
    print(f"Wrote dialogue community map to {destination}")


if __name__ == "__main__":
    main()
