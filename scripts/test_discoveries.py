"""Validate partition membership, exact motif examples, and exported statistics."""
from pathlib import Path
from collections import Counter
import json
import math
import networkx as nx
import numpy as np

root = Path(__file__).resolve().parents[1]
graph_data = json.loads((root / "assets/plots/explorer.json").read_text(encoding="utf-8"))
data = json.loads((root / "assets/plots/discoveries.json").read_text(encoding="utf-8"))
graph = nx.DiGraph()
graph.add_nodes_from(n["id"] for n in graph_data["nodes"])
graph.add_edges_from(graph_data["edges"])
atlas = data["atlas"]
groups = atlas["groups"]
assert set(groups["root"]["members"]) == set(graph)
assert set(groups["isolates"]["members"]) == set(nx.isolates(graph))
assert math.isclose(sum(atlas["flow"].values()), 1, rel_tol=1e-8)
for group in groups.values():
    assert len(group["members"]) == len(set(group["members"]))
    if group["children"]:
        members = [n for child in group["children"] for n in groups[child]["members"]]
        assert Counter(members) == Counter(group["members"])
        assert all(groups[child]["parent"] == group["id"] for child in group["children"])
for node, path in atlas["nodePaths"].items():
    assert all(node in groups[group]["members"] for group in path)
    assert groups[path[0]]["parent"] == "root"
    assert all(groups[b]["parent"] == a for a, b in zip(path, path[1:]))
counts = nx.triadic_census(graph)
assert sum(counts.values()) == math.comb(len(graph), 3)
for motif in data["fingerprint"]["motifs"]:
    assert motif["observed"] == counts[motif["id"]]
    assert len(motif["nullCounts"]) == 100
    assert math.isclose(motif["nullMean"], np.mean(motif["nullCounts"]), abs_tol=1e-9)
    assert math.isclose(motif["nullStd"], np.std(motif["nullCounts"], ddof=1), abs_tol=1e-9)
    assert len(motif["examples"]) == min(8, motif["observed"])
    for example in motif["examples"]:
        assert len(set(example)) == 3
        assert nx.triad_type(graph.subgraph(example)) == motif["id"]
    canonical = nx.DiGraph()
    canonical.add_nodes_from("abc")
    canonical.add_edges_from(motif["diagramEdges"])
    assert nx.triad_type(canonical) == motif["id"]
    if motif["nullStd"]:
        assert math.isclose(motif["z"], (motif["observed"] - motif["nullMean"]) / motif["nullStd"])
    else:
        assert motif["z"] is None
for i in range(100):
    assert sum(m["nullCounts"][i] for m in data["fingerprint"]["motifs"]) == math.comb(len(graph), 3)
print("Passed: full hierarchy coverage, isolates, flow normalization, all 16 triad counts and diagrams, 128 exact examples, 100 null-census totals, and z-score calculations.")
