"""Notebook support: directed Infomap hierarchy and exact triad comparisons."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import json
import math

import infomap
import networkx as nx
import numpy as np
from networkx.algorithms.triads import TRIAD_NAMES, TRICODES


def community_atlas(graph, names):
    isolates = sorted(nx.isolates(graph))
    active = graph.subgraph(sorted(set(graph) - set(isolates))).copy()
    result = infomap.run(
        active, directed=True, seed=42, num_trials=20, num_threads=1,
        to_nodes=True, recorded_teleportation=True, teleportation_probability=0.15,
    )
    groups = {"root": {"id": "root", "label": "All communities", "parent": None,
                       "children": [], "members": sorted(graph), "kind": "root"}}
    node_paths, flow = {}, {}
    for leaf in result.nodes(depth=-1):
        node = result.names[leaf.node_id]
        module_path = leaf.path[:-1]
        node_paths[node] = ["m-" + "-".join(map(str, module_path[:i])) for i in range(1, len(module_path) + 1)]
        flow[node] = leaf.flow
        parent = "root"
        for group_id in node_paths[node]:
            if group_id not in groups:
                groups[group_id] = {"id": group_id, "label": "Community " + group_id[2:].replace("-", "."),
                                    "parent": parent, "children": [], "members": [], "kind": "community"}
                groups[parent]["children"].append(group_id)
            groups[group_id]["members"].append(node)
            parent = group_id
    if isolates:
        groups["isolates"] = {"id": "isolates", "label": "Isolates", "parent": "root",
                              "children": [], "members": isolates, "kind": "display-group"}
        groups["root"]["children"].append("isolates")
        for node in isolates:
            node_paths[node] = ["isolates"]
            flow[node] = 0
    for group in groups.values():
        group["members"].sort()
        group["children"].sort(key=lambda key: (-len(groups[key]["members"]), key))
        group["flow"] = sum(flow.get(n, 0) for n in group["members"])
        group["leaders"] = sorted(group["members"], key=lambda n: (-flow.get(n, 0), n))[:3]
        members = set(group["members"])
        group["internalEdges"] = sum(a in members and b in members for a, b in graph.edges())
        group["outgoingEdges"] = sum(a in members and b not in members for a, b in graph.edges())
        group["incomingEdges"] = sum(a not in members and b in members for a, b in graph.edges())
    # Cross-community links per character are measurable, not a claim of social brokerage.
    bridges = []
    for n in active:
        neighbors = set(graph.predecessors(n)) | set(graph.successors(n))
        other = {v for v in neighbors if node_paths[v][0] != node_paths[n][0]}
        bridges.append({"id": n, "crossNeighbors": len(other),
                        "otherCommunities": len({node_paths[v][0] for v in other})})
    bridges.sort(key=lambda item: (-item["crossNeighbors"], item["id"]))
    return {"groups": groups, "nodePaths": node_paths, "flow": flow, "bridges": bridges,
            "method": {"algorithm": "Infomap", "version": infomap.__version__, "seed": 42,
                       "trials": 20, "directed": True, "teleportProbability": .15,
                       "teleportTargets": "uniform active nodes", "recordedTeleportation": True,
                       "activeNodes": len(active), "isolatesExcluded": len(isolates),
                       "topModules": result.num_top_modules, "levels": result.num_levels,
                       "codelength": result.codelength, "oneLevelCodelength": result.one_level_codelength}}


def motif_fingerprint(graph, samples=100, swaps_per_edge=10):
    assert not nx.number_of_selfloops(graph), "Triad analysis assumes no self-loops"
    observed = nx.triadic_census(graph)
    assert sum(observed.values()) == math.comb(len(graph), 3)
    # Collect up to eight exact induced examples per type, in node-ID order.
    # These examples illustrate the pattern; they are not a random sample.
    examples = {kind: [] for kind in TRIAD_NAMES}
    quota = {kind: min(8, observed[kind]) for kind in TRIAD_NAMES}
    successors = {n: set(graph.successors(n)) for n in graph}
    for a, b, c in combinations(sorted(graph), 3):
        bits = ((b in successors[a]) + 2 * (a in successors[b]) + 4 * (c in successors[a])
                + 8 * (a in successors[c]) + 16 * (c in successors[b]) + 32 * (b in successors[c]))
        kind = TRIAD_NAMES[TRICODES[bits] - 1]
        if len(examples[kind]) < quota[kind]:
            examples[kind].append([a, b, c])
            if all(len(examples[k]) == quota[k] for k in TRIAD_NAMES):
                break
    null_counts = []
    original_degrees = {n: (graph.in_degree(n), graph.out_degree(n)) for n in graph}
    for seed in range(4200, 4200 + samples):
        randomized = graph.copy()
        nx.directed_edge_swap(randomized, nswap=swaps_per_edge * graph.number_of_edges(),
                              max_tries=swaps_per_edge * graph.number_of_edges() * 50, seed=seed)
        assert original_degrees == {n: (randomized.in_degree(n), randomized.out_degree(n)) for n in randomized}
        assert not nx.number_of_selfloops(randomized)
        census = nx.triadic_census(randomized)
        assert sum(census.values()) == math.comb(len(graph), 3)
        null_counts.append(census)
    motifs = []
    for kind in TRIAD_NAMES:
        counts = [sample[kind] for sample in null_counts]
        mean, std = float(np.mean(counts)), float(np.std(counts, ddof=1))
        motif_graph = nx.triad_graph(kind)
        motifs.append({"id": kind, "observed": observed[kind], "nullMean": mean, "nullStd": std,
                       "z": (observed[kind] - mean) / std if std > 0 else None,
                       "nullCounts": counts, "examples": examples[kind],
                       "diagramEdges": list(motif_graph.edges()),
                       "connected": nx.is_weakly_connected(motif_graph)})
    return {"motifs": motifs, "method": {"samples": samples, "seedStart": 4200,
            "swapsPerEdge": swaps_per_edge, "successfulSwapsPerGraph": swaps_per_edge * graph.number_of_edges(),
            "preserves": "Each node's in-degree and out-degree; node roster; simple directed graph",
            "triads": math.comb(len(graph), 3), "examples": "First up to eight induced triples in node-ID order",
            "caveat": "Finite rewiring runs are an approximate null ensemble, not proven uniform samples. Z-scores are exploratory, not significance tests."}}


def build_discoveries(graph, names, output):
    atlas = community_atlas(graph, names)
    print(f"Infomap: {atlas['method']['topModules']} top communities; {atlas['method']['isolatesExcluded']} isolates shown separately")
    motifs = motif_fingerprint(graph)
    data = {"atlas": atlas, "fingerprint": motifs}
    Path(output).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Exact census: {motifs['method']['triads']:,} triples; compared with {motifs['method']['samples']} degree-preserving rewired graphs")
    return data
