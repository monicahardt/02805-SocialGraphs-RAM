"""Build an interactive movie centrality and social-bridge explorer."""

import argparse
import html
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx

if __package__:
    from .dialogue_csv_reader import DialogueRecord, read_dialogue_records
    from .interaction_graph import NON_SPECIFIC_MOVIE_LABELS
else:  # Support `python3 code/centrality_analysis.py` from the repository root.
    from dialogue_csv_reader import DialogueRecord, read_dialogue_records
    from interaction_graph import NON_SPECIFIC_MOVIE_LABELS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "movie_data" / "all_movies.csv"
DEFAULT_OUTPUT = ROOT / "graphs" / "movies" / "analysis" / "centrality.html"
RULES = {"1": 1, "2": 2, "5": 5, "chapter": None}
RULE_LABELS = {
    "1": "Next 1 dialogue line",
    "2": "Next 2 dialogue lines",
    "5": "Next 5 dialogue lines",
    "chapter": "Whole chapter co-appearance",
}
METRICS = {
    "degree": "Degree centrality",
    "closeness": "Closeness centrality",
    "betweenness": "Betweenness centrality",
    "eigenvector": "Eigenvector centrality",
}


def _safe_json(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _collect_evidence(records: list[DialogueRecord]):
    """Keep speaker turn blocks and chapter speaker sets for the edge rules."""
    turns_by_source: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    speakers_by_chapter: dict[tuple[str, str], set[str]] = defaultdict(set)
    speakers_by_movie: dict[str, set[str]] = defaultdict(set)
    active_key: tuple[str, str] | None = None
    active_turns: list[str] = []

    def finish_block() -> None:
        nonlocal active_turns
        if active_key is not None and active_turns:
            turns_by_source[active_key[0]].append((active_key[1], active_turns))
        active_turns = []

    for row in records:
        movie = row.book or "Unknown movie"
        key = (movie, row.chapter)
        if key != active_key:
            finish_block()
            active_key = key
        if row.character.casefold() == "unknown" or row.confidence.casefold() in {
            "low",
            "unattributed",
        }:
            finish_block()
            continue
        speakers_by_movie[movie].add(row.character)
        speakers_by_chapter[key].add(row.character)
        active_turns.append(row.character)
    finish_block()
    return turns_by_source, speakers_by_chapter, speakers_by_movie


def _build_graph(
    movie: str,
    rule: str,
    named_only: bool,
    turns_by_source: dict[str, list[tuple[str, list[str]]]],
    speakers_by_chapter: dict[tuple[str, str], set[str]],
    speakers_by_movie: dict[str, set[str]],
) -> tuple[nx.Graph, Counter[tuple[str, str]]]:
    selected_movies = set(speakers_by_movie) if movie == "all" else {movie}
    is_named = lambda name: name.casefold() not in NON_SPECIFIC_MOVIE_LABELS
    graph = nx.Graph()
    node_names = {
        name
        for source in selected_movies
        for name in speakers_by_movie.get(source, set())
        if not named_only or is_named(name)
    }
    graph.add_nodes_from(sorted(node_names))
    evidence: Counter[tuple[str, str]] = Counter()

    if rule == "chapter":
        for (source, _chapter), speakers in speakers_by_chapter.items():
            if source not in selected_movies:
                continue
            present = sorted(name for name in speakers if name in node_names)
            for pair in combinations(present, 2):
                graph.add_edge(*pair)
                evidence[pair] += 1
    else:
        window = RULES[rule]
        for source in selected_movies:
            for _chapter, turns in turns_by_source.get(source, []):
                for index, first in enumerate(turns):
                    if first not in node_names:
                        continue
                    for second in turns[index + 1 : index + window + 1]:
                        if second not in node_names or first == second:
                            continue
                        pair = tuple(sorted((first, second)))
                        graph.add_edge(*pair)
                        evidence[pair] += 1
    return graph, evidence


def _centralities(graph: nx.Graph) -> dict[str, dict[str, float]]:
    if graph.number_of_nodes() == 0:
        return {key: {} for key in METRICS}
    node_count = graph.number_of_nodes()
    degree = {
        node: (graph.degree[node] / (node_count - 1) if node_count > 1 else 0.0)
        for node in graph
    }
    closeness = nx.closeness_centrality(graph, wf_improved=True)
    betweenness = nx.betweenness_centrality(graph, normalized=True, weight=None)
    if graph.number_of_edges() == 0:
        eigenvector = {node: 1 / math.sqrt(graph.number_of_nodes()) for node in graph}
    else:
        try:
            eigenvector = nx.eigenvector_centrality(graph, max_iter=2000, tol=1e-9, weight=None)
        except nx.PowerIterationFailedConvergence:
            eigenvector = nx.eigenvector_centrality_numpy(graph, weight=None)
    return {
        "degree": degree,
        "closeness": closeness,
        "betweenness": betweenness,
        "eigenvector": eigenvector,
    }


def _scenario(graph: nx.Graph, evidence: Counter[tuple[str, str]]) -> dict:
    centralities = _centralities(graph)
    positions = nx.spring_layout(graph, seed=42, weight=None, iterations=100) if graph else {}
    articulation_points = set(nx.articulation_points(graph)) if graph.number_of_nodes() else set()
    components = list(nx.connected_components(graph)) if graph.number_of_nodes() else []
    component_size = {node: len(component) for component in components for node in component}
    connected_pairs = sum(len(component) * (len(component) - 1) // 2 for component in components)
    separated_pairs = {}
    for name in graph:
        pairs_before_removal = connected_pairs - max(0, component_size[name] - 1)
        reduced = graph.copy()
        reduced.remove_node(name)
        pairs_after_removal = sum(
            len(component) * (len(component) - 1) // 2
            for component in nx.connected_components(reduced)
        )
        separated_pairs[name] = max(0, pairs_before_removal - pairs_after_removal)
    edge_betweenness = nx.edge_betweenness_centrality(graph, normalized=True, weight=None)
    structural_bridges = {frozenset(edge) for edge in nx.bridges(graph)}
    nodes = []
    for name in sorted(graph.nodes, key=str.casefold):
        nodes.append({
            "id": name,
            "label": name,
            "degree": int(graph.degree[name]),
            "articulation": name in articulation_points,
            "separated_pairs": int(separated_pairs[name]),
            "scores": {metric: float(scores[name]) for metric, scores in centralities.items()},
            "x": float(positions[name][0]) * 1000,
            "y": float(positions[name][1]) * 1000,
        })
    edges = [
        {
            "from": first,
            "to": second,
            "weight": int(count),
            "edge_betweenness": float(edge_betweenness.get((first, second), edge_betweenness.get((second, first), 0.0))),
            "is_bridge": frozenset((first, second)) in structural_bridges,
            "title": f"{count} interaction evidence occurrences",
        }
        for (first, second), count in sorted(evidence.items())
    ]
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    return {
        "nodes": nodes,
        "edges": edges,
        "metrics": {
            "nodes": node_count,
            "edges": edge_count,
            "average_degree": 2 * edge_count / node_count if node_count else 0.0,
            "components": nx.number_connected_components(graph) if node_count else 0,
            "articulation_points": len(articulation_points),
            "structural_bridges": len(structural_bridges),
        },
    }


def build_scenarios(records: list[DialogueRecord]) -> tuple[dict, list[str]]:
    turns, chapter_speakers, speakers_by_movie = _collect_evidence(records)
    movies = sorted(speakers_by_movie, key=str.casefold)
    scenarios = {}
    for rule in RULES:
        scenarios[rule] = {}
        for movie in ["all", *movies]:
            scenarios[rule][movie] = {}
            for people_mode, named_only in (("all", False), ("named", True)):
                graph, evidence = _build_graph(
                    movie, rule, named_only, turns, chapter_speakers, speakers_by_movie
                )
                scenarios[rule][movie][people_mode] = _scenario(graph, evidence)
    return scenarios, movies


def write_page(csv_path: Path = DEFAULT_CSV, output_path: Path = DEFAULT_OUTPUT) -> None:
    scenarios, movies = build_scenarios(read_dialogue_records(csv_path))
    movie_options = "".join(
        f'<option value="{html.escape(movie, quote=True)}">{html.escape(movie)}</option>'
        for movie in movies
    )
    rule_options = "".join(
        f'<option value="{rule}">{label}</option>' for rule, label in RULE_LABELS.items()
    )
    metric_options = "".join(
        f'<option value="{key}">{label}</option>' for key, label in METRICS.items()
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Movie Network Centrality</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
*{{box-sizing:border-box}}body{{font:15px system-ui,sans-serif;color:#202832;margin:0}}header{{padding:15px 20px 10px}}h1{{font-size:23px;margin:0 0 6px}}header p{{color:#596775;max-width:1150px;margin:5px 0}}.controls{{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:14px 0 6px}}select{{font:inherit;padding:6px}}main{{padding:0 14px 20px}}.summary{{color:#526273;margin:7px 0 10px;font-size:14px}}.caveat{{padding:9px 12px;background:#f2f6f9;border-left:4px solid #1683a0;border-radius:3px;color:#425261;font-size:13px;margin:8px 0 12px}}.layout{{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(300px,.8fr);gap:12px}}.panel{{border:1px solid #dce3e9;border-radius:8px;overflow:hidden;background:white}}.panel h2{{font-size:16px;margin:0;padding:10px 12px;border-bottom:1px solid #e7ebef}}#network{{height:600px}}.ranking{{padding:10px 12px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px 5px;text-align:left;border-bottom:1px solid #e8edf1}}th{{color:#596775}}td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}#bridge-chart{{height:310px}}.small{{font-size:12px;color:#687887;padding:0 12px 10px}}@media(max-width:900px){{.layout{{grid-template-columns:1fr}}#network{{height:65vh;min-height:450px}}}}
</style></head><body>
<header><h1>Movie Network Centrality</h1>
<p>Each measure answers a different question: degree counts direct neighbors, closeness measures reach, betweenness identifies nodes on shortest paths between others, and eigenvector centrality rewards connections to well-connected nodes.</p>
<p>Edges use the selected dialogue rule and the same undirected speaker network as the movie interaction page. Repeated evidence changes edge thickness, not centrality. Centrality is recalculated after each selection.</p>
<div class="controls"><label>Movies: <select id="movie"><option value="all">All Movies</option>{movie_options}</select></label>
<label>Edge basis: <select id="rule">{rule_options}</select></label>
<label>Measure: <select id="metric">{metric_options}</select></label>
<label><input id="named-only" type="checkbox"> Named people only</label></div>
<div id="summary" class="summary" aria-live="polite"></div>
<div id="caveat" class="caveat"></div></header>
<main><div class="layout"><section class="panel"><h2 id="network-title">Centrality network</h2><div id="network"></div></section>
<section class="panel"><h2>Top 10 by selected centrality</h2><div class="ranking"><table><thead><tr><th>#</th><th>Character</th><th class="num">Score</th><th class="num">Degree</th></tr></thead><tbody id="ranking-body"></tbody></table></div>
<p class="small">The degree column is the raw number of distinct neighbors. Scores are recalculated for each selected graph.</p></section></div></main>
<script>
const scenarios={_safe_json(scenarios)};
const metricLabels={_safe_json(METRICS)};
const ruleLabels={_safe_json(RULE_LABELS)};
const nodeData=new vis.DataSet([]), edgeData=new vis.DataSet([]);
const network=new vis.Network(document.getElementById('network'),{{nodes:nodeData,edges:edgeData}},{{
  physics:{{enabled:false}},interaction:{{hover:true,dragNodes:false}},
  nodes:{{shape:'dot',font:{{size:12}},scaling:{{min:8,max:38}},borderWidth:1}},
  edges:{{color:{{color:'#a0a8b0',highlight:'#334155'}},scaling:{{min:1,max:8}},smooth:{{type:'continuous'}}}}
}});
const fmt=value=>Number(value).toFixed(4);
function update() {{
  const movie=document.getElementById('movie').value;
  const rule=document.getElementById('rule').value;
  const metric=document.getElementById('metric').value;
  const people=document.getElementById('named-only').checked?'named':'all';
  const scenario=scenarios[rule][movie][people], data=scenario.nodes, edges=scenario.edges, stats=scenario.metrics;
  const scores=data.map(node=>node.scores[metric]);
  const low=Math.min(...scores,0), high=Math.max(...scores,0);
  const positioned=data.map(node=>{{
    const score=node.scores[metric], fraction=high>low?(score-low)/(high-low):0.5;
    const red=Math.round(230-180*fraction), green=Math.round(241-100*fraction), blue=Math.round(248-20*fraction);
    return {{...node,value:Math.max(0.02,score),color:{{background:`rgb(${{red}},${{green}},${{blue}})`,border:'#526273'}},title:node.label+'<br>'+metricLabels[metric]+': '+fmt(score)+'<br>Degree: '+node.degree+'<br>Betweenness: '+fmt(node.scores.betweenness)}};
  }});
  nodeData.clear();nodeData.add(positioned);
  edgeData.clear();edgeData.add(edges.map(edge=>({{...edge,value:edge.weight}})));
  network.fit();
  document.getElementById('network-title').textContent=metricLabels[metric]+' · '+(movie==='all'?'All Movies':movie);
  document.getElementById('summary').textContent=stats.nodes+' nodes · '+stats.edges+' distinct edges · average degree '+stats.average_degree.toFixed(2)+' · '+stats.components+' connected components · '+ruleLabels[rule]+(people==='named'?' · named people only':'');
  let caveat='Scores describe this selected graph; edge weights are not used in shortest paths. ';
  if(metric==='degree') caveat+='Degree centrality is degree divided by N−1; raw degree is shown in the table.';
  if(metric==='closeness') caveat+='Closeness uses NetworkX’s disconnected-graph correction, so unreachable nodes reduce the score.';
  if(metric==='betweenness') caveat+='Betweenness is the normalized share of shortest paths that pass through a character. A high score suggests brokerage in this graph, not control over real communication.';
  if(metric==='eigenvector') caveat+=stats.components>1?'The graph is disconnected; the global eigenvector score is dominated by the component with the largest eigenvalue. Compare scores within connected components cautiously.':'Eigenvector centrality is high when a character is connected to other high-scoring characters.';
  document.getElementById('caveat').textContent=caveat;
  const ranked=[...data].sort((a,b)=>b.scores[metric]-a.scores[metric]||a.label.localeCompare(b.label)).slice(0,10);
  const body=document.getElementById('ranking-body');body.replaceChildren();
  ranked.forEach((node,index)=>{{
    const tr=document.createElement('tr');
    for(const [value,numeric] of [[index+1,false],[node.label,false],[fmt(node.scores[metric]),true],[node.degree,true]]){{
      const td=document.createElement('td');td.textContent=String(value);if(numeric)td.className='num';tr.appendChild(td);
    }}body.appendChild(tr);
  }});
}}
for(const id of ['movie','rule','metric','named-only'])document.getElementById(id).addEventListener('change',update);
update();
</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_page(args.csv, args.output)
    print(f"Wrote movie centrality and bridge explorer to {args.output}")


if __name__ == "__main__":
    main()
