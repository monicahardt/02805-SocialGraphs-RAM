"""Build interactive clustering-coefficient views for movie dialogue networks."""

import html
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx

if __package__:
    from .dialogue_csv_reader import read_dialogue_records
else:  # Support `python3 code/clustering_analysis.py` from the repository root.
    from dialogue_csv_reader import read_dialogue_records


RULES = ("1", "2", "5", "chapter")
RULE_LABELS = {
    "1": "Next 1 dialogue line",
    "2": "Next 2 dialogue lines",
    "5": "Next 5 dialogue lines",
    "chapter": "Whole chapter co-appearance",
}


def _safe_json(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _make_graph(csv_path: Path, movie_filter: str, rule: str) -> tuple[nx.Graph, Counter]:
    records = read_dialogue_records(csv_path)
    graph = nx.Graph()
    chapters: dict[tuple[str, str], set[str]] = defaultdict(set)
    blocks: list[tuple[tuple[str, str], list[str]]] = []
    current_key: tuple[str, str] | None = None
    current_turns: list[str] = []

    def finish_block() -> None:
        nonlocal current_turns
        if current_key is not None and current_turns:
            blocks.append((current_key, current_turns))
            current_turns = []

    for row in records:
        movie = row.book or "Unknown movie"
        if movie_filter != "all" and movie != movie_filter:
            finish_block()
            current_key = None
            continue
        key = (movie, row.chapter)
        if key != current_key:
            finish_block()
            current_key = key
        if row.character.casefold() == "unknown" or row.confidence.casefold() in {
            "low",
            "unattributed",
        }:
            finish_block()
            continue
        graph.add_node(row.character)
        chapters[key].add(row.character)
        current_turns.append(row.character)
    finish_block()

    evidence: Counter[tuple[str, str]] = Counter()
    if rule == "chapter":
        for speakers in chapters.values():
            evidence.update(combinations(sorted(speakers), 2))
    else:
        window = int(rule)
        for _key, turns in blocks:
            for index, character in enumerate(turns):
                for other in turns[index + 1 : index + window + 1]:
                    if character != other:
                        evidence[tuple(sorted((character, other)))] += 1

    graph.add_edges_from(evidence)
    return graph, evidence


def _scenario_data(graph: nx.Graph, evidence: Counter, rule: str) -> dict:
    clustering = nx.clustering(graph, weight=None)
    triangles = nx.triangles(graph)
    node_data = [
        {
            "id": name,
            "label": name,
            "degree": graph.degree[name],
            "coefficient": clustering[name],
            "triangles": triangles[name],
            "title": (
                f"{name} — degree {graph.degree[name]} — "
                f"local clustering Cᵢ = {clustering[name]:.3f} — "
                f"{triangles[name]} triangles through this character"
            ),
        }
        for name in sorted(graph.nodes, key=str.casefold)
    ]
    edge_data = [
        {
            "from": first,
            "to": second,
            "weight": count,
            "title": f"{count} shared chapters" if rule == "chapter" else f"{count} nearby turn-pair occurrences",
        }
        for (first, second), count in sorted(evidence.items())
    ]
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    density = nx.density(graph) if node_count > 1 else 0.0
    average_degree = 2 * edge_count / node_count if node_count else 0.0
    average_local = nx.average_clustering(graph, weight=None) if node_count else 0.0
    transitivity = nx.transitivity(graph) if edge_count else 0.0
    triangle_count = sum(triangles.values()) // 3
    return {
        "nodes": node_data,
        "edges": edge_data,
        "metrics": {
            "nodes": node_count,
            "edges": edge_count,
            "density": density,
            "average_degree": average_degree,
            "average_local": average_local,
            "er_benchmark": density,
            "transitivity": transitivity,
            "triangles": triangle_count,
        },
    }


def write_page(csv_path: Path, output_path: Path, collection_label: str = "Movie") -> None:
    records = read_dialogue_records(csv_path)
    movies = sorted({row.book for row in records if row.book})
    movie_options = "".join(
        f'<option value="{html.escape(movie, quote=True)}">{html.escape(movie)}</option>'
        for movie in movies
    )
    scenarios = {
        rule: {
            movie: _scenario_data(*_make_graph(csv_path, movie, rule), rule)
            for movie in ["all", *movies]
        }
        for rule in RULES
    }
    collection_plural = "books" if collection_label.casefold() == "book" else "movies"
    movie_names = {"all": f"All {collection_plural}", **{movie: movie for movie in movies}}
    title = f"{collection_label} Dialogue Network: Clustering"
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{{box-sizing:border-box}}body{{font:15px system-ui,sans-serif;margin:0;color:#222}}header{{padding:16px 20px}}h1{{font-size:22px;margin:0 0 6px}}header p{{margin:5px 0;color:#555;max-width:1100px}}.controls{{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;align-items:center}}select{{font:inherit;padding:6px}}main{{padding:0 14px 20px}}.metrics{{display:grid;grid-template-columns:repeat(6,minmax(110px,1fr));gap:9px;margin:4px 0 12px}}.metric{{background:#f4f6f8;border-radius:7px;padding:10px 12px}}.metric b{{display:block;font-size:20px}}.metric span{{font-size:12px;color:#555}}.benchmark{{border-left:4px solid #d55e00;padding:7px 11px;background:#fff8f3;margin:8px 0 12px}}.visuals{{display:grid;grid-template-columns:1.15fr 1fr;gap:12px}}.panel{{border:1px solid #ddd;border-radius:7px;overflow:hidden}}.panel h2{{font-size:16px;margin:0;padding:10px 12px;border-bottom:1px solid #eee}}#network,#scatter{{height:570px}}.legend{{display:flex;align-items:center;gap:8px;font-size:12px;color:#555;padding:7px 12px}}.gradient{{height:11px;width:170px;background:linear-gradient(90deg,#2166ac,#f7f7f7,#b2182b);border:1px solid #bbb}}.note{{font-size:13px;color:#555;margin:10px 2px}}@media(max-width:900px){{.visuals{{grid-template-columns:1fr}}.metrics{{grid-template-columns:repeat(2,minmax(125px,1fr))}}#network,#scatter{{height:480px}}}}
</style></head><body>
<header><h1>{title}</h1>
<p>Local clustering <b>Cᵢ</b> is the fraction of a character’s neighbors who are connected to each other. A value of 1 means all neighbors are linked; 0 means no links among neighbors. Characters with fewer than two neighbors are assigned 0.</p>
<p>Use the controls to compare edge definitions and {collection_plural}. The Erdős–Rényi benchmark uses <b>C ≈ p</b>, where <b>p</b> is network density and matches the random graph’s mean degree ⟨k⟩. The benchmark is an expected value, not a simulated network or confidence interval.</p>
<div class="controls"><label>{collection_label}: <select id="movie"><option value="all">All {collection_plural}</option>{movie_options}</select></label>
<label>Edge basis: <select id="rule"><option value="1">Next 1 dialogue line</option><option value="2">Next 2 dialogue lines</option><option value="5">Next 5 dialogue lines</option><option value="chapter">Whole chapter co-appearance</option></select></label></div></header>
<main><div class="metrics">
<div class="metric"><b id="avg-local">—</b><span>Average local clustering C</span></div>
<div class="metric"><b id="er-benchmark">—</b><span>ER benchmark ≈ density p</span></div>
<div class="metric"><b id="transitivity">—</b><span>Global transitivity</span></div>
<div class="metric"><b id="triangle-count">—</b><span>Network triangles</span></div>
<div class="metric"><b id="average-degree">—</b><span>Mean degree ⟨k⟩</span></div>
<div class="metric"><b id="network-size">—</b><span>Nodes · distinct edges</span></div></div>
<div class="benchmark" id="comparison"></div>
<div class="visuals"><section class="panel"><h2>Interaction network — node color is local Cᵢ; node size is degree</h2><div class="legend"><span>Lower Cᵢ</span><div class="gradient"></div><span>Higher Cᵢ</span></div><div id="network"></div></section>
<section class="panel"><h2>Local clustering versus degree — hover for character names</h2><div id="scatter"></div></section></div>
<p class="note">The graph is a simple, undirected network for measuring clustering: repeated evidence affects edge thickness, not degree or Cᵢ. Chapter co-appearance will create triangles when three characters speak in the same chapter, so interpret that mode as shared chapter participation rather than direct conversation.</p>
</main><script>
const scenarios={_safe_json(scenarios)};
const movieNames={_safe_json(movie_names)};
const networkNodes=new vis.DataSet([]), networkEdges=new vis.DataSet([]);
const network=new vis.Network(document.getElementById('network'),{{nodes:networkNodes,edges:networkEdges}},{{
  physics:{{enabled:false}},interaction:{{hover:true,dragNodes:false}},
  nodes:{{shape:'dot',font:{{size:12}},scaling:{{min:8,max:34}},borderWidth:1}},
  edges:{{color:{{color:'#9aa0a6',highlight:'#333'}},scaling:{{min:1,max:7}},smooth:{{type:'continuous'}}}}
}});
function colorFor(value) {{
  const low=[33,102,172], middle=[247,247,247], high=[178,24,43];
  const a=value<0.5?low:middle, b=value<0.5?middle:high, fraction=value<0.5?value*2:(value-0.5)*2;
  const rgb=a.map((component,index)=>Math.round(component+(b[index]-component)*fraction));
  return '#'+rgb.map(component=>component.toString(16).padStart(2,'0')).join('');
}}
function update() {{
  const movie=document.getElementById('movie').value, rule=document.getElementById('rule').value;
  const scenario=scenarios[rule][movie], metrics=scenario.metrics;
  const ordered=[...scenario.nodes].sort((a,b)=>b.degree-a.degree||a.label.localeCompare(b.label));
  const golden=Math.PI*(3-Math.sqrt(5));
  const positioned=ordered.map((node,index)=>{{const radius=index===0?0:72*Math.sqrt(index),angle=index*golden;return {{...node,x:radius*Math.cos(angle),y:radius*Math.sin(angle),value:Math.max(1,node.degree),color:{{background:colorFor(node.coefficient),border:'#555'}},title:node.label+' — degree '+node.degree+' — Cᵢ = '+node.coefficient.toFixed(3)+' — '+node.triangles+' neighbor triangles'}};}});
  networkNodes.clear();networkNodes.add(positioned);
  networkEdges.clear();networkEdges.add(scenario.edges.map(edge=>({{from:edge.from,to:edge.to,value:edge.weight,title:edge.weight+(rule==='chapter'?' shared chapters':' nearby turn-pair occurrences')}})));
  network.fit();
  document.getElementById('avg-local').textContent=metrics.average_local.toFixed(3);
  document.getElementById('er-benchmark').textContent=metrics.er_benchmark.toFixed(3);
  document.getElementById('transitivity').textContent=metrics.transitivity.toFixed(3);
  document.getElementById('triangle-count').textContent=metrics.triangles.toLocaleString();
  document.getElementById('average-degree').textContent=metrics.average_degree.toFixed(2);
  document.getElementById('network-size').textContent=metrics.nodes+' · '+metrics.edges;
  const difference=metrics.average_local-metrics.er_benchmark;
  const ratio=metrics.er_benchmark>0?metrics.average_local/metrics.er_benchmark:null;
  document.getElementById('comparison').textContent='Observed average local C is '+(difference>=0?'above':'below')+' the ER benchmark by '+Math.abs(difference).toFixed(3)+(ratio===null?'':(' ('+ratio.toFixed(1)+'× the benchmark)'))+'. Current selection: '+movieNames[movie]+' · '+{_safe_json(RULE_LABELS)}[rule]+'.';
  const points=scenario.nodes;
  const trace={{x:points.map(node=>node.degree),y:points.map(node=>node.coefficient),customdata:points.map(node=>node.label),mode:'markers',type:'scatter',name:'Characters',marker:{{size:9,color:points.map(node=>node.coefficient),cmin:0,cmax:1,colorscale:[[0,'#2166ac'],[0.5,'#f7f7f7'],[1,'#b2182b']],line:{{color:'#555',width:0.5}},showscale:false}},hovertemplate:'%{{customdata}}<br>degree = %{{x}}<br>local Cᵢ = %{{y:.3f}}<extra></extra>'}};
  Plotly.react('scatter',[trace],{{xaxis:{{title:'Degree k',rangemode:'tozero'}},yaxis:{{title:'Local clustering coefficient Cᵢ',range:[-0.03,1.03]}},margin:{{t:18,r:20,b:60,l:65}},showlegend:false}},{{responsive:true}});
}}
document.getElementById('movie').addEventListener('change',update);
document.getElementById('rule').addEventListener('change',update);
update();
</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=Path("movie_data/all_movies.csv"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    is_book = "book_data" in args.csv.parts
    collection_label = "Book" if is_book else "Movie"
    output_path = args.output or Path("graphs") / ("books" if is_book else "movies") / "analysis" / "clustering_coefficients.html"
    write_page(args.csv, output_path, collection_label)
    print(f"Wrote clustering-coefficient visualization to {output_path}")


if __name__ == "__main__":
    main()
