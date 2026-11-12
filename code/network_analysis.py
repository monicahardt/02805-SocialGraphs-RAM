"""Create the movie dialogue degree-distribution view."""

import argparse
import html
import json
import math
from collections import Counter
from pathlib import Path

import networkx as nx

if __package__:
    from .dialogue_csv_reader import read_dialogue_records
else:  # Support `python3 code/network_analysis.py` from the repository root.
    from dialogue_csv_reader import read_dialogue_records


def _safe_json(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def build_interaction_graph(csv_path: Path) -> tuple[nx.Graph, Counter[tuple[str, str]]]:
    """Build a simple undirected speaker-transition graph, retaining isolates."""
    graph = nx.Graph()
    transition_counts: Counter[tuple[str, str]] = Counter()
    previous_character: str | None = None
    previous_movie: str | None = None
    previous_chapter: str | None = None

    for row in read_dialogue_records(csv_path):
        movie = row.book or "Unknown movie"
        if row.character.casefold() == "unknown" or row.confidence.casefold() in {
            "low",
            "unattributed",
        }:
            previous_character = None
            previous_movie = movie
            previous_chapter = row.chapter
            continue

        character = row.character
        graph.add_node(character)
        if (
            movie == previous_movie
            and row.chapter == previous_chapter
            and previous_character is not None
            and previous_character != character
        ):
            edge = tuple(sorted((previous_character, character)))
            graph.add_edge(*edge)
            transition_counts[edge] += 1
        previous_character = character
        previous_movie = movie
        previous_chapter = row.chapter

    return graph, transition_counts


def degree_distribution_html(graph: nx.Graph, output_path: Path, title: str) -> None:
    """Write linear and log-log raw/binned degree distributions as an HTML plot."""
    degrees = [degree for _, degree in graph.degree()]
    counts = Counter(degrees)
    node_count = len(degrees)
    max_u = max((degree + 1 for degree in degrees), default=1)

    max_degree = max(counts, default=0)
    raw_x = list(range(max_degree + 1))
    raw_y = [counts.get(k, 0) / node_count for k in raw_x] if node_count else []
    character_points: list[dict[str, object]] = []
    degree_slots: Counter[int] = Counter()
    for character, degree in sorted(graph.degree, key=lambda item: (item[1], item[0].casefold())):
        degree_slots[degree] += 1
        character_points.append({"name": character, "degree": degree, "slot": degree_slots[degree]})
    # Use width-one bins where degree is dense, then integer-aligned doubling bins.
    bins: list[tuple[int, int]] = [(u, u + 1) for u in range(1, min(8, max_u + 1))]
    lower = 8
    while lower <= max_u:
        upper = lower * 2
        bins.append((lower, upper))
        lower = upper

    binned_x: list[float] = []
    binned_y: list[float] = []
    populated_bins: list[tuple[int, int]] = []
    for lower, upper in bins:
        in_bin = sum(count for degree, count in counts.items() if lower <= degree + 1 < upper)
        if in_bin:
            binned_x.append(math.sqrt(lower * (upper - 1)))
            binned_y.append(in_bin / node_count / (upper - lower))
            populated_bins.append((lower, upper))

    density = nx.density(graph) if node_count > 1 else 0.0

    stats = {
        "nodes": node_count,
        "edges": graph.number_of_edges(),
        "isolates": sum(1 for _, degree in graph.degree() if degree == 0),
        "average_degree": (2 * graph.number_of_edges() / node_count) if node_count else 0,
        "density": density,
    }
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — Degree Distribution</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{font:15px system-ui,sans-serif;color:#222;margin:0}}header{{padding:18px 24px}}main{{padding:0 20px}}.controls{{display:flex;gap:20px;align-items:center;flex-wrap:wrap;margin:12px 0}}select{{font:inherit;padding:5px}}#plots{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.plot{{height:540px}}#characters{{height:620px}}.stats{{color:#555}}@media(max-width:800px){{#plots{{grid-template-columns:1fr}}.plot{{height:420px}}}}</style></head>
<body><header><h1>{html.escape(title)}: degree distribution</h1>
<p>Degree is the number of distinct character neighbors in the undirected interaction graph. Isolates are included. The horizontal axis is <em>k + 1</em> so degree zero remains visible on a logarithmic axis.</p>
<p class="stats">{stats['nodes']} nodes · {stats['edges']} distinct edges · {stats['isolates']} isolates · average degree {stats['average_degree']:.2f} · density {stats['density']:.4f}</p>
<p><b>P(k)</b> is the fraction of characters with degree k. The optional power-law reference uses P(k) ∝ k⁻γ for k ≥ 1, excludes isolates, and is scaled to the observed positive-degree mass; it is an adjustable reference shape, not a fit. The binned view uses width-one bins for k + 1 = 1…7, then doubling integer bins; each count is divided by the bin width.</p>
<div class="controls"><label>View: <select id="view"><option value="distribution">Degree distribution</option><option value="characters">Characters by degree</option></select></label><label id="series-control">Distribution: <select id="series"><option value="raw">Raw P(k)</option><option value="binned">Binned density</option></select></label><label id="power-control"><input id="show-power" type="checkbox"> Overlay power law</label><label id="gamma-control">γ: <input id="gamma" type="number" min="1.1" max="5" step="0.1" value="3" style="width:68px"></label><span id="view-note">Compare linear axes at left with log–log axes at right.</span></div></header>
<main><div id="plots"><div id="linear" class="plot"></div><div id="loglog" class="plot"></div></div><div id="characters" style="display:none"></div></main>
<script>
const raw = {{x:{_safe_json([k + 1 for k in raw_x])},y:{_safe_json(raw_y)}}};
const binned = {{x:{_safe_json(binned_x)},y:{_safe_json(binned_y)}}};
const degreeValues={_safe_json(raw_x)};
const bins={_safe_json(populated_bins)};
const characterPoints={_safe_json(character_points)};
function powerLawData(gamma, mode) {{
  const maxDegree=degreeValues.length-1;
  if(maxDegree<1) return {{x:[],y:[]}};
  const positiveMass=1-(raw.y[0]||0);
  const weights=degreeValues.slice(1).map(k=>Math.pow(k,-gamma));
  const total=weights.reduce((sum,value)=>sum+value,0);
  const probability=degreeValues.map(k=>k===0?0:(positiveMass*Math.pow(k,-gamma)/total));
  if(mode==='raw') return {{x:degreeValues.slice(1).map(k=>k+1),y:probability.slice(1)}};
  const x=[],y=[];
  for(const [lower,upper] of bins) {{
    const mass=degreeValues.reduce((sum,k)=>sum+(k>0 && lower<=k+1 && k+1<upper?probability[k]:0),0);
    if(mass>0) {{x.push(Math.sqrt(lower*(upper-1)));y.push(mass/(upper-lower));}}
  }}
  return {{x,y}};
}}
function draw(mode) {{
  const data = mode === 'raw' ? raw : binned;
  const name = mode === 'raw' ? 'Raw P(k)' : 'Binned density';
  const traces=[{{x:data.x,y:data.y,type:'scatter',mode:'lines+markers',name:name,marker:{{size:8,color:'#176b87'}},line:{{color:'#176b87'}},hovertemplate:'k + 1 = %{{x}}<br>P(k) = %{{y:.4f}}<extra></extra>'}}];
  const gamma=Number(document.getElementById('gamma').value)||3;
  if(document.getElementById('show-power').checked) {{
    const power=powerLawData(gamma,mode);
    traces.push({{x:power.x,y:power.y,type:'scatter',mode:'lines',name:'Power-law reference, γ = '+gamma.toFixed(1),line:{{color:'#7b3294',width:2.5,dash:'dot'}},hovertemplate:'k + 1 = %{{x}}<br>reference P(k) = %{{y:.4f}}<extra></extra>'}});
  }}
  const legendVisible=document.getElementById('show-power').checked;
  const linearTraces=traces.filter(trace=>!trace.name.startsWith('Power-law reference'));
  Plotly.react('linear',linearTraces,{{title:name + ' · linear axes',xaxis:{{title:'k + 1'}},yaxis:{{title:'Fraction of nodes P(k)'}},margin:{{t:50,r:20,b:55,l:65}},showlegend:false}},{{responsive:true}});
  Plotly.react('loglog',traces,{{title:name + ' · log–log axes',xaxis:{{title:'k + 1',type:'log'}},yaxis:{{title:'Fraction of nodes P(k)',type:'log'}},margin:{{t:50,r:20,b:55,l:65}},showlegend:legendVisible,legend:{{orientation:'h',y:1.12}}}},{{responsive:true}});
}}
document.getElementById('series').addEventListener('change',event=>draw(event.target.value));
document.getElementById('show-power').addEventListener('change',()=>draw(document.getElementById('series').value));
document.getElementById('gamma').addEventListener('input',()=>draw(document.getElementById('series').value));
draw('raw');
function drawCharacters() {{
  const trace={{x:characterPoints.map(point=>point.degree),y:characterPoints.map(point=>point.slot),customdata:characterPoints.map(point=>point.name),type:'scatter',mode:'markers',marker:{{size:10,color:'#176b87',opacity:0.75}},hovertemplate:'%{{customdata}}<br>degree = %{{x}}<extra></extra>'}};
  Plotly.react('characters',[trace],{{title:'One point per character',xaxis:{{title:'Degree k',rangemode:'tozero',dtick:1}},yaxis:{{title:'Characters sharing this degree',showticklabels:false,gridcolor:'#eee'}},margin:{{t:55,r:25,b:65,l:70}}}},{{responsive:true}});
}}
document.getElementById('view').addEventListener('change',event=>{{
  const characterView=event.target.value==='characters';
  document.getElementById('plots').style.display=characterView?'none':'grid';
  document.getElementById('characters').style.display=characterView?'block':'none';
  document.getElementById('series-control').style.display=characterView?'none':'inline-flex';
  document.getElementById('power-control').style.display=characterView?'none':'inline-flex';
  document.getElementById('gamma-control').style.display=characterView?'none':'inline-flex';
  document.getElementById('view-note').textContent=characterView?'Each dot is one character; horizontal position is that character’s degree. Hover a dot to see the name.':'Compare linear axes at left with log–log axes at right.';
  if(characterView) drawCharacters(); else draw(document.getElementById('series').value);
}});
</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def community_graph_html(
    graph: nx.Graph,
    transition_counts: Counter[tuple[str, str]],
    output_path: Path,
    title: str,
) -> None:
    """Write a fixed-layout network with Louvain community colors and controls."""
    if graph.number_of_nodes() == 0:
        communities: list[set[str]] = []
    elif graph.number_of_edges() == 0:
        communities = [{node} for node in graph.nodes]
    else:
        communities = list(nx.community.louvain_communities(graph, weight=None, seed=42))
    communities.sort(key=lambda group: (-len(group), min(group)))
    community_of = {node: index for index, group in enumerate(communities) for node in group}
    palette = [
        "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9",
        "#F0E442", "#332288", "#88CCEE", "#AA4499", "#44AA99", "#999933",
    ]
    positions = nx.spring_layout(graph, seed=42, weight=None, iterations=100)
    nodes = [
        {
            "id": node,
            "label": node,
            "community": community_of[node],
            "degree": graph.degree[node],
            "x": float(positions[node][0]) * 1000,
            "y": float(positions[node][1]) * 1000,
            "title": f"{node} — degree {graph.degree[node]} — community {community_of[node] + 1}",
        }
        for node in sorted(graph.nodes)
    ]
    edges = [
        {
            "from": first,
            "to": second,
            "value": max(1, transition_counts[tuple(sorted((first, second)))]),
            "title": f"{transition_counts[tuple(sorted((first, second)))]} consecutive speaker transitions",
        }
        for first, second in sorted(graph.edges)
    ]
    community_sizes = [len(group) for group in communities]
    # Give each group a stable center on a large circle, then distribute its
    # members around a roomy local circle so individual nodes remain legible.
    circle_positions = {}
    for group, members in enumerate(communities):
        angle = 2 * math.pi * group / max(1, len(communities))
        center_radius = 760
        center_x = center_radius * math.cos(angle)
        center_y = center_radius * math.sin(angle)
        member_radius = max(115, 28 * math.sqrt(len(members)))
        ordered_members = sorted(members, key=lambda node: (-graph.degree[node], node.casefold()))
        for member_index, node in enumerate(ordered_members):
            member_angle = 2 * math.pi * member_index / max(1, len(ordered_members))
            circle_positions[node] = {
                "x": center_x + member_radius * math.cos(member_angle),
                "y": center_y + member_radius * math.sin(member_angle),
            }
    modularity = (
        nx.community.modularity(graph, communities, weight=None)
        if graph.number_of_edges() and communities
        else 0
    )
    nodes_json = _safe_json(nodes)
    edges_json = _safe_json(edges)
    circles_json = _safe_json(circle_positions)
    colors_json = _safe_json(palette)
    community_labels = []
    for index, group in enumerate(communities):
        prominent = sorted(group, key=lambda node: (-graph.degree[node], node.casefold()))[:3]
        community_labels.append({
            "id": index,
            "label": f"Community {index + 1} · {len(group)} characters · most connected: {', '.join(prominent)}",
            "color": palette[index % len(palette)],
        })
    legend_html = "".join(
        f'<div class="community-key"><span style="background:{entry["color"]}"></span><b>Community {entry["id"] + 1}</b> · {len(communities[entry["id"]])} characters<br><small>Most connected: {html.escape(", ".join(sorted(communities[entry["id"]], key=lambda node: (-graph.degree[node], node.casefold()))[:3]))}</small></div>'
        for entry in community_labels
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — Communities</title><script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>body{{font:15px system-ui,sans-serif;color:#222;margin:0}}header{{padding:14px 20px}}header p{{margin:5px 0}}#network{{height:calc(100vh - 270px);min-height:430px;border-top:1px solid #ddd}}.controls{{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-top:10px}}select{{font:inherit;padding:5px}}.note{{color:#555;font-size:13px}}#community-key{{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}}.community-key{{font-size:12px;background:#f5f6f7;border-radius:6px;padding:7px 9px;max-width:310px}}.community-key span{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}</style></head>
<body><header><strong>{html.escape(title)}: interaction communities</strong>
<p>{graph.number_of_nodes()} nodes · {graph.number_of_edges()} distinct edges · average degree {(2 * graph.number_of_edges() / graph.number_of_nodes()) if graph.number_of_nodes() else 0:.2f} · {len(communities)} Louvain communities · modularity Q = {modularity:.3f}</p>
<p class="note">Edges join different speakers with consecutive attributed turns in the same chapter. Louvain uses the unweighted simple graph; transition counts only control line thickness. The names below summarize high-degree members of each detected group; they are descriptive labels, not known factions. Positions are layout, not data.</p>
<div id="community-key">{legend_html}</div>
<div class="controls"><label>Layout: <select id="layout"><option value="spring">Force-directed</option><option value="community">Communities in a circle</option></select></label><label><input id="color-toggle" type="checkbox" checked> Color by community</label></div></header>
<div id="network"></div><script>
const originalNodes={nodes_json}, originalEdges={edges_json}, circle={circles_json}, palette={colors_json};
const nodes=new vis.DataSet(originalNodes.map(node=>({{...node,color:{{background:palette[node.community%palette.length],border:'#333'}}}})));
const edges=new vis.DataSet(originalEdges);
const network=new vis.Network(document.getElementById('network'),{{nodes,edges}},{{
  physics:{{enabled:false}}, interaction:{{hover:true,dragNodes:false}},
  nodes:{{shape:'dot',scaling:{{min:9,max:34}},font:{{size:13}}}},
  edges:{{color:{{color:'#9aa0a6',highlight:'#333'}},scaling:{{min:1,max:7}},smooth:{{type:'continuous'}}}}
}});
function setLayout(mode) {{
  const positions=mode==='community'?circle:Object.fromEntries(originalNodes.map(node=>[node.id,{{x:node.x,y:node.y}}]));
  for (const [id,point] of Object.entries(positions)) network.moveNode(id,point.x,point.y);
  network.fit({{animation:{{duration:350,easingFunction:'easeInOutQuad'}}}});
}}
document.getElementById('layout').addEventListener('change',event=>setLayout(event.target.value));
document.getElementById('color-toggle').addEventListener('change',event=>{{
  nodes.update(originalNodes.map(node=>({{id:node.id,color:{{background:event.target.checked?palette[node.community%palette.length]:'#8ecae6',border:'#333'}}}})));
}});
network.fit();
</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=Path("movie_data/all_movies.csv"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--communities",
        action="store_true",
        help="Also generate the optional community graph.",
    )
    args = parser.parse_args()

    input_path = args.csv
    output_dir = args.output_dir
    if output_dir is None:
        kind = "books" if "book_data" in input_path.parts else "movies"
        output_dir = Path("graphs") / kind / "analysis"
    graph, transition_counts = build_interaction_graph(input_path)
    if "book_data" in input_path.parts:
        title = "All Books" if input_path.stem == "all_books" else f"Book Dialogue Network · {input_path.stem}"
    elif input_path.stem == "all_movies":
        title = "All Movies"
    else:
        title = f"Movie Dialogue Network · {input_path.stem}"
    degree_distribution_html(graph, output_dir / "degree_distribution.html", title)
    if args.communities:
        community_graph_html(graph, transition_counts, output_dir / "interaction_communities.html", title)
        print(f"Wrote degree distribution and community graph to {output_dir}")
    else:
        print(f"Wrote degree distribution to {output_dir}")


if __name__ == "__main__":
    main()
