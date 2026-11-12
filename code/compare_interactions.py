"""Compare speaker-turn proximity with same-chapter co-appearance."""

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import networkx as nx

if __package__:
    from .dialogue_csv_reader import read_dialogue_records
else:  # Support `python3 code/compare_interactions.py` from the repository root.
    from dialogue_csv_reader import read_dialogue_records


def _safe_json(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _is_attributed(row: object) -> bool:
    return row.character.casefold() != "unknown" and row.confidence.casefold() not in {
        "low",
        "unattributed",
    }


def build_comparison(csv_path: Path) -> tuple[list[dict], dict[str, Counter]]:
    """Build turn-window graphs and chapter co-appearance counts."""
    records = read_dialogue_records(csv_path)
    characters: set[str] = set()
    chapter_speakers: dict[tuple[str, str], set[str]] = {}
    blocks: list[list[str]] = []
    current_block: list[str] = []
    current_key: tuple[str, str] | None = None

    def finish_block() -> None:
        nonlocal current_block
        if current_block:
            blocks.append(current_block)
            current_block = []

    for row in records:
        movie = row.book or "Unknown movie"
        key = (movie, row.chapter)
        if key != current_key:
            finish_block()
            current_key = key
        if not _is_attributed(row):
            # Unattributed turns break the proximity window instead of allowing
            # a false connection to jump across missing speaker information.
            finish_block()
            continue
        characters.add(row.character)
        chapter_speakers.setdefault(key, set()).add(row.character)
        current_block.append(row.character)
    finish_block()

    turn_windows: dict[str, Counter[tuple[str, str]]] = {
        str(window): Counter() for window in (1, 2, 3, 10)
    }
    for block in blocks:
        for index, character in enumerate(block):
            for window in (1, 2, 3, 10):
                for other in block[index + 1 : index + window + 1]:
                    if character != other:
                        turn_windows[str(window)][tuple(sorted((character, other)))] += 1

    chapter_edges: Counter[tuple[str, str]] = Counter()
    for speakers in chapter_speakers.values():
        chapter_edges.update(combinations(sorted(speakers), 2))

    nodes = []
    for character in sorted(characters):
        nodes.append({"id": character, "label": character, "title": character})

    edge_sets = {
        name: [
            {"from": first, "to": second, "weight": weight}
            for (first, second), weight in sorted(counts.items())
        ]
        for name, counts in turn_windows.items()
    }
    edge_sets["chapter"] = [
        {"from": first, "to": second, "weight": weight}
        for (first, second), weight in sorted(chapter_edges.items())
    ]
    return nodes, edge_sets


def write_comparison_page(csv_path: Path, output_path: Path) -> None:
    nodes, edge_sets = build_comparison(csv_path)
    union_graph = nx.Graph()
    union_graph.add_nodes_from(node["id"] for node in nodes)
    for edges in edge_sets.values():
        union_graph.add_edges_from((edge["from"], edge["to"]) for edge in edges)
    positions = nx.spring_layout(union_graph, seed=42, weight=None, iterations=120)
    for node in nodes:
        point = positions[node["id"]]
        node["x"] = float(point[0]) * 1000
        node["y"] = float(point[1]) * 1000

    payload_nodes = _safe_json(nodes)
    payload_edges = _safe_json(edge_sets)
    title = "All Movies: Interaction Definition Comparison"
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
*{{box-sizing:border-box}}body{{font:15px system-ui,sans-serif;color:#222;margin:0}}header{{padding:14px 20px}}header h1{{font-size:21px;margin:0 0 6px}}header p{{margin:5px 0;color:#555}}main{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:0 12px 12px}}section{{min-width:0;border:1px solid #ddd;border-radius:8px;overflow:hidden}}.panel-head{{padding:12px 14px}}.panel-head h2{{font-size:17px;margin:0 0 5px}}.panel-head p{{font-size:13px;color:#555;margin:5px 0}}label{{font-size:14px}}select{{font:inherit;padding:5px;margin-left:5px}}.summary{{font-size:13px;color:#333;font-weight:600}}.network{{height:calc(100vh - 280px);min-height:450px;border-top:1px solid #eee}}.legend{{font-size:12px;color:#666}}.ranking{{padding:10px 14px 14px;border-top:1px solid #eee}}.ranking h3{{font-size:15px;margin:0 0 7px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:5px 7px;border-bottom:1px solid #eee}}th{{color:#555}}td:last-child,th:last-child{{text-align:right}}@media(max-width:900px){{main{{grid-template-columns:1fr}}.network{{height:65vh}}}}
</style></head><body>
<header><h1>{title}</h1>
<p>Each panel uses the same characters and the same node positions. Compare what counts as an edge under two definitions of interaction.</p>
<p>Both are undirected. Degree is the number of distinct neighbors; thicker edges indicate more repeated evidence under that panel’s rule. Unknown or low-confidence dialogue breaks the turn sequence.</p></header>
<main>
<section><div class="panel-head"><h2>Nearby dialogue turns</h2>
<label>Connect to the next <select id="window"><option value="1">1 turn</option><option value="2">2 turns</option><option value="3">3 turns</option><option value="10">10 turns</option></select></label>
<p>An edge joins two different speakers when their attributed turns are within the selected number of following turns in the same chapter. Edge weight counts how often this happens.</p>
<div class="summary" id="turn-summary"></div><p class="legend">This captures nearby speaker changes; it cannot identify the intended addressee.</p></div><div id="turn-network" class="network"></div><div class="ranking"><h3>Top 10 characters by degree</h3><table><thead><tr><th>Rank</th><th>Character</th><th>Degree</th></tr></thead><tbody id="turn-ranking"></tbody></table></div></section>
<section><div class="panel-head"><h2>Chapter co-appearance</h2>
<p>An edge joins every pair of distinct characters who have at least one attributed dialogue row in the same movie chapter. Edge weight counts the number of chapters they share.</p>
<div class="summary" id="chapter-summary"></div><p class="legend">This captures shared chapter participation, not proof that characters directly spoke to each other or were present for every line.</p></div><div id="chapter-network" class="network"></div><div class="ranking"><h3>Top 10 characters by degree</h3><table><thead><tr><th>Rank</th><th>Character</th><th>Degree</th></tr></thead><tbody id="chapter-ranking"></tbody></table></div></section>
</main><script>
const baseNodes={payload_nodes};
const edgeSets={payload_edges};
const palette={{background:'#8ecae6',border:'#219ebc'}};
const graphViews={{}};
function updateRanking(tableId, degree) {{
  const ranking=baseNodes.map(node=>({{name:node.label,degree:degree.get(node.id)}}))
    .sort((a,b)=>b.degree-a.degree||a.name.localeCompare(b.name)).slice(0,10);
  const body=document.getElementById(tableId);
  body.replaceChildren();
  ranking.forEach((entry,index)=>{{
    const row=document.createElement('tr');
    for(const value of [index+1,entry.name,entry.degree]){{
      const cell=document.createElement('td');cell.textContent=String(value);row.appendChild(cell);
    }}
    body.appendChild(row);
  }});
}}
function drawGraph(containerId, summaryId, networkName, edges, heading) {{
  let view=graphViews[containerId];
  if(!view){{
    const nodeData=new vis.DataSet([]), edgeData=new vis.DataSet([]);
    const network=new vis.Network(document.getElementById(containerId),{{nodes:nodeData,edges:edgeData}},{{
      physics:{{enabled:false}},interaction:{{hover:true,dragNodes:false}},
      nodes:{{shape:'dot',font:{{size:12}},scaling:{{min:8,max:30}}}},
      edges:{{color:{{color:'#8a8f98',highlight:'#d55e00'}},scaling:{{min:1,max:7}},smooth:{{type:'continuous'}}}}
    }});
    view=graphViews[containerId]={{nodeData,edgeData,network,fitted:false}};
  }}
  const degree=new Map(baseNodes.map(node=>[node.id,0]));
  for(const edge of edges){{degree.set(edge.from,degree.get(edge.from)+1);degree.set(edge.to,degree.get(edge.to)+1);}}
  updateRanking(networkName==='chapter'?'chapter-ranking':'turn-ranking',degree);
  view.nodeData.clear();
  view.nodeData.add(baseNodes.map(node=>({{...node,value:Math.max(1,degree.get(node.id)),color:palette,title:node.label+' — degree '+degree.get(node.id)}})));
  view.edgeData.clear();
  view.edgeData.add(edges.map(edge=>({{from:edge.from,to:edge.to,value:edge.weight,title:edge.weight+' '+(networkName==='chapter'?'shared chapters':'nearby turn occurrences')}})));
  if(!view.fitted){{view.network.fit();view.fitted=true;}}
  const nodeCount=baseNodes.length;
  const averageDegree=nodeCount?2*edges.length/nodeCount:0;
  document.getElementById(summaryId).textContent=heading+': '+nodeCount+' nodes · '+edges.length+' distinct edges · average degree '+averageDegree.toFixed(2);
}}
function refreshTurns() {{
  const windowSize=document.getElementById('window').value;
  const edges=edgeSets[windowSize];
  drawGraph('turn-network','turn-summary','turns',edges,'Next '+windowSize+' turn'+(windowSize==='1'?'':'s'));
}}
document.getElementById('window').addEventListener('change',refreshTurns);
refreshTurns();
drawGraph('chapter-network','chapter-summary','chapter',edgeSets.chapter,'All movies');
</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=Path("movie_data/all_movies.csv"))
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("graphs/movies/analysis/interaction_definitions_comparison.html"),
    )
    args = parser.parse_args()
    write_comparison_page(args.csv, args.output)
    print(f"Wrote interaction comparison to {args.output}")


if __name__ == "__main__":
    main()
