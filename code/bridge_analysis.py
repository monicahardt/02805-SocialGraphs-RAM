"""Build an interactive movie social-bridge and structural-bottleneck explorer."""

import argparse
import html
from pathlib import Path

if __package__:
    from .centrality_analysis import DEFAULT_CSV, build_scenarios
    from .dialogue_csv_reader import read_dialogue_records
else:  # Support `python3 code/bridge_analysis.py` from the repository root.
    from centrality_analysis import DEFAULT_CSV, build_scenarios
    from dialogue_csv_reader import read_dialogue_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "graphs" / "movies" / "analysis" / "bridges.html"
VIEWS = {
    "node_betweenness": "Brokerage · node betweenness",
    "separated_pairs": "Node removal · pairs disconnected",
    "articulation": "Structural bottlenecks · articulation points",
    "edge_betweenness": "Potential link bridges · edge betweenness",
    "edge_bridge": "Structural link bridges",
}


def write_page(csv_path: Path = DEFAULT_CSV, output_path: Path = DEFAULT_OUTPUT) -> None:
    scenarios, movies = build_scenarios(read_dialogue_records(csv_path))
    movie_options = "".join(
        f'<option value="{html.escape(movie, quote=True)}">{html.escape(movie)}</option>'
        for movie in movies
    )
    view_options = "".join(
        f'<option value="{key}">{label}</option>' for key, label in VIEWS.items()
    )
    import json

    scenarios_json = json.dumps(scenarios, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Movie Social Bridges</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
*{{box-sizing:border-box}}body{{font:15px system-ui,sans-serif;color:#202832;margin:0}}header{{padding:15px 20px 10px}}h1{{font-size:23px;margin:0 0 6px}}header p{{color:#596775;max-width:1150px;margin:5px 0}}.controls{{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:14px 0 6px}}select{{font:inherit;padding:6px}}main{{padding:0 14px 20px}}.summary{{color:#526273;margin:7px 0 10px;font-size:14px}}.note{{padding:9px 12px;background:#f2f6f9;border-left:4px solid #1683a0;border-radius:3px;color:#425261;font-size:13px;margin:8px 0 12px}}.layout{{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(300px,.8fr);gap:12px}}.panel{{border:1px solid #dce3e9;border-radius:8px;overflow:hidden;background:white}}.panel h2{{font-size:16px;margin:0;padding:10px 12px;border-bottom:1px solid #e7ebef}}#network{{height:640px}}.ranking{{padding:10px 12px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px 5px;text-align:left;border-bottom:1px solid #e8edf1}}th{{color:#596775}}td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}.small{{font-size:12px;color:#687887;padding:0 12px 12px}}@media(max-width:900px){{.layout{{grid-template-columns:1fr}}#network{{height:65vh;min-height:450px}}}}
</style></head><body>
<header><h1>Movie Social Bridges &amp; Bottlenecks</h1>
<p>“Bridge” can mean different things. A broker has high node betweenness because many shortest paths pass through them. An articulation point disconnects part of the graph if removed. Edge betweenness identifies links used by many shortest paths; a structural bridge is a link whose removal disconnects the graph.</p>
<p>All calculations use the selected undirected interaction network. Choose a movie, edge definition, and bridge measure; the graph and ranking recalculate together.</p>
<div class="controls"><label>Movies: <select id="movie"><option value="all">All Movies</option>{movie_options}</select></label>
<label>Edge basis: <select id="rule"><option value="1">Next 1 dialogue line</option><option value="2">Next 2 dialogue lines</option><option value="5">Next 5 dialogue lines</option><option value="chapter">Whole chapter co-appearance</option></select></label>
<label>Bridge measure: <select id="view">{view_options}</select></label>
<label>Layout: <select id="layout"><option value="bridge">Bridge candidates centered</option><option value="spring">Spring layout</option></select></label>
<label><input id="named-only" type="checkbox"> Named people only</label></div>
<div id="summary" class="summary" aria-live="polite"></div><div id="note" class="note"></div></header>
<main><div class="layout"><section class="panel"><h2 id="network-title">Bridge network</h2><div id="network"></div></section>
<section class="panel"><h2 id="ranking-title">Highest-scoring nodes</h2><div class="ranking"><table><thead id="ranking-head"></thead><tbody id="ranking-body"></tbody></table></div>
<p class="small">Node removal impact counts pairs of other characters that were connected before removal but become disconnected afterward. It does not count pairs involving the removed character itself.</p></section></div></main>
<script>
const scenarios={scenarios_json};
const viewLabels={json.dumps(VIEWS,ensure_ascii=False)};
const ruleLabels={{'1':'Next 1 dialogue line','2':'Next 2 dialogue lines','5':'Next 5 dialogue lines','chapter':'Whole chapter co-appearance'}};
const nodeData=new vis.DataSet([]),edgeData=new vis.DataSet([]);
const network=new vis.Network(document.getElementById('network'),{{nodes:nodeData,edges:edgeData}},{{
 physics:{{enabled:false}},interaction:{{hover:true,dragNodes:false}},
 nodes:{{shape:'dot',font:{{size:12}},scaling:{{min:8,max:38}},borderWidth:1}},
 edges:{{color:{{color:'#a0a8b0',highlight:'#334155'}},scaling:{{min:1,max:9}},smooth:{{type:'continuous'}}}}
}});
const fmt=value=>Number(value).toFixed(4);
function update(){{
 const movie=document.getElementById('movie').value,rule=document.getElementById('rule').value;
 const view=document.getElementById('view').value,layout=document.getElementById('layout').value;
 const people=document.getElementById('named-only').checked?'named':'all';
 const data=scenarios[rule][movie][people],nodes=data.nodes,edges=data.edges,stats=data.metrics;
 const edgeMode=view==='edge_betweenness'||view==='edge_bridge';
 const values=edgeMode?edges.map(e=>e.edge_betweenness):nodes.map(n=>view==='separated_pairs'?n.separated_pairs:n.scores.betweenness);
 const maximum=Math.max(0,...values);
 const color=(value, max)=>{{const t=max>0?value/max:0;return `rgb(${{Math.round(226-180*t)}},${{Math.round(239-150*t)}},${{Math.round(247-115*t)}})`;}};
 const bridgeStrength=new Map(nodes.map(node=>[node.id,0]));
 if(view==='node_betweenness')nodes.forEach(node=>bridgeStrength.set(node.id,node.scores.betweenness));
 else if(view==='separated_pairs')nodes.forEach(node=>bridgeStrength.set(node.id,node.separated_pairs));
 else if(view==='articulation')nodes.forEach(node=>bridgeStrength.set(node.id,node.articulation?Math.max(1,node.separated_pairs):0));
 else edges.forEach(edge=>{{
   const score=view==='edge_bridge'?(edge.is_bridge?1:0):edge.edge_betweenness;
   bridgeStrength.set(edge.from,bridgeStrength.get(edge.from)+score);
   bridgeStrength.set(edge.to,bridgeStrength.get(edge.to)+score);
 }});
 const bridgeOrder=[...nodes].sort((a,b)=>bridgeStrength.get(b.id)-bridgeStrength.get(a.id)||b.degree-a.degree||a.label.localeCompare(b.label));
 const bridgePositions=new Map(bridgeOrder.map((node,index)=>{{
   const angle=index*2.399963229728653, radius=index===0?0:82*Math.sqrt(index);
   return [node.id,{{x:radius*Math.cos(angle),y:radius*Math.sin(angle)}}];
 }}));
 const shownNodes=nodes.map(node=>{{
   const score=view==='separated_pairs'?node.separated_pairs:node.scores.betweenness;
   const selected=view==='articulation'?node.articulation:(!edgeMode&&score>0);
   const fill=view==='articulation'?(node.articulation?'#c43c39':'#c9d3dc'):(!edgeMode?color(score,maximum):'#8ecae6');
   const point=layout==='bridge'?bridgePositions.get(node.id):{{x:node.x,y:node.y}};
   return {{...node,...point,value:edgeMode?Math.max(1,node.degree):Math.max(1,score),color:{{background:fill,border:'#596775'}},title:node.label+'<br>Degree: '+node.degree+'<br>Betweenness: '+fmt(node.scores.betweenness)+'<br>Pairs disconnected on removal: '+node.separated_pairs+'<br>Articulation point: '+(node.articulation?'yes':'no')}};
 }});
 const shownEdges=edges.map(edge=>{{
   const stroke=view==='edge_bridge'?(edge.is_bridge?'#c43c39':'#cbd5df'):(view==='edge_betweenness'?color(edge.edge_betweenness,maximum):'#a0a8b0');
   return {{...edge,value:view==='edge_betweenness'?1+8*edge.edge_betweenness/Math.max(maximum,1e-12):Math.max(1,edge.weight),color:{{color:stroke,highlight:'#111827'}},title:edge.from+' — '+edge.to+'<br>Edge betweenness: '+fmt(edge.edge_betweenness)+'<br>Structural bridge: '+(edge.is_bridge?'yes':'no')+'<br>Evidence count: '+edge.weight}};
 }});
 nodeData.clear();nodeData.add(shownNodes);edgeData.clear();edgeData.add(shownEdges);network.fit();
 document.getElementById('network-title').textContent=viewLabels[view]+' · '+(movie==='all'?'All Movies':movie);
 document.getElementById('summary').textContent=stats.nodes+' nodes · '+stats.edges+' edges · average degree '+stats.average_degree.toFixed(2)+' · '+stats.components+' connected components · '+stats.articulation_points+' articulation points · '+stats.structural_bridges+' structural bridge links · '+(layout==='bridge'?'bridge-ranked layout':'spring layout');
 const notes={{
   node_betweenness:'Node betweenness ranks brokers: characters lying on many shortest paths. High betweenness does not require the character to disconnect the graph when removed.',
   separated_pairs:'Removal impact counts how many pairs among the remaining characters lose all paths when a character is removed. A larger value indicates a stronger structural bottleneck.',
   articulation:'Articulation points are strict bottlenecks: removing one increases the number of connected components. Red nodes are articulation points.',
   edge_betweenness:'Edge betweenness ranks links by how many shortest paths use them. A high score suggests a potential bridge, even when alternate routes mean removing the link does not disconnect the graph.',
   edge_bridge:'Structural bridges are links whose removal increases the number of connected components. Red links are structural bridges.'
 }};
 document.getElementById('note').textContent=notes[view]+' NetworkX betweenness is normalized; results are descriptive and are not compared with a degree-preserving null model.';
 const head=document.getElementById('ranking-head'),body=document.getElementById('ranking-body');head.replaceChildren();body.replaceChildren();
 const headers=edgeMode?['#','Bridge link','Edge betweenness','Evidence']:['#','Character',view==='articulation'?'Articulation point':view==='separated_pairs'?'Pairs separated':'Betweenness','Degree'];
 const hr=document.createElement('tr');headers.forEach((name,index)=>{{const th=document.createElement('th');th.textContent=name;if(index>=2)th.className='num';hr.appendChild(th);}});head.appendChild(hr);
 let entries;
 if(edgeMode) entries=edges.filter(e=>view==='edge_bridge'?e.is_bridge:true).sort((a,b)=>(view==='edge_bridge'?Number(b.is_bridge)-Number(a.is_bridge):b.edge_betweenness-a.edge_betweenness)||a.from.localeCompare(b.from)).slice(0,10).map(e=>({{name:e.from+' — '+e.to,score:view==='edge_bridge'?'Yes':fmt(e.edge_betweenness),degree:e.weight}}));
 else entries=nodes.filter(n=>view!=='articulation'||n.articulation).sort((a,b)=>(view==='separated_pairs'?b.separated_pairs-a.separated_pairs:b.scores.betweenness-a.scores.betweenness)||a.label.localeCompare(b.label)).slice(0,10).map(n=>({{name:n.label,score:view==='articulation'?'Yes':view==='separated_pairs'?n.separated_pairs:fmt(n.scores.betweenness),degree:n.degree}}));
 if(entries.length===0){{const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=4;td.textContent=view==='articulation'?'No articulation points found in this graph.':'No structural bridges found for this graph.';tr.appendChild(td);body.appendChild(tr);}}
 entries.forEach((entry,index)=>{{const tr=document.createElement('tr');for(const [value,numeric] of [[index+1,false],[entry.name,false],[entry.score,true],[entry.degree,true]]){{const td=document.createElement('td');td.textContent=String(value);if(numeric)td.className='num';tr.appendChild(td);}}body.appendChild(tr);}});
}}
for(const id of ['movie','rule','view','layout','named-only'])document.getElementById(id).addEventListener('change',update);
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
    print(f"Wrote movie social-bridges explorer to {args.output}")


if __name__ == "__main__":
    main()
