"""Create an interactive HTML character network from dialogue turns."""

import html
import json
from collections import Counter
from pathlib import Path

if __package__:
    from .dialogue_csv_reader import DialogueRecord
else:  # Support importing from a script launched inside the code directory.
    from dialogue_csv_reader import DialogueRecord


def _js_json(value: object) -> str:
    """Encode JSON safely for embedding inside an HTML script element."""
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def write_html(dialogue: list[DialogueRecord], output_path: str | Path) -> None:
    """Write a weighted, undirected network to an interactive HTML file.

    An edge is added when two different attributed speakers have consecutive
    dialogue rows in the same chapter. Unknown and low-confidence rows break
    the sequence so the graph does not jump across unattributed dialogue.
    """
    characters: set[str] = set()
    edge_weights: Counter[tuple[str, str]] = Counter()
    previous_character: str | None = None
    previous_chapter: str | None = None

    for row in dialogue:
        if row.character.casefold() == "unknown" or row.confidence.casefold() in {
            "low",
            "unattributed",
        }:
            previous_character = None
            previous_chapter = row.chapter
            continue

        character = row.character
        characters.add(character)
        if (
            row.chapter == previous_chapter
            and previous_character is not None
            and previous_character != character
        ):
            edge = tuple(sorted((previous_character, character)))
            edge_weights[edge] += 1
        previous_character = character
        previous_chapter = row.chapter

    degrees: Counter[str] = Counter()
    for first, second in edge_weights:
        degrees[first] += 1
        degrees[second] += 1

    nodes = [
        {
            "id": character,
            "label": character,
            "value": max(1, degrees[character]),
            "title": f"{character} — degree {degrees[character]}",
        }
        for character in sorted(characters)
    ]
    edges = [
        {
            "from": first,
            "to": second,
            "value": weight,
            "title": f"{weight} consecutive speaker transitions",
        }
        for (first, second), weight in sorted(edge_weights.items())
    ]
    title = dialogue[0].book if dialogue else "Character dialogue network"

    document = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — Character Network</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>body{{font:15px system-ui;margin:0;color:#222}}header{{padding:12px 20px}}#network{{height:calc(100vh - 75px);border-top:1px solid #ddd}}</style>
</head><body><header><strong>{html.escape(title)}</strong> — character dialogue network<br>
Each edge counts consecutive attributed turns by different speakers in the same chapter. Unknown and low-confidence turns are skipped; thicker edges mean more transitions.</header>
<div id="network"></div><script>
const nodes = new vis.DataSet({_js_json(nodes)});
const edges = new vis.DataSet({_js_json(edges)});
new vis.Network(document.getElementById('network'), {{nodes, edges}}, {{
  nodes:{{shape:'dot',font:{{size:14}},scaling:{{min:10,max:38}},color:{{background:'#8ecae6',border:'#219ebc'}}}},
  edges:{{smooth:{{type:'dynamic'}},color:{{color:'#777',highlight:'#e76f51'}},scaling:{{min:1,max:8}}}},
  interaction:{{hover:true}},
  physics:{{solver:'forceAtlas2Based',forceAtlas2Based:{{gravitationalConstant:-80,centralGravity:0.015,springLength:130,springConstant:0.08,avoidOverlap:0.8}},stabilization:{{iterations:800}}}}
}});
</script></body></html>
"""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
