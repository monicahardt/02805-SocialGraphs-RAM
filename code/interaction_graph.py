"""Create an interactive HTML character network from dialogue turns."""

import html
import json
from pathlib import Path

if __package__:
    from .dialogue_csv_reader import DialogueRecord
else:  # Support importing from a script launched inside the code directory.
    from dialogue_csv_reader import DialogueRecord


# These labels in the movie CSV are group labels, generic roles, objects, or
# creatures/locations rather than a specific character. Keep this explicit so
# named characters with a single-word name (for example Dobby) remain included.
NON_SPECIFIC_MOVIE_LABELS = {
    "all", "announcer", "basilisk", "boy", "boy 2", "centaur", "class",
    "crowd", "death eater", "diary", "gang thug", "ghost", "ghosts", "girl",
    "goblin", "gryffindors", "guard", "howler", "maid", "man",
    "man in a painting", "merpeople", "other", "patronus", "pixie",
    "professors", "shrunken head", "skinny kid", "snake", "snatcher",
    "station guard", "student", "students", "teacher", "teachers",
    "random guard", "random teacher", "professor", "trolley witch", "waiter",
    "waitress", "whomping willow", "witch", "wizard", "woman", "old man",
}


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


def write_html(
    dialogue: list[DialogueRecord],
    output_path: str | Path,
    *,
    edge_rule_dashboard: bool = False,
    collection_label: str = "Movie",
) -> None:
    """Write an interactive dialogue network, optionally with edge-rule filters.

    Turn-window edges connect different attributed speakers within the chosen
    number of following turns in the same movie and chapter. In dashboard mode,
    a chapter co-appearance edge also joins each pair of speakers in a chapter.
    Unknown and low-confidence rows break the turn sequence.
    """
    characters_by_movie: dict[str, set[str]] = {}
    chapter_speakers: dict[tuple[str, str], set[str]] = {}
    turn_blocks: list[tuple[tuple[str, str], list[str]]] = []
    current_key: tuple[str, str] | None = None
    current_turns: list[str] = []

    def finish_turn_block() -> None:
        nonlocal current_turns
        if current_key is not None and current_turns:
            turn_blocks.append((current_key, current_turns))
            current_turns = []

    for row in dialogue:
        movie = row.book or "Unknown movie"
        key = (movie, row.chapter)
        if key != current_key:
            finish_turn_block()
            current_key = key
        characters_by_movie.setdefault(movie, set())
        if row.character.casefold() == "unknown" or row.confidence.casefold() in {
            "low",
            "unattributed",
        }:
            finish_turn_block()
            continue

        character = row.character
        characters_by_movie[movie].add(character)
        chapter_speakers.setdefault(key, set()).add(character)
        current_turns.append(character)
    finish_turn_block()

    # For each undirected pair, retain evidence counts separately for each
    # forward-turn window and for chapter co-appearance.
    rule_weights: dict[tuple[str, str], dict[str, dict[str, int]]] = {}
    for (movie, _chapter), turns in turn_blocks:
        for window in (1, 2, 5):
            rule = str(window)
            for index, character in enumerate(turns):
                for other in turns[index + 1 : index + window + 1]:
                    if character == other:
                        continue
                    pair = tuple(sorted((character, other)))
                    per_rule = rule_weights.setdefault(pair, {})
                    per_movie = per_rule.setdefault(rule, {})
                    per_movie[movie] = per_movie.get(movie, 0) + 1

    if edge_rule_dashboard:
        for (movie, _chapter), speakers in chapter_speakers.items():
            ordered_speakers = sorted(speakers)
            for index, first in enumerate(ordered_speakers):
                for second in ordered_speakers[index + 1 :]:
                    per_rule = rule_weights.setdefault((first, second), {})
                    per_movie = per_rule.setdefault("chapter", {})
                    per_movie[movie] = per_movie.get(movie, 0) + 1

    movies = sorted(characters_by_movie)
    collection_plural = "Books" if collection_label.casefold() == "book" else "Movies"
    collection_lower_plural = collection_plural.casefold()
    characters = sorted({character for movie_chars in characters_by_movie.values() for character in movie_chars})
    nodes = [
        {
            "id": character,
            "label": character,
            "specific_person": (
                collection_label.casefold() != "movie"
                or character.casefold() not in NON_SPECIFIC_MOVIE_LABELS
            ),
            "movies": sorted(movie for movie in movies if character in characters_by_movie[movie]),
            "value": max(1, sum(1 for edge in rule_weights if character in edge and "1" in rule_weights[edge])),
            "title": f"{character} — appears in " + ", ".join(
                movie for movie in movies if character in characters_by_movie[movie]
            ),
        }
        for character in characters
    ]
    edges = []
    for (first, second), modes in sorted(rule_weights.items()):
        if not edge_rule_dashboard and "1" not in modes:
            continue
        consecutive_weights = modes.get("1", {})
        edges.append(
            {
                "from": first,
                "to": second,
                "weights": consecutive_weights,
                "rule_weights": modes,
                "value": sum(consecutive_weights.values()),
                "title": "Consecutive speaker transitions",
            }
        )
    title = f"{collection_label} Dialogue Interaction Network"
    movie_options = "".join(
        f'<option value="{html.escape(movie, quote=True)}">{html.escape(movie)}</option>'
        for movie in movies
    )
    rule_control = (
        '<label>Edge basis: <select id="edge-rule">'
        '<option value="1">Next 1 dialogue line</option>'
        '<option value="2">Next 2 dialogue lines</option>'
        '<option value="5">Next 5 dialogue lines</option>'
        '<option value="chapter">Whole chapter co-appearance</option>'
        "</select></label>"
        if edge_rule_dashboard
        else ""
    )
    person_filter_control = (
        '<label><input type="checkbox" id="named-people-only"> Named people only</label>'
        if collection_label.casefold() == "movie"
        else ""
    )
    edge_description = (
        "Use the controls to compare short-range speaker turns with everyone who speaks in the same chapter."
        if edge_rule_dashboard
        else f"Each edge counts consecutive attributed turns by different speakers in the same {collection_label.casefold()} and chapter. Thicker edges mean more transitions."
    )
    ranking_section = """<section id="ranking"><h2>Top 10 characters by degree</h2>
<p>Degree counts distinct character neighbors in the currently selected graph.</p>
<table><thead><tr><th>Rank</th><th>Character</th><th>Degree</th></tr></thead><tbody id="ranking-body"></tbody></table></section>""" if len(movies) > 1 else ""

    document = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>body{{font:15px system-ui;margin:0;color:#222}}header{{padding:12px 20px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}}header p{{margin:4px 0 0}}#graph-summary{{width:100%;color:#555;font-size:14px}}#visualization{{display:flex;align-items:stretch;min-height:calc(100vh - 130px)}}#network{{height:calc(100vh - 130px);min-height:420px;flex:1;min-width:0;border-top:1px solid #ddd}}select{{font:inherit;padding:6px}}#ranking{{width:310px;flex-shrink:0;margin:0;padding:16px 14px;border-left:1px solid #ddd}}#ranking h2{{font-size:18px;margin:0 0 4px}}#ranking p{{font-size:13px;color:#555}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:7px 4px;border-bottom:1px solid #ddd}}td:last-child,th:last-child{{text-align:right}}@media(max-width:800px){{#visualization{{display:block}}#network{{height:65vh}}#ranking{{width:auto;border-left:0;border-top:1px solid #ddd}}}}</style>
</head><body><header><div><strong>{title}</strong><p>{edge_description} Nodes with more distinct neighbors are placed closer to the center; positions update with the selected filters.</p></div>
<label for="movie-filter">{collection_label}: <select id="movie-filter"><option value="all">All {collection_plural}</option>{movie_options}</select></label>{rule_control}{person_filter_control}<div id="graph-summary" aria-live="polite"></div></header>
<div id="visualization"><div id="network"></div>{ranking_section}</div><script>
const allNodes = {_js_json(nodes)};
const allEdges = {_js_json(edges)};
const nodeData = new vis.DataSet([]);
const edgeData = new vis.DataSet([]);
const network = new vis.Network(document.getElementById('network'), {{nodes:nodeData, edges:edgeData}}, {{
  nodes:{{shape:'dot',font:{{size:14}},scaling:{{min:10,max:38}},color:{{background:'#8ecae6',border:'#219ebc'}}}},
  edges:{{smooth:{{type:'dynamic'}},color:{{color:'#777',highlight:'#e76f51'}},scaling:{{min:1,max:8}}}},
  interaction:{{hover:true,dragNodes:false}},
  physics:{{enabled:false}}
}});
function filterMovie(movie) {{
  const selectedRule = {"document.getElementById('edge-rule').value" if edge_rule_dashboard else "'1'"};
  const peopleFilter = document.getElementById('named-people-only');
  const namedPeopleOnly = Boolean(peopleFilter && peopleFilter.checked);
  const visibleNodes = allNodes.filter(node =>
    (movie === 'all' || node.movies.includes(movie)) && (!namedPeopleOnly || node.specific_person)
  );
  const visibleIds = new Set(visibleNodes.map(node => node.id));
  const visibleEdges = allEdges.map(edge => {{
    const weights = edge.rule_weights ? (edge.rule_weights[selectedRule] || {{}}) : edge.weights;
    const weight = movie === 'all'
      ? Object.values(weights).reduce((total, count) => total + count, 0)
      : (weights[movie] || 0);
    return {{...edge, activeWeight:weight, value:weight, weights}};
  }}).filter(edge => visibleIds.has(edge.from) && visibleIds.has(edge.to) && edge.activeWeight > 0);
  const degree = new Map(visibleNodes.map(node => [node.id, 0]));
  for (const edge of visibleEdges) {{ degree.set(edge.from, degree.get(edge.from) + 1); degree.set(edge.to, degree.get(edge.to) + 1); }}
  const rankingBody = document.getElementById('ranking-body');
  if (rankingBody) {{
    const topTen = [...visibleNodes].sort((a, b) => degree.get(b.id) - degree.get(a.id) || a.label.localeCompare(b.label)).slice(0, 10);
    rankingBody.replaceChildren();
    topTen.forEach((node, index) => {{
      const row = document.createElement('tr');
      for (const value of [index + 1, node.label, degree.get(node.id)]) {{
        const cell = document.createElement('td'); cell.textContent = String(value); row.appendChild(cell);
      }}
      rankingBody.appendChild(row);
    }});
  }}
  const orderedNodes = [...visibleNodes].sort((a, b) => degree.get(b.id) - degree.get(a.id) || a.label.localeCompare(b.label));
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const positionedNodes = orderedNodes.map((node, index) => {{
    const radius = index === 0 ? 0 : 75 * Math.sqrt(index);
    const angle = index * goldenAngle;
    const nodeDegree = degree.get(node.id);
    return {{...node, x:radius * Math.cos(angle), y:radius * Math.sin(angle), value:Math.max(1,nodeDegree), title:node.label + ' — degree ' + nodeDegree}};
  }});
  const filteredEdges = visibleEdges.map(edge => {{
    const basis = selectedRule === 'chapter' ? 'shared chapters' : 'nearby dialogue-turn pairs';
    return {{...edge, value:edge.activeWeight, title:edge.activeWeight + ' ' + basis + (movie === 'all' ? ' across all {collection_lower_plural}' : ' in ' + movie)}};
  }});
  const averageDegree = visibleNodes.length ? (2 * visibleEdges.length / visibleNodes.length) : 0;
  document.getElementById('graph-summary').textContent = visibleNodes.length + ' nodes · ' + visibleEdges.length + ' distinct edges · average degree ' + averageDegree.toFixed(2);
  nodeData.clear(); nodeData.add(positionedNodes);
  edgeData.clear(); edgeData.add(filteredEdges);
  network.fit();
}}
document.getElementById('movie-filter').addEventListener('change', event => filterMovie(event.target.value));
{"document.getElementById('edge-rule').addEventListener('change', () => filterMovie(document.getElementById('movie-filter').value));" if edge_rule_dashboard else ""}
{"document.getElementById('named-people-only').addEventListener('change', () => filterMovie(document.getElementById('movie-filter').value));" if collection_label.casefold() == "movie" else ""}
filterMovie('all');
</script></body></html>
"""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
