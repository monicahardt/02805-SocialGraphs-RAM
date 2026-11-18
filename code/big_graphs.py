"""Build the three network–language research visualizations for the homepage."""

import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx

if __package__:
    from .dialogue_csv_reader import read_dialogue_records
    from .interaction_graph import NON_SPECIFIC_MOVIE_LABELS
else:
    from dialogue_csv_reader import read_dialogue_records
    from interaction_graph import NON_SPECIFIC_MOVIE_LABELS


ROOT = Path(__file__).resolve().parents[1]
BOOK_CSV = ROOT / "book_data" / "all_books.csv"
MOVIE_CSV = ROOT / "movie_data" / "all_movies.csv"
OUT = ROOT / "graphs" / "research"
BERT_MAP = ROOT / "results" / "bert_character_map" / "pacmap_coordinates.json"
RULES = ("1", "2", "5", "chapter")
RULE_LABELS = {
    "1": "Next 1 dialogue line",
    "2": "Next 2 dialogue lines",
    "5": "Next 5 dialogue lines",
    "chapter": "Whole chapter co-appearance",
}
TITLES = (
    "Philosopher's Stone", "Chamber of Secrets", "Prisoner of Azkaban",
    "Goblet of Fire", "Order of the Phoenix", "Half-Blood Prince", "Deathly Hallows",
)
ALIASES = {
    "Tom Riddle": "Voldemort", "Garrick Ollivander": "Ollivander",
    "Poppy Pomfrey": "Madam Pomfrey", "Rolanda Hooch": "Madam Hooch",
    "Stanley Shunpike": "Stan Shunpike", "Collin Creevey": "Colin Creevey",
    "Ernest Macmillan": "Ernie Macmillan",
}
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
STOP = set("""a about above after again against all am an and any are as at be because been before being below between both but by can could did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor of off on once only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why will with would you your yours yourself yourselves also's""".split())
NAME_TERMS = set()


def _safe_json(value):
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _canonical(name):
    return ALIASES.get(name.strip(), name.strip())


def _installment(title):
    return next((i for i, suffix in enumerate(TITLES, 1) if suffix in title), None)


def _tokens(text, *, remove_names=True):
    result = []
    for raw in TOKEN_RE.findall(text.lower().replace("’", "'")):
        word = raw.strip("'")
        if len(word) < 3 or word in STOP:
            continue
        if remove_names and word in NAME_TERMS:
            continue
        result.append(word)
    return result


def _read_inputs():
    corpora = {
        "books": read_dialogue_records(BOOK_CSV),
        "movies": read_dialogue_records(MOVIE_CSV),
    }
    all_names = {_canonical(row.character) for rows in corpora.values() for row in rows if row.character.casefold() != "unknown"}
    global NAME_TERMS
    NAME_TERMS = {part.lower() for name in all_names for part in TOKEN_RE.findall(name) if len(part) >= 3}
    return corpora


def _line_data(corpora):
    """Tokenize supplied lines once; aggregate profiles and installment profiles."""
    lines = []
    profile_counts = defaultdict(Counter)
    installment_counts = defaultdict(Counter)
    raw_words = Counter()
    utterances = Counter()
    sequences = defaultdict(list)
    rows_by_medium = {}
    for medium, records in corpora.items():
        rows = []
        for row in records:
            name = _canonical(row.character)
            if name.casefold() == "unknown":
                continue
            if medium == "movies" and name.casefold() in NON_SPECIFIC_MOVIE_LABELS:
                continue
            source = row.book
            installment = _installment(source)
            tokens = _tokens(row.line)
            raw_count = len(TOKEN_RE.findall(row.line))
            rows.append({"medium": medium, "source": source, "chapter": row.chapter,
                         "installment": installment, "character": name, "tokens": tokens,
                         "raw_words": raw_count, "line": row.line})
            if not tokens:
                continue
            lines.append((medium, name, tokens))
            profile_counts[(medium, name)].update(tokens)
            if installment is not None:
                installment_counts[(medium, name, installment)].update(tokens)
                raw_words[(medium, name, installment)] += raw_count
            raw_words[(medium, name)] += raw_count
            utterances[(medium, name)] += 1
            sequences[(medium, name)].extend(tokens)
        rows_by_medium[medium] = rows
    return lines, profile_counts, installment_counts, raw_words, utterances, sequences, rows_by_medium


def _tfidf_vectors(count_docs, *, min_df=2):
    """Small sparse-dictionary implementation of course TF-IDF and cosine."""
    docs = list(count_docs)
    n = len(docs)
    df = Counter(word for counts in docs for word in counts)
    idf = {word: math.log(n / frequency) for word, frequency in df.items() if frequency >= min_df and frequency < n}
    vectors = []
    for counts in docs:
        total = sum(counts.values())
        weights = {word: (count / total) * idf[word] for word, count in counts.items() if word in idf and total}
        norm = math.sqrt(sum(value * value for value in weights.values()))
        vectors.append({word: value / norm for word, value in weights.items()} if norm else {})
    return vectors


def _cosine(first, second):
    if len(first) > len(second):
        first, second = second, first
    return sum(value * second.get(word, 0.0) for word, value in first.items())


def _mattr(tokens, window=100):
    if not tokens:
        return None
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    counts = Counter(tokens[:window])
    unique_sum = len(counts)
    steps = len(tokens) - window
    for index in range(steps):
        leaving, entering = tokens[index], tokens[index + window]
        counts[leaving] -= 1
        if counts[leaving] == 0:
            del counts[leaving]
        counts[entering] += 1
        unique_sum += len(counts)
    return unique_sum / (steps + 1) / window


def _utterance_dispersion(lines):
    """Mean cosine distance of each utterance from its character's TF-IDF centroid."""
    df = Counter()
    for _medium, _character, tokens in lines:
        df.update(set(tokens))
    n = len(lines)
    idf = {word: math.log(n / count) for word, count in df.items() if count < n}
    sums = defaultdict(Counter)
    normalized_lines = []
    for medium, character, tokens in lines:
        counts = Counter(tokens)
        total = len(tokens)
        weights = {word: count / total * idf[word] for word, count in counts.items() if word in idf}
        norm = math.sqrt(sum(value * value for value in weights.values()))
        vector = {word: value / norm for word, value in weights.items()} if norm else {}
        sums[(medium, character)].update(vector)
        normalized_lines.append(((medium, character), vector))
    centroid_norms = {key: math.sqrt(sum(value * value for value in total.values())) for key, total in sums.items()}
    similarities = Counter()
    counts = Counter()
    for key, vector in normalized_lines:
        norm = centroid_norms[key]
        if not vector or not norm:
            continue
        centroid = sums[key]
        similarities[key] += sum(value * centroid.get(word, 0.0) for word, value in vector.items()) / norm
        counts[key] += 1
    return {key: max(0.0, min(1.0, 1 - similarities[key] / counts[key]))
            for key in counts if counts[key] >= 5}


def _make_graph(rows, medium, rule, installment=None):
    graph = nx.Graph()
    chapter_speakers = defaultdict(set)
    blocks = []
    active_key = None
    turns = []

    def finish():
        nonlocal turns
        if active_key is not None and turns:
            blocks.append((active_key, turns))
        turns = []

    for row in rows:
        if installment is not None and row["installment"] != installment:
            continue
        key = (row["source"], row["chapter"])
        if key != active_key:
            finish()
            active_key = key
        name = row["character"]
        graph.add_node(name)
        chapter_speakers[key].add(name)
        turns.append(name)
    finish()

    evidence = Counter()
    if rule == "chapter":
        for speakers in chapter_speakers.values():
            evidence.update(combinations(sorted(speakers), 2))
    else:
        window = int(rule)
        for _key, speaker_turns in blocks:
            for index, first in enumerate(speaker_turns):
                for second in speaker_turns[index + 1:index + window + 1]:
                    if first != second:
                        evidence[tuple(sorted((first, second)))] += 1
    graph.add_edges_from(evidence)
    return graph, evidence


def _community_positions(graph, communities):
    result = {}
    for community_index, group in enumerate(communities):
        angle = 2 * math.pi * community_index / max(1, len(communities))
        radius = 620 if len(communities) > 1 else 0
        cx, cy = radius * math.cos(angle), radius * math.sin(angle)
        members = sorted(group, key=lambda name: (-graph.degree[name], name.casefold()))
        inner = max(90, 28 * math.sqrt(len(members)))
        for index, name in enumerate(members):
            a = 2 * math.pi * index / max(1, len(members))
            result[name] = {"x": cx + inner * math.cos(a), "y": cy + inner * math.sin(a)}
    return result


def _matched_profiles(installment_counts, raw_words):
    names = sorted({key[1] for key in installment_counts if key[0] == "books"} &
                   {key[1] for key in installment_counts if key[0] == "movies"})
    matched_counts = {}
    for name in names:
        common = [inst for inst in range(1, 8)
                  if installment_counts.get(("books", name, inst))
                  and installment_counts.get(("movies", name, inst))]
        b = Counter(); m = Counter(); bw = mw = 0
        for inst in common:
            b.update(installment_counts[("books", name, inst)])
            m.update(installment_counts[("movies", name, inst)])
            bw += raw_words[("books", name, inst)]
            mw += raw_words[("movies", name, inst)]
        if common and bw >= 100 and mw >= 100:
            matched_counts[("books", name)] = (b, bw, common)
            matched_counts[("movies", name)] = (m, mw, common)
    keys = sorted(matched_counts)
    vectors = _tfidf_vectors([matched_counts[key][0] for key in keys])
    return {key: {"vector": vector, "words": matched_counts[key][1], "installments": matched_counts[key][2]}
            for key, vector in zip(keys, vectors)}


def _profile_vectors(profile_counts, minimum=100):
    keys = sorted(key for key, counts in profile_counts.items() if sum(counts.values()) >= minimum)
    vectors = _tfidf_vectors([profile_counts[key] for key in keys])
    return {key: vector for key, vector in zip(keys, vectors)}


def _partition(graph):
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        return [{node} for node in graph]
    result = list(nx.community.louvain_communities(graph, weight=None, seed=42))
    return sorted(result, key=lambda group: (-len(group), min(group, key=str.casefold).casefold()))


def _build_graph_scenarios(rows_by_medium, profile_counts, raw_words, utterances, sequences,
                           dispersion, profile_vectors):
    graph1, graph2 = {}, {}
    raw_graphs = {}
    for medium, rows in rows_by_medium.items():
        graph1[medium] = {}
        graph2[medium] = {}
        for rule in RULES:
            graph, evidence = _make_graph(rows, medium, rule)
            raw_graphs[(medium, rule)] = graph
            betweenness = nx.betweenness_centrality(graph, normalized=True, weight=None)
            partition = _partition(graph)
            community_of = {node: index for index, group in enumerate(partition) for node in group}
            positions = _community_positions(graph, partition)
            candidates = [name for name in graph if (medium, name) in profile_vectors]
            pair_sims = {}
            within, between = [], []
            within_scores = defaultdict(list)
            outside_scores = defaultdict(list)
            for i, first in enumerate(candidates):
                for second in candidates[i + 1:]:
                    score = _cosine(profile_vectors[(medium, first)], profile_vectors[(medium, second)])
                    same = community_of[first] == community_of[second]
                    pair_sims[(first, second)] = score
                    if same:
                        within.append(score)
                        within_scores[first].append(score)
                        within_scores[second].append(score)
                    else:
                        between.append(score)
                        outside_scores[first].append(score)
                        outside_scores[second].append(score)
            nodes = []
            for name in sorted(graph, key=str.casefold):
                inside = within_scores[name]
                outside = outside_scores[name]
                in_mean = sum(inside) / len(inside) if inside else None
                out_mean = sum(outside) / len(outside) if outside else None
                nodes.append({
                    "id": name, "label": name, "degree": graph.degree[name],
                    "betweenness": betweenness[name], "community": community_of[name],
                    "within_similarity": in_mean, "outside_similarity": out_mean,
                    "similarity_gap": in_mean - out_mean if in_mean is not None and out_mean is not None else None,
                    "words": raw_words[(medium, name)],
                    **positions[name],
                })
            edges = [{"from": a, "to": b, "weight": int(count)} for (a, b), count in sorted(evidence.items())]
            graph1[medium][rule] = [
                {"name": name, "degree": graph.degree[name], "degree_centrality": graph.degree[name] / max(1, graph.number_of_nodes() - 1),
                 "betweenness": betweenness[name], "mattr": _mattr(sequences[(medium, name)]),
                 "dispersion": dispersion.get((medium, name)), "words": raw_words[(medium, name)],
                 "utterances": utterances[(medium, name)], "community": community_of[name]}
                for name in sorted(graph, key=str.casefold)
                if (medium, name) in profile_vectors and len(sequences[(medium, name)]) >= 100
                and utterances[(medium, name)] >= 5 and dispersion.get((medium, name)) is not None
            ]
            graph2[medium][rule] = {
                "nodes": nodes, "edges": edges, "within": within, "between": between,
                "metrics": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
                            "communities": len(partition),
                            "mean_within": sum(within) / len(within) if within else None,
                            "mean_between": sum(between) / len(between) if between else None},
            }
    return graph1, graph2, raw_graphs


def _build_adaptation_data(rows_by_medium, installment_counts, raw_words, matched_profiles):
    by_rule = {rule: {} for rule in RULES}
    eligible = {name for medium, name in matched_profiles if medium == "books"}
    eligible &= {name for medium, name in matched_profiles if medium == "movies"}
    for rule in RULES:
        per_character = defaultdict(list)
        for installment in range(1, 8):
            book_graph, _ = _make_graph(rows_by_medium["books"], "books", rule, installment)
            movie_graph, _ = _make_graph(rows_by_medium["movies"], "movies", rule, installment)
            shared = set(book_graph) & set(movie_graph)
            if not shared:
                continue
            book_betweenness = nx.betweenness_centrality(book_graph, normalized=True, weight=None)
            movie_betweenness = nx.betweenness_centrality(movie_graph, normalized=True, weight=None)
            for name in eligible & shared:
                book_neighbors = set(book_graph.neighbors(name)) & shared
                movie_neighbors = set(movie_graph.neighbors(name)) & shared
                union = book_neighbors | movie_neighbors
                if not union:
                    continue
                jaccard = len(book_neighbors & movie_neighbors) / len(union)
                per_character[name].append({
                    "installment": installment, "neighbor_jaccard": jaccard,
                    "betweenness_change": movie_betweenness.get(name, 0.0) - book_betweenness.get(name, 0.0),
                })
        points = []
        for name in sorted(eligible, key=str.casefold):
            items = per_character.get(name, [])
            if not items:
                continue
            b = matched_profiles[("books", name)]
            m = matched_profiles[("movies", name)]
            cosine = _cosine(b["vector"], m["vector"])
            points.append({
                "name": name,
                "neighbor_retention": sum(item["neighbor_jaccard"] for item in items) / len(items),
                "language_cosine": cosine,
                "betweenness_change": sum(item["betweenness_change"] for item in items) / len(items),
                "book_words": b["words"], "movie_words": m["words"],
                "minimum_words": min(b["words"], m["words"]),
                "shared_installments": [item["installment"] for item in items],
                "installment_count": len(items),
            })
        by_rule[rule] = points
    return by_rule


def build_data():
    corpora = _read_inputs()
    lines, profile_counts, installment_counts, raw_words, utterances, sequences, rows_by_medium = _line_data(corpora)
    dispersion = _utterance_dispersion(lines)
    profile_vectors = _profile_vectors(profile_counts)
    graph1, graph2, _graphs = _build_graph_scenarios(
        rows_by_medium, profile_counts, raw_words, utterances, sequences, dispersion, profile_vectors
    )
    matched_profiles = _matched_profiles(installment_counts, raw_words)
    graph3 = _build_adaptation_data(rows_by_medium, installment_counts, raw_words, matched_profiles)
    counts = {medium: {"lines": len(rows), "speakers": len({r['character'] for r in rows})}
              for medium, rows in rows_by_medium.items()}
    return graph1, graph2, graph3, counts


def _write_big_graph_1(data):
    if not BERT_MAP.exists():
        raise FileNotFoundError(
            f"Missing {BERT_MAP.relative_to(ROOT)}. Run `python3 code/build_bert_character_map.py` first."
        )
    bert_map = json.loads(BERT_MAP.read_text(encoding="utf-8"))
    for medium in ("books", "movies"):
        for rule in RULES:
            for point in data[medium][rule]:
                coords = bert_map.get(medium, {}).get(point["name"])
                if coords:
                    point["pacmap_x"], point["pacmap_y"] = coords
                else:
                    point["pacmap_x"] = point["pacmap_y"] = None
    medium_options = '<option value="both">Books + Movies</option><option value="books">Books</option><option value="movies">Movies</option>'
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Big Graph 1 · Bridge centrality and language diversity</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>*{{box-sizing:border-box}}body{{font:15px system-ui,sans-serif;color:#202832;margin:0}}header{{padding:18px 22px}}h1{{font-size:23px;margin:0 0 8px}}header p{{max-width:1100px;color:#596775;margin:6px 0}}.controls{{display:flex;gap:20px;flex-wrap:wrap;align-items:center;margin-top:14px}}select{{font:inherit;padding:6px}}main{{padding:0 8px 20px}}.pair{{display:grid;grid-template-columns:minmax(0,1fr);gap:12px}}.panel{{border:1px solid #dce3e9;border-radius:8px;overflow:hidden;width:100%}}.panel h2{{font-size:16px;margin:0;padding:10px 12px;border-bottom:1px solid #e7ebef}}.plot{{height:540px}}#mattr.plot{{height:760px}}#dispersion.plot{{height:520px}}.note{{font-size:12px;color:#657482;padding:0 13px 12px}}@media(max-width:900px){{.plot{{height:460px}}#mattr.plot{{height:620px}}#dispersion.plot{{height:460px}}}}</style></head><body>
<header><h1>Big Graph 1 · Bridge position and linguistic diversity</h1>
<p>Do characters whose dialogue-network position makes them a bridge also use more varied language? The first plot places each character by their dialogue’s BERT embedding, projected into two dimensions with PaCMAP; color shows betweenness centrality on a shared logarithmic scale. Hover to see the original value. The second plot compares degree and betweenness.</p>
<div class="controls"><label>Edge basis: <select id="rule">{''.join(f'<option value="{r}">{RULE_LABELS[r]}</option>' for r in RULES)}</select></label><label>Corpus: <select id="medium">{medium_options}</select></label><span>Point size represents attributed dialogue word count.</span></div>
<p id="summary"></p></header><main><div class="pair"><section class="panel"><h2>BERT character embeddings · position = language, color = betweenness</h2><div id="mattr" class="plot"></div><p class="note">Each character’s dialogue is represented by token-weighted mean sentence-transformer embeddings (all-MiniLM-L6-v2), then projected together with PaCMAP. Nearby points have more similar BERT representations approximately; PaCMAP axes have no direct meaning. NetworkX already normalizes betweenness to 0–1; colors use a fixed logarithmic transform to make low values easier to distinguish. The scale is shared across books, movies, and edge rules; hover reports the original value.</p></section>
<section class="panel"><h2>Degree vs. betweenness, colored by utterance dispersion</h2><div id="dispersion" class="plot"></div><p class="note">Utterance dispersion is mean cosine distance from a speaker’s TF-IDF utterance centroid. It measures lexical/topic spread, not personality or all meanings of “linguistic diversity.”</p></section></div>
<p class="note">Compare betweenness relative to degree: high degree alone can produce high betweenness. The network connects nearby attributed turns (or same-chapter speakers under that option), not confirmed addressees. This view is descriptive and does not include a degree-preserving null-model score.</p></main>
<script>
const data={_safe_json(data)};
const colors={{books:'#176b87',movies:'#b64a3f'}};
const betweennessColor=value=>Math.log10(1+1000*Math.max(0,value))/Math.log10(1001);
const centralityTicks=[0,0.001,0.005,0.01,0.02,0.05,0.1,1];
function update(){{
 const rule=document.getElementById('rule').value,medium=document.getElementById('medium').value;
 const sources=medium==='both'?['books','movies']:[medium];
 const tracesM=[],tracesD=[];
 for(const source of sources){{
  const points=data[source][rule];
  const custom=points.map(p=>[p.name,p.degree,p.words,p.utterances,p.dispersion,p.community,p.betweenness]);
  const sizes=points.map(p=>Math.max(7,Math.min(26,5+Math.sqrt(p.words)/4)));
  tracesM.push({{x:points.map(p=>p.pacmap_x),y:points.map(p=>p.pacmap_y),customdata:custom,mode:'markers',type:'scatter',name:source==='books'?'Books':'Movies',marker:{{size:sizes,color:points.map(p=>betweennessColor(p.betweenness)),colorscale:'Viridis',cmin:0,cmax:1,showscale:source===sources[0],colorbar:{{title:'Betweenness',tickvals:centralityTicks.map(betweennessColor),ticktext:centralityTicks.map(v=>v===0||v===1?String(v):v.toFixed(3))}},symbol:source==='books'?'circle':'triangle-up',line:{{color:colors[source],width:1}}}},hovertemplate:'%{{customdata[0]}} · '+(source==='books'?'Books':'Movies')+'<br>betweenness = %{{customdata[6]:.4f}}<br>degree = %{{customdata[1]}}<br>dialogue words = %{{customdata[2]}}<br>utterances = %{{customdata[3]}}<extra></extra>'}});
  tracesD.push({{x:points.map(p=>p.degree),y:points.map(p=>p.betweenness),customdata:custom,mode:'markers',type:'scatter',name:source==='books'?'Books':'Movies',marker:{{size:sizes,color:points.map(p=>p.dispersion),colorscale:'Viridis',cmin:0,cmax:1,showscale:source===sources[0],colorbar:{{title:'Utterance<br>dispersion'}},line:{{color:colors[source],width:1}}}},hovertemplate:'%{{customdata[0]}}<br>degree = %{{x}}<br>betweenness = %{{y:.4f}}<br>dispersion = %{{customdata[4]:.3f}}<br>words = %{{customdata[2]}}<extra></extra>'}});
 }}
 Plotly.react('mattr',tracesM,{{xaxis:{{title:'PaCMAP dimension 1',zeroline:false}},yaxis:{{title:'PaCMAP dimension 2',zeroline:false}},margin:{{t:12,r:75,b:60,l:65}},showlegend:sources.length>1,legend:{{orientation:'h',y:1.12}}}},{{responsive:true}});
 Plotly.react('dispersion',tracesD,{{xaxis:{{title:'Degree (distinct neighbors)',rangemode:'tozero'}},yaxis:{{title:'Normalized betweenness',rangemode:'tozero'}},margin:{{t:12,r:75,b:60,l:65}},showlegend:sources.length>1,legend:{{orientation:'h',y:1.12}}}},{{responsive:true}});
 document.getElementById('summary').textContent=sources.map(s=>s==='books'?'Books: ':'Movies: ')+sources.map(s=>data[s][rule].length+' characters with at least 100 content tokens and 5 dialogue lines').join(' · ')+'. Rule: '+{_safe_json(RULE_LABELS)}[rule]+'.';
}}
for(const id of ['rule','medium'])document.getElementById(id).addEventListener('change',update);update();
</script></body></html>"""
    (OUT / "big_graph1.html").write_text(document, encoding="utf-8")


def _write_big_graph_2(data):
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Big Graph 2 · Community language similarity</title><script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>*{{box-sizing:border-box}}body{{font:15px system-ui,sans-serif;color:#202832;margin:0}}header{{padding:17px 20px 10px}}h1{{font-size:23px;margin:0 0 7px}}header p{{color:#596775;max-width:1150px;margin:5px 0}}.controls{{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin:12px 0}}select{{font:inherit;padding:6px}}main{{padding:0 14px 20px}}.layout{{display:grid;grid-template-columns:1.2fr 1fr;gap:12px}}.panel{{border:1px solid #dce3e9;border-radius:8px;overflow:hidden}}.panel h2{{font-size:16px;margin:0;padding:10px 12px;border-bottom:1px solid #e7ebef}}#network{{height:610px}}#similarity{{height:430px}}.legend{{display:flex;align-items:center;gap:10px;padding:8px 12px;color:#596775;font-size:12px}}.bar{{height:12px;width:180px;background:linear-gradient(90deg,#2166ac,#f7f7f7,#b2182b)}}.note{{font-size:12px;color:#657482;padding:0 13px 12px}}@media(max-width:950px){{.layout{{grid-template-columns:1fr}}#network{{height:65vh;min-height:450px}}}}</style></head><body>
<header><h1>Big Graph 2 · Communities and language similarity</h1>
<p>Communities are detected from interaction edges, without using language. Each node’s color compares its average TF-IDF cosine similarity with other characters in its community against its similarity to characters outside the community. The distribution chart compares all eligible within-community and between-community character pairs.</p>
<div class="controls"><label>Corpus: <select id="medium"><option value="books">Books</option><option value="movies">Movies</option></select></label><label>Edge basis: <select id="rule">{''.join(f'<option value="{r}">{RULE_LABELS[r]}</option>' for r in RULES)}</select></label></div>
<div id="summary"></div></header><main><div class="layout"><section class="panel"><h2>Interaction communities · color = within minus between similarity</h2><div class="legend"><span>Less similar to community</span><span class="bar"></span><span>More similar to community</span></div><div id="network"></div><p class="note">Node placement groups Louvain communities for readability. Positions do not encode language similarity. Hover for within/outside similarity and degree.</p></section><section class="panel"><h2>Pairwise TF-IDF cosine similarity</h2><div id="similarity"></div><p class="note">Character names and common English words are excluded from TF-IDF. Name-component filtering can also remove ordinary words such as “black” or “wood.” This is an exploratory comparison; scene/topic overlap can make voices look similar.</p></section></div></main>
<script>
const data={_safe_json(data)};
const nodes=new vis.DataSet([]),edges=new vis.DataSet([]);
const network=new vis.Network(document.getElementById('network'),{{nodes,edges}},{{physics:{{enabled:false}},interaction:{{hover:true,dragNodes:false}},nodes:{{shape:'dot',font:{{size:12}},scaling:{{min:8,max:32}}}},edges:{{color:{{color:'#a6afb8'}},scaling:{{min:1,max:6}},smooth:{{type:'continuous'}}}}}});
function update(){{
 const medium=document.getElementById('medium').value,rule=document.getElementById('rule').value,s=data[medium][rule];
 const deltas=s.nodes.map(n=>Math.abs(n.similarity_gap||0)),max=Math.max(0,...deltas);
 const mapped=s.nodes.map(n=>{{const g=n.similarity_gap,ratio=g===null||!max?0.5:0.5+0.5*g/max;const color=ratio<0.5?`rgb(${{Math.round(33+(247-33)*ratio*2)}},${{Math.round(102+(247-102)*ratio*2)}},${{Math.round(172+(247-172)*ratio*2)}})`: `rgb(${{Math.round(247+(178-247)*(ratio-.5)*2)}},${{Math.round(247+(24-247)*(ratio-.5)*2)}},${{Math.round(247+(43-247)*(ratio-.5)*2)}})`;return {{...n,value:Math.max(1,n.degree),color:{{background:g===null?'#b8bec5':color,border:'#596775'}},title:n.label+'<br>Community '+(n.community+1)+'<br>Within-community cosine: '+(n.within_similarity===null?'not available':n.within_similarity.toFixed(3))+'<br>Outside-community cosine: '+(n.outside_similarity===null?'not available':n.outside_similarity.toFixed(3))+'<br>Difference: '+(g===null?'not available':g.toFixed(3))+'<br>Degree: '+n.degree}};}});
 nodes.clear();nodes.add(mapped);edges.clear();edges.add(s.edges.map(e=>({{...e,value:e.weight,title:e.weight+' interaction evidence occurrences'}})));network.fit();
 const stats=s.metrics,delta=stats.mean_within===null||stats.mean_between===null?null:stats.mean_within-stats.mean_between;
 document.getElementById('summary').textContent=stats.nodes+' nodes · '+stats.edges+' edges · '+stats.communities+' communities · mean within similarity '+(stats.mean_within===null?'n/a':stats.mean_within.toFixed(3))+' · mean between similarity '+(stats.mean_between===null?'n/a':stats.mean_between.toFixed(3))+' · gap '+(delta===null?'n/a':delta.toFixed(3));
 Plotly.react('similarity',[{{y:s.within,name:'Within same community',type:'violin',box:{{visible:true}},meanline:{{visible:true}},points:'outliers',line:{{color:'#176b87'}}}},{{y:s.between,name:'Across communities',type:'violin',box:{{visible:true}},meanline:{{visible:true}},points:'outliers',line:{{color:'#b64a3f'}}}}],{{yaxis:{{title:'Character-pair TF-IDF cosine',range:[0,1]}},xaxis:{{title:'Pair group'}},violinmode:'group',margin:{{t:15,r:15,b:55,l:65}},showlegend:false}},{{responsive:true}});
}}
for(const id of ['medium','rule'])document.getElementById(id).addEventListener('change',update);update();
</script></body></html>"""
    (OUT / "big_graph2.html").write_text(document, encoding="utf-8")


def _write_big_graph_3(data):
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Big Graph 3 · Adaptation and network retention</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>*{{box-sizing:border-box}}body{{font:15px system-ui,sans-serif;color:#202832;margin:0}}header{{padding:18px 22px}}h1{{font-size:23px;margin:0 0 8px}}header p{{max-width:1150px;color:#596775;margin:6px 0}}.controls{{display:flex;gap:20px;align-items:center;flex-wrap:wrap;margin:14px 0}}select{{font:inherit;padding:6px}}main{{padding:0 14px 20px}}.panel{{border:1px solid #dce3e9;border-radius:8px;overflow:hidden}}.panel h2{{font-size:16px;margin:0;padding:10px 12px;border-bottom:1px solid #e7ebef}}#plot{{height:650px}}.note{{font-size:12px;color:#657482;padding:0 13px 12px}}@media(max-width:800px){{#plot{{height:520px}}}}</style></head><body>
<header><h1>Big Graph 3 · Network and language across adaptations</h1>
<p>For matched characters, does retaining the same interaction neighbors across book and film go with greater book–film language similarity? Each point is one character. Neighbor retention averages Jaccard overlap across installments where the character has dialogue in both media.</p>
<div class="controls"><label>Edge basis: <select id="rule">{''.join(f'<option value="{r}">{RULE_LABELS[r]}</option>' for r in RULES)}</select></label><span>Point size = smaller book/film word count · color = film-minus-book betweenness change.</span></div><p id="summary"></p></header>
<main><section class="panel"><h2>Neighbor retention vs. book–film TF-IDF cosine</h2><div id="plot"></div><p class="note">Network neighborhoods are compared only among characters present in both media within each installment. Book–film TF-IDF uses each character’s dialogue from installments where they have text in both media, with character names and common words excluded. These are matched character-level comparisons, not matched-scene comparisons; token thresholds and aliases limit coverage.</p></section></main>
<script>
const data={_safe_json(data)};
function pearson(x,y){{if(x.length<2)return 0;const mx=x.reduce((a,b)=>a+b,0)/x.length,my=y.reduce((a,b)=>a+b,0)/y.length;let n=0,a=0,b=0;for(let i=0;i<x.length;i++){{n+=(x[i]-mx)*(y[i]-my);a+=(x[i]-mx)**2;b+=(y[i]-my)**2;}}return a&&b?n/Math.sqrt(a*b):0;}}
function update(){{
 const rule=document.getElementById('rule').value,points=data[rule],x=points.map(p=>p.neighbor_retention),y=points.map(p=>p.language_cosine),r=pearson(x,y);
 const trace={{x,y,customdata:points.map(p=>[p.name,p.betweenness_change,p.book_words,p.movie_words,p.installment_count,p.shared_installments.join(', ')]),mode:'markers',type:'scatter',marker:{{size:points.map(p=>Math.max(8,Math.min(28,6+Math.sqrt(p.minimum_words)/3))),color:points.map(p=>p.betweenness_change),colorscale:'RdBu',reversescale:true,showscale:true,colorbar:{{title:'Betweenness<br>change'}},line:{{color:'#465563',width:.7}}}},hovertemplate:'%{{customdata[0]}}<br>neighbor Jaccard = %{{x:.3f}}<br>book–film TF-IDF cosine = %{{y:.3f}}<br>betweenness change = %{{customdata[1]:.4f}}<br>book/film words = %{{customdata[2]}} / %{{customdata[3]}}<br>installments = %{{customdata[5]}}<extra></extra>'}};
 Plotly.react('plot',[trace],{{xaxis:{{title:'Interaction-neighbor retention (mean Jaccard)',range:[-0.03,1.03]}},yaxis:{{title:'Book–film TF-IDF cosine similarity',range:[-0.03,1.03]}},margin:{{t:12,r:85,b:70,l:75}},showlegend:false}},{{responsive:true}});
 document.getElementById('summary').textContent=points.length+' matched characters · Pearson correlation between neighbor retention and language cosine = '+r.toFixed(3)+' · '+{_safe_json(RULE_LABELS)}[rule]+'. Descriptive only; characters are not independent story samples.';
}}
document.getElementById('rule').addEventListener('change',update);update();
</script></body></html>"""
    (OUT / "big_graph3.html").write_text(document, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    graph1, graph2, graph3, counts = build_data()
    _write_big_graph_1(graph1)
    _write_big_graph_2(graph2)
    _write_big_graph_3(graph3)
    (OUT / "summary.json").write_text(json.dumps({"counts": counts, "big_graph_1": {m: {r: len(v) for r, v in rs.items()} for m, rs in graph1.items()}, "big_graph_3_matched_characters": {r: len(v) for r, v in graph3.items()}}, indent=2), encoding="utf-8")
    print("Built Big Graphs 1–3 in graphs/research")


if __name__ == "__main__":
    main()
