"""Build slide-ready conversation topic figures and their pair-level counts."""

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path

from scripts.extract_book_dialogue import BOOK_TITLES

from .dialogue_csv_reader import read_dialogue_records
from .interaction_graph import NON_SPECIFIC_MOVIE_LABELS
from .topic_conversation_figures import plot_spell_networks, plot_topic_pairs, plot_topic_trends
from .word_counts import tokenize


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "topic_conversations"
SOURCES = {"books": "book_data/all_books.csv", "films": "movie_data/all_movies.csv"}
EXCLUDED = NON_SPECIFIC_MOVIE_LABELS | {"unknown", "wakanda"}
STAGE_DIRECTIONS = re.compile(r"\[[^\[\]]*\]")
NIMBUS_MODEL = re.compile(r"\bnimbus\s+200([01])\b", re.IGNORECASE)


@dataclass
class Turn:
    speaker: str
    rows: list[str]


@dataclass
class PairSummary:
    turns: int = 0
    hits: Counter = field(default_factory=Counter)


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def normalize(text):
    text = NIMBUS_MODEL.sub(lambda match: "Nimbus two thousand and one" if match[1] == "1" else "Nimbus two thousand", text)
    return " ".join(tokenize(text))


def partition_dialogue(records, aliases):
    """Return uninterrupted same-segment blocks and the named cast per installment."""
    casts = defaultdict(set)
    blocks = []
    for (title, _), rows in groupby(records, lambda record: (record.book, record.chapter)):
        story = next(number for number, name in BOOK_TITLES.items() if title.startswith(name))
        turns = []
        for record in rows:
            speaker = aliases.get(record.character, record.character)
            spoken = STAGE_DIRECTIONS.sub(" ", record.line)
            if speaker.casefold() in EXCLUDED or record.confidence.casefold() in {"low", "unattributed"} or not tokenize(spoken):
                if turns:
                    blocks.append((story, turns))
                    turns = []
                continue
            casts[story].add(speaker)
            if turns and turns[-1].speaker == speaker:
                turns[-1].rows.append(spoken)
            else:
                turns.append(Turn(speaker, [spoken]))
        if turns:
            blocks.append((story, turns))
    return blocks, casts


def topic_rules():
    topics = read_json("config/conversation_topics.json")
    spells = read_json("config/spells.json")
    rules = {}
    for topic in topics:
        cues = list(topic["cues"])
        if topic.get("include_spell_inventory"):
            cues += spells["incantations"] + list(spells["references"])
            for variants in ("ocr_variants", "reference_variants", "incantation_transcript_variants"):
                cues.extend(variant for names in spells[variants].values() for variant in names)
        phrases = sorted({normalize(cue) for cue in cues}, key=lambda phrase: (-len(phrase), phrase))
        alternatives = "|".join(map(re.escape, phrases))
        rules[topic["id"]] = re.compile(rf"(?<!\S)(?:{alternatives})(?=(?:'s)?(?: |$))")
    return topics, rules


def cue_matches(turn, rules):
    """Count each topic at most once, with phrases matched within individual rows."""
    rows = [normalize(text) for text in turn.rows]
    return [topic for topic, rule in rules.items() if any(rule.search(row) for row in rows)]


def summarize(blocks, shared_casts, rules):
    summaries = defaultdict(lambda: defaultdict(PairSummary))
    for story, turns in blocks:
        cast = shared_casts[story]
        for before, turn, after in zip(turns, turns[1:], turns[2:]):
            partner = before.speaker
            if partner != after.speaker or turn.speaker not in cast or partner not in cast:
                continue
            pair = tuple(sorted((turn.speaker, partner)))
            summary = summaries[story][pair]
            summary.turns += 1
            summary.hits.update(cue_matches(turn, rules))
    return summaries


def build_comparison():
    aliases = read_json("results/vocabulary_map/metrics.json")["aliases"]
    topics, rules = topic_rules()
    loaded = {medium: partition_dialogue(read_dialogue_records(ROOT / path), aliases) for medium, path in SOURCES.items()}
    shared_casts = {story: loaded["books"][1][story] & loaded["films"][1][story] for story in BOOK_TITLES}
    by_medium = {medium: summarize(blocks, shared_casts, rules) for medium, (blocks, _) in loaded.items()}
    stories = {story: {medium: data[story] for medium, data in by_medium.items()} for story in BOOK_TITLES}
    return topics, stories


def write_pair_data(topics, stories, path):
    topic_ids = [topic["id"] for topic in topics]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["installment", "medium", "character_a", "character_b", "all_turns", *topic_ids])
        for story, media in stories.items():
            for medium, pairs in media.items():
                for (a, b), summary in sorted(pairs.items()):
                    writer.writerow([story, medium, a, b, summary.turns, *(summary.hits[topic] for topic in topic_ids)])


def main():
    topics, stories = build_comparison()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_pair_data(topics, stories, OUTPUT / "topic_conversation_pairs.csv")
    plot_spell_networks(stories, OUTPUT)
    plot_topic_trends(topics, stories, OUTPUT)
    plot_topic_pairs(topics, stories, OUTPUT)
    print(f"Wrote three static figures and pair counts to {OUTPUT}")


if __name__ == "__main__":
    main()
