"""Build the directed, source-backed conversation map."""

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .dialogue_csv_reader import read_dialogue_records
from .dialogue_partners import BOOK_TITLES, HARRY, INPUTS, OUTPUT, ROOT, blocks
from .interaction_graph import _js_json
from .word_counts import tokenize

RON, HERMIONE = "Ron Weasley", "Hermione Granger"
CAST_SIZE = 15
SKIP = set("""
a an the and or but to in of on at by for from as with into up out if then than
i me my you your we our he his she her it its they their us him them mine yours
what which who where when why how whose am is was are were be being been does
do did done have has had can could should would shall will must might may
not no yes well now here there just all any some each either only also
this that these those so very much more most got get go going gone get tell
look know see said say come thing things something really alright think let
mean means doing saw please suppose remember telling probably want make right anyway
wouldn't couldn't shouldn't don't didn't can't isn't wasn't haven't aren't maybe
yeah oh ah um hey mm ooh okay
""".split())


@dataclass(frozen=True)
class Exchange:
    installment: int
    source: str
    segment: str
    speaker: str
    partner: str
    start: int
    text: str

    @property
    def question(self):
        return "?" in self.text


def load_exchanges():
    aliases = json.loads((ROOT / "results" / "vocabulary_map" / "metrics.json").read_text(
        encoding="utf-8"
    ))["aliases"]
    by_medium = {}
    for medium, path in INPUTS.items():
        rows = []
        for (title, segment), turns in blocks(read_dialogue_records(path), aliases):
            installment = next(i for i, name in BOOK_TITLES.items() if title.startswith(name))
            for before, spoken, after in zip(turns, turns[1:], turns[2:]):
                if before.speaker == after.speaker:
                    rows.append(Exchange(installment, title, segment, spoken.speaker,
                                         before.speaker, spoken.records[0], spoken.text))
        by_medium[medium] = rows
    return by_medium


def choose_cast(exchanges):
    participation = {
        medium: Counter(name for row in rows for name in (row.speaker, row.partner))
        for medium, rows in exchanges.items()
    }
    shared = set.intersection(*(set(counts) for counts in participation.values()))
    return sorted(shared, key=lambda name: (
        -min(counts[name] for counts in participation.values()), name,
    ))[:CAST_SIZE]


def node_positions(cast):
    center = {
        HARRY: (468, 278), HERMIONE: (631, 420), RON: (306, 420),
    }
    others = [name for name in cast if name not in center]
    for i, name in enumerate(others):
        angle = math.tau * i / len(others) - math.pi / 2
        center[name] = (round(470 + 355*math.cos(angle), 1), round(350 + 265*math.sin(angle), 1))
    return [{"name":name, "x":center[name][0], "y":center[name][1]} for name in cast]


def terms(row):
    own_names = set(tokenize(f"{row.speaker} {row.partner}"))
    return Counter(
        word for word in tokenize(row.text)
        if len(word) >= 3 and word not in SKIP
        and "'" not in word and word not in own_names
    )


def excerpt(row):
    text = " ".join(row.text.split())
    if len(text) > 230:
        text = text[:229].rsplit(" ", 1)[0] + "…"
    part = " · Part " + row.source[-1] if row.source.endswith(("Part 1", "Part 2")) else ""
    return {"text":text, "chapter":row.segment, "installment":row.installment,
            "part":part, "record":row.start}


def describe(rows, phrase_docs, token_counts):
    counts = sum((token_counts[row] for row in rows), Counter())
    vocabulary = sorted(
        (word for word, support in counts.items() if support >= 2),
        key=lambda word: (
            -counts[word] * math.log(1 + phrase_docs[0] / phrase_docs[1][word]), word,
        ),
    )[:5]
    ranking = sorted(
        rows,
        key=lambda row: (
            -sum(word in token_counts[row] for word in vocabulary),
            abs(len(row.text) - 86), row.start,
        ),
    )
    examples = []
    chapters = set()
    for row in ranking:
        place = row.source, row.segment
        if place in chapters:
            continue
        chapters.add(place)
        examples.append(excerpt(row))
        if len(examples) == 3:
            break
    return {"words": [[word, counts[word]] for word in vocabulary], "examples":examples}


def summarize_scope(rows, ids, token_counts):
    pairs = defaultdict(list)
    for row in rows:
        pairs[ids[row.speaker], ids[row.partner]].append(row)
    questions = {pair: [r for r in group if r.question] for pair, group in pairs.items()}
    answer = {pair: {"a":pair[0], "b":pair[1], "turns":len(group),
                     "questions":len(questions[pair])} for pair, group in pairs.items()}
    for mode, groups in (("all", pairs), ("question", questions)):
        df = Counter()
        for group in groups.values():
            df.update({word for row in group for word in token_counts[row]})
        docs = sum(bool(group) for group in groups.values())
        for pair, group in groups.items():
            answer[pair][mode] = describe(group, (docs, df), token_counts)
    return list(answer.values())


def main():
    exchanges = load_exchanges()
    cast = choose_cast(exchanges)
    ids = {name: i for i, name in enumerate(cast)}
    data = {}
    for medium, rows in exchanges.items():
        shown = [row for row in rows if row.speaker in ids and row.partner in ids]
        token_counts = {row: terms(row) for row in shown}
        data[medium] = {
            str(i): summarize_scope(
                [row for row in shown if i == 0 or row.installment == i], ids, token_counts
            )
            for i in (0, *BOOK_TITLES)
        }

    def example(medium, i, name):
        return next(r for r in data[medium][str(i)] if (r["a"], r["b"]) == (ids[HARRY], ids[name]))

    for medium, name, totals in (
        ("Book", HERMIONE, (707, 243)), ("Film", HERMIONE, (195, 76)),
        ("Book", RON, (919, 328)), ("Film", RON, (180, 44)),
    ):
        record = example(medium, 0, name)
        assert (record["turns"], record["questions"]) == totals
    third = example("Film", 3, HERMIONE)
    assert (third["turns"], third["questions"]) == (25, 11)

    payload = {
        "nodes":node_positions(cast),
        "data":data,
        "installments":{i: name.removeprefix("Harry Potter and the ") for i, name in BOOK_TITLES.items()},
    }
    template = (ROOT / "code" / "templates" / "conversation_map.html").read_text(encoding="utf-8")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "conversation_map.html"
    destination.write_text(template.replace("__CONVERSATION_DATA__", _js_json(payload)), encoding="utf-8")
    print(f"Wrote {len(cast)} characters and eight installment views per medium to {destination}")


if __name__ == "__main__":
    main()
