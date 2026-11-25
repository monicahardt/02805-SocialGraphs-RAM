"""Compare Harry's question rates with Ron and Hermione in books and films."""

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.extract_book_dialogue import ALIAS_PATTERN, ALIAS_TO_NAME, BOOK_TITLES, QUOTE_RE, chapter_sections

from .dialogue_csv_reader import read_dialogue_records
from .interaction_graph import NON_SPECIFIC_MOVIE_LABELS
from .word_counts import tokenize

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "analysis" / "partner_questions"
INPUTS = {
    "Book": ROOT / "book_data" / "all_books.csv",
    "Film": ROOT / "movie_data" / "all_movies.csv",
}
HARRY = "Harry Potter"
PARTNERS = ("Ron Weasley", "Hermione Granger")
ALTERNATING = "Same speaker before and after"
NEXT = "Next speaker"
TAGGED = "Same speaker before and after, named tags"
EXCLUDED = NON_SPECIFIC_MOVIE_LABELS | {"unknown", "wakanda"}
SPEECH_TAG = "said|asked|replied|whispered|shouted|yelled|cried|muttered|answered|exclaimed|snapped|added|continued|told"
NAMED_TAG = rf"(?:(?P<name_first>{ALIAS_PATTERN})\s+(?:{SPEECH_TAG})\b|\b(?:{SPEECH_TAG})\s+(?P<verb_first>{ALIAS_PATTERN})(?!\w))"
AFTER_QUOTE = re.compile(r"^[\s,]*" + NAMED_TAG, re.I)
BEFORE_QUOTE = re.compile(r"(?:^|[.!?]\s+)\s*" + NAMED_TAG + r"[\s,:]*$", re.I)


@dataclass
class Turn:
    speaker: str
    records: list[int]
    parts: list[str]

    @property
    def text(self):
        return " ".join(self.parts)


def blocks(records, aliases):
    """Build uninterrupted blocks, merging consecutive rows by one speaker."""
    segment = None
    block = []
    for index, record in enumerate(records, 1):
        key = (record.book, record.chapter)
        speaker = aliases.get(record.character, record.character)
        excluded = (
            speaker.casefold() in EXCLUDED
            or record.confidence.casefold() in {"low", "unattributed"}
            or not tokenize(record.line)
        )
        if block and (key != segment or excluded):
            yield segment, block
            block = []
        segment = key
        if excluded:
            continue
        if block and block[-1].speaker == speaker:
            block[-1].records.append(index)
            block[-1].parts.append(record.line)
        else:
            block.append(Turn(speaker, [index], [record.line]))
    if block:
        yield segment, block


def named_tag(paragraph, quotes, index, aliases):
    quote = quotes[index]
    previous_end = quotes[index - 1].end() if index else 0
    next_start = quotes[index + 1].start() if index + 1 < len(quotes) else len(paragraph)
    before = BEFORE_QUOTE.search(paragraph[previous_end:quote.start()])
    after = AFTER_QUOTE.search(paragraph[quote.end():next_start])
    names = {
        ALIAS_TO_NAME[(match["name_first"] or match["verb_first"]).casefold()]
        for match in (before, after) if match
    }
    canonical = {aliases.get(name, name) for name in names}
    return next(iter(canonical)) if len(canonical) == 1 else None


def book_tags(records, aliases):
    """Re-align the original OCR quotes to the CSV and collect immediate named tags."""
    tags = {}
    index = 0
    for installment in BOOK_TITLES:
        raw = (ROOT / "book_data" / f"Book{installment}.txt").read_text(encoding="utf-8-sig")
        for chapter, body in chapter_sections(raw):
            for paragraph in re.split(r"(\n\s*\n)", body):
                paragraph = re.sub(r"\s+", " ", paragraph).strip()
                quotes = list(QUOTE_RE.finditer(paragraph))
                for position, quote in enumerate(quotes):
                    speech = (quote.group(1) or quote.group(2) or "").strip()
                    if not speech:
                        continue
                    record = records[index]
                    if (record.chapter, record.line) != (chapter, speech):
                        raise ValueError(f"Book dialogue row {index + 1} differs from the original quotation.")
                    index += 1
                    tags[index] = named_tag(paragraph, quotes, position, aliases)
    if index != len(records):
        raise ValueError("The raw book quotations do not cover the combined dialogue file.")
    return tags


def measure(aliases):
    counts = defaultdict(Counter)
    film_questions_by_segment = Counter()
    for medium, path in INPUTS.items():
        records = read_dialogue_records(path)
        tags = book_tags(records, aliases) if medium == "Book" else {}
        for (title, chapter), turns in blocks(records, aliases):
            installment = next(i for i, book in BOOK_TITLES.items() if title.startswith(book))
            for index, turn in enumerate(turns[:-1]):
                following = turns[index + 1]
                partner = following.speaker
                if turn.speaker != HARRY or partner not in PARTNERS:
                    continue
                rules = [NEXT]
                if index and turns[index - 1].speaker == partner:
                    rules.append(ALTERNATING)
                    if tags and all(
                        any(tags[row] == item.speaker for row in item.records)
                        for item in (turns[index - 1], turn, following)
                    ):
                        rules.append(TAGGED)
                question_marks = turn.text.count("?")
                if medium == "Film" and question_marks and ALTERNATING in rules:
                    film_questions_by_segment[installment, partner, chapter] += 1
                for rule in rules:
                    bucket = counts[medium, installment, rule, partner]
                    bucket["turns"] += 1
                    bucket["words"] += len(tokenize(turn.text))
                    bucket["question_turns"] += bool(question_marks)
                    bucket["question_marks"] += question_marks

    rows = []
    for (medium, installment, rule, partner), values in sorted(counts.items()):
        rows.append({
            "medium": medium, "installment": installment, "rule": rule,
            "speaker": HARRY, "candidate_partner": partner, **values,
            "question_turn_percentage": 100 * values["question_turns"] / values["turns"],
        })
    return rows, film_questions_by_segment


def draw(rows):
    stats = {
        (r["medium"], r["installment"], r["candidate_partner"]): r
        for r in rows if r["rule"] == ALTERNATING
    }
    fig, ax = plt.subplots(figsize=(10.7, 5.9))
    fig.subplots_adjust(left=.245, right=.965, top=.76, bottom=.285)
    fig.text(.245, .93, "Harry’s questions shift toward Hermione on screen",
             fontsize=19, weight="bold", color="#202b3b")
    fig.text(.245, .877, "Share of Harry’s likely turns with a question mark, compared across matched installments.",
             color="#536176", fontsize=10)
    blue, orange = "#276788", "#b14c30"
    for i in BOOK_TITLES:
        book_ron, book_hermione = [stats["Book", i, person] for person in PARTNERS]
        film_ron, film_hermione = [stats["Film", i, person] for person in PARTNERS]
        book = book_hermione["question_turn_percentage"] - book_ron["question_turn_percentage"]
        film = film_hermione["question_turn_percentage"] - film_ron["question_turn_percentage"]
        ax.plot([book, film], [i, i], color="#bdc7d1", lw=2, zorder=1)
        ax.scatter(book, i, color=blue, s=46, zorder=3)
        low_count = min(film_hermione["turns"], film_ron["turns"]) < 10
        ax.scatter(film, i, facecolor="white" if low_count else orange, edgecolor=orange,
                   marker="D", s=47, linewidth=1.2, zorder=4)
        ax.text(40, i, f'{film_hermione["turns"]} / {film_ron["turns"]}',
                va="center", fontsize=8.5, color="#536176")
    ax.text(40, .48, "Film turns¹", color="#536176", fontsize=8.5)
    ax.set_xlim(-20, 52)
    ax.set_ylim(7.6, .32)
    ax.set_xticks([-20, -10, 0, 10, 20, 30], ["−20", "−10", "0", "+10", "+20", "+30"])
    ax.axvline(37, color="#e1e6ee", lw=.9)
    ax.set_yticks(list(BOOK_TITLES), [title.removeprefix("Harry Potter and the ") for title in BOOK_TITLES.values()])
    ax.tick_params(axis="y", length=0, pad=10)
    ax.grid(axis="x", color="#eceff3")
    ax.set_axisbelow(True)
    ax.axvline(0, color="#8794a6", ls="--", lw=1.1)
    ax.set_xlabel("Question-turn rate with Hermione minus with Ron (percentage points)", labelpad=8)
    for edge in ("top", "left", "right"):
        ax.spines[edge].set_visible(False)
    ax.spines["bottom"].set_color("#b2bdcb")
    ax.scatter([], [], color=blue, label="Book")
    ax.scatter([], [], color=orange, marker="D", label="Film")
    ax.scatter([], [], facecolor="white", edgecolor=orange, marker="D", label="Film with fewer than 10 turns for either partner")
    ax.legend(loc="upper left", bbox_to_anchor=(-.01, -.23), frameon=False, ncol=3, fontsize=8.7)
    fig.text(.245, .046,
             "¹ Eligible film turns with Hermione / Ron. A likely partner speaks on both sides of Harry’s turn.\n"
             "A turn is a run of consecutive lines by one named speaker. The final two films form installment seven.",
             color="#536176", fontsize=8.3)
    for ext in ("png", "svg"):
        fig.savefig(OUTPUT / f"harry_question_gap.{ext}", dpi=200)
    plt.close(fig)


def main():
    aliases = json.loads((ROOT / "results" / "vocabulary_map" / "metrics.json").read_text(encoding="utf-8"))["aliases"]
    rows, film_segments = measure(aliases)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "question_rates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    draw(rows)
    totals = defaultdict(Counter)
    for row in rows:
        totals[row["medium"], row["rule"], row["candidate_partner"]].update(
            turns=row["turns"], questions=row["question_turns"], words=row["words"], marks=row["question_marks"]
        )
    for key, values in totals.items():
        print(*key, dict(values))
    print("Azkaban, Harry and Hermione:",
          {chapter: n for (i, partner, chapter), n in film_segments.items() if i == 3 and partner == PARTNERS[1]})
    print(f"Wrote figure and table to {OUTPUT}")


if __name__ == "__main__":
    main()
