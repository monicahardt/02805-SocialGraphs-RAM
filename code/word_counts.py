"""Count dialogue words overall and for each character within a book or film.

Example from the repository root:
    python -m code.word_counts book_data/all_books.csv --output-dir results/word_counts/books

Words are lowercase English letter sequences, including internal apostrophes.
Curly apostrophes are normalized; punctuation, numbers and hyphens separate words.
Names, common words, stage-direction words and Unknown-speaker dialogue are kept.
No stemming, stopword removal or speaker relabeling is performed.
"""

import argparse
import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

if __package__:
    from .dialogue_csv_reader import DialogueRecord, read_dialogue_records
else:
    from dialogue_csv_reader import DialogueRecord, read_dialogue_records


WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")


def tokenize(text: str) -> list[str]:
    """For example, 'Harry's wand!' becomes [\"harry's\", 'wand']."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return WORD_RE.findall(text.lower())


def count_words(records: list[DialogueRecord]) -> tuple[Counter, dict, Counter]:
    """Return corpus counts, counts per (source, character), and utterance totals.

    Keeping each source separate allows later methods to use a character's
    dialogue within one book/film as a document.
    """
    overall = Counter()
    by_character = {}
    utterances = Counter()
    for record in records:
        key = (record.book, record.character)
        tokens = tokenize(record.line)
        overall.update(tokens)
        by_character.setdefault(key, Counter()).update(tokens)
        utterances[key] += 1
    return overall, by_character, utterances


def sorted_counts(counts: Counter) -> list[tuple[str, int]]:
    """Highest count first, with alphabetical ordering for ties."""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def write_counts(records: list[DialogueRecord], output_dir: Path) -> Counter:
    overall, by_character, utterances = count_words(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_words = overall.total()

    with (output_dir / "word_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["word", "count", "per_1000_words"])
        for word, count in sorted_counts(overall):
            writer.writerow([word, count, round(1000 * count / total_words, 6)])

    with (output_dir / "character_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "character", "utterances", "words", "unique_words"])
        for (source, character), counts in sorted(by_character.items()):
            writer.writerow([source, character, utterances[source, character], counts.total(), len(counts)])

    with (output_dir / "character_word_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "character", "word", "count", "per_1000_words"])
        for (source, character), counts in sorted(by_character.items()):
            for word, count in sorted_counts(counts):
                writer.writerow([source, character, word, count, round(1000 * count / counts.total(), 6)])

    print(f"Dialogue rows: {len(records):,}")
    print(f"Word occurrences: {total_words:,}; distinct words: {len(overall):,}")
    print("Most frequent words: " + ", ".join(f"{word} ({count:,})" for word, count in sorted_counts(overall)[:10]))
    print(f"Wrote counts to {output_dir}")
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=Path("book_data/all_books.csv"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or Path("results/word_counts") / args.csv.stem
    write_counts(read_dialogue_records(args.csv), output_dir)


if __name__ == "__main__":
    main()
