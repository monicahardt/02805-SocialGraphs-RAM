"""Read dialogue CSV files and group dialogue by speaker."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class DialogueLine:
    book: str
    chapter: str
    line: str


@dataclass(frozen=True)
class DialogueRecord:
    book: str
    chapter: str
    character: str
    line: str
    confidence: str = "high"


def _records(csv_path: str | Path) -> tuple[TextIO, csv.DictReader]:
    """Open a UTF-8 CSV and validate its expected columns."""
    handle = Path(csv_path).open("r", encoding="utf-8-sig", newline="")
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        handle.close()
        raise ValueError(f"CSV file is empty: {csv_path}")

    columns = {name.strip().lower() for name in reader.fieldnames if name}
    has_book = "movie" in columns or "book" in columns
    required = {"chapter", "character", "dialog"}
    if not has_book or not required.issubset(columns):
        handle.close()
        raise ValueError(
            "CSV must contain movie/book, chapter, character, and dialog columns: "
            f"{csv_path}"
        )
    return handle, reader


def _column_names(reader: csv.DictReader) -> dict[str, str]:
    """Map normalized names to the actual header spelling."""
    names = {name.strip().lower(): name for name in reader.fieldnames or [] if name}
    if "movie" not in names and "book" in names:
        names["movie"] = names["book"]
    return names


def read_dialogue_records(csv_path: str | Path) -> list[DialogueRecord]:
    """Return valid dialogue rows in their original order."""
    handle, reader = _records(csv_path)
    names = _column_names(reader)
    result: list[DialogueRecord] = []
    try:
        for row in reader:
            character = (row.get(names["character"]) or "").strip()
            if not character:
                continue
            result.append(
                DialogueRecord(
                    book=row.get(names["movie"], "") or "",
                    chapter=row.get(names["chapter"], "") or "",
                    character=character,
                    line=row.get(names["dialog"], "") or "",
                    confidence=(
                        row.get(names["confidence"], "high") or "high"
                    ).strip()
                    if "confidence" in names
                    else "high",
                )
            )
    finally:
        handle.close()
    return result


def read_entries_by_character(
    csv_path: str | Path,
) -> dict[str, list[DialogueLine]]:
    """Group each character's lines with their source book and chapter."""
    entries: dict[str, list[DialogueLine]] = {}
    for row in read_dialogue_records(csv_path):
        entries.setdefault(row.character, []).append(
            DialogueLine(row.book, row.chapter, row.line)
        )
    return entries


def read_lines_by_character(csv_path: str | Path) -> dict[str, list[str]]:
    """Group dialogue text by character, preserving the order in the CSV."""
    return {
        character: [entry.line for entry in entries]
        for character, entries in read_entries_by_character(csv_path).items()
    }
