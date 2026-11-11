"""Small console utilities for dialogue data."""

import random

if __package__:
    from .dialogue_csv_reader import DialogueLine
else:  # Support importing from a script launched inside the code directory.
    from dialogue_csv_reader import DialogueLine


def print_summary(entries_by_character: dict[str, list[DialogueLine]]) -> None:
    """Print dataset counts and one random line for up to five characters."""
    all_entries = [entry for entries in entries_by_character.values() for entry in entries]
    books = {entry.book for entry in all_entries}
    chapters = {entry.chapter for entry in all_entries}

    print(f"Books: {len(books)}")
    print(f"Chapters: {len(chapters)}")
    print(f"Characters: {len(entries_by_character)}")
    print(f"Lines: {len(all_entries)}")

    sample = random.sample(
        list(entries_by_character), k=min(5, len(entries_by_character))
    )
    for character in sample:
        entry = random.choice(entries_by_character[character])
        print(f"\n{character} [{entry.book} | {entry.chapter}] {entry.line}")
