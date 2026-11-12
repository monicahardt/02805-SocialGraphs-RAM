"""Combine all book dialogue CSVs and generate the all-books interaction graph."""

import csv
import re
from pathlib import Path

if __package__:
    from .dialogue_csv_reader import read_dialogue_records
    from .interaction_graph import write_html
else:  # Support `python3 code/combine_books.py` from the repository root.
    from dialogue_csv_reader import read_dialogue_records
    from interaction_graph import write_html

ROOT = Path(__file__).resolve().parent.parent
BOOK_DIR = ROOT / "book_data"
COMBINED_CSV = BOOK_DIR / "all_books.csv"
GRAPH_HTML = ROOT / "graphs" / "books" / "all_books_networks.html"


def main() -> None:
    book_files = sorted(
        BOOK_DIR.glob("book*_dialogue.csv"),
        key=lambda path: int(re.match(r"book(\d+)", path.stem).group(1)),
    )
    if not book_files:
        raise SystemExit(f"No book dialogue CSV files found in {BOOK_DIR}")

    records = []
    for path in book_files:
        records.extend(read_dialogue_records(path))

    with COMBINED_CSV.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["book", "chapter", "character", "dialog"])
        writer.writerows((row.book, row.chapter, row.character, row.line) for row in records)

    write_html(records, GRAPH_HTML, edge_rule_dashboard=True, collection_label="Book")
    print(
        f"Combined {len(records)} dialogue rows from {len(book_files)} books "
        f"into {COMBINED_CSV.relative_to(ROOT)}"
    )
    print(f"Wrote interactive network to {GRAPH_HTML.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
