"""Combine movie dialogue CSVs and generate the all-movies interaction graph."""

import csv
from pathlib import Path

from interaction_graph import write_html
from dialogue_csv_reader import read_dialogue_records

ROOT = Path(__file__).resolve().parent.parent
MOVIE_DIR = ROOT / "movie_data"
COMBINED_CSV = MOVIE_DIR / "all_movies.csv"
GRAPH_HTML = ROOT / "graphs" / "movies" / "all_movies_networks.html"


def main() -> None:
    movie_files = sorted(MOVIE_DIR.glob("hp*.csv"), key=lambda path: int(path.stem[2:]))
    if not movie_files:
        raise SystemExit(f"No movie dialogue CSV files found in {MOVIE_DIR}")

    records = []
    for path in movie_files:
        records.extend(read_dialogue_records(path))

    with COMBINED_CSV.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["movie", "chapter", "character", "dialog"])
        writer.writerows((row.book, row.chapter, row.character, row.line) for row in records)

    write_html(records, GRAPH_HTML, edge_rule_dashboard=True)
    print(f"Combined {len(records)} dialogue rows from {len(movie_files)} movies into {COMBINED_CSV.relative_to(ROOT)}")
    print(f"Wrote interactive network to {GRAPH_HTML.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
