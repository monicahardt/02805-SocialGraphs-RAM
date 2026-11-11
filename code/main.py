"""Generate an interactive character dialogue network from a CSV."""

import argparse
from pathlib import Path

if __package__:
    from .dialogue_csv_reader import read_dialogue_records
    from .interaction_graph import write_html
else:  # Support `python3 code/main.py` from the repository root.
    from dialogue_csv_reader import read_dialogue_records
    from interaction_graph import write_html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=Path("movie_data/hp1.csv"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        name = args.csv.stem
        suffix = "_book_networks.html" if "book_data" in args.csv.parts else "_movie_networks.html"
        output_path = Path("graphs") / f"{name}{suffix}"

    write_html(read_dialogue_records(args.csv), output_path)
    print(f"Wrote character network to {output_path}")


if __name__ == "__main__":
    main()
