"""Fail before deployment if a published page references a missing local file."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
missing = []

class Links(HTMLParser):
    def __init__(self, page):
        super().__init__()
        self.page = page

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        for key in ("src", "href"):
            url = urlsplit(attrs.get(key, ""))
            if url.scheme or url.netloc or not url.path:
                continue
            target = (self.page.parent / unquote(url.path)).resolve()
            if not target.is_relative_to(root) or not target.exists():
                missing.append(f"{self.page.relative_to(root)}: {attrs[key]}")

pages = [root / "index.html", root / "week1.html", root / "notebooks/published/week1.html"]
pages += sorted((root / "assets/plots").glob("*.html"))
for page in pages:
    if not page.is_file():
        missing.append(str(page.relative_to(root)))
    else:
        Links(page).feed(page.read_text(encoding="utf-8"))
# These resources are fetched by the explorer rather than declared in HTML.
for resource in ["assets/plots/explorer.json", "assets/plots/discoveries.json", "assets/network-model.mjs"]:
    if not (root / resource).is_file():
        missing.append(resource)
if missing:
    raise SystemExit("Missing published resources:\n" + "\n".join(missing))
print(f"Checked {len(pages)} HTML pages: all local links and chart resources exist.")
