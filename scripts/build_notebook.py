"""Execute the source notebook and export a readable HTML companion."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
notebook = nbformat.read(ROOT / "notebooks/week1.ipynb", as_version=4)
NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
output = ROOT / "notebooks/published"
output.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, output / "week1.ipynb")
exporter = HTMLExporter()
exporter.exclude_input = True
body, _ = exporter.from_notebook_node(notebook)
(output / "week1.html").write_text(body, encoding="utf-8")
print("Published four interactive figures and notebooks/published/week1.html")
