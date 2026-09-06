# Social Graphs Assemble · Group RAM

A static GitHub Pages site for 02805 Social Graphs. The first post explores the frozen week-one Marvel Wikipedia network with degree rankings, linear/log–log distributions, and a network map including all isolates.

## Reproduce

The graphs are authored in **[notebooks/week1.ipynb](notebooks/week1.ipynb)** using Python, NetworkX, and Plotly. Open it in Jupyter or VS Code and run all cells to regenerate the interactive HTML figures in `assets/plots/`.

To build everything from the command line with Python 3.13:

```sh
python -m venv .venv
# Activate: .venv\Scripts\Activate.ps1 on Windows; source .venv/bin/activate on macOS/Linux
python -m pip install -r requirements.txt
python scripts/build_notebook.py
python -m http.server 8000
```

Open http://localhost:8000 to preview the homepage and post. The build executes the notebook, exports its four interactive figures, and creates `notebooks/published/week1.html` with the code hidden, plus an executed notebook for inspection. Generated outputs are ignored by Git and rebuilt during deployment. The source notebook remains editable and downloadable.

For notebook editing, use VS Code's Jupyter extension with the `.venv` Python interpreter, or install JupyterLab separately. The website serves exported HTML and JavaScript; visitors do not need Python or a notebook server. Hover, zoom, and Plotly axis buttons work on GitHub Pages; Python callbacks would need a live server.

The original independent JavaScript analysis remains available with `node scripts/analyze.mjs` (Node.js 18+). It writes `assets/week1-analysis.json` and can be used to cross-check the Python results.

The full roster is loaded before adding edges, preserving isolates. Both analyses validate node IDs, duplicate edges, and endpoints.

## Interactive experiments

The week-one page has fixed section navigation, a selectable network map, directed or undirected shortest routes, and a node-removal experiment. Selecting a scatter point, lasso group, degree bucket, or ranking bar also highlights those characters on the map.

The notebook exports `assets/plots/explorer.json`: seeded coordinates, the directed graph, and robustness curves. Targeted removal ranks once by **original in-degree**, breaking ties by node ID. The random reference uses 30 permutations with seeds 42–71; the band shows their min–max, not a confidence interval. The displayed random experiment uses seed 42. Metrics count surviving nodes and weak components in the entire roster. Map positions and node sizes stay fixed to their original values. The scatter, distributions, and rankings remain views of the original snapshot.

Browser interactions use `assets/explorer.js` and the small graph functions in `assets/network-model.mjs`; Python still produces the notebook figures, graph data, and comparison curves. After building, run `node scripts/test-network-model.mjs` to check paths and compare every slider step against the NetworkX results. GitHub Actions runs this check before publishing. The experiment can be downloaded as JSON with its settings, removed IDs, metrics, and route.

## Publish

In the repository's **Settings → Pages**, an administrator should choose **GitHub Actions** as the source. Push changes to `main`; the included workflow executes the notebook and deploys the full site, including the interactive figures, week-one post, and notebook HTML. A notebook error fails the build before deployment.

The `github-pages` environment must allow deployments from `main`. After the **Deploy site to GitHub Pages** workflow succeeds in the Actions tab, refresh the site to see the update.

Expected URL after successful deployment: https://monicahardt.github.io/02805-SocialGraphs-RAM/

Publication has not been verified. The group should add member names if desired, review the post, share the live link in Teams by Monday evening, and leave constructive feedback on another group's post.

## Data and interpretation

The supplied TSV headers identify the frozen release as 2026-08-26. Source: https://sunelehmann.com/socialgraphs2026-web/data/

303 nodes, 1,784 directed edges; weak components have sizes 277, 9, and seventeen singletons. Links describe article references within this roster, not social relationships. The site includes original decorative superhero-inspired illustrations; the measured network appears in the week-one post.
