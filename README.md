# Social Graphs Assemble · Group RAM

A static GitHub Pages site for 02805 Social Graphs. The first post explores the frozen week-one Marvel Wikipedia network with degree rankings, linear/log–log distributions, and a network map including all isolates.

## Reproduce

Install Node.js 18 or later, then run `node scripts/analyze.mjs`. No npm packages are needed. The script reads the full node roster before adding edges, validates endpoints and uniqueness, and writes `assets/week1-analysis.json`.

Serve the repository with any local HTTP server to view `index.html` and `week1.html`. Opening the files directly will prevent the charts from fetching their JSON in some browsers.

## Publish

In the repository's **Settings → Pages**, choose **GitHub Actions** as the source. Push these changes to `week1_anton_llm`; the included workflow builds the analysis and deploys the public files. No push to `main` is needed.

If the `github-pages` environment restricts deployment branches, allow `week1_anton_llm` under **Settings → Environments → github-pages → Deployment branches and tags**.

Expected URL after successful deployment: https://monicahardt.github.io/02805-SocialGraphs-RAM/

Publication has not been verified. The group should add member names if desired, review the post, share the live link in Teams by Monday evening, and leave constructive feedback on another group's post.

## Data and interpretation

The supplied TSV headers identify the frozen release as 2026-08-26. Source: https://sunelehmann.com/socialgraphs2026-web/data/

303 nodes, 1,784 directed edges; weak components have sizes 277, 9, and seventeen singletons. Links describe article references within this roster, not social relationships. The site includes original decorative superhero-inspired illustrations; the measured network appears in the week-one post.
