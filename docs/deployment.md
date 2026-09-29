# Public Streamlit deployment

The public interface is deliberately separated from the research databases.
It reads `data/public_viewer/accepted_experiment_v1.json`, a sanitized snapshot
containing only:

- the verified overview of the accepted 300-question experiment;
- the six curated questions offered by the replay selector;
- the stored responses, reconstructed messages, votes and usage needed to show
  those six debates;
- hashes and version labels that identify the source experiment.

It does **not** contain the OpenRouter key, `.env`, pilot runs, development
runs, database backups, OpenRouter account identifiers or the rest of
`storage/cache.sqlite`. An account identifier embedded in one stored upstream
error remains unchanged in the private audit database but is replaced with a
redaction marker in this presentation artifact. The public app makes no API
calls and cannot modify experimental evidence.

## Safety checks

The original `storage/results.sqlite` and `storage/cache.sqlite` remain
gitignored. Never force-add either file. Before a deployment commit, run:

```bash
.venv/bin/python scripts/check_public_deployment.py
.venv/bin/python -m pytest -q
git diff --check
```

To regenerate the public artifact after an intentional accepted-result change:

```bash
.venv/bin/python scripts/export_public_viewer_bundle.py --force
```

The exporter opens the source databases read-only, hashes them before and after
the export, refuses private strings and writes only the public JSON artifact.
Do not regenerate it merely for presentation edits.

## Test locally

Run from the repository root, matching Streamlit Community Cloud's working
directory:

```bash
.venv/bin/streamlit run app/viewer.py
```

Check the overview, all six curated cases and all nine replay steps. The app
must work even when the private `storage/` directory is unavailable.

## Deploy on Streamlit Community Cloud

1. Commit and push the verified deployment files to `main`.
2. Sign in at <https://share.streamlit.io> and connect the GitHub repository.
3. Create an app from the existing repository.
4. Select branch `main` and entrypoint `app/viewer.py`.
5. In Advanced settings select Python 3.13, matching the tested local runtime.
6. Do not add any secrets; the viewer does not need an API key.
7. Choose a public subdomain and deploy.
8. Open the final URL in a private browser window and repeat the local checks.

The root `requirements.txt` pins the tested Python packages. The root
`.streamlit/config.toml` supplies the presentation theme. Streamlit Community
Cloud runs the entrypoint from the repository root, so these paths match both
local and hosted execution.

## What remains private

Keep these only in protected local/off-machine backups:

- `.env`;
- `storage/results.sqlite`;
- `storage/cache.sqlite`;
- `storage/backups/`;
- any raw operational logs.

The tracked public bundle is a presentation artifact, not a replacement for
those research backups.
