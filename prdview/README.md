# PRD View

This is an auxiliary MkDocs configuration that serves Markdown files from selected repo paths while preserving directory structure.

## Select what to index

Override scan inputs (repo-relative, space-separated; paths or globs):

- Pass explicit paths:
  - `pixi run bash prdview/refresh-docs-tree.sh <scan_path_or_glob_1> <scan_path_or_glob_2>`
- Or set `PRDVIEW_DIRS`:
  - `PRDVIEW_DIRS="<scan_path_or_glob_1> <scan_path_or_glob_2>" pixi run bash prdview/refresh-docs-tree.sh`
- Or edit `prdview/docview.yml` to change the default scan include/exclude patterns.

If this work dir is **not** inside the git repo, also set the repo root:

- `PRDVIEW_REPO_ROOT="/abs/path/to/repo" pixi run bash prdview/refresh-docs-tree.sh <scan_path_or_glob_1> <scan_path_or_glob_2>`

Defaults (when no args / env override):

- `.`

## Run

From the repo root:

- `pixi run bash prdview/refresh-docs-tree.sh`
- `pixi run mkdocs serve -f prdview/mkdocs.yml`

Notes:
- Staged files live under `prdview/_staged/` and are generated as symlinks by `prdview/refresh-docs-tree.sh`.
- `prdview/scan-files-to-stage.py` discovers Markdown files and image assets referenced by Markdown.
- `prdview/docview.yml` stores the default scan patterns and gitignore policy (`scan.respect_gitignore`, `scan.auto_exclude_generated`, `scan.include_globs`, `scan.force_globs`, `scan.exclude_globs`).
- `prdview/mkdocs.yml` is generated only if missing (existing configs are preserved).
- Bind address/port uses MkDocs defaults unless overridden (typically `127.0.0.1:8000`).
- Mermaid and KaTeX are enabled by default via `pymdown-extensions` + `extra_javascript`/`extra_css` in the generated MkDocs config.
- Mermaid code fences are supported via `pymdownx.superfences` with a `mermaid` custom fence.
- Math is supported via `pymdownx.arithmatex` (generic mode) + KaTeX auto-render.
- Search is enabled via the built-in MkDocs `search` plugin.
