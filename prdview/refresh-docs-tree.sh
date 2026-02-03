\
#!/usr/bin/env bash
set -euo pipefail

# Build a staged tree made of symlinks to repo Markdown files (plus referenced image assets).
#
# Selection:
# - If arguments are provided, those paths/globs (relative to repo root) are indexed.
# - Otherwise, patterns from `docview.yml` are used.
# - You can also override defaults with `PRDVIEW_DIRS` (space-separated list).
#
# Discovery:
# - By default, Markdown discovery respects .gitignore (configurable via `docview.yml`).
# - Referenced image assets from selected Markdown are staged even if gitignored.

work_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

repo_root="${PRDVIEW_REPO_ROOT:-}"
if [[ -z "${repo_root}" && -f "${work_dir}/repo-root.txt" ]]; then
  repo_root="$(cat -- "${work_dir}/repo-root.txt" 2>/dev/null || true)"
fi
if [[ -z "${repo_root}" ]]; then
  repo_root="$(git -C "${work_dir}" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "${repo_root}" ]]; then
  repo_root="$(cd -- "${work_dir}/.." && pwd)"
else
  repo_root="$(cd -- "${repo_root}" && pwd)"
fi

mkdocs_cfg="${work_dir}/mkdocs.yml"
site_dir="${work_dir}/site"
scan_py="${work_dir}/scan-files-to-stage.py"
manifest_yml="${work_dir}/docview.yml"
staging_dir="_staged"
out_dir="${work_dir}/${staging_dir}"

rm -rf -- "${out_dir}"
mkdir -p -- "${out_dir}"
rm -rf -- "${site_dir}"

if [[ ! -f "${manifest_yml}" ]]; then
  echo "prdview: missing manifest: ${manifest_yml}" >&2
  echo "prdview: run the scaffolder to create it (or re-run with --force)" >&2
  exit 1
fi

positional_targets=()
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'USAGE'
Usage: bash prdview/refresh-docs-tree.sh [paths_or_globs...]

Builds a staged symlink tree of Markdown files (plus referenced image assets) from repo paths.

Arguments:
  paths_or_globs...   Optional list of repo-root-relative paths or glob patterns to index. When omitted, `docview.yml` is used.
USAGE
      exit 0
      ;;
    *)
      positional_targets+=("$arg")
      ;;
  esac
done

targets=()
if [[ "${#positional_targets[@]}" -gt 0 ]]; then
  targets=("${positional_targets[@]}")
elif [[ -n "${PRDVIEW_DIRS:-}" ]]; then
  # shellcheck disable=SC2206
  targets=(${PRDVIEW_DIRS})
else
  # Let the manifest decide defaults.
  targets=()
fi

 # Note: targets may be globs; the scanner handles resolution and ignore rules.

# Link all required files (Markdown + referenced image assets).
#
# The scanner prints repo-relative paths. We convert them back to absolute paths under repo_root.
if [[ -f "${scan_py}" ]]; then
  run_in_repo_root=false
  py_cmd=()
  if command -v pixi >/dev/null 2>&1; then
    # Prefer running inside the project's Pixi environment.
    # Pixi resolves the project manifest from the current working directory.
    if (cd -- "${repo_root}" && pixi run python -c 'import sys' >/dev/null 2>&1); then
      py_cmd=(pixi run python)
      run_in_repo_root=true
    fi
  fi

  if [[ "${#py_cmd[@]}" -eq 0 ]]; then
    py_cmd=(python3)
    if ! command -v "${py_cmd[0]}" >/dev/null 2>&1; then
      py_cmd=(python)
    fi
  fi

  scan_args=("${py_cmd[@]}" "${scan_py}" --repo-root "${repo_root}" --exclude "${work_dir}" --print0 --manifest "${manifest_yml}")
  scan_args+=(-- "${targets[@]}")

  if [[ "${run_in_repo_root}" == true ]]; then
    (cd -- "${repo_root}" && "${scan_args[@]}") \
    | while IFS= read -r -d '' rel; do
        src="${repo_root}/${rel}"
        if [[ ! -e "${src}" ]]; then
          echo "prdview: missing staged file: ${rel}" >&2
          continue
        fi
        dest="${out_dir}/${rel}"
        mkdir -p -- "$(dirname -- "${dest}")"
        ln -sf -- "${src}" "${dest}"
      done
  else
    "${scan_args[@]}" \
    | while IFS= read -r -d '' rel; do
        src="${repo_root}/${rel}"
        if [[ ! -e "${src}" ]]; then
          echo "prdview: missing staged file: ${rel}" >&2
          continue
        fi
        dest="${out_dir}/${rel}"
        mkdir -p -- "$(dirname -- "${dest}")"
        ln -sf -- "${src}" "${dest}"
      done
  fi
else
  echo "prdview: missing scanner script: ${scan_py}" >&2
  exit 1
fi

# Provide a stable MkDocs homepage.
if [[ -f "${repo_root}/README.md" ]]; then
  ln -sf -- "${repo_root}/README.md" "${out_dir}/index.md"
fi

# Stage local JS helpers required by mkdocs.yml (Mermaid + KaTeX init).
#
# MkDocs expects extra_javascript paths to exist under docs_dir.
if [[ -d "${work_dir}/javascripts" ]]; then
  mkdir -p -- "${out_dir}/javascripts"
  if [[ -f "${work_dir}/javascripts/mermaid-init.js" ]]; then
    ln -sf -- "${work_dir}/javascripts/mermaid-init.js" "${out_dir}/javascripts/mermaid-init.js"
  fi
  if [[ -f "${work_dir}/javascripts/katex-init.js" ]]; then
    ln -sf -- "${work_dir}/javascripts/katex-init.js" "${out_dir}/javascripts/katex-init.js"
  fi
fi

# Verification: fail if any broken symlinks were created.
broken_links="$(find "${out_dir}" -xtype l -print || true)"
if [[ -n "${broken_links}" ]]; then
  echo "prdview: broken symlinks detected under ${out_dir}:" >&2
  echo "${broken_links}" >&2
  exit 1
fi

# Generate MkDocs config (kept out of git).
if [[ ! -f "${mkdocs_cfg}" ]]; then
cat > "${mkdocs_cfg}" <<'YAML'
site_name: 'PRD View'

docs_dir: _staged
site_dir: site
theme:
  name: material
  features:
    - search.suggest
    - search.highlight
    - search.share

plugins:
  - search

markdown_extensions:
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format

extra_css:
  - https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css

extra_javascript:
  - https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js
  - https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js
  - javascripts/katex-init.js
  - https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js
  - javascripts/mermaid-init.js

use_directory_urls: true
YAML
fi

echo "prdview: linked staged files into ${out_dir}"
