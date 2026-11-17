#!/usr/bin/env bash
set -euo pipefail

# Adapted from https://stackoverflow.com/questions/10622179/how-to-find-identify-large-commits-in-git-history
# Displays sorted git blob sizes across all reachable objects in the repository.

usage() {
  cat <<'EOF'
Usage: display_commits_size.sh [-h|--help]

Display sorted git blob sizes across all reachable objects in the repository.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  usage >&2
  exit 2
fi

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "Error: must be run inside a git repository." >&2
  exit 2
fi

numfmt_cmd=""
if command -v gnumfmt >/dev/null 2>&1; then
  numfmt_cmd="gnumfmt"
elif command -v numfmt >/dev/null 2>&1; then
  numfmt_cmd="numfmt"
fi

if [[ -n "$numfmt_cmd" ]]; then
  git rev-list --objects --all |
    git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
    sed -n 's/^blob //p' |
    sort --numeric-sort --key=2 |
    "$numfmt_cmd" --field=2 --to=iec-i --suffix=B --padding=7 --round=nearest
else
  git rev-list --objects --all |
    git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
    sed -n 's/^blob //p' |
    sort --numeric-sort --key=2
fi
