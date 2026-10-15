#!/usr/bin/env bash

set -euo pipefail

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-main}"
WORK_REMOTE="${WORK_REMOTE:-origin}"
WORK_BRANCH="${WORK_BRANCH:-dev}"

PUSH=1
[[ "${1:-}" == "--no-push" ]] && PUSH=0

# --- Run from the repo root regardless of where the script is called from ----
cd "$(git rev-parse --show-toplevel)"

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- Safety: refuse to run on a dirty tree ----------------------------------
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
  die "You have uncommitted changes. Commit or stash them first, then re-run."
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  info "Note: you have untracked files (they won't be touched by the merge)."
fi

# --- Fetch the latest --------------------------------------------
info "Fetching ${UPSTREAM_REMOTE}..."
git fetch --prune "$UPSTREAM_REMOTE"

# --- Switch to your working branch ------------------------------------------
info "Switching to '${WORK_BRANCH}'..."
git checkout "$WORK_BRANCH"

# --- Merge the new labs in ---------------------------------------------------
info "Merging ${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH} into '${WORK_BRANCH}'..."
if git merge --no-edit "${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}"; then
  info "Merge clean."
else
  echo
  die "Merge hit conflicts. Resolve them, then run:
       git add <files> && git commit
       git push ${WORK_REMOTE} ${WORK_BRANCH}
     Or abort with: git merge --abort"
fi

# --- Push to fork -------------------------------------------------------
if [[ "$PUSH" -eq 1 ]]; then
  info "Pushing '${WORK_BRANCH}' to ${WORK_REMOTE}..."
  git push "$WORK_REMOTE" "$WORK_BRANCH"
else
  info "Skipping push (--no-push). Push later with: git push ${WORK_REMOTE} ${WORK_BRANCH}"
fi

info "Done. '${WORK_BRANCH}' is up to date with ${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}."
