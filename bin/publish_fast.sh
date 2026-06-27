#!/usr/bin/env bash
set -euo pipefail

message="${1:-Update website}"
repo_root="$(git rev-parse --show-toplevel)"
token_file="${GITHUB_TOKEN_FILE:-${repo_root}/../git_token.txt}"

if [[ ! -f "${token_file}" ]]; then
  echo "Token file not found: ${token_file}" >&2
  exit 1
fi

git add -A

if git diff --cached --quiet; then
  echo "No changes to publish."
  exit 0
fi

git commit -m "${message}"

askpass="$(mktemp)"
trap 'rm -f "${askpass}"' EXIT

cat >"${askpass}" <<'ASKPASS'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\n' 'yeop-giraffe' ;;
  *Password*) cat "${GITHUB_TOKEN_FILE}" ;;
  *) printf '\n' ;;
esac
ASKPASS
chmod 700 "${askpass}"

GIT_ASKPASS="${askpass}" GITHUB_TOKEN_FILE="${token_file}" GIT_TERMINAL_PROMPT=0 git push origin master
