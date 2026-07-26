#!/usr/bin/env bash
#
# ============================================================================
#  bootstrap.sh  —  The ONE thing that must run before everything else.
# ============================================================================
#
#  Its only job: make sure `uv` is installed. That's it.
#
#  Why does this exist as a shell script (and not Python)?
#    The real tool is `./mlx-pi`, a Python program with a `uv run` shebang.
#    But that shebang needs uv to already exist — and a fresh Mac ships
#    neither uv nor a reliable python3. Bash is the only thing guaranteed to
#    be present, so this tiny script does the irreducible bootstrap, then
#    hands off to ./mlx-pi for all the real work.
#
#  After this runs once:   ./mlx-pi setup   →   ./mlx-pi up   →   ./mlx-pi pi
# ============================================================================

set -euo pipefail

# --- minimal colored output ------------------------------------------------
if [[ -t 1 ]]; then
  B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; C=$'\033[36m'; D=$'\033[2m'; R=$'\033[0m'
else B=""; G=""; Y=""; C=""; D=""; R=""; fi

echo "${B}${C}🍎 mlx-pi bootstrap — ensuring uv is installed${R}"

if command -v uv >/dev/null 2>&1; then
  echo "  ${G}✅ uv already installed ($(uv --version))${R}"
else
  echo "  ${Y}⬇️  Installing uv (Astral's static binary → ~/.local/bin)…${R}"
  # Security: this trusts Astral's official HTTPS installer; no independent
  # checksum verification is performed here.
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Make uv visible to the rest of THIS shell session.
  export PATH="$HOME/.local/bin:$PATH"
  echo "  ${G}✅ uv installed ($(uv --version))${R}"
fi

# Ensure ~/.local/bin is on PATH for future shells (where uv tools live).
uv tool update-shell >/dev/null 2>&1 || true

# Make the Python CLI executable so you can run it directly (./mlx-pi).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$SCRIPT_DIR/mlx-pi" ]] && chmod +x "$SCRIPT_DIR/mlx-pi"

echo
echo "${B}✅ Bootstrap done.${R} Next:"
echo "  ${C}./mlx-pi setup${R}      ${D}# install both MLX backends + pi; configure (no model download)${R}"
echo "  ${C}./mlx-pi up${R}         ${D}# start the local model server${R}"
echo "  ${C}./mlx-pi pi${R}         ${D}# launch the coding agent against it${R}"
echo
echo "${D}First run of ./mlx-pi downloads its Python deps (rich, httpx) via uv — a few seconds, once.${R}"
