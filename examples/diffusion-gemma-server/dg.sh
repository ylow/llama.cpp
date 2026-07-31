#!/usr/bin/env bash
# Run DiffusionGemma locally behind the Anthropic Messages API.
#
#   ./dg.sh init    build the binaries and download the weights
#   ./dg.sh start   run the server in the foreground (Ctrl-C to stop)
#   ./dg.sh code    point Claude Code at the running server
#
# See README-anthropic-adapter.md for what the endpoint does and does not support.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Everything below is overridable from the environment.
DG_REPO="${DG_REPO:-unsloth/diffusiongemma-26B-A4B-it-GGUF}"
DG_QUANT="${DG_QUANT:-Q4_K_M}"
DG_MODEL_DIR="${DG_MODEL_DIR:-$REPO_ROOT/models/diffusiongemma}"
DG_MODEL="${DG_MODEL:-$DG_MODEL_DIR/diffusiongemma-26B-A4B-it-$DG_QUANT.gguf}"
DG_BUILD="${DG_BUILD:-$REPO_ROOT/build}"
DG_HOST="${DG_HOST:-127.0.0.1}"
DG_PORT="${DG_PORT:-8080}"
DG_NGL="${DG_NGL:-99}"
DG_MODEL_NAME="${DG_MODEL_NAME:-diffusiongemma}"
DG_API_KEY="${DG_API_KEY:-}"

BACKEND="$DG_BUILD/bin/llama-diffusion-gemma-visual-server"
ADAPTER="$SCRIPT_DIR/anthropic_adapter.py"
BASE_URL="http://$DG_HOST:$DG_PORT"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m  %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "$1 not found in PATH. $2"; }

# The HF CLI was renamed; accept either.
hf_cli() {
    if command -v hf >/dev/null 2>&1; then hf "$@"
    elif command -v huggingface-cli >/dev/null 2>&1; then huggingface-cli "$@"
    else die "neither 'hf' nor 'huggingface-cli' found. Install with: pip install huggingface_hub"
    fi
}

server_up() {
    curl -fsS --max-time 3 "$BASE_URL/health" >/dev/null 2>&1
}

# ---------------------------------------------------------------------------

cmd_init() {
    need cmake "Install it (brew install cmake) and re-run."
    need curl "Install it and re-run."

    say "Configuring the build in $DG_BUILD"
    cmake -B "$DG_BUILD" -S "$REPO_ROOT" -DCMAKE_BUILD_TYPE=Release

    # The adapter drives the visual server; the CLI is handy for a quick sanity check
    # outside the HTTP path.
    say "Building llama-diffusion-gemma-visual-server and llama-diffusion-cli"
    cmake --build "$DG_BUILD" -j --target \
        llama-diffusion-gemma-visual-server llama-diffusion-cli

    if [ -f "$DG_MODEL" ]; then
        say "Weights already present: $DG_MODEL"
    else
        say "Downloading $DG_QUANT weights from $DG_REPO (about 17 GB, this takes a while)"
        hf_cli download "$DG_REPO" --include "*$DG_QUANT*" --local-dir "$DG_MODEL_DIR"
    fi

    [ -f "$DG_MODEL" ] || die "expected weights at $DG_MODEL but they are not there.
Check what landed in $DG_MODEL_DIR and set DG_MODEL to the right file."

    say "Running the adapter's test suites"
    python3 "$SCRIPT_DIR/test_adapter.py"
    python3 "$SCRIPT_DIR/test_tools.py"

    say "Ready. Start the server with:  $0 start"
}

cmd_start() {
    [ -x "$BACKEND" ] || die "backend not built: $BACKEND
Run '$0 init' first."
    [ -f "$DG_MODEL" ] || die "weights not found: $DG_MODEL
Run '$0 init' first."

    if server_up; then
        warn "something is already serving $BASE_URL -- stop it first, or set DG_PORT."
    fi

    say "Serving $DG_MODEL_NAME on $BASE_URL/v1/messages  (Ctrl-C to stop)"
    args=(--model "$DG_MODEL" --binary "$BACKEND"
          --host "$DG_HOST" --port "$DG_PORT" --ngl "$DG_NGL")
    if [ -n "$DG_API_KEY" ];        then args+=(--api-key "$DG_API_KEY"); fi
    if [ -n "${DG_MAXTOK:-}" ];     then args+=(--maxtok "$DG_MAXTOK");   fi
    if [ -n "${DG_FLASH_ATTN:-}" ]; then args+=(--flash-attn);            fi
    if [ -n "${DG_VERBOSE:-}" ];    then args+=(-v);                      fi

    # exec so Ctrl-C reaches the adapter directly and it can shut the model down cleanly.
    exec python3 "$ADAPTER" "${args[@]}" "$@"
}

cmd_code() {
    need claude "Install Claude Code first: https://claude.com/claude-code"

    server_up || die "nothing is serving $BASE_URL.
Start it in another terminal with:  $0 start"

    # A local endpoint needs no real credential, but Claude Code insists on one being set,
    # so pass the adapter's key if configured and a placeholder otherwise.
    local token="${DG_API_KEY:-local}"

    say "Pointing Claude Code at $BASE_URL (model: $DG_MODEL_NAME)"
    warn "Claude Code's system prompt plus its tool definitions measured 23476 tokens here,"
    warn "against a 20480-token budget. Short prompts work; anything using the full tool set"
    warn "returns 'conversation too long'. See README-anthropic-adapter.md."

    ANTHROPIC_BASE_URL="$BASE_URL" \
    ANTHROPIC_AUTH_TOKEN="$token" \
    ANTHROPIC_API_KEY="$token" \
    ANTHROPIC_MODEL="$DG_MODEL_NAME" \
    ANTHROPIC_DEFAULT_HAIKU_MODEL="$DG_MODEL_NAME" \
    ANTHROPIC_SMALL_FAST_MODEL="$DG_MODEL_NAME" \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    DISABLE_TELEMETRY=1 \
        exec claude "$@"
}

usage() {
    cat <<EOF
Usage: $0 {init|start|code} [args...]

  init    Configure the build, compile the diffusion binaries, download the
          $DG_QUANT weights, and run the adapter's tests. Safe to re-run.

  start   Run the Anthropic Messages API server in the foreground on
          $BASE_URL. Ctrl-C stops it. Extra args go to the adapter.

  code    Launch Claude Code against the running server. Extra args go to
          claude. Requires '$0 start' in another terminal.

Environment overrides:
  DG_PORT ($DG_PORT), DG_HOST ($DG_HOST), DG_NGL ($DG_NGL), DG_QUANT ($DG_QUANT)
  DG_MODEL, DG_MODEL_DIR, DG_BUILD, DG_MODEL_NAME, DG_API_KEY
  DG_MAXTOK (context budget; default is auto-sized), DG_FLASH_ATTN, DG_VERBOSE
EOF
}

case "${1:-}" in
    init)  shift; cmd_init  "$@" ;;
    start) shift; cmd_start "$@" ;;
    code)  shift; cmd_code  "$@" ;;
    ""|-h|--help|help) usage ;;
    *) warn "unknown subcommand: $1"; echo; usage; exit 1 ;;
esac
