#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_multilandmark_certificate_probe_v3_aabb
SRC="$ROOT/cpu_multilandmark_probe.cpp"
BIN="$ROOT/cpu_multilandmark_probe"
OUT="$ROOT/result.json"
[[ -f "$SRC" && ! -L "$SRC" ]] || exit 65
[[ ! -e "$BIN" && ! -L "$BIN" && ! -e "$OUT" && ! -L "$OUT" ]] || exit 66
g++ -std=c++17 -O3 -Wall -Wextra -Werror "$SRC" -o "$BIN"
"$BIN" >"$OUT"
cat "$OUT"
