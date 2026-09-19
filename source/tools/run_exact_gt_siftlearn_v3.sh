#!/usr/bin/env bash
# CPU-only immutable-reference generation for the C2 v3 fresh compact learn workload.
set -euo pipefail
umask 077
ROOT='/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v3'
IN="$ROOT/inputs/sift_learn_compact10k_v1"
SRC="$ROOT/tools/build_exact_gt_siftlearn_v3.cpp"
BIN="$ROOT/tools/build_exact_gt_siftlearn_v3.bin"
FINALIZER="$ROOT/tools/finalize_exact_gt_siftlearn_v3.py"
BASE='/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs'
QUERY="$IN/sift_learn_clean10k.fvecs"
OUT="$IN/exact_oracle_v1"
LOG="$IN/exact_oracle_v1_cpu.log"
CMD="$IN/exact_oracle_v1_command.txt"
[[ ! -e "$OUT" && ! -L "$OUT" && ! -e "$LOG" && ! -L "$LOG" && ! -e "$CMD" && ! -L "$CMD" ]] || { echo 'refuse existing oracle output/log/command' >&2; exit 64; }
for p in "$SRC" "$BIN" "$FINALIZER" "$BASE" "$QUERY" "$IN/selection_manifest_pre_gt.json"; do [[ -f "$p" && ! -L "$p" ]] || { echo "missing/symlink $p" >&2; exit 65; }; done
[[ "$(sha256sum "$QUERY" | awk '{print $1}')" == '318e5085dfb4f50831d8f6620cffc736da9c11fb44454b507d2f5e7019d2e11e' ]] || { echo 'compact query SHA mismatch' >&2; exit 66; }
[[ "$(sha256sum "$BASE" | awk '{print $1}')" == '21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816' ]] || { echo 'base SHA mismatch' >&2; exit 67; }
cat > "$CMD" <<EOF
scope=CPU-only exact reference; no GPU/C2 traversal/gamma/recall/timing
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
host=$(hostname)
source_sha256=$(sha256sum "$SRC" | awk '{print $1}')
binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')
compile_flags=-O3 -march=native -fopenmp -ffp-contract=off -fno-fast-math -std=c++20
nice=10
omp_num_threads=32
command=taskset -c 0-31 nice -n 10 "$BIN" --base "$BASE" --query "$QUERY" --out "$OUT"
EOF
# CPU preflight only: do not query or use GPU.  Nice+fixed affinity deliberately avoids consuming all shared cores.
env OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores taskset -c 0-31 nice -n 10 "$BIN" --base "$BASE" --query "$QUERY" --out "$OUT" >"$LOG" 2>&1
python3 "$FINALIZER" --input-root "$IN" --oracle-dir "$OUT" --source "$SRC" --binary "$BIN" --command-file "$CMD"
