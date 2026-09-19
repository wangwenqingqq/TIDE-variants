#!/usr/bin/env bash
# One-shot CPU-only Safe-C2 v4 exact-oracle / tie-admission input generator.
#
# This launcher compiles and runs ONLY tools/c2_v4_cpu_exact_oracle.cpp with
# g++. It never invokes a CUDA/GTS binary, nvidia-smi, Nsight, or a formal
# ledger/stage. It refuses any pre-existing output, allowing no append/retry.
set -Eeuo pipefail
umask 077

ROOT=/workspace/experiments/tide_safe_c1_20260727/c2_speculative_fallback_v4
IN="$ROOT/inputs/fresh_selection_pre_gt_v1"
SRC="$ROOT/tools/c2_v4_cpu_exact_oracle.cpp"
FINALIZER="$ROOT/tools/finalize_c2_v4_cpu_oracle.py"
PROTOCOL="$ROOT/protocols/c2_v4_oracle_admission_protocol_v1.json"
BIN="$ROOT/tools/c2_v4_cpu_exact_oracle.bin"
BASE=/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.fvecs
QUERY="$IN/sift_learn_fresh12524.fvecs"
ORACLE_DIR="$IN/exact_oracle_v1"
RUN_DIR="$ROOT/provenance/c2_v4_cpu_oracle_run_v1"
COMMAND="$RUN_DIR/oracle_command.txt"
LOG="$RUN_DIR/oracle_cpu.log"
STATE="$RUN_DIR/run_state.json"
TMP_BIN="$ROOT/tools/.c2_v4_cpu_exact_oracle.bin.tmp.$$"
SUCCESS=0

write_state() {
  local status=$1
  local reason=$2
  python3 - "$STATE" "$status" "$reason" <<'PY'
import json,os,sys
p,status,reason=sys.argv[1:]
obj={
  "schema":"safe-c2-v4-cpu-oracle-run-state-v1",
  "status":status,
  "reason":reason,
  "scope":"CPU-only oracle; no CUDA/GPU/nvidia-smi/GTS/formal ledger/stage execution",
  "gpu_binary_executed":False,
  "nvidia_smi_called":False,
}
tmp=p+".tmp"
with open(tmp,"x",encoding="utf-8") as f:
    json.dump(obj,f,sort_keys=True,indent=2);f.write("\n")
os.replace(tmp,p)
PY
}

on_exit() {
  local rc=$?
  trap - EXIT
  rm -f -- "$TMP_BIN" || true
  if [[ "$SUCCESS" != 1 && -d "$RUN_DIR" ]]; then
    set +e
    rm -f -- "$STATE" "$STATE.tmp"
    write_state "FAILED_CPU_ONLY_ORACLE" "outer_launcher_exit_$rc"
    set -e
  fi
  exit "$rc"
}
trap on_exit EXIT

for path in "$SRC" "$FINALIZER" "$PROTOCOL" "$BASE" "$QUERY" "$IN/selection_manifest_pre_gt.json"; do
  [[ -f "$path" && ! -L "$path" ]] || { echo "missing/symlink input: $path" >&2; exit 64; }
done
for path in "$BIN" "$ORACLE_DIR" "$RUN_DIR" "$TMP_BIN"; do
  [[ ! -e "$path" && ! -L "$path" ]] || { echo "refusing pre-existing output: $path" >&2; exit 65; }
done
[[ -d "$ROOT/tools" && ! -L "$ROOT/tools" && -d "$IN" && ! -L "$IN" ]] || {
  echo "root/tool/input directory is not direct canonical" >&2; exit 66;
}
[[ "$(sha256sum "$BASE" | awk '{print $1}')" == "21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816" ]] || {
  echo "base SHA mismatch" >&2; exit 67;
}
[[ "$(sha256sum "$QUERY" | awk '{print $1}')" == "6b8e480bcd7247e13d1a8b4248d2e86d619abea6919bc9d27fced93e6c3fe3c2" ]] || {
  echo "fresh query SHA mismatch" >&2; exit 68;
}
[[ "$(sha256sum "$IN/selection_manifest_pre_gt.json" | awk '{print $1}')" == "98e3627c7dba033f5ab73b9df90b794789c9e39046491e5cc1745b6ac87b7c39" ]] || {
  echo "pre-GT selection SHA mismatch" >&2; exit 69;
}

mkdir "$RUN_DIR"
write_state "PREPARING_CPU_ONLY_ORACLE" "no CUDA/GPU/formal stage action"
{
  echo "started_utc=$(date -u -Is)"
  echo "scope=CPU-only independent exact FP32+int64 oracle; no CUDA/GTS/gamma/timing/trace/ledger"
  echo "host=$(hostname)"
  echo "compiler=$(g++ --version | head -n 1)"
  g++ -std=c++20 -O3 -march=native -fopenmp -ffp-contract=off -fno-fast-math "$SRC" -o "$TMP_BIN"
  mv "$TMP_BIN" "$BIN"
  "$BIN" --self-test
} >"$LOG" 2>&1

cat >"$COMMAND" <<EOF
schema=safe-c2-v4-cpu-oracle-command-v1
scope=CPU-only independent exact reference and deterministic tie-admission input only; no CUDA/GTS/gamma/recall/timing/trace/ledger
created_utc=$(date -u -Is)
host=$(hostname)
source_path=$SRC
source_sha256=$(sha256sum "$SRC" | awk '{print $1}')
binary_path=$BIN
binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')
finalizer_path=$FINALIZER
finalizer_sha256=$(sha256sum "$FINALIZER" | awk '{print $1}')
protocol_path=$PROTOCOL
protocol_sha256=$(sha256sum "$PROTOCOL" | awk '{print $1}')
base_path=$BASE
base_sha256=$(sha256sum "$BASE" | awk '{print $1}')
query_path=$QUERY
query_sha256=$(sha256sum "$QUERY" | awk '{print $1}')
selection_manifest_sha256=$(sha256sum "$IN/selection_manifest_pre_gt.json" | awk '{print $1}')
compile_flags=-std=c++20 -O3 -march=native -fopenmp -ffp-contract=off -fno-fast-math
cpu_set=0-31
nice=10
command=env OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores taskset -c 0-31 nice -n 10 $BIN --base $BASE --query $QUERY --out $ORACLE_DIR
EOF

write_state "RUNNING_CPU_ONLY_ORACLE" "oracle output is fresh and unadmitted"
env OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores \
  taskset -c 0-31 nice -n 10 "$BIN" --base "$BASE" --query "$QUERY" --out "$ORACLE_DIR" >>"$LOG" 2>&1

[[ -d "$ORACLE_DIR" && ! -L "$ORACLE_DIR" ]] || { echo "oracle did not create output directory" >&2; exit 70; }
for path in "$ORACLE_DIR/groundtruth_fp32_top100.ivecs" "$ORACLE_DIR/top101_int64_fp32_audit.bin" "$ORACLE_DIR/tie_admission.tsv" "$ORACLE_DIR/oracle_summary.json"; do
  [[ -s "$path" && ! -L "$path" ]] || { echo "oracle output missing/empty: $path" >&2; exit 71; }
done
rm -f -- "$STATE"
write_state "COMPLETE_CPU_ONLY_ORACLE_UNADMITTED" "finalization still required; no formal stage has been claimed"
SUCCESS=1
printf '%s\n' "$RUN_DIR"
