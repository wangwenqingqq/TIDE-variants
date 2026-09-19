#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/TIDE/active/certigraph_gate0_20260827
EXPORT=$(cat "$ROOT/CURRENT_EXPORT_RUN.txt")/export
RUN="$ROOT/raw/algorithm_formal_$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS="$RUN/results"
mkdir -p "$RUN" "$RESULTS"
UUID=GPU-CONFIGURE-ARCHIVE-DEVICE
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,pstate,power.draw --format=csv,noheader,nounits > "$RUN/gpus_pre.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits > "$RUN/apps_pre.csv" || true
if grep -Fq "$UUID" "$RUN/apps_pre.csv"; then
  echo "ABORT: physical GPU 4 is occupied" | tee "$RUN/ABORTED.txt"
  exit 17
fi
PY=/workspace/external-tools/baselines/cuvs-venv/bin/python
printf "%s\n" "env CUDA_VISIBLE_DEVICES=4 $PY $ROOT/src/run_algorithmic_gate.py --export $EXPORT --manifest $ROOT/data/prepared/query_manifest.json --raw $RUN/evidence --results $RESULTS" > "$RUN/command.sh"
echo $$ > "$RUN/pid.txt"
date -Is > "$RUN/start_time.txt"
$PY - <<"PY" > "$RUN/python_packages.txt"
import numpy, cupy, cuvs
print("numpy", numpy.__version__)
print("cupy", cupy.__version__)
print("cuvs", getattr(cuvs,"__version__","unknown"))
PY
nvidia-smi pmon -i 4 -s um -d 1 > "$RUN/pmon.log" 2>&1 &
PMON=$!
cleanup() { kill "$PMON" 2>/dev/null || true; wait "$PMON" 2>/dev/null || true; }
trap cleanup EXIT
mkdir -p "$RUN/evidence"
flock /tmp/certigraph_gate0_gpu4.lock env CUDA_VISIBLE_DEVICES=4 \
  "$PY" "$ROOT/src/run_algorithmic_gate.py" \
    --export "$EXPORT" \
    --manifest "$ROOT/data/prepared/query_manifest.json" \
    --raw "$RUN/evidence" --results "$RESULTS" \
  > "$RUN/stdout.log" 2> "$RUN/stderr.log"
date -Is > "$RUN/end_time.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits > "$RUN/apps_post.csv" || true
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,pstate,power.draw --format=csv,noheader,nounits > "$RUN/gpus_post.csv"
find "$RUN" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN/files.sha256"
echo "$RUN" | tee "$ROOT/CURRENT_ALGORITHM_RUN.txt"
cat "$RESULTS/RESULTS.md"
