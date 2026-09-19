#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/TIDE/active/certigraph_gate0_20260827
RUN="$ROOT/raw/export_formal_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN/export"
UUID=GPU-CONFIGURE-ARCHIVE-DEVICE
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,pstate,power.draw --format=csv,noheader,nounits > "$RUN/gpus_pre.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits > "$RUN/apps_pre.csv" || true
if grep -Fq "$UUID" "$RUN/apps_pre.csv"; then
  echo "ABORT: physical GPU 4 is occupied" | tee "$RUN/ABORTED.txt"
  exit 17
fi
printf "%s\n" "env CUDA_VISIBLE_DEVICES=4 $ROOT/bin/export_edit_evidence --base $ROOT/data/prepared/words_normalized.txt --queries $ROOT/data/prepared/queries.txt --out-dir $RUN/export --pivots 64 --seed 20260827" > "$RUN/command.sh"
echo $$ > "$RUN/pid.txt"
date -Is > "$RUN/start_time.txt"
nvidia-smi pmon -i 4 -s um -d 1 > "$RUN/pmon.log" 2>&1 &
PMON=$!
cleanup() { kill "$PMON" 2>/dev/null || true; wait "$PMON" 2>/dev/null || true; }
trap cleanup EXIT
flock /tmp/certigraph_gate0_gpu4.lock env CUDA_VISIBLE_DEVICES=4 \
  "$ROOT/bin/export_edit_evidence" \
    --base "$ROOT/data/prepared/words_normalized.txt" \
    --queries "$ROOT/data/prepared/queries.txt" \
    --out-dir "$RUN/export" --pivots 64 --seed 20260827 \
  > "$RUN/stdout.log" 2> "$RUN/stderr.log"
date -Is > "$RUN/end_time.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits > "$RUN/apps_post.csv" || true
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,pstate,power.draw --format=csv,noheader,nounits > "$RUN/gpus_post.csv"
find "$RUN/export" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN/export.sha256"
echo "$RUN" | tee "$ROOT/CURRENT_EXPORT_RUN.txt"
cat "$RUN/export/export_metadata.json"
