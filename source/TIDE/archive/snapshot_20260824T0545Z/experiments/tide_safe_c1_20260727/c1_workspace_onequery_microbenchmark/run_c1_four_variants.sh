#!/usr/bin/env bash
# Isolated C1 execution engine.  It is callable only from the hardened guard.
# It never selects a numeric CUDA device: each binary receives the physical
# GPU0 UUID established and rechecked by the guard immediately before launch.
set -uo pipefail
umask 077

ROOT='/workspace/experiments/tide_safe_c1_20260727/c1_workspace_onequery_microbenchmark'
BASE='/workspace/legacy_workspace/GTS/Datasets/sift1m/sift_base.txt'
TRACE="$ROOT/inputs/sift1m_10k_442_first1024_type2_query_only.txt"
GUARD_EXPECTED="$ROOT/run_c1_workspace_microbenchmark_guard.sh"
PYTHON="$(command -v python3 || true)"

die() { printf 'C1-ENGINE BLOCKED: %s\n' "$*" >&2; exit 64; }
[[ -n "$PYTHON" && -x "$PYTHON" ]] || die 'python3 unavailable'
[[ "${C1_ALLOW_GPU0:-}" == YES ]] || die 'C1_ALLOW_GPU0=YES required'
[[ "${C1_GUARD_SCRIPT:-}" == "$GUARD_EXPECTED" && -x "$GUARD_EXPECTED" ]] || die 'must be launched by canonical hardened guard'
[[ -n "${C1_RUN_OUT:-}" && -d "$C1_RUN_OUT" ]] || die 'guard-created C1_RUN_OUT required'
[[ -n "${C1_GUARD_SESSION_FILE:-}" && -f "$C1_GUARD_SESSION_FILE" ]] || die 'guard session required'
[[ -n "${C1_GUARD_SESSION_NONCE:-}" ]] || die 'guard nonce required'
[[ "${C1_GPU_UUID:-}" == GPU-* ]] || die 'physical GPU UUID mapping required'
[[ "${CUDA_VISIBLE_DEVICES:-}" == "$C1_GPU_UUID" ]] || die 'CUDA_VISIBLE_DEVICES must equal guarded physical GPU UUID'
[[ "${NVIDIA_VISIBLE_DEVICES:-}" == "$C1_GPU_UUID" ]] || die 'NVIDIA_VISIBLE_DEVICES must equal guarded physical GPU UUID'
[[ "$(hostname)" == CONFIGURE_ARCHIVE_HOST ]] || die 'wrong host'

run_variant() {
  local rep="$1" variant="$2" mode="$3" limit="$4"
  local bin="$ROOT/builds/$variant/bin/C1Microbench"
  local out="$C1_RUN_OUT/rep$rep/$variant"
  local label="rep$rep/$variant"
  [[ -x "$bin" && ! -L "$bin" ]] || die "missing isolated binary: $bin"
  [[ ! -e "$out" && ! -L "$out" ]] || die "variant output exists; no overwrite: $out"
  mkdir -p -- "$(dirname "$out")" || die 'cannot create replicate directory'
  mkdir -- "$out" || die 'cannot create fresh variant output'

  # The helper performs a fresh binary pin check plus physical-GPU0 UUID,
  # utilization, memory and compute-app checks immediately before this launch.
  "$GUARD_EXPECTED" --before-launch "$label" "$variant" || exit $?
  local args=(--base "$BASE" --trace "$TRACE" --radius 500 --out "$out" --replicate "$rep" --variant "$variant")
  if [[ "$mode" == smoke ]]; then args+=(--smoke --trace-limit "$limit"); fi
  env CUDA_VISIBLE_DEVICES="$C1_GPU_UUID" \
      NVIDIA_VISIBLE_DEVICES="$C1_GPU_UUID" \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      "$bin" "${args[@]}" >"$out/stdout.log" 2>"$out/stderr.log"
  local rc=$?
  "$GUARD_EXPECTED" --after-launch "$label" "$variant" || exit $?
  [[ "$rc" -eq 0 ]] || { printf 'C1 binary failed rc=%s: %s\n' "$rc" "$label" >&2; exit "$rc"; }
  "$PYTHON" - "$out/completion.json" <<'PY'
import json,sys,pathlib
completion_path=pathlib.Path(sys.argv[1])
p=json.load(completion_path.open())
card=json.load((completion_path.parent/'run_card.json').open())
assert p['tree_invariance_pass'] is True, p
assert p.get('run_mode') in {'UNMEASURED_SMOKE_DO_NOT_USE', 'MEASURED_PROTOCOL'}, p
assert card.get('C2_residual_mode') == 0, card
assert card.get('requested_variant') == completion_path.parent.name, card
PY
  [[ $? -eq 0 ]] || die "completion semantic contract failed: $label"
}

if [[ "${C1_SMOKE:-NO}" == YES ]]; then
  smoke_ops="${C1_SMOKE_OPS:-16}"
  [[ "$smoke_ops" =~ ^([1-9]|[1-9][0-9]|1[0-1][0-9]|12[0-8])$ ]] || die 'C1_SMOKE_OPS must be 1..128'
  # Parser precondition is replicate >= 1; smoke stays unmeasured but uses rep1.
  run_variant 1 E_G_c1_off_reference smoke "$smoke_ops"
  echo 'UNMEASURED_SMOKE_COMPLETE: C1-only diagnostic; do not use for statistics or claims.'
  exit 0
fi

[[ "${C1_REPLICATES:-5}" == 5 ]] || die 'protocol fixes C1_REPLICATES=5'
orders=(
  'E_G_c1_off_reference P_G_workspace_only E_F_fastpath_only P_F_full_C1'
  'P_G_workspace_only E_F_fastpath_only P_F_full_C1 E_G_c1_off_reference'
  'E_F_fastpath_only P_F_full_C1 E_G_c1_off_reference P_G_workspace_only'
  'P_F_full_C1 E_G_c1_off_reference P_G_workspace_only E_F_fastpath_only'
  'E_G_c1_off_reference E_F_fastpath_only P_G_workspace_only P_F_full_C1'
)
for i in "${!orders[@]}"; do
  rep=$((i+1))
  for variant in ${orders[$i]}; do run_variant "$rep" "$variant" measured 1024; done
done
"$PYTHON" "$ROOT/verify_c1_semantics.py" --run-root "$C1_RUN_OUT" || exit $?
"$PYTHON" "$ROOT/analyze_c1_microbenchmark.py" --run-root "$C1_RUN_OUT" || exit $?
echo 'C1_MEASURED_PROTOCOL_COMPLETE'
