#!/usr/bin/env bash
# Reproducible driver for the hipVS/cuVS CAGRA work on an AMD Radeon Cloud
# instance (ROCm + hipVS already installed, source tree = this checkout).
#
#   bash docs/gpu-vector-index-hipvs-cuvs/validation/build_and_measure.sh check    # toolchain + GPU gate
#   ... build_and_measure.sh bridge   # build libseekdb_cuvs_bridge.so, verify symbols
#   ... build_and_measure.sh seekdb   # cmake configure + compile seekdb with the bridge
#   ... build_and_measure.sh smoke    # functional gates (DDL, search, snapshot reload)
#   ... build_and_measure.sh bench    # CPU baseline + CAGRA parameter sweep
#   ... build_and_measure.sh report   # render the table in PERF_REPORT.md
#
# Every subcommand is independent; only "bridge" output is reused by the rest.
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../../.." && pwd)
HIPVS_ROOT=${HIPVS_ROOT:-/opt/hipvs}
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
RUN_ROOT=${RUN_ROOT:-$HOME/run}
BRIDGE_OUT=${BRIDGE_OUT:-$RUN_ROOT/bridge}
OBSERVER=${OBSERVER:-$REPO/build_cuvs/src/observer/seekdb}
EVIDENCE=${EVIDENCE:-$RUN_ROOT/evidence}
PORT=${PORT:-2989}
JOBS=${JOBS:-$(nproc)}

die() { echo "error: $*" >&2; exit 2; }
note() { printf '\n== %s\n' "$*"; }

sub_check() {
  note "toolchain"
  command -v cmake >/dev/null || die "cmake not found"
  command -v python3 >/dev/null || die "python3 not found"
  command -v mysql >/dev/null || die "mysql client not found"
  [[ -x "$ROCM_ROOT/bin/hipcc" ]] || die "hipcc missing: $ROCM_ROOT/bin/hipcc"
  [[ -f "$HIPVS_ROOT/include/cuvs/neighbors/cagra.h" ]] || die "hipVS headers missing under $HIPVS_ROOT/include"
  [[ -f "$HIPVS_ROOT/lib/libcuvs_c.so" ]] || die "libcuvs_c.so missing under $HIPVS_ROOT/lib"
  ls /dev/kfd >/dev/null 2>&1 || die "/dev/kfd missing (no amdgpu driver)"
  compgen -G "/dev/dri/renderD*" >/dev/null || die "no /dev/dri/renderD* node"

  note "cuVS symbol gate"
  # A shared library links fine with unresolved symbols; a missing graph
  # export/import only shows up at dlopen time. Probe the two entry points the
  # snapshot path depends on before spending an hour on the seekdb build.
  local probe
  probe=$(mktemp -d)/probe.c
  cat > "$probe" <<'EOF'
#include <cuvs/neighbors/cagra.h>
int main(void) {
  void *a = (void *)cuvsCagraIndexGetGraph;
  void *b = (void *)cuvsCagraIndexFromArgs;
  return (a && b) ? 0 : 1;
}
EOF
  "$ROCM_ROOT/bin/hipcc" -D__HIP_PLATFORM_AMD__ -I"$HIPVS_ROOT/include" \
    -I"$HIPVS_ROOT/include/cuvs" -I"$ROCM_ROOT/include" "$probe" -o "${probe%.c}" \
    -L"$HIPVS_ROOT/lib" -lcuvs_c -Wl,-rpath,"$HIPVS_ROOT/lib:$ROCM_ROOT/lib" \
    || die "cuVS probe failed to compile/link"
  "${probe%.c}" || die "cuVS build lacks the graph export/import entry points"
  echo "ok: hipVS C API exposes the snapshot entry points"
}

sub_bridge() {
  note "bridge"
  mkdir -p "$BRIDGE_OUT"
  bash "$REPO/tools/hipvs_bridge/build.sh" "$BRIDGE_OUT/libseekdb_cuvs_bridge.so"
  local lib=$BRIDGE_OUT/libseekdb_cuvs_bridge.so
  for sym in seekdb_cuvs_build seekdb_cuvs_search seekdb_cuvs_free \
             seekdb_cuvs_export_graph seekdb_cuvs_import seekdb_cuvs_free_host; do
    nm -D "$lib" | grep -q " $sym\$" || die "missing symbol: $sym"
  done
  # RTLD_NOW: fail here instead of mid-DDL if a cuVS symbol is unresolved.
  LD_LIBRARY_PATH="$HIPVS_ROOT/lib:$ROCM_ROOT/lib" python3 - "$lib" <<'PY'
import ctypes, sys
ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
print("ok: bridge loads with RTLD_NOW")
PY
}

sub_seekdb() {
  note "seekdb"
  local lib=$BRIDGE_OUT/libseekdb_cuvs_bridge.so
  [[ -f "$lib" ]] || die "build the bridge first"
  cd "$REPO"
  # OB_VSAG_TRACE is a no-op unless the tracer is compiled in, and the bench
  # script uses it to prove a restart rehydrated the snapshot instead of
  # rebuilding. It only writes when the observer is started with OB_VSAG_TRACE=1.
  ./build.sh release \
    -DOB_BUILD_CUVS=ON \
    -DOB_BUILD_CUVS_TRACE=ON \
    -DCUVS_BRIDGE_LIB="$lib" \
    --make -j"$JOBS"
}

sub_smoke() {
  note "functional smoke"
  local lib=$BRIDGE_OUT/libseekdb_cuvs_bridge.so
  [[ -x "$OBSERVER" ]] || die "observer not found: $OBSERVER (set OBSERVER=)"
  export LD_LIBRARY_PATH="$BRIDGE_OUT:$HIPVS_ROOT/lib:$ROCM_ROOT/lib"
  bash "$SCRIPT_DIR/run_gpu_smoke.sh" \
    --observer "$OBSERVER" --bridge "$lib" \
    --base-dir "$RUN_ROOT/smoke" --port "$PORT" \
    --render "$(ls /dev/dri/renderD* | head -1)" \
    --evidence "$EVIDENCE/smoke"
  echo
  echo "snapshot reload is also covered by 'bench --restart-check' below;"
  echo "run_ddl_persistence_smoke.sh is the docker/TSV variant of the same check."
}

sub_bench() {
  note "bench"
  local lib=$BRIDGE_OUT/libseekdb_cuvs_bridge.so
  [[ -x "$OBSERVER" ]] || die "observer not found: $OBSERVER (set OBSERVER=)"
  bash "$SCRIPT_DIR/bench_cagra_build.sh" \
    --observer "$OBSERVER" --bridge "$lib" \
    --base-dir "$RUN_ROOT/bench" --port "$PORT" \
    --evidence "$EVIDENCE/bench" \
    --restart-check
}

sub_report() {
  note "report"
  local restart=()
  [[ -f "$EVIDENCE/bench/restart.tsv" ]] && restart=(--restart "$EVIDENCE/bench/restart.tsv")
  python3 "$SCRIPT_DIR/render_perf_report.py" \
    --sweep "$EVIDENCE/bench/sweep.tsv" \
    "${restart[@]}" \
    --report "$REPO/docs/gpu-vector-index-hipvs-cuvs/PERF_REPORT.md"
}

case "${1:-}" in
  check)  sub_check ;;
  bridge) sub_bridge ;;
  seekdb) sub_seekdb ;;
  smoke)  sub_smoke ;;
  bench)  sub_bench ;;
  report) sub_report ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac
