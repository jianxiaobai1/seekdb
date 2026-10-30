#!/usr/bin/env bash
# One-shot setup for the hipVS CAGRA build-acceleration patch on a Radeon Cloud
# seekdb-hipvs workspace.  Usage:
#   bash hipvs_setup.sh check     # env + hipVS C API gate (fails fast, zero cost)
#   bash hipvs_setup.sh bridge    # apply patch + build libseekdb_cuvs_bridge.so
#   bash hipvs_setup.sh seekdb    # cmake configure + build the seekdb binary
#   bash hipvs_setup.sh smoke     # functional gate: run_gpu_smoke.sh on the built binary
#   bash hipvs_setup.sh bench     # CPU baseline + CAGRA build-parameter sweep
#   bash hipvs_setup.sh report    # render PERF_REPORT.md table from sweep.tsv
set -uo pipefail

# Canonical copy is docs/gpu-vector-index-hipvs-cuvs/validation/hipvs_setup.sh in
# the seekdb tree; this one is the same file, kept so the driver can live outside
# the repo (it is also the thing that applies the patch).
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

PATCH_FILE=${PATCH_FILE:-$HOME/cagra-fast-build.patch}
BENCH_FILE=${BENCH_FILE:-$SCRIPT_DIR/bench_cagra_build.sh}
[ -f "$BENCH_FILE" ] || BENCH_FILE=$HOME/seekdb_bench_cagra.sh
BRIDGE_OUT=${BRIDGE_OUT:-$HOME/out}
BUILD_DIR=${BUILD_DIR:-build_release}
JOBS=${JOBS:-$(nproc)}
TRACE_FLAG=${TRACE_FLAG:-ON}
HIPVS_ROOT=${HIPVS_ROOT:-/opt/hipvs}
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
# observer preallocates datafile_size=40G + clog per run dir, so the run dirs
# must live on the volume with the most free space: /root (the overlay) here.
RUN_ROOT=${RUN_ROOT:-$HOME/run}
CAGRA_H=$HIPVS_ROOT/include/cuvs/neighbors/cagra.h
# The OBCUVS03 patch makes the bridge call these two cuVS entry points directly.
# A shared library links happily with them undefined, so a hipVS build that
# predates them only blows up at dlopen time on the instance -- gate it here.
REQUIRED_CAGRA_SYMS="cuvsCagraBuild cuvsCagraSerialize cuvsCagraIndexGetGraph cuvsCagraIndexFromArgs"

log() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\n[FAIL] %s\n' "$*" >&2; exit 1; }

find_cuvs_lib() {
  local f
  for f in "$HIPVS_ROOT/lib/libcuvs_c.so" "$HIPVS_ROOT/lib64/libcuvs_c.so" \
           "$HIPVS_ROOT"/lib/libcuvs_c.so.*.* "$HIPVS_ROOT"/lib64/libcuvs_c.so.*.* \
           "$HIPVS_ROOT/lib/libcuvs_c.a"; do
    [ -e "$f" ] && { echo "$f"; return 0; }
  done
  f=$(ldconfig -p 2>/dev/null | awk '/libcuvs_c\.so/ {print $NF; exit}')
  [ -n "$f" ] && { echo "$f"; return 0; }
  return 1
}

# Directories to search at load time when probing a .so that was built with
# -lcuvs_c: the cuVS library itself plus the HIP runtime.
load_path() {
  local lib dirs="$ROCM_ROOT/lib:$ROCM_ROOT/lib64:/opt/hipvs/lib"
  lib=$(find_cuvs_lib) && dirs="$(dirname "$lib"):$dirs"
  echo "$dirs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

find_repo() {
  if [ -n "${REPO:-}" ] && [ -f "$REPO/tools/hipvs_bridge/seekdb_cuvs_bridge.c" ]; then
    return 0
  fi
  local c
  c=$(find "$HOME" /workspace /root /opt -maxdepth 7 -name seekdb_cuvs_bridge.c 2>/dev/null | head -1)
  [ -n "$c" ] || die "seekdb source tree not found; set REPO=/path/to/seekdb"
  REPO=$(cd "$(dirname "$c")/../.." && pwd)
}

observer_bin() { echo "$REPO/$BUILD_DIR/src/observer/seekdb"; }

render_node() {
  local r
  r=$(ls /dev/dri/renderD* 2>/dev/null | head -1)
  [ -n "$r" ] || die "no /dev/dri/renderD* node"
  echo "$r"
}

# Header grep is informational; the compile+link probe is the authoritative
# check because it fails when the header and the library disagree.
gate_api() {
  local s missing="" tmp cc lib libdir
  log "hipVS CAGRA C API"
  [ -f "$CAGRA_H" ] || die "header not found: $CAGRA_H
hipVS is not installed here, or HIPVS_ROOT is wrong (currently $HIPVS_ROOT)."
  for s in $REQUIRED_CAGRA_SYMS; do
    if grep -qw "$s" "$CAGRA_H"; then
      printf '  header   OK    %s\n' "$s"
    else
      printf '  header   MISSING  %s\n' "$s"; missing="$missing $s"
    fi
  done
  if [ -n "$missing" ]; then
    die "CAGRA header $CAGRA_H lacks:$missing
This hipVS build predates the graph export/import entry points the OBCUVS03
snapshot format needs, so libseekdb_cuvs_bridge.so would build but fail to
dlopen. Install a newer hipVS (the one shipping cuvsCagraIndexFromArgs)."
  fi
  lib=$(find_cuvs_lib) || die "libcuvs_c not found under $HIPVS_ROOT/lib{,64}; set HIPVS_ROOT"
  libdir=$(dirname "$lib")
  echo "  library  $lib"
  tmp=$(mktemp -d)
  cc=${CC:-}
  if [ -z "$cc" ]; then
    if command -v hipcc >/dev/null; then cc=hipcc; else cc=cc; fi
  fi
  # Same include order as the bridge: the CUDA-type shim hipVS ships lives at
  # include/cuvs/cuda_runtime.h, so include/cuvs must be on the search path.
  cat > "$tmp/probe.c" <<'EOF'
#include <cuvs/neighbors/cagra.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#endif
int main(void)
{
  void *get_graph = (void *)cuvsCagraIndexGetGraph;
  void *from_args = (void *)cuvsCagraIndexFromArgs;
  return (get_graph == 0 || from_args == 0) ? 1 : 0;
}
EOF
  if ! "$cc" -x c -D__HIP_PLATFORM_AMD__ \
       -I"$HIPVS_ROOT/include" -I"$HIPVS_ROOT/include/cuvs" -I"$ROCM_ROOT/include" "$tmp/probe.c" \
       -o "$tmp/probe" -L"$libdir" -Wl,-rpath,"$libdir" -lcuvs_c 2>"$tmp/cc.log"; then
    sed 's/^/  /' "$tmp/cc.log"
    rm -rf "$tmp"
    die "probe does not compile+link against $(basename "$lib") -- header and library disagree"
  fi
  if ! LD_LIBRARY_PATH="$(load_path)" "$tmp/probe"; then
    rm -rf "$tmp"
    die "probe linked but did not run cleanly -- either the two symbols are null
or the HIP runtime is not loadable; check LD_LIBRARY_PATH ($ROCM_ROOT/lib)."
  fi
  rm -rf "$tmp"
  echo "  probe    OK    compile+link+resolve against $(basename "$lib")"
}

step_check() {
  log "host"
  nproc; free -g | head -2; df -h "$HOME" | tail -1
  log "gpu"
  (command -v rocm-smi >/dev/null && rocm-smi --showproductname --showmeminfo vram) || echo "rocm-smi unavailable"
  ls -l /dev/kfd /dev/dri 2>&1 | head
  log "roots"
  ls -d "$HIPVS_ROOT" "$ROCM_ROOT" 2>&1
  log "toolchain"
  for t in cmake gcc g++ make python3 mysql; do printf '%s: ' "$t"; command -v "$t" || echo missing; done
  python3 -c 'import numpy; print("numpy", numpy.__version__)' 2>&1
  find_repo
  log "repo"
  echo "REPO=$REPO"
  git -C "$REPO" log --oneline -3 2>/dev/null
  git -C "$REPO" status --short | head -20
  log "patch marker"
  # The baseline already carries an OBCUVS02 snapshot, so the snapshot magic
  # symbol exists before the patch. The v3 marker is the graph export call.
  if grep -q "seekdb_cuvs_export_graph" "$REPO/src/oblib/lib/vector/ob_vsag_adaptor.cpp" 2>/dev/null; then
    echo "patched: OBCUVS03 graph snapshot present"
  else
    echo "not patched"
  fi
  gate_api
}

step_bridge() {
  find_repo
  gate_api
  log "apply patch"
  if grep -q "seekdb_cuvs_export_graph" "$REPO/src/oblib/lib/vector/ob_vsag_adaptor.cpp" 2>/dev/null; then
    echo "patch already applied; skipping"
  else
    [ -f "$PATCH_FILE" ] || die "patch not found: $PATCH_FILE"
    git -C "$REPO" apply --check "$PATCH_FILE" || die "patch does not apply cleanly"
    git -C "$REPO" apply "$PATCH_FILE" || die "git apply failed"
  fi
  git -C "$REPO" diff --stat
  mkdir -p "$BRIDGE_OUT"
  log "build bridge"
  ( cd "$REPO" && HIPVS_ROOT="$HIPVS_ROOT" ROCM_ROOT="$ROCM_ROOT" \
      bash tools/hipvs_bridge/build.sh "$BRIDGE_OUT/libseekdb_cuvs_bridge.so" ) \
    || die "bridge build failed"
  log "symbols"
  nm -D "$BRIDGE_OUT/libseekdb_cuvs_bridge.so" | grep seekdb_cuvs_ || die "symbols missing"
  log "dynamic deps"
  readelf -d "$BRIDGE_OUT/libseekdb_cuvs_bridge.so" 2>/dev/null | grep -E 'NEEDED|RPATH|RUNPATH' || true
  log "dlopen check (RTLD_NOW, so undefined cuVS symbols fail here)"
  LD_LIBRARY_PATH="$(load_path)" python3 - "$BRIDGE_OUT/libseekdb_cuvs_bridge.so" <<'PY' \
    || die "bridge cannot be dlopen'ed; the missing symbol is named above"
import ctypes, sys
ctypes.CDLL(sys.argv[1])   # ctypes uses RTLD_NOW: every symbol must resolve now
print("  dlopen   OK   ", sys.argv[1])
PY
}

step_seekdb() {
  find_repo
  [ -f "$BRIDGE_OUT/libseekdb_cuvs_bridge.so" ] || die "build the bridge first"
  # build.sh owns the third-party dependency layout. Point it at the image's
  # download cache so the build does not refetch several GB.
  local dep_cache=${DEP_CACHE_DIR:-}
  if [ -z "$dep_cache" ] && [ -d /opt/seekdb-deps-cache ]; then dep_cache=/opt/seekdb-deps-cache; fi
  log "build seekdb (-j$JOBS, deps cache ${dep_cache:-none})"
  ( cd "$REPO" && DEP_CACHE_DIR="$dep_cache" ./build.sh release --init \
      -DOB_BUILD_CUVS=ON -DOB_BUILD_CUVS_TRACE="$TRACE_FLAG" \
      -DCUVS_BRIDGE_LIB="$BRIDGE_OUT/libseekdb_cuvs_bridge.so" \
      --make -j "$JOBS" ) || die "seekdb build failed"
  ls -l "$(observer_bin)"
}

step_smoke() {
  find_repo
  local s="$REPO/docs/gpu-vector-index-hipvs-cuvs/validation/run_gpu_smoke.sh"
  [ -f "$s" ] || die "smoke script missing: $s"
  [ -x "$(observer_bin)" ] || die "seekdb binary missing; run: $0 seekdb"
  local render; render=$(render_node)
  log "run GPU smoke"
  ( cd "$REPO" && bash "$s" \
      --observer "$(observer_bin)" \
      --bridge "$BRIDGE_OUT/libseekdb_cuvs_bridge.so" \
      --base-dir "$RUN_ROOT/smoke" \
      --port 2981 --render "$render" \
      --evidence "$HOME/evidence/smoke" )
}

step_bench() {
  find_repo
  [ -f "$BENCH_FILE" ] || die "bench script missing: $BENCH_FILE"
  [ -x "$(observer_bin)" ] || die "seekdb binary missing; run: $0 seekdb"
  log "CPU baseline + CAGRA build-parameter sweep"
  # SETTINGS replaces the whole row list, CPU baseline included, so a caller can
  # sweep only GPU rows. The default keeps lib=vsag first: a build-speed claim
  # needs its CPU reference in the same table.
  local settings
  if [ -n "${SETTINGS:-}" ]; then
    settings=$SETTINGS
  else
    settings="lib=vsag;deg=128,inter=128,niter=32;deg=128,inter=192,niter=32;deg=128,inter=256,niter=100"
  fi
  bash "$BENCH_FILE" \
    --observer "$(observer_bin)" \
    --bridge "$BRIDGE_OUT/libseekdb_cuvs_bridge.so" \
    --base-dir "$RUN_ROOT/bench" \
    --port 2989 \
    --evidence "$HOME/evidence/bench" \
    --rows "${ROWS:-250000}" --queries "${NQ:-200}" --topk "${TOPK:-10}" \
    --settings "$settings" \
    ${RESTART_CHECK:+--restart-check}
}

step_report() {
  find_repo
  local r="$REPO/docs/gpu-vector-index-hipvs-cuvs/validation/render_perf_report.py"
  [ -f "$r" ] || r="$SCRIPT_DIR/render_perf_report.py"
  [ -f "$r" ] || die "report renderer missing (looked in the repo and $SCRIPT_DIR)"
  [ -f "$HOME/evidence/bench/sweep.tsv" ] || die "no sweep yet; run: $0 bench"
  log "render PERF_REPORT.md"
  python3 "$r" \
    --sweep "$HOME/evidence/bench/sweep.tsv" \
    ${RESTART_TSV:+--restart "$RESTART_TSV"} \
    --meta "rows=${ROWS:-250000} dim=${DIM:-128} queries=${NQ:-200} topk=${TOPK:-10}" \
    --report "$REPO/docs/gpu-vector-index-hipvs-cuvs/PERF_REPORT.md"
}

case "${1:-check}" in
  check)  step_check ;;
  bridge) step_bridge ;;
  seekdb) step_seekdb ;;
  smoke)  step_smoke ;;
  bench)  step_bench ;;
  report) step_report ;;
  all)    step_check && step_bridge && step_seekdb ;;
  *)      die "unknown step: $1" ;;
esac
