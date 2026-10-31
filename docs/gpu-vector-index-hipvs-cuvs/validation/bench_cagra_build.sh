#!/usr/bin/env bash
# On-instance benchmark: CAGRA build parameters vs build window, QPS, recall.
# Runs one seekdb observer per setting against the same synthetic dataset.
# A setting may name lib=vsag (CPU HNSW baseline) or lib=cuvs (default, GPU
# CAGRA build); the remaining keys deg/inter/niter/algo only apply to cuvs.
set -Eeuo pipefail

OBSERVER=""; BRIDGE=""; BASE_DIR=""; PORT=2989; EVIDENCE=""
ROWS=250000; DIM=128; NQ=200; TOPK=10
SETTINGS=("lib=vsag" "deg=128,inter=256,niter=100" "deg=128,inter=192,niter=32" "deg=128,inter=128,niter=32")
RESTART_CHECK=0

while (($# > 0)); do
  case "$1" in
    --observer) OBSERVER=$2; shift 2 ;;
    --bridge) BRIDGE=$2; shift 2 ;;
    --base-dir) BASE_DIR=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --evidence) EVIDENCE=$2; shift 2 ;;
    --rows) ROWS=$2; shift 2 ;;
    --dim) DIM=$2; shift 2 ;;
    --queries) NQ=$2; shift 2 ;;
    --topk) TOPK=$2; shift 2 ;;
    --settings) IFS=';' read -r -a SETTINGS <<< "$2"; shift 2 ;;
    --restart-check) RESTART_CHECK=1; shift ;;
    -h|--help) sed -n '2,4p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -x "$OBSERVER" ]] || { echo "observer not executable: $OBSERVER" >&2; exit 2; }
[[ -f "$BRIDGE" ]] || { echo "bridge missing: $BRIDGE" >&2; exit 2; }
[[ -n "$BASE_DIR" && "$BASE_DIR" != "/" ]] || { echo "unsafe base-dir" >&2; exit 2; }
[[ -n "$EVIDENCE" && "$EVIDENCE" != "/" ]] || { echo "unsafe evidence" >&2; exit 2; }

DATA=$EVIDENCE/data
mkdir -p "$BASE_DIR" "$EVIDENCE" "$DATA"
RESULT=$EVIDENCE/sweep.tsv
printf 'setting\tddl_s\tddl_build_s\tfirst_query_s\tbuild_window_s\tqps\trecall@%s\tcuvs_build\tcuvs_serve\n' "$TOPK" > "$RESULT"

mysql_cmd() { timeout 600 mysql --connect-timeout=5 --protocol=TCP -h127.0.0.1 -P"$1" -uroot -A -N -B "${@:2}"; }
now() { date +%s.%N; }
elapsed() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f", b-a}'; }

if [[ ! -f "$DATA/done" ]]; then
  python3 - "$DATA" "$ROWS" "$DIM" "$NQ" "$TOPK" <<'PY'
import pathlib, sys
import numpy as np

out = pathlib.Path(sys.argv[1]); n, dim, nq, topk = map(int, sys.argv[2:6])
rng = np.random.default_rng(20261029)
base = rng.normal(size=(n, dim)).astype(np.float32)
base /= np.linalg.norm(base, axis=1, keepdims=True)
idx = rng.choice(n, size=nq, replace=False)
queries = base[idx] + rng.normal(scale=0.02, size=(nq, dim)).astype(np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)
with (out / "base.tsv").open("w") as f:
    for i, row in enumerate(base):
        f.write(str(i) + "\t[" + ",".join(f"{v:.7f}" for v in row) + "]\n")
np.save(out / "queries.npy", queries)
gt = np.empty((nq, 100), dtype=np.int32)
for s in range(0, nq, 25):
    d = np.sum((base[None, :, :] - queries[s:s+25, None, :]) ** 2, axis=2)
    gt[s:s+25] = np.argsort(d, axis=1)[:, :100]
np.save(out / "gt.npy", gt)
(out / "query0.txt").write_text("[" + ",".join(f"{v:.7f}" for v in queries[0]) + "]")
with (out / "q.sql").open("w") as f:
    f.write("use bench;\n")
    for row in queries:
        f.write("select id from t order by l2_distance(v,'[" + ",".join(f"{v:.7f}" for v in row) + "]') approximate limit " + str(topk) + ";\n")
(out / "done").write_text("1")
PY
fi

QUERY=$(cat "$DATA/query0.txt")

start_observer() {
  local dir=$1 port=$2 trace=$3 log=$4 env_prefix=$5
  ulimit -n 65535
  env -u OB_VSAG_USE_CUVS $env_prefix \
    "LD_LIBRARY_PATH=$(dirname "$BRIDGE"):/opt/hipvs/lib:/opt/rocm/lib" \
    "OB_VSAG_TRACE=1" "OB_VSAG_TRACE_FILE=$trace" \
    "$OBSERVER" --nodaemon --port "$port" \
      --data-dir "$dir/store" --base-dir "$dir" \
      --parameter __min_full_resource_pool_memory=2147483648 \
      --parameter memory_budget=16G --parameter datafile_size=40G \
      --parameter cpu_count=32 --parameter local_ip=127.0.0.1 >"$log" 2>&1 &
  PID=$!
  for _ in $(seq 1 180); do
    [[ -d "/proc/$PID" ]] || { echo "observer exited, see $log" >&2; return 1; }
    mysql_cmd "$port" -e 'select 1' >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

stop_observer() {
  [[ -n "${PID:-}" && -d "/proc/$PID" ]] || return 0
  kill -TERM "$PID" 2>/dev/null || true
  for _ in $(seq 1 30); do [[ -d "/proc/$PID" ]] || break; sleep 1; done
  [[ -d "/proc/$PID" ]] && kill -KILL "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
  PID=""
}
trap 'stop_observer' EXIT

for setting in "${SETTINGS[@]}"; do
  IFS=',' read -r -a kv <<< "$setting"
  lib=cuvs; deg=128; inter=""; niter=32; algo=nn_descent; itopk=""; width=""; label_override=""
  for pair in "${kv[@]}"; do
    case "$pair" in
      lib=*) lib=${pair#*=} ;;
      deg=*) deg=${pair#*=} ;;
      inter=*) inter=${pair#*=} ;;
      niter=*) niter=${pair#*=} ;;
      algo=*) algo=${pair#*=} ;;
      itopk=*) itopk=${pair#*=} ;;
      width=*) width=${pair#*=} ;;
      label=*) label_override=${pair#*=} ;;
    esac
  done
  # lib=vsag is the CPU HNSW builder and is the comparison baseline: same DDL,
  # same m/ef_construction, no CAGRA build parameters apply.
  if [[ "$lib" == "vsag" ]]; then
    label="cpu_vsag_m16_efc200"
    lib_clause=""
    build_env=""
  else
    label="gpu_cuvs_deg${deg}_inter${inter:-$deg}_niter${niter}_${algo}"
    [[ -n "$itopk" ]] && label="${label}_itopk${itopk}"
    [[ -n "$width" ]] && label="${label}_w${width}"
    lib_clause=",lib=cuvs"
    build_env="SEEKDB_CUVS_GRAPH_DEGREE=$deg SEEKDB_CUVS_INTERMEDIATE_DEGREE=${inter:-$deg} SEEKDB_CUVS_NN_DESCENT_NITER=$niter SEEKDB_CUVS_BUILD_ALGO=$algo"
    # Serving knobs apply on top of the same graph; they steer QPS vs recall only.
    [[ -n "$itopk" ]] && build_env="$build_env SEEKDB_CUVS_SEARCH_ITOPK=$itopk"
    [[ -n "$width" ]] && build_env="$build_env SEEKDB_CUVS_SEARCH_WIDTH=$width"
  fi
  # label= overrides the generated name, so an unpatched baseline checkout can
  # be swept by the same script and land in the same sweep.tsv.
  [[ -n "$label_override" ]] && label=$label_override
  run="$BASE_DIR/$label"
  rm -rf "$run"; mkdir -p "$run"
  trace="$EVIDENCE/$label.trace"; log="$EVIDENCE/$label.log"
  : > "$trace"

  start_observer "$run" "$PORT" "$trace" "$log" "$build_env" \
    || { echo "start failed: $label" >&2; stop_observer; continue; }

  mysql_cmd "$PORT" -e 'alter system set ob_vector_memory_limit_percentage=30; create database if not exists bench; use bench; create table t(id bigint primary key, v vector('"$DIM"'));' >/dev/null
  mysql --local-infile=1 --connect-timeout=5 --protocol=TCP -h127.0.0.1 -P"$PORT" -uroot -N -B \
    -e "use bench; load data local infile '$DATA/base.tsv' into table t fields terminated by '\\t' lines terminated by '\\n' (id,v);" >/dev/null

  t0=$(now)
  mysql_cmd "$PORT" -e 'use bench; create vector index idx on t(v) with (distance=l2,type=hnsw'"$lib_clause"',m=16,ef_construction=200,ef_search=64);' >/dev/null
  t1=$(now)
  # The CAGRA build inside the DDL is timed by the adaptor itself; the trace only
  # records lazy post-DDL rebuilds, so pull the GPU build time from the log.
  build_us=$(grep -ho 'built=true, build_us=[0-9]*' "$run/log/"*.log 2>/dev/null | tail -1 | grep -o '[0-9]*$' || true)
  if [[ -n "$build_us" ]]; then
    ddl_build=$(awk -v u="$build_us" 'BEGIN{printf "%.3f", u/1000000}')
  else
    ddl_build="-"
  fi

  q0=$(now)
  mysql_cmd "$PORT" -e "use bench; select id from t order by l2_distance(v,'$QUERY') approximate limit $TOPK;" > "$EVIDENCE/$label.first.ids"
  q1=$(now)

  b0=$(now)
  mysql_cmd "$PORT" < "$DATA/q.sql" > "$EVIDENCE/$label.topk.tsv"
  b1=$(now)

  python3 - "$DATA/gt.npy" "$EVIDENCE/$label.topk.tsv" "$TOPK" > "$EVIDENCE/$label.recall" <<'PY'
import sys, numpy as np
gt = np.load(sys.argv[1]); topk = int(sys.argv[3])
lines = [l.strip() for l in open(sys.argv[2]) if l.strip()]
ids_per_line = [[int(x) for x in line.split("\t") if x] for line in lines]
# mysql -N -B prints one id per row, so a query's top-k spans topk lines;
# accept the one-line-per-query layout too.
if len(ids_per_line) != len(gt):
    flat = [i for ids in ids_per_line for i in ids]
    per_q = len(flat) // len(gt)
    ids_per_line = [flat[i * per_q:(i + 1) * per_q] for i in range(len(gt))]
hits = total = 0
for truth_row, ids in zip(gt, ids_per_line):
    truth = set(truth_row[:topk].tolist())
    hits += len(truth & set(ids[:topk])); total += topk
print(f"{hits/total:.4f}" if total else "0")
PY
  recall=$(cat "$EVIDENCE/$label.recall")
  ddl_s=$(elapsed "$t0" "$t1")
  first_s=$(elapsed "$q0" "$q1")
  window_s=$(awk -v a="$ddl_s" -v b="$first_s" 'BEGIN{printf "%.3f", a+b}')
  qps=$(awk -v a="$b0" -v b="$b1" -v n="$NQ" 'BEGIN{d=b-a; printf "%.1f", (d>0? n/d : 0)}')
  builds=$(grep -c '^cuvs_build' "$trace" || true)
  serves=$(grep -c '^cuvs_serve' "$trace" || true)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$ddl_s" "$ddl_build" "$first_s" "$window_s" "$qps" "$recall" "$builds" "$serves" >> "$RESULT"

  if ((RESTART_CHECK == 1)) && [[ "$lib" != "vsag" ]]; then
    stop_observer
    mysql_cmd "$PORT" -e 'select 1' >/dev/null 2>&1 && { echo "port still open" >&2; continue; }
    r0=$(now)
    start_observer "$run" "$PORT" "$trace" "$log.restart" "$build_env" || continue
    mysql_cmd "$PORT" -e "use bench; select id from t order by l2_distance(v,'$QUERY') approximate limit $TOPK;" > "$EVIDENCE/$label.restart.ids"
    r1=$(now)
    rb=$(tail -n +1 "$trace" | awk '$1=="cuvs_build"{c++} END{print c+0}')
    # A restart must rehydrate the index from the snapshot, not rebuild it: the
    # adaptor logs the import with the byte count it validated.
    ri=$(grep -ho 'deserialized GPU CAGRA snapshot.*graph_bytes=[0-9]*' "$run/log/"*.log 2>/dev/null | tail -1 | grep -o '[0-9]*$' || true)
    [[ -f "$EVIDENCE/restart.tsv" ]] || printf 'run\trestart_first_query_s\tbuilds_in_trace\timported_graph_bytes\n' > "$EVIDENCE/restart.tsv"
    echo -e "${label}_restart\t$(elapsed "$r0" "$r1")s\tbuilds_in_trace=$rb\t${ri:--}" >> "$EVIDENCE/restart.tsv"
    stop_observer
  fi

  stop_observer
  mysql_cmd "$PORT" -e 'select 1' >/dev/null 2>&1 && { echo "port $PORT still open" >&2; exit 1; }
  echo "done $label ddl=${ddl_s}s first=${first_s}s qps=${qps} recall=${recall}"
done

if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$RESULT"
else
  cat "$RESULT"
fi
