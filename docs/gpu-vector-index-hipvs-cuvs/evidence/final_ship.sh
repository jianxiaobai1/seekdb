#!/bin/bash
# Certification run for the configuration that is actually submitted:
#   - candidate binary (branch amd-cagra-v2) + repo-exact bridge (/workspace/out_final)
#   - NO build-knob environment variables at all -> shipped defaults (nn_descent,
#     intermediate=2*degree, niter=100)
#   - same 250k x 256 table as the same-round baseline arm
# Then a graceful stop + restart on the same data-dir, to prove the OBCUVS03 snapshot
# is what serves queries after a restart (no rebuild) and that results are unchanged.
set -u
P=/workspace/final_ship
SRC=/workspace/ab256
PORT=2992
CAND_BIN=/workspace/src2/build_release/src/observer/seekdb
LIBDIR=/workspace/out_final

unset SEEKDB_CUVS_BUILD_ALGO SEEKDB_CUVS_AN_FP16 SEEKDB_CUVS_GRAPH_DEGREE
unset SEEKDB_CUVS_INTERMEDIATE_DEGREE SEEKDB_CUVS_NN_DESCENT_NITER
export SEEKDB_CUVS_DEBUG=1

mkdir -p $P
ln -sf $SRC/base.tsv $SRC/queries.txt $SRC/exact.txt $P/

sql() { mysql --connect-timeout=15 --protocol=TCP -h127.0.0.1 -P$PORT -uroot -A -N -B "$@"; }
up() {
  for i in $(seq 1 600); do sql -e "select 1" >/dev/null 2>&1 && return 0; sleep 1; done
  return 1
}
start_obs() {
  LD_LIBRARY_PATH=$LIBDIR:/opt/hipvs/lib:/opt/rocm/lib nohup $CAND_BIN --nodaemon --port $PORT \
    --data-dir $P/store --base-dir $P \
    --parameter __min_full_resource_pool_memory=2147483648 --parameter memory_budget=12G \
    --parameter datafile_size=12G --parameter cpu_count=32 --parameter local_ip=127.0.0.1 \
    >> $P/observer.log 2>&1 &
  echo $! > $P/obs.pid
  up || { echo "FAIL start"; exit 1; }
}
stop_obs() {
  pid=$(cat $P/obs.pid); kill -TERM $pid 2>/dev/null
  for i in $(seq 1 300); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; sleep 8
}

rm -rf $P/store $P/log; mkdir -p $P/store
: > $P/observer.log; : > $P/store.marks
bash /workspace/vram_sampler.sh $P/store.vram & echo $! > $P/vrampid

start_obs
sql -e "alter system set ob_vector_memory_limit_percentage=30; create database if not exists fs;"
for g in ob_query_timeout ob_trx_timeout ob_trx_idle_timeout; do
  sql -e "set global $g=6000000000"
done
sql -e "use fs; create table items(id bigint primary key, v vector(256));"
mysql --local-infile=1 --protocol=TCP -h127.0.0.1 -P$PORT -uroot \
  -e "use fs; load data local infile '$SRC/base.tsv' into table items fields terminated by '\t' lines terminated by '\n' (id,v);" \
  || { echo "FAIL load"; exit 1; }
echo "ROWS=$(sql -e 'select count(*) from fs.items')"

echo "$(date +%s.%N) ddl_start" >> $P/store.marks
t0=$(date +%s%N)
sql -e "use fs; create vector index idx on items(v) with (distance=l2,type=hnsw,lib=cuvs,m=16,ef_construction=200,ef_search=64);" \
  || { echo "FAIL ddl"; exit 1; }
Q=$(head -1 $SRC/queries.txt)
t1=$(date +%s%N)
sql -e "use fs; select id from items order by l2_distance(v,$Q) approximate limit 10" > /dev/null
t2=$(date +%s%N)
echo "$(date +%s.%N) ddl_end" >> $P/store.marks
echo "RESULT ship ddl_ms=$(( (t1-t0)/1000000 )) first_q_ms=$(( (t2-t1)/1000000 )) window_ms=$(( (t2-t0)/1000000 ))"
grep -ao 'serialized GPU CAGRA snapshot([^)]*)' $P/log/observer.log | tail -2
grep -ao 'all_neighbors[^)]*)' $P/log/observer.log | tail -1

apx() {
  K=$1; sfx=$2; out=$P/$sfx.k$K.txt; : > $out
  t0=$(date +%s%N)
  while IFS= read -r q; do
    sql -e "use fs; select id from items order by l2_distance(v,$q) approximate limit $K" >> $out
  done < $SRC/queries.txt
  t1=$(date +%s%N)
  echo "$sfx QPS top$K = $(python3 -c "print(round(100/((($t1-$t0)/1000000)/1000),1))") q/s"
  python3 /workspace/recall_ab.py $SRC/exact.txt $out | sed "s|^|$sfx RECALL |"
}
apx 100 pre
apx 10 pre

stop_obs
kill $(cat $P/vrampid) 2>/dev/null
python3 - <<'PY'
marks = {}
for ln in open('/workspace/final_ship/store.marks'):
    t, k = ln.split()
    marks[k] = float(t)
lo, hi = marks['ddl_start'], marks['ddl_end']
allw = {}; win = {}; n = 0
for ln in open('/workspace/final_ship/store.vram'):
    p = ln.split()
    if len(p) != 3:
        continue
    t, card, v = float(p[0]), p[1], int(p[2])
    n += 1
    allw[card] = max(allw.get(card, 0), v)
    if lo - 0.5 <= t <= hi + 0.5:
        win[card] = max(win.get(card, 0), v)
print('VRAM samples=%d ddl_peak=%.3fGB cards=%s' % (
    n, max(win.values()) / 2.0**30,
    ' '.join('%s:%.3f' % (c, v / 2.0**30) for c, v in sorted(win.items(), key=lambda x: -x[1])[:3])))
PY

echo "=== PHASE 2: restart on the same data-dir (no load, no DDL) ==="
cp $P/log/observer.log $P/phase1.observer.log
before=$(grep -ac 'serialized GPU CAGRA snapshot' $P/log/observer.log)
start_obs
t0=$(date +%s%N)
sql -e "use fs; select id from items order by l2_distance(v,$Q) approximate limit 10" > /dev/null \
  || { echo "FAIL first query after restart"; exit 1; }
t1=$(date +%s%N)
echo "RESULT restarted first_q_ms=$(( (t1-t0)/1000000 ))"
apx 100 post
apx 10 post
diff -q $P/pre.k100.txt $P/post.k100.txt > /dev/null \
  && echo "POST top100 ids identical to pre-restart" || echo "POST top100 ids DIFFER"
diff -q $P/pre.k10.txt $P/post.k10.txt > /dev/null \
  && echo "POST top10 ids identical to pre-restart" || echo "POST top10 ids DIFFER"
echo "snapshot_serialize_lines_before=$before after=$(grep -ac 'serialized GPU CAGRA snapshot' $P/log/observer.log)"
grep -ao 'deserialized GPU CAGRA snapshot([^)]*)' $P/log/observer.log | tail -2
stop_obs
echo FINAL_SHIP_DONE
