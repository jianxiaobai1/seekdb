#!/bin/bash
# Final same-round A/B at 250k x 256: frozen baseline build vs the shipped bridge
# (default graph builder = nn_descent niter=80, SEEKDB_CUVS_BUILD_ALGO=0),
# with VRAM peak sampling and Top-10/Top-100 QPS for the query gate.
P=/workspace/ab256
PORT=2992
BASE_BIN=/workspace/src_base/build_release/src/observer/seekdb
CAND_BIN=/workspace/src2/build_release/src/observer/seekdb
export SEEKDB_CUVS_BUILD_ALGO=0
export SEEKDB_CUVS_DEBUG=1

sql() { mysql --connect-timeout=15 --protocol=TCP -h127.0.0.1 -P$PORT -uroot -A -N -B "$@"; }
up() { for i in $(seq 1 400); do sql -e "select 1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
vram_on() { : > $P/$1.vram; : > $P/$1.marks; bash /workspace/vram_sampler.sh $P/$1.vram & echo $! > $P/$1.vrampid; }
vram_off() { kill $(cat $P/$1.vrampid) 2>/dev/null; sleep 1; awk '{if($3+0>m[$2]+0)m[$2]=$3}END{for(k in m)printf "PEAK %s %s=%d ",k,m[k];print ""}' $P/$1.vram; }
mark() { echo "$(date +%s.%N) $2" >> $P/$1.marks; }
start_obs() {
  tag=$1; bin=$2; libdir=$3; dd=$4
  rm -rf $dd; mkdir -p $dd
  vram_on $tag
  LD_LIBRARY_PATH=$libdir:/opt/hipvs/lib:/opt/rocm/lib nohup $bin --nodaemon --port $PORT --data-dir $dd --base-dir $(dirname $dd) --parameter __min_full_resource_pool_memory=2147483648 --parameter memory_budget=12G --parameter datafile_size=12G --parameter cpu_count=32 --parameter local_ip=127.0.0.1 >> $P/$tag.observer.log 2>&1 &
  echo $! > $P/$tag.pid
  up || { echo "FAIL start $tag"; exit 1; }
}
stop_obs() {
  tag=$1; pid=$(cat $P/$tag.pid); kill -TERM $pid 2>/dev/null
  for i in $(seq 1 180); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; sleep 5; vram_off $tag
}
load_data() {
  mark $1 load_start
  sql -e "alter system set ob_vector_memory_limit_percentage=30; create database if not exists ab;"
  sql -e "set global ob_query_timeout=6000000000"; sql -e "set global ob_trx_timeout=6000000000"; sql -e "set global ob_trx_idle_timeout=6000000000"; sql -e "set global max_allowed_packet=2000000000"
  sql -e "use ab; create table items(id bigint primary key, v vector(256));"
  mysql --local-infile=1 --protocol=TCP -h127.0.0.1 -P$PORT -uroot -e "use ab; load data local infile '$P/base.tsv' into table items fields terminated by '\t' lines terminated by '\n' (id,v);" || { echo "FAIL load"; exit 1; }
  echo "ROWS=$(sql -e 'select count(*) from ab.items')"
  mark $1 load_end
}
build_index() {
  tag=$1
  mark $1 ddl_start
  t0=$(date +%s%N)
  sql -e "use ab; create vector index idx on items(v) with (distance=l2,type=hnsw,lib=cuvs,m=16,ef_construction=200,ef_search=64);" || { echo "FAIL ddl $tag"; return 1; }
  t1=$(date +%s%N)
  Q=$(head -1 $P/queries.txt)
  sql -e "use ab; select id from items order by l2_distance(v,$Q) approximate limit 10" > /dev/null || return 1
  t2=$(date +%s%N)
  mark $1 ddl_end
  echo "RESULT $tag ddl_ms=$(( (t1-t0)/1000000 )) first_q_ms=$(( (t2-t1)/1000000 )) window_ms=$(( (t2-t0)/1000000 ))"
  grep -h all_neighbors $P/$tag.observer.log | tail -2
}
approx() {
  tag=$1; K=$2
  : > $P/$tag.k$K.txt
  t0=$(date +%s%N)
  while IFS= read -r q; do sql -e "use ab; select id from items order by l2_distance(v,$q) approximate limit $K" >> $P/$tag.k$K.txt; done < $P/queries.txt
  t1=$(date +%s%N)
  echo "QPS $tag top$K = $(python3 -c "print(round(100/((($t1-$t0)/1000000)/1000),1))") q/s"
  if [ "$K" = "100" ]; then python3 /workspace/recall_ab.py $P/exact.txt $P/$tag.k$K.txt | sed "s/^/RECALL $tag /"; fi
}

echo "### SHIP ARM A: frozen baseline binary + baseline bridge"
start_obs nn_base $BASE_BIN /workspace/out_base $P/nn_base/store
load_data nn_base
build_index nn_base
approx nn_base 100
approx nn_base 10
stop_obs nn_base

echo "### SHIP ARM B: candidate binary + shipped bridge (default = nn_descent niter=80)"
start_obs nn_cand $CAND_BIN /workspace/out_ship $P/nn_cand/store
load_data nn_cand
build_index nn_cand
approx nn_cand 100
approx nn_cand 10
stop_obs nn_cand
echo NN_AB_DONE
