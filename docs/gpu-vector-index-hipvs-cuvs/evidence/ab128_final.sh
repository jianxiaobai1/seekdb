#!/bin/bash
# Same-round A/B at 250k x 128 (SIFT-scale): frozen baseline binary vs the candidate at
# the shipped defaults (nn_descent, inter=2xdegree, niter=100), with VRAM sampling.
P=/workspace/ab
PORT=2991
BASE_BIN=/workspace/src_base/build_release/src/observer/seekdb
CAND_BIN=/workspace/src2/build_release/src/observer/seekdb
export SEEKDB_CUVS_BUILD_ALGO=0
export SEEKDB_CUVS_DEBUG=1
sql() { mysql --connect-timeout=15 --protocol=TCP -h127.0.0.1 -P$PORT -uroot -A -N -B "$@"; }
up() { for i in $(seq 1 400); do sql -e "select 1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
vram_on() { : > $P/$1.vram; : > $P/$1.marks; bash /workspace/vram_sampler.sh $P/$1.vram & echo $! > $P/$1.vrampid; }
vram_off() { kill $(cat $P/$1.vrampid) 2>/dev/null; sleep 1; }
mark() { echo "$(date +%s.%N) $2" >> $P/$1.marks; }
arm() {
  tag=$1; bin=$2; lib=$3; niter=$4
  export SEEKDB_CUVS_NN_DESCENT_NITER=$niter
  rm -rf $P/$tag/store; mkdir -p $P/$tag/store
  vram_on $tag
  LD_LIBRARY_PATH=$lib:/opt/hipvs/lib:/opt/rocm/lib nohup $bin --nodaemon --port $PORT --data-dir $P/$tag/store --base-dir $P/$tag --parameter __min_full_resource_pool_memory=2147483648 --parameter memory_budget=12G --parameter datafile_size=12G --parameter cpu_count=32 --parameter local_ip=127.0.0.1 >> $P/$tag.observer.log 2>&1 &
  echo $! > $P/$tag.pid
  up || { echo "FAIL start $tag"; exit 1; }
  sql -e "alter system set ob_vector_memory_limit_percentage=30; create database if not exists ab;"
  sql -e "set global ob_query_timeout=6000000000"; sql -e "set global ob_trx_timeout=6000000000"; sql -e "set global ob_trx_idle_timeout=6000000000"
  sql -e "use ab; drop table if exists items; create table items(id bigint primary key, v vector(128));"
  mark $tag load_start
  mysql --local-infile=1 --protocol=TCP -h127.0.0.1 -P$PORT -uroot -e "use ab; load data local infile '$P/base.tsv' into table items fields terminated by '\t' lines terminated by '\n' (id,v);" || { echo "FAIL load"; exit 1; }
  echo "ROWS=$(sql -e 'select count(*) from ab.items')"
  mark $tag load_end
  mark $tag ddl_start
  t0=$(date +%s%N)
  sql -e "use ab; create vector index idx on items(v) with (distance=l2,type=hnsw,lib=cuvs,m=16,ef_construction=200,ef_search=64);" || { echo "FAIL ddl $tag"; return 1; }
  t1=$(date +%s%N)
  Q=$(head -1 $P/queries.txt)
  sql -e "use ab; select id from items order by l2_distance(v,$Q) approximate limit 10" > /dev/null || return 1
  t2=$(date +%s%N)
  mark $tag ddl_end
  echo "RESULT $tag niter=$niter ddl_ms=$(( (t1-t0)/1000000 )) first_q_ms=$(( (t2-t1)/1000000 )) window_ms=$(( (t2-t0)/1000000 ))"
  : > $P/$tag.k100.txt
  t3=$(date +%s%N)
  while IFS= read -r q; do sql -e "use ab; select id from items order by l2_distance(v,$q) approximate limit 100" >> $P/$tag.k100.txt; done < $P/queries.txt
  t4=$(date +%s%N)
  echo "QPS $tag top100 = $(python3 -c "print(round(100/((($t4-$t3)/1000000)/1000),1))") q/s"
  python3 /workspace/recall_ab.py $P/exact.txt $P/$tag.k100.txt | sed "s/^/RECALL $tag /"
  : > $P/$tag.k10.txt
  t5=$(date +%s%N)
  while IFS= read -r q; do sql -e "use ab; select id from items order by l2_distance(v,$q) approximate limit 10" >> $P/$tag.k10.txt; done < $P/queries.txt
  t6=$(date +%s%N)
  echo "QPS $tag top10 = $(python3 -c "print(round(100/((($t6-$t5)/1000000)/1000),1))") q/s"
  pid=$(cat $P/$tag.pid); kill -TERM $pid 2>/dev/null
  for i in $(seq 1 180); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; sleep 5; vram_off $tag
  rm -rf $P/$tag/store
}
arm b128base $BASE_BIN /workspace/out_base 100
arm c128i100 $CAND_BIN /workspace/out_final 100
echo D128_AB_DONE
