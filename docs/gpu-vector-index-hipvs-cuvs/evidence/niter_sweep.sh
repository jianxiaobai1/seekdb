#!/bin/bash
# Candidate-only arms at 250k x 256 and 250k x 128, nn_descent with different refine
# round counts, so the recall drop vs the same-round baseline can be picked per niter.
P=/workspace/ab256
PORT=2992
CAND_BIN=/workspace/src2/build_release/src/observer/seekdb
export SEEKDB_CUVS_BUILD_ALGO=0
export SEEKDB_CUVS_DEBUG=1
sql() { mysql --connect-timeout=15 --protocol=TCP -h127.0.0.1 -P$PORT -uroot -A -N -B "$@"; }
up() { for i in $(seq 1 400); do sql -e "select 1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
vram_on() { : > $P/$1.vram; : > $P/$1.marks; bash /workspace/vram_sampler.sh $P/$1.vram & echo $! > $P/$1.vrampid; }
vram_off() { kill $(cat $P/$1.vrampid) 2>/dev/null; sleep 1; }
mark() { echo "$(date +%s.%N) $2" >> $P/$1.marks; }
start_obs() {
  tag=$1; dim=$2; libdir=$3; dd=$4
  rm -rf $dd; mkdir -p $dd
  vram_on $tag
  LD_LIBRARY_PATH=$libdir:/opt/hipvs/lib:/opt/rocm/lib nohup $CAND_BIN --nodaemon --port $PORT --data-dir $dd --base-dir $(dirname $dd) --parameter __min_full_resource_pool_memory=2147483648 --parameter memory_budget=12G --parameter datafile_size=12G --parameter cpu_count=32 --parameter local_ip=127.0.0.1 >> $P/$tag.observer.log 2>&1 &
  echo $! > $P/$tag.pid
  up || { echo "FAIL start $tag"; exit 1; }
  sql -e "alter system set ob_vector_memory_limit_percentage=30; create database if not exists ab;"
  sql -e "set global ob_query_timeout=6000000000"; sql -e "set global ob_trx_timeout=6000000000"; sql -e "set global ob_trx_idle_timeout=6000000000"
  sql -e "use ab; create table items(id bigint primary key, v vector($dim));"
}
stop_obs() {
  tag=$1; pid=$(cat $P/$tag.pid); kill -TERM $pid 2>/dev/null
  for i in $(seq 1 180); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; sleep 5; vram_off $tag
}
arm() {
  tag=$1; niter=$2; dim=$3
  export SEEKDB_CUVS_NN_DESCENT_NITER=$niter
  if [ "$dim" = "256" ]; then SRC=$P; else SRC=/workspace/ab; fi
  start_obs $tag $dim /workspace/out_ship $SRC/$tag/store
  mark $tag load_start
  mysql --local-infile=1 --protocol=TCP -h127.0.0.1 -P$PORT -uroot -e "use ab; load data local infile '$SRC/base.tsv' into table items fields terminated by '\t' lines terminated by '\n' (id,v);" || { echo "FAIL load"; exit 1; }
  echo "ROWS=$(sql -e 'select count(*) from ab.items')"
  mark $tag load_end
  mark $tag ddl_start
  t0=$(date +%s%N)
  sql -e "use ab; create vector index idx on items(v) with (distance=l2,type=hnsw,lib=cuvs,m=16,ef_construction=200,ef_search=64);" || { echo "FAIL ddl $tag"; return 1; }
  t1=$(date +%s%N)
  Q=$(head -1 $SRC/queries.txt)
  sql -e "use ab; select id from items order by l2_distance(v,$Q) approximate limit 10" > /dev/null || return 1
  t2=$(date +%s%N)
  mark $tag ddl_end
  echo "RESULT $tag niter=$niter dim=$dim ddl_ms=$(( (t1-t0)/1000000 )) first_q_ms=$(( (t2-t1)/1000000 )) window_ms=$(( (t2-t0)/1000000 ))"
  : > $SRC/$tag.k100.txt
  t3=$(date +%s%N)
  while IFS= read -r q; do sql -e "use ab; select id from items order by l2_distance(v,$q) approximate limit 100" >> $SRC/$tag.k100.txt; done < $SRC/queries.txt
  t4=$(date +%s%N)
  echo "QPS $tag top100 = $(python3 -c "print(round(100/((($t4-$t3)/1000000)/1000),1))") q/s"
  python3 /workspace/recall_ab.py $SRC/exact.txt $SRC/$tag.k100.txt | sed "s/^/RECALL $tag /"
  : > $SRC/$tag.k10.txt
  t5=$(date +%s%N)
  while IFS= read -r q; do sql -e "use ab; select id from items order by l2_distance(v,$q) approximate limit 10" >> $SRC/$tag.k10.txt; done < $SRC/queries.txt
  t6=$(date +%s%N)
  echo "QPS $tag top10 = $(python3 -c "print(round(100/((($t6-$t5)/1000000)/1000),1))") q/s"
  stop_obs $tag
}
arm n256i100 100 256
arm n256i64 64 256
arm d128i100 100 128
arm d128i64 64 128
arm d128i80 80 128
echo NITER_AB_DONE
