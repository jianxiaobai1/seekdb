#!/bin/bash
# Two things the report still owed:
#  (1) DDL contract checks from the baseline's CI-safe mysqltest
#      (tools/deploy/mysql_test/test_suite/vector_index/t/vector_index_cuvs_ddl.test)
#      re-run against the candidate binary.
#  (2) A real CPU (lib=vsag HNSW) build on the same instance, same binary tree, same
#      250k x 128 table as the GPU arms -> the CPU/GPU comparison the task asks for,
#      and proof that the untouched vsag path still works after the adaptor rewrite.
set -u
P=/workspace/cpu_vsag
SRC=/workspace/ab
PORT=2993
CAND_BIN=/workspace/src2/build_release/src/observer/seekdb
LIBDIR=/workspace/out_final
unset SEEKDB_CUVS_BUILD_ALGO SEEKDB_CUVS_AN_FP16 SEEKDB_CUVS_GRAPH_DEGREE
unset SEEKDB_CUVS_INTERMEDIATE_DEGREE SEEKDB_CUVS_NN_DESCENT_NITER
export SEEKDB_CUVS_DEBUG=1
mkdir -p $P
: > $P/out.log

sql() { mysql --connect-timeout=15 --protocol=TCP -h127.0.0.1 -P$PORT -uroot -A -N -B "$@"; }
up() { for i in $(seq 1 600); do sql -e "select 1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }

LD_LIBRARY_PATH=$LIBDIR:/opt/hipvs/lib:/opt/rocm/lib nohup $CAND_BIN --nodaemon --port $PORT \
  --data-dir $P/store --base-dir $P \
  --parameter __min_full_resource_pool_memory=2147483648 --parameter memory_budget=12G \
  --parameter datafile_size=12G --parameter cpu_count=32 --parameter local_ip=127.0.0.1 \
  >> $P/observer.log 2>&1 &
echo $! > $P/obs.pid
up || { echo "FAIL start"; exit 1; }

echo "=== DDL CONTRACT (expect 1235 on the three rejected forms) ==="
sql -e "create database if not exists ddlc;"
for st in \
  "create table ddlc.t_cuvs_ddl(c1 int primary key, c2 vector(4), vector index idx(c2) with (distance=l2, type=hnsw, lib=cuvs))" \
  "create table ddlc.t_cuvs_cos(c1 int primary key, c2 vector(4), vector index idx(c2) with (distance=cosine, type=hnsw, lib=cuvs))" \
  "create table ddlc.t_cuvs_ip(c1 int primary key, c2 vector(4), vector index idx(c2) with (distance=inner_product, type=hnsw, lib=cuvs))" \
  "create table ddlc.t_vsag_cos(c1 int primary key, c2 vector(4), vector index idx(c2) with (distance=cosine, type=hnsw, lib=vsag))" \
  "create table ddlc.t_vsag_ip(c1 int primary key, c2 vector(4), vector index idx(c2) with (distance=inner_product, type=hnsw, lib=vsag))" \
  "create table ddlc.t_cuvs_ivf(c1 int primary key, c2 vector(4), vector index idx(c2) with (distance=l2, type=ivf_flat, lib=cuvs))"
do
  sql -e "$st;" > /dev/null 2>&1 && rc=OK || rc=$(sql -e "$st;" 2>&1 | head -1)
  echo "STMT: $st"
  echo "  -> $rc"
done

echo "=== CPU (lib=vsag) build on the same 250k x 128 table ==="
sql -e "alter system set ob_vector_memory_limit_percentage=30; create database if not exists cpuab;"
for g in ob_query_timeout ob_trx_timeout ob_trx_idle_timeout; do sql -e "set global $g=6000000000"; done
sql -e "use cpuab; create table items(id bigint primary key, v vector(128));"
mysql --local-infile=1 --protocol=TCP -h127.0.0.1 -P$PORT -uroot \
  -e "use cpuab; load data local infile '$SRC/base.tsv' into table items fields terminated by '\t' lines terminated by '\n' (id,v);" \
  || { echo "FAIL load"; exit 1; }
echo "ROWS=$(sql -e 'select count(*) from cpuab.items')"
t0=$(date +%s%N)
sql -e "use cpuab; create vector index idx on items(v) with (distance=l2,type=hnsw,lib=vsag,m=16,ef_construction=200,ef_search=64);" \
  || { echo "FAIL vsag ddl"; exit 1; }
Q=$(head -1 $SRC/queries.txt)
t1=$(date +%s%N)
sql -e "use cpuab; select id from items order by l2_distance(v,$Q) approximate limit 10" > /dev/null
t2=$(date +%s%N)
echo "RESULT vsag250k128 ddl_ms=$(( (t1-t0)/1000000 )) first_q_ms=$(( (t2-t1)/1000000 )) window_ms=$(( (t2-t0)/1000000 ))"

: > $P/vsag.k100.txt
t0=$(date +%s%N)
while IFS= read -r q; do
  sql -e "use cpuab; select id from items order by l2_distance(v,$q) approximate limit 100" >> $P/vsag.k100.txt
done < $SRC/queries.txt
t1=$(date +%s%N)
echo "vsag QPS top100 = $(python3 -c "print(round(100/((($t1-$t0)/1000000)/1000),1))") q/s"
python3 /workspace/recall_ab.py $SRC/exact.txt $P/vsag.k100.txt | sed 's/^/vsag RECALL /'

t0=$(date +%s%N)
while IFS= read -r q; do
  sql -e "use cpuab; select id from items order by l2_distance(v,$q) approximate limit 10" >> $P/vsag.k10.txt
done < $SRC/queries.txt
t1=$(date +%s%N)
echo "vsag QPS top10 = $(python3 -c "print(round(100/((($t1-$t0)/1000000)/1000),1))") q/s"
kill -TERM $(cat $P/obs.pid) 2>/dev/null
sleep 10
echo CPU_VSAG_DONE
