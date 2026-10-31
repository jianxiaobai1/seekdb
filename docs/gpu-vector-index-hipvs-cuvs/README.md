# hipVS/cuVS GPU vector backend

Optional hipVS/cuVS acceleration behind seekdb's existing vector adaptor. The
backend is off by default; a dense L2 HNSW/HGRAPH index opts in per index with
`WITH (lib=cuvs)`.

| Piece | Where |
|---|---|
| C ABI bridge (all cuVS/ROCm state) | [`tools/hipvs_bridge`](../../tools/hipvs_bridge/) |
| Adaptor integration (mirror, search, snapshot) | [`ob_vsag_adaptor.cpp`](../../src/oblib/lib/vector/ob_vsag_adaptor.cpp) |
| `lib=cuvs` DDL mapping / validation | `ob_plugin_vector_index_adaptor.cpp`, `ob_vector_index_util.cpp` |
| Batched GPU ANN | `ob_dbms_vector_mysql.cpp` (`dbms_vector.batch_knn`) |
| Reproducible build + measurement | [`validation/build_and_measure.sh`](validation/build_and_measure.sh) |
| Results | [`PERF_REPORT.md`](PERF_REPORT.md) |

## What the GPU path does

- `lib=cuvs` indexes mirror the rows they receive (bulk build and incremental
  adds) and serve kNN from a CAGRA index on the GPU.
- `lib=vsag` indexes are untouched. Cosine and inner-product metrics are
  rejected for `lib=cuvs` at DDL time.
- Queries fall back to the CPU index when there is no fresh GPU index: fewer
  than 256 rows, rows added since the last build, an excluded row under a query
  filter, or no GPU.
- Snapshots store the adjacency matrix plus the rows. Loading transplants the
  graph with `cuvsCagraIndexFromArgs`, so a restart never repeats the build.

## Build

```bash
# 1. bridge (needs hipcc + hipVS headers/libs)
tools/hipvs_bridge/build.sh /work/bridge/libseekdb_cuvs_bridge.so

# 2. seekdb against that bridge
./build.sh release \
  -DOB_BUILD_CUVS=ON \
  -DOB_BUILD_CUVS_TRACE=OFF \
  -DCUVS_BRIDGE_LIB=/work/bridge/libseekdb_cuvs_bridge.so \
  --make -j64

# CPU-only build: leave the defaults (OB_BUILD_CUVS=OFF).
```

The bridge exports `seekdb_cuvs_build`, `seekdb_cuvs_search`,
`seekdb_cuvs_free`, `seekdb_cuvs_export_graph`, `seekdb_cuvs_import`,
`seekdb_cuvs_free_host`, plus `seekdb_cuvs_serialize` /
`seekdb_cuvs_deserialize` for the standalone persistence tool. seekdb never
sees a cuVS type.

Build knobs are read from the environment on the observer process so that a
parameter sweep does not require a rebuild:

| Variable | Default | Effect |
|---|---|---|
| `SEEKDB_CUVS_GRAPH_DEGREE` | 128 | persisted graph degree |
| `SEEKDB_CUVS_INTERMEDIATE_DEGREE` | = graph degree | pre-prune degree |
| `SEEKDB_CUVS_NN_DESCENT_NITER` | 32 | refinement rounds |
| `SEEKDB_CUVS_BUILD_ALGO` | `nn_descent` | or `auto` |
| `SEEKDB_CUVS_SEARCH_ITOPK` | `max(4*topk, 512)` | candidate list size |
| `SEEKDB_CUVS_SEARCH_WIDTH` | 4 | graph walk width |

## SQL usage

```sql
CREATE TABLE items(
  id BIGINT PRIMARY KEY,
  embedding VECTOR(128),
  VECTOR INDEX idx_embedding(embedding)
    WITH (distance=l2, type=hnsw, lib=cuvs)
);
```

Explicit batch ANN writes results to a caller-created output table:

```sql
CALL dbms_vector.batch_knn("items", "query_vectors", 10, "batch_results");
```

See [`examples/batch_knn.sql`](examples/batch_knn.sql) for the table contract.

## Reproducing the measurements

```bash
bash docs/gpu-vector-index-hipvs-cuvs/validation/build_and_measure.sh check
bash docs/gpu-vector-index-hipvs-cuvs/validation/build_and_measure.sh bridge
bash docs/gpu-vector-index-hipvs-cuvs/validation/build_and_measure.sh seekdb
bash docs/gpu-vector-index-hipvs-cuvs/validation/build_and_measure.sh smoke
bash docs/gpu-vector-index-hipvs-cuvs/validation/build_and_measure.sh bench
bash docs/gpu-vector-index-hipvs-cuvs/validation/build_and_measure.sh report
```

`check` gates the toolchain and, before an hour of compilation, proves that
`libcuvs_c` actually exports the graph export/import entry points the snapshot
path calls. `bridge` verifies the exported symbols and dlopens the result with
`RTLD_NOW`. `bench` runs the CPU HNSW baseline and a CAGRA parameter sweep, one
observer per setting, and `report` renders the table in `PERF_REPORT.md` from
the raw `sweep.tsv`. The observer preallocates `datafile_size=40G`, so point
`RUN_ROOT` at the volume with the most free space.

`validation/probe_cagra.c` is the cuVS-only probe used to find the recall
boundary of the build parameters independently of seekdb:

```bash
hipcc -O3 -fopenmp -I/opt/hipvs/include \
  docs/gpu-vector-index-hipvs-cuvs/validation/probe_cagra.c -o probe_cagra \
  -L/opt/hipvs/lib -lcuvs_c -Wl,-rpath,/opt/hipvs/lib
./probe_cagra --n 250000 --dim 128 --deg 128 --inter 128 --niter 32 --cluster 0
```

Snapshots use format `OBCUVS04`; older snapshots are rejected and rebuilt
rather than misread.

## Current limitations

- Dense float32 L2 indexes only.
- GPU indexes are process-local and rebuilt lazily; only the snapshot path
  avoids a rebuild.
- A query whose filter excludes any of the GPU top-k falls back to the CPU
  index, because the true filtered top-k may rank beyond k.
- `dbms_vector.batch_knn` builds its CAGRA index per call.
- Multi-GPU and long-running load tests are out of scope.
