#!/usr/bin/env python3
"""Skeleton-scale benchmark mirroring the official shape:
10,000 rows x 128 dims, batch INSERT of 100, 1,000 TopK (K=10) queries,
half of them with an equality filter. Also measures recovery time after a
clean shutdown (snapshot load).

Usage: python3 scripts/bench.py [path-to-binary]
"""
import os
import random
import subprocess
import sys
import tempfile
import time

BIN = sys.argv[1] if len(sys.argv) > 1 else "./build/minivectordb"
SENTINEL = "__MVDB_END__"

N = 10_000
DIM = 128
BATCH = 100
NQ = 1_000
K = 10
CATEGORIES = 100
SEED = 20261109


def vec_literal(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def start(data_dir):
    proc = subprocess.Popen(
        [BIN, "--data-dir", data_dir],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    ready = proc.stdout.readline().strip()
    assert ready == "READY", ready
    return proc


def send(proc, sql):
    proc.stdin.write(sql + ";\n")
    proc.stdin.flush()
    lines = []
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("database closed stdout")
        line = line.rstrip("\n")
        if line == SENTINEL:
            break
        lines.append(line)
    return lines


def rss_kb(proc):
    try:
        with open(f"/proc/{proc.pid}/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


rng = random.Random(SEED)
rows = []
for i in range(1, N + 1):
    vec = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
    rows.append((i, f"cat{i % CATEGORIES}", vec))

# Queries use categories that actually have enough rows for K=10.
from collections import Counter  # noqa: E402

counts = Counter(row[1] for row in rows)
query_categories = [c for c, n in counts.items() if n >= K]
assert len(query_categories) >= 20

data_dir = tempfile.mkdtemp(prefix="mvdb-bench-")
print(f"data_dir: {data_dir}")
print(f"rows={N} dim={DIM} batch={BATCH} queries={NQ} K={K}")

proc = start(data_dir)

t0 = time.perf_counter()
send(proc, "CREATE TABLE bench (id BIGINT PRIMARY KEY, "
           f"category VARCHAR(16), embedding VECTOR({DIM}))")
for start_i in range(0, N, BATCH):
    batch = rows[start_i:start_i + BATCH]
    values = ",".join(
        f"({i},'{cat}','{vec_literal(vec)}')" for i, cat, vec in batch
    )
    send(proc, f"INSERT INTO bench VALUES {values}")
t_write = time.perf_counter() - t0

rng_q = random.Random(SEED + 1)
queries = []
for qi in range(NQ):
    qv = vec_literal([rng_q.gauss(0.0, 1.0) for _ in range(DIM)])
    if qi % 2 == 0:
        cat = rng_q.choice(query_categories)
        queries.append(
            f"SELECT id, L2_DISTANCE(embedding, '{qv}') AS d "
            f"FROM bench WHERE category = '{cat}' ORDER BY d LIMIT {K}"
        )
    else:
        queries.append(
            f"SELECT id, L2_DISTANCE(embedding, '{qv}') AS d "
            f"FROM bench ORDER BY d LIMIT {K}"
        )

t1 = time.perf_counter()
for sql in queries:
    out = send(proc, sql)
    assert len(out) == K + 1, f"query returned {len(out)} lines"
    ids = [line.split("\t")[0] for line in out[1:]]
    assert len(set(ids)) == K, "duplicate ids in TopK"
t_query = time.perf_counter() - t1

peak_mb = rss_kb(proc) / 1024.0

t2 = time.perf_counter()
proc.stdin.close()
assert proc.wait(timeout=60) == 0
t_close = time.perf_counter() - t2

t3 = time.perf_counter()
proc = start(data_dir)
t_recover = time.perf_counter() - t3
out = send(proc, f"SELECT id FROM bench LIMIT {K}")
assert len(out) == K + 1
proc.stdin.close()
proc.wait(timeout=60)

print(f"write      : {t_write:8.3f} s  ({N / t_write:,.0f} rows/s)")
print(f"query      : {t_query:8.3f} s  ({NQ / t_query:,.0f} q/s, "
      f"{t_query * 1000 / NQ:.2f} ms/query)")
print(f"peak RSS   : {peak_mb:8.1f} MiB")
print(f"close+ckpt : {t_close:8.3f} s")
print(f"recovery   : {t_recover:8.3f} s")
