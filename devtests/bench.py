#!/usr/bin/env python3
"""Local benchmark at the official small scale: 10k rows x 128 dims, 1000 TopK.

Measures batch-write wall time, query latency, peak RSS and WAL size.
Development only; the platform measures end-to-end time itself.
"""
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import time

BIN = os.environ.get('MVD_BIN', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'minivectordb'))
ROWS = int(os.environ.get('MVD_ROWS', '10000'))
DIM = int(os.environ.get('MVD_DIM', '128'))
QUERIES = int(os.environ.get('MVD_QUERIES', '1000'))
BATCH = int(os.environ.get('MVD_BATCH', '100'))
DATA = os.path.join(tempfile.gettempdir(), 'mvd-bench')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crash_test import Session  # noqa: E402


def fmt_vec(v):
    # 9 significant digits round-trip float32 exactly.
    return '[' + ','.join(f'{x:.9g}' for x in v) + ']'


def main():
    rng = random.Random(20261110)
    rows = []
    for i in range(1, ROWS + 1):
        vec = [rng.uniform(-1.0, 1.0) for _ in range(DIM)]
        rows.append((i, f'g{i % 10}', vec))

    shutil.rmtree(DATA, ignore_errors=True)
    os.makedirs(DATA)
    s = Session(DATA)
    dims = ','.join(f'd{i}' for i in range(DIM))
    s.ok(f'CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(8), embedding VECTOR({DIM}));')

    t0 = time.perf_counter()
    for start in range(0, ROWS, BATCH):
        chunk = rows[start:start + BATCH]
        values = ','.join(f"({i},'{c}',{fmt_vec(v)})" for i, c, v in chunk)
        resp = s.ok(f'INSERT INTO docs VALUES {values};')
        assert resp['affected_rows'] == len(chunk)
    write_s = time.perf_counter() - t0

    assert len(s.rows('SELECT id FROM docs;')) == ROWS

    queries = []
    for i in range(QUERIES):
        qv = [rng.uniform(-1.0, 1.0) for _ in range(DIM)]
        category = 'g5' if i % 2 else None
        queries.append((qv, category))

    t0 = time.perf_counter()
    for qv, category in queries:
        where = f" WHERE category = '{category}'" if category else ''
        sql = f"SELECT id, L2_DISTANCE(embedding, {fmt_vec(qv)}) AS d FROM docs{where} ORDER BY d LIMIT 10;"
        res = s.ok(sql)
        out = res['rows']
        assert len(out) == 10, f'expected 10 rows, got {len(out)}'
        ids = [r[0] for r in out]
        assert len(set(ids)) == 10
        assert all(out[k][1] <= out[k + 1][1] + 1e-9 for k in range(9))
    query_s = time.perf_counter() - t0

    # Cross-check a small sample of exact results against an independent scan.
    for qv, category in queries[:5]:
        best = []
        for i, c, v in rows:
            if category and c != category:
                continue
            d = math.dist([struct.unpack('f', struct.pack('f', x))[0] for x in v],
                          [struct.unpack('f', struct.pack('f', x))[0] for x in qv])
            best.append((d, i))
        best.sort()
        wanted = [i for _, i in best[:10]]
        where = f" WHERE category = '{category}'" if category else ''
        got = [r[0] for r in s.ok(f"SELECT id, L2_DISTANCE(embedding, {fmt_vec(qv)}) AS d FROM docs{where} ORDER BY d LIMIT 10;")['rows']]
        assert got == wanted, f'exact mismatch: {got} != {wanted}'

    # ann mode must stay exact for our implementation.
    for qv, category in queries[:20]:
        where = f" WHERE category = '{category}'" if category else ''
        sql = f"SELECT id, L2_DISTANCE(embedding, {fmt_vec(qv)}) AS d FROM docs{where} ORDER BY d LIMIT 10;"
        exact = s.ok(sql)['rows']
        ann = s.ok(sql, 'ann')['rows']
        assert ann == exact, 'ann differs from exact'

    with open(f'/proc/{s.proc.pid}/status') as f:
        status = f.read()
    hwm = next(line for line in status.splitlines() if line.startswith('VmHWM'))
    wal = os.path.join(DATA, 'minivectordb.wal')
    print(f'rows={ROWS} dim={DIM} queries={QUERIES}')
    print(f'write: {write_s:.3f}s ({ROWS / write_s:.0f} rows/s), batched {BATCH}, autocommit')
    print(f'query: {query_s * 1000 / QUERIES:.2f} ms/query exact (K=10, half filtered)')
    print(f'peak RSS: {hwm.split()[1]} kB')
    print(f'wal size: {os.path.getsize(wal) / 1048576:.2f} MiB')

    t0 = time.perf_counter()
    s.shutdown()
    s = Session(DATA)
    recover_s = time.perf_counter() - t0
    assert len(s.rows('SELECT id FROM docs;')) == ROWS
    print(f'restart (recovery + ready): {recover_s:.3f}s')
    s.shutdown()
    shutil.rmtree(DATA, ignore_errors=True)


if __name__ == '__main__':
    main()
