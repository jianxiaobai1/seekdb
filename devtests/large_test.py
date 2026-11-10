#!/usr/bin/env python3
"""Streaming-path check: dataset larger than the 32 MiB row cache.

Verifies that post-cache rows, updates and deletes are still exact, that a
large transaction works, and that peak memory stays bounded. Development only.
"""
import math
import os
import random
import shutil
import struct
import sys
import tempfile
import time

BIN = os.environ.get('MVD_BIN', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'minivectordb'))
ROWS = int(os.environ.get('MVD_ROWS', '80000'))
DIM = int(os.environ.get('MVD_DIM', '128'))
BATCH = 200
DATA = os.path.join(tempfile.gettempdir(), 'mvd-large')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crash_test import Session  # noqa: E402


def fmt_vec(v):
    return '[' + ','.join(f'{x:.9g}' for x in v) + ']'


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def brute(rows, qv, category, k=10):
    best = []
    for i, c, v in rows:
        if category is not None and c != category:
            continue
        d = math.dist(v, qv)
        best.append((d, i))
    best.sort()
    return [i for _, i in best[:k]]


def main():
    rng = random.Random(7)
    rows = []
    for i in range(1, ROWS + 1):
        vec = [f32(rng.uniform(-1.0, 1.0)) for _ in range(DIM)]
        rows.append((i, f'c{i % 5}', vec))

    shutil.rmtree(DATA, ignore_errors=True)
    os.makedirs(DATA)
    s = Session(DATA)
    s.ok(f'CREATE TABLE big (id BIGINT PRIMARY KEY, category VARCHAR(4), embedding VECTOR({DIM}));')
    t0 = time.perf_counter()
    for start in range(0, ROWS, BATCH):
        chunk = rows[start:start + BATCH]
        values = ','.join(f"({i},'{c}',{fmt_vec(v)})" for i, c, v in chunk)
        s.ok(f'INSERT INTO big VALUES {values};')
    print(f'loaded {ROWS} rows in {time.perf_counter() - t0:.2f}s')

    # Mutate late rows after the cache has been dropped, then query.
    s.ok(f"UPDATE big SET embedding = {fmt_vec([5.0] + [0.0] * (DIM - 1))} WHERE id = {ROWS};")
    s.ok(f'DELETE FROM big WHERE id = {ROWS - 1};')
    rows = [(i, c, ([5.0] + [0.0] * (DIM - 1)) if i == ROWS else v)
            for i, c, v in rows if i != ROWS - 1]
    assert s.rows(f'SELECT id FROM big WHERE id = {ROWS - 1};') == []

    def check(qv, category):
        where = f" WHERE category = '{category}'" if category else ''
        got = [r[0] for r in s.ok(
            f'SELECT id, L2_DISTANCE(embedding, {fmt_vec(qv)}) AS d FROM big{where} ORDER BY d LIMIT 10;'
        )['rows']]
        want = brute(rows, qv, category)
        assert got == want, f'mismatch ({category}): {got} != {want}'
        print(f'  query {"filter=" + category if category else "unfiltered"}: OK')

    print('exact checks after cache overflow:')
    for category in (None, 'c3', 'c4'):
        qv = [f32(rng.uniform(-1.0, 1.0)) for _ in range(DIM)]
        check(qv, category)

    assert len(s.rows('SELECT id FROM big;')) == len(rows)

    with open(f'/proc/{s.proc.pid}/status') as f:
        status = f.read()
    hwm = int(next(line for line in status.splitlines() if line.startswith('VmHWM')).split()[1])
    print(f'peak RSS while serving: {hwm / 1024:.1f} MiB (rows+WAL: ~{os.path.getsize(os.path.join(DATA, "minivectordb.wal")) / 1048576:.1f} MiB)')
    assert hwm < 512 * 1024, f'peak RSS too high: {hwm} kB'

    # Large explicit transaction, then rollback and verify.
    s.ok('BEGIN;')
    for start in range(0, 5000, 100):
        values = ','.join(f"({900000 + i},'tx',{fmt_vec([float(i % 7)] * DIM)})" for i in range(start, start + 100))
        s.ok(f'INSERT INTO big VALUES {values};')
    assert len(s.rows("SELECT id FROM big WHERE category = 'tx';")) == 5000
    s.ok('ROLLBACK;')
    assert s.rows("SELECT id FROM big WHERE category = 'tx';") == []

    s.shutdown()
    t0 = time.perf_counter()
    s = Session(DATA)
    print(f'restart with {os.path.getsize(os.path.join(DATA, "minivectordb.wal")) / 1048576:.1f} MiB WAL: {time.perf_counter() - t0:.2f}s')
    for category in (None, 'c3'):
        qv = [f32(rng.uniform(-1.0, 1.0)) for _ in range(DIM)]
        check(qv, category)
    s.shutdown()
    shutil.rmtree(DATA, ignore_errors=True)


if __name__ == '__main__':
    main()
