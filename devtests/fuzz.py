#!/usr/bin/env python3
"""Randomized differential test against a Python model of the task semantics.

Covers mixed INSERT/UPDATE/DELETE/SELECT, filters, ordering, ties, LIMIT,
explicit transactions, rollback and restarts on one table. Development only.
"""
import math
import os
import random
import shutil
import struct
import sys
import tempfile

BIN = os.environ.get('MVD_BIN', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'minivectordb'))
DATA = os.path.join(tempfile.gettempdir(), 'mvd-fuzz')
DIM = 4
VCAP = 4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crash_test import Session  # noqa: E402

rng = random.Random(int(os.environ.get('MVD_FUZZ_SEED', '1234')))
OPS = int(os.environ.get('MVD_FUZZ_OPS', '400'))


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def vec():
    return [f32(rng.choice([0, 1, 2, 0.5, 1.5, -1])) for _ in range(DIM)]


def jsonish(v):
    return [float(x) for x in v]


def gval():
    return rng.choice(['a', 'b', 'c', '', 'zz'])


class Model:
    def __init__(self):
        self.rows = {}
        self.active = False
        self.snapshot = None
        self.overlay = {}

    def view(self):
        out = dict(self.rows)
        for pk, row in self.overlay.items():
            if row is None:
                out.pop(pk, None)
            else:
                out[pk] = row
        return out

    def begin(self):
        assert not self.active
        self.active = True
        self.snapshot = dict(self.rows)
        self.overlay = {}

    def commit(self):
        for pk, row in self.overlay.items():
            if row is None:
                self.rows.pop(pk, None)
            else:
                self.rows[pk] = row
        self.active = False
        self.overlay = {}
        self.snapshot = None

    def rollback(self):
        self.active = False
        self.overlay = {}
        self.rows = self.snapshot
        self.snapshot = None


def sql_vec(v, quoted=True):
    body = '[' + ','.join(f'{x:.9g}' for x in v) + ']'
    return f"'{body}'" if quoted else body


def sql_str(s):
    return "'" + s.replace('\\', '\\\\').replace("'", "''") + "'"


def pred_match(row, preds):
    pk, (g, n, v) = row
    for col, op, val in preds:
        x = {'id': pk, 'g': g, 'n': n}[col]
        if op == '=':
            ok = x == val
        elif op == '<':
            ok = x < val
        elif op == '<=':
            ok = x <= val
        elif op == '>':
            ok = x > val
        elif op == '>=':
            ok = x >= val
        else:
            ok = x != val
        if not ok:
            return False
    return True


def pred_sql(preds):
    parts = []
    for col, op, val in preds:
        lit = sql_str(val) if isinstance(val, str) else str(val)
        parts.append(f'{col} {op} {lit}')
    return ' WHERE ' + ' AND '.join(parts) if parts else ''


def dist(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def rand_preds():
    preds = []
    for _ in range(rng.choice([0, 1, 1, 2])):
        col = rng.choice(['id', 'n', 'g'])
        if col == 'g':
            val = gval()
            op = rng.choice(['=', '<>', '<', '>=', '>', '<='])
        else:
            val = rng.randint(-3, 12) if col == 'id' else rng.randint(-5, 5)
            op = rng.choice(['=', '<', '<=', '>', '>=', '<>'])
        preds.append((col, op, val))
    return preds


def gen_insert(ids):
    batch = []
    seen = set()
    count = rng.choice([1, 1, 2, 3, 5])
    for _ in range(count):
        pk = rng.choice(ids)
        if pk in seen and rng.random() < 0.7:
            # Re-draw at most once; a collision can survive and then exercises
            # duplicate handling / statement atomicity.
            pk = rng.choice(ids)
        row = (gval(), rng.randint(-5, 5), vec())
        batch.append((pk, row))
        seen.add(pk)
    return batch


def main():
    ids = list(range(1, 16))
    shutil.rmtree(DATA, ignore_errors=True)
    os.makedirs(DATA)
    s = Session(DATA)
    s.ok('CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(%d), n BIGINT, v VECTOR(%d));' % (VCAP, DIM))
    m = Model()
    seed = list(range(1, 9))
    for pk in seed:
        row = ('seed', 0, vec())
        m.rows[pk] = row
        s.ok(f'INSERT INTO t VALUES ({pk}, {sql_str(row[0])}, {row[1]}, {sql_vec(row[2])});')

    def check(query, sql, expected_cols, expected_rows):
        resp = s.request(sql)
        if resp.get('ok') is not True:
            raise AssertionError(f'{sql}: unexpected error {resp}')
        cols = resp.get('columns')
        assert cols == expected_cols, f'{sql}: columns {cols} != {expected_cols}'
        rows = resp.get('rows')
        assert len(rows) == len(expected_rows), f'{sql}: {len(rows)} rows != {len(expected_rows)}\n{rows}\n{expected_rows}'
        for got, want in zip(rows, expected_rows):
            assert len(got) == len(want)
            for a, b in zip(got, want):
                if isinstance(b, list):
                    assert isinstance(a, list) and len(a) == len(b)
                    assert all(abs(float(x) - float(y)) <= 1e-5 * max(1.0, abs(float(y))) for x, y in zip(a, b)), \
                        f'{sql}: vector {a} != {b}'
                elif isinstance(b, float):
                    assert abs(float(a) - b) <= 1e-5 * max(1.0, abs(b)), f'{sql}: {a} != {b}'
                else:
                    assert a == b and type(a) is type(b), f'{sql}: {a!r} != {b!r}'
        return resp

    for step in range(OPS):
        action = rng.choices(
            ['insert', 'update', 'delete', 'select', 'txn', 'restart'],
            weights=[20, 20, 12, 35, 8, 1],
        )[0]

        if action == 'restart':
            if m.active:
                m.rollback()
                s.ok('ROLLBACK;')
            s.shutdown()
            s = Session(DATA)
            view = m.view()
            got = s.ok('SELECT id FROM t ORDER BY id;')['rows']
            assert got == [[pk] for pk in sorted(view)], f'restart mismatch: {got}'
            continue

        if action == 'txn':
            sub = rng.choice(['begin', 'commit', 'rollback'])
            if sub == 'begin':
                if m.active:
                    resp = s.request('BEGIN;')
                    assert resp['ok'] is False
                else:
                    s.ok('BEGIN;')
                    m.begin()
            elif sub == 'commit':
                s.ok('COMMIT;')
                if m.active:
                    m.commit()
            else:
                s.ok('ROLLBACK;')
                if m.active:
                    m.rollback()
            continue

        view = m.view()

        if action == 'insert':
            batch = gen_insert(ids)
            sql = 'INSERT INTO t VALUES ' + ', '.join(
                f'({pk}, {sql_str(row[0])}, {row[1]}, {sql_vec(row[2], quoted=rng.random() < 0.5)})'
                for pk, row in batch) + ';'
            ok = True
            seen = set()
            for pk, row in batch:
                if pk in seen or pk in view:
                    ok = False
                seen.add(pk)
            resp = s.request(sql)
            if not ok:
                assert resp['ok'] is False, f'{sql} should fail: {resp}'
                # A failed multi-row INSERT must not apply any of its rows.
                ref = m.view()
                got = s.ok('SELECT id FROM t ORDER BY id;')['rows']
                assert got == [[pk] for pk in sorted(ref)], \
                    f'failed INSERT changed state: {sql}\n{got}\nsorted={sorted(ref)}'
                continue
            assert resp['ok'] and resp['affected_rows'] == len(batch), resp
            if m.active:
                for pk, row in batch:
                    m.overlay[pk] = row
            else:
                for pk, row in batch:
                    m.rows[pk] = row
            continue

        if action == 'update':
            preds = rand_preds()
            matches = [pk for pk, row in view.items() if pred_match((pk, row), preds)]
            col = rng.choice(['g', 'n', 'v'])
            bad = col != 'n' and rng.random() < 0.12
            if col == 'g':
                val = 'toolong' if bad else gval()
                assign = f'g = {sql_str(val)}'
            elif col == 'n':
                val = rng.choice([-3, 0, 9])
                assign = f'n = {val}'
            else:
                val = [1, 2, 3] if bad else vec()
                assign = f'v = {sql_vec(val, quoted=rng.random() < 0.5)}'
            sql = f'UPDATE t SET {assign}{pred_sql(preds)};'
            resp = s.request(sql)
            if bad:
                assert resp['ok'] is False, f'{sql} should fail: {resp}'
                continue
            assert resp['ok'] and resp['affected_rows'] == len(matches), (sql, resp, matches)
            for pk in matches:
                g, n, v = view[pk]
                if col == 'g':
                    row = (val, n, v)
                elif col == 'n':
                    row = (g, val, v)
                else:
                    row = (g, n, val)
                if m.active:
                    m.overlay[pk] = row
                else:
                    m.rows[pk] = row
            continue

        if action == 'delete':
            # Exercise PK updates/unknown columns rarely to ensure they error.
            if rng.random() < 0.05:
                resp = s.request('UPDATE t SET id = 100 WHERE id = 1;')
                assert resp['ok'] is False, resp
                continue
            preds = rand_preds()
            matches = [pk for pk, row in view.items() if pred_match((pk, row), preds)]
            sql = 'DELETE FROM t' + pred_sql(preds) + ';'
            resp = s.request(sql)
            assert resp['ok'] and resp['affected_rows'] == len(matches), (sql, resp, matches)
            for pk in matches:
                if m.active:
                    m.overlay[pk] = None
                else:
                    m.rows.pop(pk, None)
            continue

        # select
        preds = rand_preds()
        q = vec()
        mode = rng.choice(['id', 'idn', 'star', 'idd', 'idnd', 'expr'])
        order = rng.choice([None, 'id', 'n', 'g', 'd'])
        desc = rng.random() < 0.5
        limit = rng.choice([None, 0, 1, 2, 3, 10, 100])
        if mode == 'id':
            proj, cols, names = 'id', ['id'], ['id']
        elif mode == 'idn':
            proj, cols, names = 'id, n', ['id', 'n'], ['id', 'n']
        elif mode == 'star':
            proj, cols, names = '*', ['id', 'g', 'n', 'v'], ['id', 'g', 'n', 'v']
        elif mode == 'idd':
            proj, cols, names = f"id, L2_DISTANCE(v, {sql_vec(q)}) AS d", ['id', 'd'], ['id', 'd']
        elif mode == 'idnd':
            proj, cols, names = f"id, n, L2_DISTANCE(v, {sql_vec(q)}) AS d", ['id', 'n', 'd'], ['id', 'n', 'd']
        else:
            proj, cols, names = f"id, L2_DISTANCE(v, {sql_vec(q)})", ['id', 'L2_DISTANCE(v)'], ['id', 'dist']
        if order is not None and order == 'd' and 'd' not in [n for n in names if n in ('d',)]:
            if mode != 'expr':
                order = 'id'
        if order == 'd' and mode == 'expr':
            order_sql = f' ORDER BY L2_DISTANCE(v, {sql_vec(q)})'
        elif order is not None:
            order_sql = f' ORDER BY {order}'
        else:
            order_sql = ''
        if order is not None and desc:
            order_sql += ' DESC'
        limit_sql = f' LIMIT {limit}' if limit is not None else ''
        sql = f'SELECT {proj} FROM t{pred_sql(preds)}{order_sql}{limit_sql};'

        filtered = [(pk, row) for pk, row in view.items() if pred_match((pk, row), preds)]
        # The engine sorts by primary key when no ORDER BY is given, and every
        # ORDER BY key tie-breaks on primary key ascending (even with DESC).
        # A stable sort on a pk-ascending list expresses both rules directly:
        # sort() is stable, so equal keys keep their pk-ascending position.
        filtered.sort(key=lambda r: r[0])
        if order == 'id':
            filtered.sort(key=lambda r: r[0], reverse=desc)
        elif order == 'n':
            filtered.sort(key=lambda r: r[1][1], reverse=desc)
        elif order == 'g':
            filtered.sort(key=lambda r: r[1][0], reverse=desc)
        elif order == 'd':
            filtered.sort(key=lambda r: dist(r[1][2], q), reverse=desc)
        if limit is not None:
            filtered = filtered[:limit]

        expected = []
        for pk, (g, n, v) in filtered:
            if mode == 'id':
                expected.append([pk])
            elif mode == 'idn':
                expected.append([pk, n])
            elif mode == 'star':
                expected.append([pk, g, n, jsonish(v)])
            elif mode == 'idd':
                expected.append([pk, dist(v, q)])
            elif mode == 'idnd':
                expected.append([pk, n, dist(v, q)])
            else:
                expected.append([pk, dist(v, q)])
        check(sql, sql, cols, expected)

        if step % 50 == 49:
            ref = m.view()
            got = s.ok('SELECT * FROM t ORDER BY id;')['rows']
            want = [[pk, g, n, jsonish(v)] for pk, (g, n, v) in sorted(ref.items())]
            assert len(got) == len(want), f'step {step}: {len(got)} rows != {len(want)}'
            for a, b in zip(got, want):
                assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2], f'step {step}: {a} != {b}'
                assert all(abs(float(x) - float(y)) <= 1e-5 for x, y in zip(a[3], b[3])), \
                    f'step {step}: {a} != {b}'

    if m.active:
        s.ok('ROLLBACK;')
        m.rollback()

    # Final restart and full comparison.
    s.shutdown()
    s = Session(DATA)
    got = s.ok('SELECT * FROM t;')['rows']
    want = [[pk, g, n, jsonish(v)] for pk, (g, n, v) in sorted(m.rows.items())]
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]
        assert all(abs(float(x) - float(y)) <= 1e-5 for x, y in zip(a[3], b[3]))
    s.shutdown()
    print(f'fuzz: PASS ({OPS} ops, {len(m.rows)} live rows)')
    shutil.rmtree(DATA, ignore_errors=True)


if __name__ == '__main__':
    main()
