#!/usr/bin/env python3
"""Local crash-recovery checks: SIGKILL, truncated commit tail, restart loop.

Drives the database over the JSONL stdio protocol. Development only.
"""
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time

BIN = os.environ.get(
    'MVD_BIN',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'minivectordb'),
)
DATA = os.environ.get('MVD_CRASH_DIR', os.path.join(tempfile.gettempdir(), 'mvd-crash'))
NEXT_ID = [0]


class Session:
    def __init__(self, datadir):
        self.proc = subprocess.Popen(
            [BIN, '--data-dir', datadir],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding='utf-8', bufsize=1,
        )
        line = self.proc.stdout.readline()
        msg = json.loads(line)
        assert msg == {'type': 'ready'}, f'bad handshake: {line!r}'

    def request(self, sql, mode='exact'):
        NEXT_ID[0] += 1
        rid = NEXT_ID[0]
        req = {'id': rid, 'op': 'execute', 'sql': sql, 'search_mode': mode}
        self.proc.stdin.write(json.dumps(req) + '\n')
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert line, f'no response for {sql!r}'
        resp = json.loads(line)
        assert resp.get('id') == rid, f'id mismatch: {resp}'
        return resp

    def ok(self, sql, mode='exact'):
        resp = self.request(sql, mode)
        assert resp.get('ok') is True, f'{sql}: {resp}'
        return resp

    def rows(self, sql, mode='exact'):
        return self.ok(sql, mode).get('rows')

    def shutdown(self):
        NEXT_ID[0] += 1
        self.proc.stdin.write(json.dumps({'id': NEXT_ID[0], 'op': 'shutdown'}) + '\n')
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert json.loads(line) == {'id': NEXT_ID[0], 'ok': True}, line
        assert self.proc.wait(timeout=10) == 0

    def kill(self):
        self.proc.send_signal(signal.SIGKILL)
        self.proc.wait(timeout=10)


def fresh(datadir):
    shutil.rmtree(datadir, ignore_errors=True)
    os.makedirs(datadir)


CHECKS = []


def check(name, fn):
    CHECKS.append((name, fn))


def scenario_open_txn_rollback(datadir):
    fresh(datadir)
    s = Session(datadir)
    s.ok('CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2));')
    s.ok("INSERT INTO t VALUES (1,'[1,0]');")
    s.ok('BEGIN;')
    s.ok("INSERT INTO t VALUES (2,'[0,1]');")
    s.ok('DELETE FROM t WHERE id=1;')
    s.kill()
    s = Session(datadir)
    assert s.rows('SELECT id FROM t ORDER BY id;') == [[1]]
    s.shutdown()


def scenario_autocommit_durable(datadir):
    fresh(datadir)
    s = Session(datadir)
    s.ok('CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(8));')
    s.ok("INSERT INTO t VALUES (7,'seven');")
    s.kill()
    s = Session(datadir)
    assert s.rows('SELECT id, s FROM t;') == [[7, 'seven']]
    s.shutdown()


def scenario_commit_durable(datadir):
    fresh(datadir)
    s = Session(datadir)
    s.ok('CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2));')
    s.ok('BEGIN;')
    for i in range(1, 101):
        s.ok(f"INSERT INTO t VALUES ({i},'[{i},0]');")
    s.ok('COMMIT;')
    s.kill()
    s = Session(datadir)
    assert s.rows('SELECT id FROM t ORDER BY id LIMIT 3;') == [[1], [2], [3]]
    assert s.rows('SELECT id, L2_DISTANCE(v, \'[0,0]\') AS d FROM t ORDER BY d DESC LIMIT 1;') == [[100, 100.0]]
    s.shutdown()


def scenario_kill_mid_commit(datadir, rounds=6):
    """Commit response never observed: the transaction must be all-or-nothing."""
    for r in range(rounds):
        fresh(datadir)
        s = Session(datadir)
        s.ok('CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(8));')
        s.ok('BEGIN;')
        batch = 200
        for b in range(0, 4000, batch):
            values = ','.join(
                f"({i},'[{i},1,2,3,4,5,6,7]')" for i in range(b + 1, b + batch + 1)
            )
            s.ok(f'INSERT INTO t VALUES {values};')
        # Fire COMMIT and kill without reading the response.
        NEXT_ID[0] += 1
        s.proc.stdin.write(json.dumps({'id': NEXT_ID[0], 'op': 'execute', 'sql': 'COMMIT;'}) + '\n')
        s.proc.stdin.flush()
        time.sleep(random.uniform(0.0, 0.02))
        s.kill()
        s = Session(datadir)
        n = s.rows('SELECT id FROM t;')
        assert len(n) in (0, 4000), f'round {r}: partial transaction visible: {len(n)} rows'
        # Database must stay usable after discarding a torn tail.
        s.ok("INSERT INTO t VALUES (9001,'[0,0,0,0,0,0,0,9]');")
        assert s.rows('SELECT id FROM t ORDER BY id DESC LIMIT 1;') == [[9001]]
        s.shutdown()


def scenario_restart_loop(datadir):
    """Kill repeatedly while recovery runs, then require a clean start."""
    fresh(datadir)
    s = Session(datadir)
    s.ok('CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(4));')
    s.ok('BEGIN;')
    for b in range(0, 20000, 100):
        values = ','.join(f"({i},'[{i},0,0,0]')" for i in range(b + 1, b + 101))
        s.ok(f'INSERT INTO t VALUES {values};')
    s.ok('COMMIT;')
    s.shutdown()
    for _ in range(8):
        p = subprocess.Popen([BIN, '--data-dir', datadir],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        time.sleep(random.uniform(0.001, 0.05))
        p.send_signal(signal.SIGKILL)
        p.wait(timeout=10)
    s = Session(datadir)
    assert s.rows('SELECT id FROM t;') and len(s.rows('SELECT id FROM t;')) == 20000
    assert s.rows('SELECT id, v FROM t WHERE id = 20000;') == [[20000, [20000, 0, 0, 0]]]
    s.ok('BEGIN;')
    s.ok('DELETE FROM t;')
    s.ok('ROLLBACK;')
    assert len(s.rows('SELECT id FROM t;')) == 20000
    s.shutdown()


def scenario_truncated_tail(datadir):
    """A torn last record must be discarded, never partially applied."""
    fresh(datadir)
    s = Session(datadir)
    s.ok('CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2));')
    s.ok('BEGIN;')
    for i in range(1, 2001):
        s.ok(f"INSERT INTO t VALUES ({i},'[{i},0]');")
    s.ok('COMMIT;')
    s.shutdown()
    wal = os.path.join(datadir, 'minivectordb.wal')
    size = os.path.getsize(wal)
    for cut in (1, 2, 3, 5, 8, 13, 21, 34, 55, 100, 500, 2000):
        variant = os.path.join(datadir + f'-cut{cut}')
        shutil.rmtree(variant, ignore_errors=True)
        shutil.copytree(datadir, variant)
        path = os.path.join(variant, 'minivectordb.wal')
        with open(path, 'r+b') as f:
            f.truncate(size - cut)
        s = Session(variant)
        assert s.rows('SELECT id FROM t;') == [], f'cut {cut}: torn transaction applied'
        s.ok("INSERT INTO t VALUES (9999,'[0,0]');")
        assert s.rows('SELECT id FROM t;') == [[9999]]
        s.shutdown()
        shutil.rmtree(variant, ignore_errors=True)


def scenario_short_header(datadir):
    """A header torn by a kill during the very first start is repaired."""
    for content in (b'', b'M', b'MVD'):
        fresh(datadir)
        with open(os.path.join(datadir, 'minivectordb.wal'), 'wb') as f:
            f.write(content)
        s = Session(datadir)
        s.ok('CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2));')
        s.ok("INSERT INTO t VALUES (1,'[1,2]');")
        s.shutdown()
        s = Session(datadir)
        assert s.rows('SELECT id FROM t;') == [[1]]
        s.shutdown()


def scenario_shutdown_rollback(datadir):
    fresh(datadir)
    s = Session(datadir)
    s.ok('CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(4));')
    s.ok("INSERT INTO t VALUES (1,'a');")
    s.ok('BEGIN;')
    s.ok("INSERT INTO t VALUES (2,'b');")
    s.shutdown()
    s = Session(datadir)
    assert s.rows('SELECT id FROM t;') == [[1]]
    s.shutdown()


def main():
    signal.alarm(600)
    for name, fn in [
        ('open_txn_rollback', scenario_open_txn_rollback),
        ('autocommit_durable', scenario_autocommit_durable),
        ('commit_durable', scenario_commit_durable),
        ('kill_mid_commit', scenario_kill_mid_commit),
        ('restart_loop', scenario_restart_loop),
        ('truncated_tail', scenario_truncated_tail),
        ('short_header', scenario_short_header),
        ('shutdown_rollback', scenario_shutdown_rollback),
    ]:
        t0 = time.time()
        fn(os.path.join(DATA, name))
        print(f'{name}: PASS ({time.time() - t0:.2f}s)', flush=True)
    shutil.rmtree(DATA, ignore_errors=True)


if __name__ == '__main__':
    main()
