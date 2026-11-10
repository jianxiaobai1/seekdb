#!/usr/bin/env python3
"""Run MiniVectorDB inside a memory-capped, read-only Docker container.

Loads more rows than the row cache (4-32 MiB, sized from the container
memory limit) so queries must stream the WAL, then restarts the container
on the same data directory and checks that the data and exact TopK results
survive.

Environment:
  MVD_IMAGE   image name (default minivectordb:local)
  MVD_MEM     docker --memory value (default 128m)
  MVD_ROWS    rows to load (default 100000)
  MVD_DIM     vector dimension (default 128)
  MVD_QUERIES queries after load (default 20)
  MVD_DATA_ROOT parent of the bind-mounted data directory (default /var/tmp).
              It must be disk-backed: on tmpfs the whole WAL is charged as
              unreclaimable memory and the container OOMs regardless of the
              database cache policy.
"""
import json
import os
import random
import select
import shutil
import subprocess
import sys
import tempfile
import time

IMAGE = os.environ.get('MVD_IMAGE', 'minivectordb:local')
MEM = os.environ.get('MVD_MEM', '128m')
ROWS = int(os.environ.get('MVD_ROWS', '100000'))
DIM = int(os.environ.get('MVD_DIM', '128'))
QUERIES = int(os.environ.get('MVD_QUERIES', '20'))
BATCH = 100


def fmt_vec(v):
    return '[' + ','.join(f'{x:.9g}' for x in v) + ']'


class Container:
    def __init__(self, data_dir, name):
        self.name = name
        self.stderr = open(os.path.join(data_dir, name + '.stderr'), 'wb')
        cmd = [
            'docker', 'run', '--rm', '-i', '--name', name,
            '--network=none', '--cpus=2',
            '--memory=' + MEM, '--memory-swap=' + MEM,
            '--pids-limit=128', '--read-only',
            '--cap-drop=ALL', '--security-opt=no-new-privileges',
            '--mount', f'type=bind,src={data_dir},dst=/data',
            IMAGE,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=self.stderr)
        line = self._readline(60)
        msg = json.loads(line)
        assert msg == {'type': 'ready'}, msg
        self.next_id = 1

    def _readline(self, timeout):
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError(f'no response from {self.name} in {timeout}s')
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f'container {self.name} exited')
        return line

    def execute(self, sql, mode='exact'):
        ident = self.next_id
        self.next_id += 1
        req = {'id': ident, 'op': 'execute', 'sql': sql, 'search_mode': mode}
        self.proc.stdin.write((json.dumps(req, ensure_ascii=False) + '\n').encode())
        self.proc.stdin.flush()
        resp = json.loads(self._readline(120))
        assert resp.get('id') == ident, (resp, ident)
        assert resp.get('ok') is True, resp
        return resp

    def shutdown(self):
        ident = self.next_id
        self.next_id += 1
        req = {'id': ident, 'op': 'shutdown'}
        self.proc.stdin.write((json.dumps(req) + '\n').encode())
        self.proc.stdin.flush()
        resp = json.loads(self._readline(120))
        assert resp == {'id': ident, 'ok': True}, resp
        self.proc.stdin.close()
        rc = self.proc.wait(timeout=60)
        self.stderr.close()
        assert rc == 0, rc

    def peak_memory(self):
        out = subprocess.check_output(
            ['docker', 'inspect', '-f', '{{.Id}}', self.name], text=True).strip()
        paths = [
            f'/sys/fs/cgroup/system.slice/docker-{out}.scope/memory.peak',
            f'/sys/fs/cgroup/docker/{out}/memory.peak',
        ]
        for path in paths:
            try:
                with open(path) as f:
                    return int(f.read().strip())
            except OSError:
                pass
        return None

    def cache_limit(self):
        try:
            with open(self.stderr.name, 'r', errors='replace') as f:
                for line in f:
                    if 'row cache limit' in line:
                        return line.strip()
        except OSError:
            pass
        return 'cache limit: n/a'

    def memory_now(self):
        try:
            return subprocess.check_output(
                ['docker', 'stats', '--no-stream', '--format', '{{.MemUsage}}',
                 self.name], text=True).strip()
        except subprocess.CalledProcessError:
            return 'n/a'


def main():
    rng = random.Random(20261111)
    rows = [(i, f'g{i % 10}', [rng.uniform(-1, 1) for _ in range(DIM)])
            for i in range(1, ROWS + 1)]
    data_root = os.environ.get('MVD_DATA_ROOT', '/var/tmp')
    if not os.path.isdir(data_root):
        data_root = tempfile.gettempdir()
    data_dir = tempfile.mkdtemp(prefix='mvd-mem-', dir=data_root)
    os.chmod(data_dir, 0o777)
    name = 'mvd-mem-' + str(os.getpid())
    t0 = time.perf_counter()
    try:
        c = Container(data_dir, name)
        cache_line = c.cache_limit()
        cols = ','.join(f'd{i}' for i in range(DIM))
        c.execute(f'CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(8), embedding VECTOR({DIM}));')
        for start in range(0, ROWS, BATCH):
            chunk = rows[start:start + BATCH]
            values = ','.join(f"({i},'{g}',{fmt_vec(v)})" for i, g, v in chunk)
            resp = c.execute(f'INSERT INTO docs VALUES {values};')
            assert resp['affected_rows'] == len(chunk), resp
            if start and start % 10000 == 0:
                print(f'  at {start}: {c.memory_now()}', flush=True)
        load_s = time.perf_counter() - t0

        queries = []
        for i in range(QUERIES):
            qv = [rng.uniform(-1, 1) for _ in range(DIM)]
            queries.append((qv, 'g5' if i % 2 else None))
        t0 = time.perf_counter()
        for qv, cat in queries:
            where = f" WHERE category = '{cat}'" if cat else ''
            sql = f"SELECT id, L2_DISTANCE(embedding, {fmt_vec(qv)}) AS d FROM docs{where} ORDER BY d LIMIT 10;"
            out = c.execute(sql)['rows']
            assert len(out) == 10, len(out)
            assert all(out[k][1] <= out[k + 1][1] + 1e-9 for k in range(9))
        query_ms = (time.perf_counter() - t0) * 1000 / QUERIES
        peak = c.peak_memory()

        c.shutdown()
        c = Container(data_dir, name)
        first = c.execute('SELECT id, category FROM docs ORDER BY id LIMIT 5;')['rows']
        assert first == [[i, f'g{i % 10}'] for i in range(1, 6)], first
        qv, cat = queries[-1]
        where = f" WHERE category = '{cat}'" if cat else ''
        out = c.execute(f"SELECT id, L2_DISTANCE(embedding, {fmt_vec(qv)}) AS d FROM docs{where} ORDER BY d LIMIT 10;")['rows']
        assert len(out) == 10, len(out)
        c.shutdown()
        wal = os.path.join(data_dir, 'minivectordb.wal')
        print(f'image={IMAGE} mem={MEM} rows={ROWS} dim={DIM}')
        print(cache_line)
        print(f'load: {load_s:.2f}s')
        print(f'query: {query_ms:.2f} ms/query (streaming, K=10)')
        print(f'peak cgroup memory: {peak} bytes' if peak else 'peak cgroup memory: n/a')
        print(f'wal: {os.path.getsize(wal) / 1048576:.1f} MiB')
        print('restart: PASS')
        print('MEM TEST PASS')
    except Exception:
        try:
            with open(os.path.join(data_dir, name + '.stderr'), 'r',
                      errors='replace') as f:
                print('--- stderr tail ---')
                print(''.join(f.readlines()[-20:]))
        except OSError:
            pass
        raise
    finally:
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
