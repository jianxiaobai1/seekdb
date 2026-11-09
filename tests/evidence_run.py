#!/usr/bin/env python3
"""Three small, self-checking experiments used as submission evidence.

Every printed line is the database's real response.  The script builds its own
data directories under the system temp dir, so it never touches /data.

Run from anywhere:

    python3 tests/evidence_run.py

Set MVDB_BIN to test a different binary (default: build/minivectordb).
"""
import math
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("MVDB_BIN", os.path.join(ROOT, "build", "minivectordb"))
SENTINEL = "__MVDB_END__"
_TMP = []


def _data_dir(prefix):
    d = tempfile.mkdtemp(prefix=prefix)
    _TMP.append(d)
    return d


def start(data_dir, extra=None):
    err_path = os.path.join(data_dir, "stderr.log")
    err = open(err_path, "ab")
    cmd = [BIN, "--data-dir", data_dir] + (extra or [])
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=err, text=True, bufsize=1)
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
            raise RuntimeError("db closed stdout")
        line = line.rstrip("\n")
        if line == SENTINEL:
            break
        lines.append(line)
    for line in lines:
        assert not line.startswith("ERROR"), (sql, line)
    return lines


def query(proc, sql):
    lines = send(proc, sql)
    header = lines[0].split("\t") if lines else []
    return header, [ln.split("\t") for ln in lines[1:]]


def kill9(proc):
    proc.kill()
    proc.wait()


def close(proc):
    proc.stdin.close()
    assert proc.wait(timeout=30) == 0


def print_query(proc, sql):
    h, body = query(proc, sql)
    print("SQL>", sql)
    print("    header:", h)
    for row in body:
        print("    row   :", row)
    return h, body


def exp1_ties():
    print("\n=== E1: distance ties are ordered by primary key ===")
    d = _data_dir("mvdb-e1-")
    p = start(d)
    send(p, "CREATE TABLE t (id BIGINT PRIMARY KEY, tag VARCHAR(8), v VECTOR(2))")
    send(p, "INSERT INTO t VALUES (5,'a','[1,0]'),(2,'a','[0,1]'),"
            "(9,'a','[-1,0]'),(4,'a','[0,-1]'),(1,'a','[1,0]')")
    _, body = print_query(
        p, "SELECT id, L2_DISTANCE(v, '[1,0]') AS d FROM t "
           "WHERE tag = 'a' ORDER BY d LIMIT 3")
    assert [r[0] for r in body] == ["1", "5", "2"], body
    _, body = print_query(
        p, "SELECT id, L2_DISTANCE(v, '[1,0]') AS d FROM t "
           "ORDER BY d DESC LIMIT 5")
    assert [r[0] for r in body] == ["9", "2", "4", "1", "5"], body
    close(p)
    print("E1 expected: ASC 1,5,2 (tie at 0 -> pk 1 before 5; tie at sqrt2 -> 2 before 4);"
          " DESC 9,2,4,1,5")


def exp2_crash():
    print("\n=== E2: kill -9 keeps committed rows, drops the open transaction ===")
    d = _data_dir("mvdb-e2-")
    p = start(d)
    send(p, "CREATE TABLE r (id BIGINT PRIMARY KEY, note VARCHAR(16))")
    send(p, "INSERT INTO r VALUES (1,'committed'),(2,'committed'),(3,'committed')")
    send(p, "BEGIN")
    send(p, "INSERT INTO r VALUES (4,'uncommitted')")
    _, body = print_query(p, "SELECT id, note FROM r ORDER BY id")
    assert [r[0] for r in body] == ["1", "2", "3", "4"], body
    print("    killing -9 with row 4 only inside the open transaction ...")
    kill9(p)

    p = start(d)
    _, body = print_query(p, "SELECT id, note FROM r ORDER BY id")
    assert [r[0] for r in body] == ["1", "2", "3"], body
    send(p, "INSERT INTO r VALUES (5,'after restart')")
    print("    inserted row 5 (auto-commit), killing -9 again ...")
    kill9(p)

    p = start(d)
    _, body = print_query(p, "SELECT id, note FROM r ORDER BY id")
    assert [r[0] for r in body] == ["1", "2", "3", "5"], body
    close(p)
    print("E2 expected: after first crash 1,2,3 (row 4 rolled back);"
          " after second crash 1,2,3,5 (row 5 survived)")


def vec(i):
    return [i % 97, (i * 31) % 89, (i * 17) % 83, (i * 7) % 79]


def vlit(v):
    return "[" + ",".join(str(x) for x in v) + "]"


def exp3_spill():
    print("\n=== E3: 1 MiB budget forces 20k rows to disk; TopK stays exact ===")
    d = _data_dir("mvdb-e3-")
    p = start(d, ["--mem-budget", "1"])
    send(p, "CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(4), "
            "v VECTOR(4))")
    N = 20000
    for base in range(1, N + 1, 500):
        values = ",".join(
            "(%d,'%s','%s')" % (i, "abc"[i % 3], vlit(vec(i)))
            for i in range(base, min(base + 500, N + 1)))
        send(p, "INSERT INTO docs VALUES " + values)
    close(p)

    # Restart: startup stats should show the whole table spilled.
    p = start(d, ["--mem-budget", "1"])
    with open(os.path.join(d, "stderr.log"), encoding="utf-8", errors="replace") as f:
        stats = [ln.strip() for ln in f if "minivectordb:" in ln][-1]
    print("    startup stats:", stats)
    assert "spilled_rows=20000" in stats, stats
    spools = [n for n in os.listdir(d) if n.endswith(".spool")]
    print("    spool files  :", [(n, os.path.getsize(os.path.join(d, n))) for n in spools])
    assert spools

    _, body = print_query(p, "SELECT id, category, v FROM docs WHERE id = 12345")
    assert body == [["12345", "abc"[12345 % 3], vlit(vec(12345))]], body

    q = [3, 5, 8, 13]
    _, body = print_query(
        p, "SELECT id, L2_DISTANCE(v, '%s') AS d FROM docs "
           "ORDER BY d LIMIT 10" % vlit(q))
    got = [r[0] for r in body]

    def dist(i):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec(i), q)))

    want = [str(i) for i in sorted(range(1, N + 1), key=lambda i: (dist(i), i))[:10]]
    print("    expected IDs :", want)
    assert got == want, (got, want)

    _, body = print_query(
        p, "SELECT id, L2_DISTANCE(v, '%s') AS d FROM docs "
           "WHERE category = 'b' ORDER BY d LIMIT 7" % vlit(q))
    got = [r[0] for r in body]
    want = [str(i) for i in sorted((i for i in range(1, N + 1) if i % 3 == 1),
                                   key=lambda i: (dist(i), i))[:7]]
    print("    expected IDs :", want)
    assert got == want, (got, want)
    close(p)
    print("E3 expected: point lookup and both TopK lists match the Python truth"
          " even though every row lives in the spool")


if __name__ == "__main__":
    print("binary:", BIN)
    print("EVIDENCE_BEGIN")
    try:
        exp1_ties()
        exp2_crash()
        exp3_spill()
    finally:
        for d in _TMP:
            shutil.rmtree(d, ignore_errors=True)
    print("EVIDENCE_END")
    print("ALL EXPERIMENTS PASSED", file=sys.stderr)
