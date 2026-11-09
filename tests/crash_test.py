#!/usr/bin/env python3
"""Crash-recovery test for MiniVectorDB (SIGKILL model).

Covers:
  1. an open transaction is rolled back after SIGKILL
  2. an acknowledged COMMIT survives SIGKILL
  3. a torn WAL tail is ignored while earlier committed records replay
"""
import os
import subprocess
import sys
import tempfile

BIN = sys.argv[1] if len(sys.argv) > 1 else "./build/minivectordb"
SENTINEL = "__MVDB_END__"


def start(data_dir, extra=None):
    proc = subprocess.Popen(
        [BIN, "--data-dir", data_dir] + (extra or []),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    ready = proc.stdout.readline().strip()
    assert ready == "READY", f"unexpected ready line: {ready!r}"
    return proc


def send(proc, sql):
    proc.stdin.write(sql + ";\n")
    proc.stdin.flush()
    lines = []
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("database closed stdout unexpectedly")
        line = line.rstrip("\n")
        if line == SENTINEL:
            break
        lines.append(line)
    return lines


def kill9(proc):
    proc.kill()
    proc.wait()


def assert_rows(proc, sql, expected, label):
    rows = send(proc, sql)
    assert rows == expected, f"{label}: got {rows!r}, want {expected!r}"


data = tempfile.mkdtemp(prefix="mvdb-crash-")

# --- case 1: uncommitted transaction dies with the process -----------------
p = start(data)
send(p, "CREATE TABLE c (id BIGINT PRIMARY KEY, v VECTOR(2))")
send(p, "INSERT INTO c VALUES (1,'[0,0]'),(2,'[1,1]')")
send(p, "BEGIN")
send(p, "INSERT INTO c VALUES (3,'[2,2]')")
kill9(p)

p = start(data)
assert_rows(p, "SELECT id FROM c ORDER BY id", ["id", "1", "2"],
            "open transaction must roll back")

# --- case 2: acknowledged COMMIT survives ---------------------------------
send(p, "BEGIN")
send(p, "INSERT INTO c VALUES (3,'[2,2]')")
send(p, "COMMIT")
kill9(p)

p = start(data)
assert_rows(p, "SELECT id FROM c ORDER BY id", ["id", "1", "2", "3"],
            "committed data must survive")

# --- case 3: torn WAL tail -------------------------------------------------
send(p, "INSERT INTO c VALUES (4,'[3,3]')")
kill9(p)

with open(os.path.join(data, "wal.log"), "ab") as fh:
    fh.write(b"\x7f\x00\x00\x00garbage-tail-without-valid-record")

p = start(data)
assert_rows(p, "SELECT id FROM c ORDER BY id", ["id", "1", "2", "3", "4"],
            "torn WAL tail must be ignored")

# --- case 4: commits made after a torn-tail recovery must survive ----------
# Recovery has to cut the garbage tail; otherwise these new records would be
# appended after it and silently skipped by the next replay.
send(p, "INSERT INTO c VALUES (5,'[4,4]')")
kill9(p)

p = start(data)
assert_rows(p, "SELECT id FROM c ORDER BY id", ["id", "1", "2", "3", "4", "5"],
            "commits after torn-tail recovery must survive the next crash")

# clean shutdown -> checkpoint, then verify snapshot path once more
p.stdin.close()
assert p.wait(timeout=10) == 0
p = start(data)
assert_rows(p, "SELECT id FROM c ORDER BY id", ["id", "1", "2", "3", "4", "5"],
            "snapshot must reload after clean shutdown")
p.stdin.close()
p.wait(timeout=10)

# --- case 5: size-triggered WAL checkpoint ---------------------------------
w = tempfile.mkdtemp(prefix="mvdb-walck-")
p = start(w, ["--wal-checkpoint-mb", "1"])
send(p, "CREATE TABLE w (id BIGINT PRIMARY KEY, v VECTOR(2))")
for batch in range(4):
    values = ",".join(
        f"({i},'[{i % 7},{i % 5}]')" for i in range(batch * 12000 + 1,
                                                     batch * 12000 + 12001)
    )
    send(p, f"INSERT INTO w VALUES {values}")
for pk in (1, 12000, 12001, 24000, 36000, 48000):
    assert_rows(p, f"SELECT id FROM w WHERE id = {pk}", ["id", str(pk)],
                "checkpoint mid-run must keep every committed row")
kill9(p)

# Roughly three batches fit in the first snapshot; the fourth is WAL-only.
p = start(w)
for pk in (1, 12001, 36001, 48000):
    assert_rows(p, f"SELECT id FROM w WHERE id = {pk}", ["id", str(pk)],
                "snapshot + WAL tail after checkpoint must recover")
p.stdin.close()
assert p.wait(timeout=30) == 0

print("crash tests ok")
