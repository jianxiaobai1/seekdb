#!/usr/bin/env python3
"""Memory-budget test: rows beyond the budget live in the mmap spool and must
still answer point lookups, filtered/unfiltered TopK, updates, deletes,
transactions and crash recovery with exact results.

Run with a 1 MiB budget and 20k rows so the table is guaranteed to spill.
"""
import math
import os
import subprocess
import sys
import tempfile

BIN = sys.argv[1] if len(sys.argv) > 1 else "./build/minivectordb"
SENTINEL = "__MVDB_END__"
BUDGET_MB = 1
N = 20000


def start(data_dir, err_path):
    err = open(err_path, "ab")
    proc = subprocess.Popen(
        [BIN, "--data-dir", data_dir, "--mem-budget", str(BUDGET_MB)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=err,
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
    for line in lines:
        assert not line.startswith("ERROR"), f"{sql!r} -> {line}"
    return lines


def query(proc, sql):
    lines = send(proc, sql)
    assert lines, f"no header for {sql!r}"
    return lines[0].split("\t"), [ln.split("\t") for ln in lines[1:]]


def ids(proc, sql):
    _, body = query(proc, sql)
    return [r[0] for r in body]


def kill9(proc):
    proc.kill()
    proc.wait()


def close(proc):
    proc.stdin.close()
    assert proc.wait(timeout=30) == 0


def vec(i):
    return [i % 97, (i * 31) % 89, (i * 17) % 83, (i * 7) % 79]


def vlit(v):
    return "[" + ",".join(str(x) for x in v) + "]"


def row(i, name=None, cat=None, v=None):
    name = name if name is not None else f"n{i}"
    cat = cat if cat is not None else "abc"[i % 3]
    v = v if v is not None else vec(i)
    return f"({i},'{name}','{cat}','{vlit(v)}')"


def topk(model, q, k, cat=None, desc=False):
    dist = lambda i: math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_of(i, model), q)))
    picked = [i for i in model if cat is None or model[i][1] == cat]
    picked.sort(key=lambda i: (-dist(i), i) if desc else (dist(i), i))
    return picked[:k]


model = {}
data = tempfile.mkdtemp(prefix="mvdb-spill-")
err_log = os.path.join(data, "stderr.log")


def vec_of(i, m):
    return m[i][2]


def check_all(proc, tag):
    """Exact-state checks shared by the first process and every restart."""
    live = sorted(model)

    # point lookup on rows that are definitely far past the memory budget
    for i in (1, 12345, live[-1]):
        h, body = query(proc, f"SELECT id, name, category, v FROM docs WHERE id = {i}")
        assert h == ["id", "name", "category", "v"]
        assert body == [[str(i), model[i][0], model[i][1], vlit(model[i][2])]], (
            f"{tag}: point lookup {i} -> {body!r}")

    # unfiltered TopK against Python truth
    q = [3, 5, 8, 13]
    got = ids(proc, f"SELECT id, L2_DISTANCE(v, '{vlit(q)}') AS d "
                    f"FROM docs ORDER BY d LIMIT 10")
    want = [str(i) for i in topk(model, q, 10)]
    assert got == want, f"{tag}: TopK {got!r} != {want!r}"

    # filtered TopK
    got = ids(proc, f"SELECT id, L2_DISTANCE(v, '{vlit(q)}') AS d FROM docs "
                    f"WHERE category = 'b' ORDER BY d LIMIT 7")
    want = [str(i) for i in topk(model, q, 7, cat="b")]
    assert got == want, f"{tag}: filtered TopK {got!r} != {want!r}"

    # distance DESC (generic ordered path, decodes spilled rows)
    got = ids(proc, f"SELECT id, L2_DISTANCE(v, '{vlit(q)}') AS d "
                    f"FROM docs ORDER BY d DESC LIMIT 5")
    want = [str(i) for i in topk(model, q, 5, desc=True)]
    assert got == want, f"{tag}: DESC TopK {got!r} != {want!r}"

    # ORDER BY primary key, both directions
    got = ids(proc, "SELECT id FROM docs ORDER BY id DESC LIMIT 5")
    assert got == [str(i) for i in live[-5:][::-1]], f"{tag}: id DESC {got!r}"
    got = ids(proc, f"SELECT id FROM docs WHERE id >= {N - 3} ORDER BY id LIMIT 2")
    assert got == [str(i) for i in live if i >= N - 3][:2], f"{tag}: id ASC {got!r}"

    # row count fingerprint (0 rows expected when the filter matches nothing)
    h, body = query(proc, f"SELECT id FROM docs WHERE category = 'zzz' LIMIT 5")
    assert h == ["id"] and body == [], f"{tag}: empty result {body!r}"


def err_log_stats():
    with open(err_log, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln for ln in fh if "minivectordb:" in ln]
    assert lines, "startup stats line missing"
    stats = {}
    for tok in lines[-1].split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            stats[k] = v
    return stats


def corrupt_spools(data_dir):
    """The spool is only a rebuildable cache: scribbling over it must not
    change any query result after restart."""
    hit = False
    for name in os.listdir(data_dir):
        if not name.endswith(".spool"):
            continue
        path = os.path.join(data_dir, name)
        with open(path, "r+b") as fh:
            fh.seek(0)
            fh.write(b"\xff" * min(4096, os.path.getsize(path)))
        hit = True
    assert hit, "no spool file found; the spill path was never exercised"


# ---------------------------------------------------------------------------
# phase 1: create, insert well past the budget, verify
# ---------------------------------------------------------------------------
p = start(data, err_log)
send(p, "CREATE TABLE docs (id BIGINT PRIMARY KEY, name VARCHAR(8), "
        "category VARCHAR(4), v VECTOR(4))")

BATCH = 500
for base in range(1, N + 1, BATCH):
    values = ",".join(row(i) for i in range(base, min(base + BATCH, N + 1)))
    send(p, "INSERT INTO docs VALUES " + values)
    for i in range(base, min(base + BATCH, N + 1)):
        model[i] = (f"n{i}", "abc"[i % 3], vec(i))

check_all(p, "phase1")

# UPDATE rows that are spilled
for i in range(15000, 15011):
    model[i] = ("upd", "z", [5, 6, 7, 8])
send(p, "UPDATE docs SET name = 'upd', category = 'z', v = '[5,6,7,8]' "
        "WHERE id >= 15000 AND id <= 15010")

q = [5, 6, 7, 8]
got = ids(p, f"SELECT id, L2_DISTANCE(v, '{vlit(q)}') AS d "
             f"FROM docs ORDER BY d LIMIT 15")
want = [str(i) for i in topk(model, q, 15)]
assert got == want, f"post-update TopK {got!r} != {want!r}"
assert got[:11] == [str(i) for i in range(15000, 15011)], got

# DELETE a spilled range and confirm it is really gone
for i in range(18000, 18021):
    model.pop(i)
send(p, "DELETE FROM docs WHERE id >= 18000 AND id <= 18020")
assert ids(p, "SELECT id FROM docs WHERE id >= 18000 AND id <= 18020") == []

# transaction: UPDATE + DELETE then ROLLBACK restores spilled rows in place
send(p, "BEGIN")
send(p, "UPDATE docs SET v = '[0,0,0,0]' WHERE id >= 16000 AND id <= 16005")
send(p, "DELETE FROM docs WHERE id >= 17000 AND id <= 17005")
assert ids(p, "SELECT id FROM docs WHERE id = 16000") == ["16000"]
assert ids(p, "SELECT id FROM docs WHERE id = 17000") == []
send(p, "ROLLBACK")
for i in range(16000, 16006):
    _, body = query(p, f"SELECT id, v FROM docs WHERE id = {i}")
    assert body == [[str(i), vlit(model[i][2])]], f"rollback update {i}: {body!r}"
assert ids(p, "SELECT id FROM docs WHERE id = 17000") == ["17000"]

# transaction: INSERT then ROLLBACK must erase the new spilled rows
send(p, "BEGIN")
send(p, "INSERT INTO docs VALUES (30001,'tx','q','[1,1,1,1]'),"
       "(30002,'tx','q','[1,1,1,1]')")
assert ids(p, "SELECT id FROM docs WHERE id = 30001") == ["30001"]
send(p, "ROLLBACK")
assert ids(p, "SELECT id FROM docs WHERE id = 30001") == []
assert ids(p, "SELECT id FROM docs WHERE id = 30002") == []

# transaction: COMMIT after spilling must persist
send(p, "BEGIN")
model[16000] = ("n16000", model[16000][1], [9, 9, 9, 9])
send(p, "UPDATE docs SET v = '[9,9,9,9]' WHERE id = 16000")
send(p, "COMMIT")

check_all(p, "phase1-after-txn")
close(p)

# ---------------------------------------------------------------------------
# phase 2: clean restart (snapshot -> spool) must reproduce the exact state
# ---------------------------------------------------------------------------
corrupt_spools(data)
p = start(data, err_log)
stats = err_log_stats()
assert stats.get("spilled_rows") == str(len(model)), (
    f"restart did not spill rows: {stats.get('spilled_rows')} != {len(model)}")
check_all(p, "phase2")
_, body = query(p, "SELECT id, v FROM docs WHERE id = 16000")
assert body == [["16000", "[9,9,9,9]"]], body

# ---------------------------------------------------------------------------
# phase 3: committed rows survive SIGKILL while spilling; WAL replay is exact
# ---------------------------------------------------------------------------
for base in range(40000, 40100, 50):
    values = ",".join(row(i) for i in range(base, min(base + 50, 40100)))
    send(p, "INSERT INTO docs VALUES " + values)
    for i in range(base, min(base + 50, 40100)):
        model[i] = (f"n{i}", "abc"[i % 3], vec(i))
kill9(p)
corrupt_spools(data)

p = start(data, err_log)
stats = err_log_stats()
assert stats.get("spilled_rows") == str(len(model)), (
    f"replay did not spill rows: {stats.get('spilled_rows')} != {len(model)}")
check_all(p, "phase3")
assert ids(p, "SELECT id FROM docs ORDER BY id DESC LIMIT 1") == ["40099"]
_, body = query(p, "SELECT id, name FROM docs WHERE id = 1")
assert body == [["1", "n1"]], body

close(p)
print("spill tests ok")
