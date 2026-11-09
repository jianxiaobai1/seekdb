#!/usr/bin/env python3
"""SQL semantics hardening tests.

Covers the hidden-case risks of the competition gate: string escaping across
the statement splitter, comments, Unicode VARCHAR lengths, LIMIT edges,
ordering ties, unsupported statements, transaction atomicity, vector literal
validation, and identifier case handling.
"""
import math
import os
import subprocess
import sys
import tempfile

BIN = sys.argv[1] if len(sys.argv) > 1 else "./build/minivectordb"
SENTINEL = "__MVDB_END__"


def start(data_dir, err_path):
    err = open(err_path, "ab")
    proc = subprocess.Popen(
        [BIN, "--data-dir", data_dir],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=err,
        text=True,
        bufsize=1,
    )
    ready = proc.stdout.readline().strip()
    assert ready == "READY", f"unexpected ready line: {ready!r}"
    return proc


def read_response(proc):
    lines = []
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("database closed stdout unexpectedly")
        line = line.rstrip("\n")
        if line == SENTINEL:
            return lines
        lines.append(line)


def send(proc, sql):
    proc.stdin.write(sql + ";\n")
    proc.stdin.flush()
    lines = read_response(proc)
    for line in lines:
        assert not line.startswith("ERROR"), f"{sql!r} -> {line}"
    return lines


def send_raw(proc, text, nstatements):
    proc.stdin.write(text)
    proc.stdin.flush()
    return [read_response(proc) for _ in range(nstatements)]


def query(proc, sql):
    lines = send(proc, sql)
    assert lines, f"no header for {sql!r}"
    return lines[0].split("\t"), [ln.split("\t") for ln in lines[1:]]


def ids(proc, sql):
    _, body = query(proc, sql)
    return [r[0] for r in body]


def expect_error(proc, sql, code):
    proc.stdin.write(sql + ";\n")
    proc.stdin.flush()
    lines = read_response(proc)
    assert len(lines) == 1, f"{sql!r} -> {lines}"
    assert lines[0].startswith(f"ERROR {code}:"), f"{sql!r} -> {lines[0]}"


def close(proc):
    proc.stdin.close()
    assert proc.wait(timeout=30) == 0


def setup(proc):
    send(
        proc,
        "CREATE TABLE sem (id BIGINT PRIMARY KEY, name VARCHAR(4), "
        "tag VARCHAR(16), v VECTOR(3))",
    )


def test_strings_and_comments(proc):
    setup(proc)
    # '' and \' escapes, plus ';' and '--' inside literals: the statement
    # splitter must not cut or comment them out.
    send(proc, "INSERT INTO sem VALUES (1, 'it''s', 'a;b', '[1,0,0]')")
    send(proc, "INSERT INTO sem VALUES (2, 'it\\'s', 'c--d', '[0,1,0]')")
    _, rows = query(proc, "SELECT id, name FROM sem WHERE id = 1")
    assert rows == [["1", "it's"]], rows
    _, rows = query(proc, "SELECT id, name, tag FROM sem WHERE id = 2")
    assert rows == [["2", "it's", "c--d"]], rows

    raw = ("/* lead */ INSERT INTO sem VALUES (3, 'aa', 'x', '[0,0,1]') "
           "-- trailing comment\n;\n")
    send_raw(proc, raw, 1)
    assert ids(proc, "SELECT id FROM sem WHERE id = 3") == ["3"]

    # hash comment inside a statement
    raw = "INSERT INTO sem VALUES (4, 'bb', 'y', '[0,0,1]') # tail\n;\n"
    send_raw(proc, raw, 1)
    assert ids(proc, "SELECT id FROM sem WHERE id = 4") == ["4"]

    # Two statements in one chunk produce two responses.
    res = send_raw(
        proc,
        "SELECT id FROM sem WHERE id = 3;\nSELECT id FROM sem WHERE id = 4;\n",
        2,
    )
    assert res[0][1].split("\t") == ["3"], res
    assert res[1][1].split("\t") == ["4"], res


def test_unicode_length(proc):
    # VARCHAR(4) counts code points, not UTF-8 bytes.
    send(proc, "INSERT INTO sem VALUES (5, '中文测试', 'u', '[0,0,0]')")
    expect_error(
        proc,
        "INSERT INTO sem VALUES (6, '中文字符串', 'u', '[0,0,0]')",
        1406,
    )
    assert ids(proc, "SELECT id FROM sem WHERE id = 5") == ["5"]
    assert ids(proc, "SELECT id FROM sem WHERE id = 6") == []


def test_limit_edges(proc):
    _, rows = query(proc, "SELECT id FROM sem ORDER BY id LIMIT 0")
    assert rows == [], rows
    expect_error(proc, "SELECT id FROM sem LIMIT -1", 1265)
    _, rows = query(proc, "SELECT id FROM sem ORDER BY id LIMIT 100")
    assert [r[0] for r in rows] == ["1", "2", "3", "4", "5"], rows


def test_order_ties(proc):
    send(proc, "CREATE TABLE ord (id BIGINT PRIMARY KEY, k VARCHAR(4))")
    for pk, k in [(1, "b"), (2, "a"), (3, "b"), (4, "é"), (5, "aa")]:
        send(proc, f"INSERT INTO ord VALUES ({pk}, '{k}')")
    assert ids(proc, "SELECT id FROM ord ORDER BY k") == ["2", "5", "1", "3", "4"]
    assert ids(proc, "SELECT id FROM ord ORDER BY k DESC LIMIT 2") == ["4", "1"]
    assert ids(proc, "SELECT id FROM ord ORDER BY k LIMIT 3") == ["2", "5", "1"]
    assert ids(proc, "SELECT id FROM ord ORDER BY id DESC LIMIT 2") == ["5", "4"]
    # String equality is by Unicode code point, i.e. case sensitive.
    send(proc, "INSERT INTO ord VALUES (6, 'A')")
    assert ids(proc, "SELECT id FROM ord WHERE k = 'a'") == ["2"]
    assert ids(proc, "SELECT id FROM ord WHERE k >= 'b' ORDER BY k") == [
        "1",
        "3",
        "4",
    ]


def test_unsupported(proc):
    expect_error(proc, "DROP TABLE sem", 1064)
    expect_error(proc, "ALTER TABLE sem ADD COLUMN x BIGINT", 1064)
    expect_error(proc, "CREATE INDEX i ON sem (id)", 1064)
    expect_error(proc, "SELECT id FROM sem WHERE id = 1 OR id = 2", 1064)
    expect_error(proc, "SELECT COUNT(*) FROM sem", 1064)
    expect_error(proc, "SELECT id FROM sem ORDER BY id LIMIT 1 OFFSET 1", 1064)
    expect_error(proc, "SELECT id FROM sem JOIN ord ON sem.id = ord.id", 1064)
    expect_error(proc, "INSERT INTO sem VALUES (7, 'x')", 1265)
    expect_error(proc, "UPDATE sem SET id = 9 WHERE id = 1", 1235)
    expect_error(proc, "SELECT id FROM sem WHERE v = '[1,0,0]'", 1235)
    expect_error(proc, "SELECT id FROM sem ORDER BY v", 1235)


def test_errors(proc):
    expect_error(
        proc,
        "CREATE TABLE sem (id BIGINT PRIMARY KEY)",
        1050,
    )
    expect_error(proc, "SELECT id FROM missing", 1146)
    expect_error(proc, "SELECT nope FROM sem", 1054)
    expect_error(proc, "SELECT id FROM sem WHERE nope = 1", 1054)
    expect_error(proc, "INSERT INTO sem (id, nope) VALUES (8, 'x')", 1054)
    expect_error(
        proc,
        "CREATE TABLE bad (id BIGINT PRIMARY KEY, id2 BIGINT, id BIGINT)",
        1265,
    )
    expect_error(
        proc,
        "CREATE TABLE bad2 (a BIGINT, b BIGINT PRIMARY KEY, c BIGINT PRIMARY KEY)",
        1265,
    )
    expect_error(
        proc,
        "CREATE TABLE bad3 (id BIGINT)",
        1235,
    )


def test_insert_atomicity(proc):
    # One bad tuple rejects the whole statement.
    expect_error(
        proc,
        "INSERT INTO sem VALUES (20, 'a', 'a', '[0,0,0]'), "
        "(20, 'b', 'b', '[0,0,0]')",
        1062,
    )
    assert ids(proc, "SELECT id FROM sem WHERE id = 20") == []
    send(
        proc,
        "INSERT INTO sem VALUES (20, 'a', 'a', '[0,0,0]'), "
        "(21, 'b', 'b', '[0,0,0]')",
    )
    assert ids(proc, "SELECT id FROM sem WHERE id >= 20 ORDER BY id") == [
        "20",
        "21",
    ]
    # Explicit column list must carry the primary key (no DEFAULT in subset).
    expect_error(
        proc,
        "INSERT INTO sem (name, tag, v) VALUES ('x', 'y', '[0,0,0]')",
        1364,
    )
    send(proc, "INSERT INTO sem (v, id, name) VALUES ('[0,0,0]', 22, 'x')")
    assert ids(proc, "SELECT id FROM sem WHERE id = 22") == ["22"]


def test_transactions(proc):
    send(proc, "BEGIN")
    send(proc, "INSERT INTO sem VALUES (40, 'tx', 't', '[0,0,0]')")
    assert ids(proc, "SELECT id FROM sem WHERE id = 40") == ["40"]  # own writes
    expect_error(proc, "INSERT INTO sem VALUES (40, 'x', 'x', '[0,0,0]')", 1062)
    # The failed statement leaves the earlier one intact.
    assert ids(proc, "SELECT id FROM sem WHERE id = 40") == ["40"]
    send(proc, "COMMIT")
    assert ids(proc, "SELECT id FROM sem WHERE id = 40") == ["40"]

    send(proc, "BEGIN")
    send(proc, "DELETE FROM sem WHERE id = 40")
    send(proc, "UPDATE sem SET name = 'zz' WHERE id = 1")
    assert ids(proc, "SELECT id FROM sem WHERE id = 40") == []
    _, rows = query(proc, "SELECT name FROM sem WHERE id = 1")
    assert rows == [["zz"]], rows
    send(proc, "ROLLBACK")
    assert ids(proc, "SELECT id FROM sem WHERE id = 40") == ["40"]
    _, rows = query(proc, "SELECT name FROM sem WHERE id = 1")
    assert rows == [["it's"]], rows

    # COMMIT / ROLLBACK outside a transaction are harmless no-ops.
    send(proc, "COMMIT")
    send(proc, "ROLLBACK")
    send(proc, "BEGIN")
    expect_error(proc, "BEGIN", 1235)
    send(proc, "ROLLBACK")

    # START TRANSACTION / END are the MySQL spellings of BEGIN / COMMIT.
    send(proc, "START TRANSACTION")
    send(proc, "INSERT INTO sem VALUES (41, 'st', 't', '[0,0,0]')")
    send(proc, "END")
    assert ids(proc, "SELECT id FROM sem WHERE id = 41") == ["41"]


def test_vector_validation(proc):
    for bad in ["[nan,0,0]", "[inf,0,0]", "[1e39,0,0]", "[1,2]", "[1,2,3,4]",
                "[1,0]", "[]", "[1,0,x]"]:
        expect_error(
            proc, f"INSERT INTO sem VALUES (60, 'a', 'a', '{bad}')", 1265
        )
    expect_error(
        proc, "INSERT INTO sem VALUES (9223372036854775808, 'a', 'a', '[0,0,0]')",
        1264,
    )
    expect_error(
        proc, "SELECT id FROM sem WHERE id = 9223372036854775808", 1264
    )
    expect_error(
        proc,
        "SELECT L2_DISTANCE(v, '[1,2]') AS d FROM sem",
        1265,
    )
    expect_error(
        proc,
        "SELECT L2_DISTANCE(id, '[1,2,3]') AS d FROM sem",
        1265,
    )


def test_topk(proc):
    send(proc, "CREATE TABLE tk (id BIGINT PRIMARY KEY, v VECTOR(2))")
    data = [(1, [1, 0]), (2, [0, 1]), (3, [-1, 0]), (4, [0, -1]), (5, [3, 4])]
    for pk, v in data:
        lit = "[" + ",".join(str(x) for x in v) + "]"
        send(proc, f"INSERT INTO tk VALUES ({pk}, '{lit}')")
    # All unit vectors are exactly distance 1 from the origin -> pk order.
    assert ids(
        proc, "SELECT id FROM tk ORDER BY L2_DISTANCE(v, '[0,0]') LIMIT 2"
    ) == ["1", "2"]
    assert ids(
        proc,
        "SELECT id FROM tk WHERE id >= 2 "
        "ORDER BY L2_DISTANCE(v, '[0,0]') LIMIT 2",
    ) == ["2", "3"]
    assert ids(
        proc, "SELECT id FROM tk ORDER BY L2_DISTANCE(v, '[0,0]') LIMIT 3"
    ) == ["1", "2", "3"]
    _, rows = query(
        proc,
        "SELECT id, L2_DISTANCE(v, '[0,0]') AS distance FROM tk WHERE id = 5",
    )
    assert rows == [["5", "5.000000"]], rows
    _, rows = query(proc, "SELECT * FROM tk WHERE id = 5")
    assert rows == [["5", "[3,4]"]], rows
    # Distance descending order still breaks ties by ascending pk.
    assert ids(
        proc, "SELECT id FROM tk ORDER BY L2_DISTANCE(v, '[0,0]') DESC LIMIT 2"
    ) == ["5", "1"]
    # Projection alias is usable in ORDER BY.
    assert ids(
        proc,
        "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM tk ORDER BY d LIMIT 1",
    ) == ["1"]

    # A transaction sees its own vector changes and rollback restores them.
    send(proc, "BEGIN")
    send(proc, "INSERT INTO tk VALUES (6, '[0,0]')")
    assert ids(
        proc, "SELECT id FROM tk ORDER BY L2_DISTANCE(v, '[0,0]') LIMIT 1"
    ) == ["6"]
    send(proc, "ROLLBACK")
    assert ids(
        proc, "SELECT id FROM tk ORDER BY L2_DISTANCE(v, '[0,0]') LIMIT 1"
    ) == ["1"]

    # Vector index DDL is accepted and does not change exact results.
    send(proc, "CREATE VECTOR INDEX tk_vec ON tk (v)")
    expect_error(proc, "CREATE VECTOR INDEX tk_vec2 ON tk (id)", 1265)
    expect_error(proc, "CREATE VECTOR INDEX tk_dup ON tk (v)", 1061)
    assert ids(proc, "SELECT id FROM tk WHERE id >= 3 "
                     "ORDER BY L2_DISTANCE(v, '[0,0]') LIMIT 1") == ["3"]


def test_empty_table(proc):
    send(proc, "CREATE TABLE empty_t (id BIGINT PRIMARY KEY, v VECTOR(2))")
    _, rows = query(proc, "SELECT id FROM empty_t")
    assert rows == [], rows
    _, rows = query(
        proc, "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM empty_t "
              "ORDER BY d LIMIT 3"
    )
    assert rows == [], rows
    send(proc, "DELETE FROM empty_t")
    send(proc, "UPDATE empty_t SET v = '[0,0]' WHERE id = 1")
    assert ids(proc, "SELECT id FROM empty_t LIMIT 5") == []


def test_case_and_backticks(proc):
    assert ids(proc, "SELECT ID FROM SEM WHERE Id = 1") == ["1"]
    assert ids(proc, "SELECT id FROM Sem WHERE ID = 2") == ["2"]
    send(proc, "CREATE TABLE `bt` (`id` BIGINT PRIMARY KEY, `v` VECTOR(2))")
    send(proc, "INSERT INTO `bt` VALUES (1, '[1,2]')")
    _, rows = query(proc, "SELECT `id`, `v` FROM `bt`")
    assert rows == [["1", "[1,2]"]], rows


def main():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        proc = start(data_dir, os.path.join(tmp, "stderr.log"))
        test_strings_and_comments(proc)
        test_unicode_length(proc)
        test_limit_edges(proc)
        test_order_ties(proc)
        test_unsupported(proc)
        test_errors(proc)
        test_insert_atomicity(proc)
        test_transactions(proc)
        test_vector_validation(proc)
        test_topk(proc)
        test_empty_table(proc)
        test_case_and_backticks(proc)
        close(proc)

        # Everything above must survive a clean restart (snapshot round trip).
        proc = start(data_dir, os.path.join(tmp, "stderr2.log"))
        _, rows = query(proc, "SELECT id, name FROM sem WHERE id = 2")
        assert rows == [["2", "it's"]], rows
        assert ids(proc, "SELECT id FROM ord ORDER BY k") == [
            "6", "2", "5", "1", "3", "4"
        ]
        assert ids(
            proc, "SELECT id FROM tk ORDER BY L2_DISTANCE(v, '[0,0]') LIMIT 2"
        ) == ["1", "2"]
        close(proc)
    print("semantics tests ok")


if __name__ == "__main__":
    main()
