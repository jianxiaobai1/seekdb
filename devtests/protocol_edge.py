#!/usr/bin/env python3
"""Protocol edge cases: multi-line SQL, quoted identifiers, ids, escapes, big lines.

Development only. Uses the JSONL stdio contract from STDIO_PROTOCOL.md.
"""
import json
import os
import shutil
import subprocess
import tempfile

BIN = os.environ.get('MVD_BIN', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'minivectordb'))
DATA = os.path.join(tempfile.gettempdir(), 'mvd-proto')


class Raw:
    def __init__(self):
        shutil.rmtree(DATA, ignore_errors=True)
        os.makedirs(DATA)
        self.proc = subprocess.Popen(
            [BIN, '--data-dir', DATA],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding='utf-8', bufsize=1,
        )
        assert json.loads(self.proc.stdout.readline()) == {'type': 'ready'}

    def send(self, payload):
        self.proc.stdin.write(payload + '\n')
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert line.endswith('\n')
        return json.loads(line)

    def req(self, rid, sql, mode=None):
        body = {'id': rid, 'op': 'execute', 'sql': sql}
        if mode is not None:
            body['search_mode'] = mode
        return self.send(json.dumps(body, ensure_ascii=False))


def main():
    r = Raw()

    # Multi-line SQL in one request (task example style).
    resp = r.req(1, "CREATE TABLE docs (\n"
                    "  id BIGINT PRIMARY KEY,\n"
                    "  note VARCHAR(16),\n"
                    "  embedding VECTOR(2)\n"
                    ");")
    assert resp == {'id': 1, 'ok': True}, resp

    resp = r.req('two', "INSERT INTO docs VALUES\n"
                        "  (1, 'it''s', '[1, 0]'),\n"
                        "  (2, \"a\\nb\", '[0, 1]');")
    assert resp == {'id': 'two', 'ok': True, 'affected_rows': 2}, resp

    resp = r.req(3, "SELECT id, note,\n"
                    "       L2_DISTANCE(embedding, '[0, 0]') AS distance\n"
                    "FROM docs\n"
                    "ORDER BY distance\n"
                    "LIMIT 2;")
    assert resp['ok'] and resp['columns'] == ['id', 'note', 'distance'], resp
    assert resp['rows'] == [[1, "it's", 1], [2, 'a\nb', 1]], resp

    # Backtick identifiers, aliases and mixed case keywords.
    assert r.req(4, 'CREATE TABLE `odd name` (`id` BIGINT PRIMARY KEY, `v` VECTOR(1));')['ok']
    assert r.req(5, "INSERT INTO `odd name` VALUES (7, '[2]');")['affected_rows'] == 1
    resp = r.req(6, 'select `id`, l2_distance(`v`, \'[0]\') as d from `odd name` order by d desc;')
    assert resp['columns'] == ['id', 'd'] and resp['rows'] == [[7, 2]], resp

    # Comments, extra semicolons and an empty statement list.
    assert r.req(7, '-- leading comment\nSELECT id /* inline */ FROM docs WHERE id = 1;;')['rows'] == [[1]]
    assert r.req(8, '-- only a comment')['ok']

    # String escapes: backslash forms and doubled quote.
    assert r.req(9, "INSERT INTO docs VALUES (10, 'x\\\\y%z', '[0,0]'), (11, 'l\\tm', '[0,0]');")['affected_rows'] == 2
    resp = r.req(10, "SELECT note FROM docs WHERE id >= 10 ORDER BY id;")
    assert resp['rows'] == [['x\\y%z'], ['l\tm']], resp

    # Large single-line request (~1.5 MiB).
    rows = ','.join(f"({1000 + i},'n{i}','[{i % 7},0]')" for i in range(20000))
    resp = r.req(11, f'INSERT INTO docs VALUES {rows};')
    assert resp['ok'] and resp['affected_rows'] == 20000, resp
    assert r.req(12, 'SELECT id FROM docs WHERE id = 20999;')['rows'] == [[20999]]

    # A request without search_mode still works; ann is accepted.
    assert r.req(13, "SELECT id FROM docs WHERE id = 1000;")['rows'] == [[1000]]
    assert r.req(14, "SELECT id FROM docs WHERE id = 1000;", 'ann')['rows'] == [[1000]]
    assert r.req(15, "SELECT id FROM docs WHERE id = 1000;", 'exact')['rows'] == [[1000]]

    # Unknown op gets one error response and the session continues.
    resp = r.send(json.dumps({'id': 16, 'op': 'explain'}))
    assert resp['id'] == 16 and resp['ok'] is False and resp['error']['code'], resp
    assert r.req(17, 'SELECT id FROM docs WHERE id = 1000;')['rows'] == [[1000]]

    # Case-insensitive table lookup.
    assert r.req(18, 'SELECT id FROM DOCS WHERE id = 1000;')['rows'] == [[1000]]

    # Shutdown: exactly one response, then exit 0, no trailing stdout.
    NEXT = 19
    resp = r.send(json.dumps({'id': NEXT, 'op': 'shutdown'}))
    assert resp == {'id': NEXT, 'ok': True}, resp
    assert r.proc.wait(timeout=10) == 0
    rest = r.proc.stdout.read()
    assert rest == '', f'unexpected stdout after shutdown: {rest!r}'

    # SIGTERM with an open transaction must not corrupt committed data.
    p = subprocess.Popen([BIN, '--data-dir', DATA], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, encoding='utf-8', bufsize=1)
    assert json.loads(p.stdout.readline()) == {'type': 'ready'}
    for i, sql in enumerate(['BEGIN;', "INSERT INTO docs VALUES (99999,'tx','[0,0]');"], 1):
        p.stdin.write(json.dumps({'id': i, 'op': 'execute', 'sql': sql}) + '\n')
        p.stdin.flush()
        p.stdout.readline()
    p.terminate()
    assert p.wait(timeout=10) == 0
    s = subprocess.Popen([BIN, '--data-dir', DATA], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, encoding='utf-8', bufsize=1)
    assert json.loads(s.stdout.readline()) == {'type': 'ready'}
    s.stdin.write(json.dumps({'id': 1, 'op': 'execute', 'sql': 'SELECT id FROM docs WHERE id = 99999;'}) + '\n')
    s.stdin.flush()
    resp = json.loads(s.stdout.readline())
    assert resp['ok'] and resp['rows'] == [], resp
    s.stdin.write(json.dumps({'id': 2, 'op': 'shutdown'}) + '\n')
    s.stdin.flush()
    s.stdout.readline()
    assert s.wait(timeout=10) == 0

    print('protocol_edge: PASS')
    shutil.rmtree(DATA, ignore_errors=True)


if __name__ == '__main__':
    main()
