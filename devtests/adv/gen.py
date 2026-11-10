#!/usr/bin/env python3
"""Generate local adversarial cases in the public mysqltest package format.

Local development only; not part of the submitted package. Each expected
result was derived by hand from the task statement (tie -> primary key asc,
filter before TopK, float32 storage, Unicode code-point semantics).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = {}


def case(cid, test, result):
    statements = [line for line in test if not line.startswith('--')]
    assert len(statements) == len(result), f'{cid}: {len(statements)} sql vs {len(result)} results'
    CASES[cid] = (test, result)


case('adv_order_variants', [
    "CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2));",
    "INSERT INTO t VALUES (1,'[1,0]'),(2,'[0,1]'),(3,'[1,0]'),(4,'[2,2]');",
    "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM t ORDER BY d DESC;",
    "SELECT id FROM t WHERE id >= 2 ORDER BY id DESC LIMIT 2;",
    "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM t ORDER BY d LIMIT 0;",
    "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM t ORDER BY d;",
    "SELECT id, L2_DISTANCE(v, '[0,0]') AS distance FROM t WHERE id <= 2 ORDER BY distance DESC LIMIT 1;",
], [
    {},
    {'affected_rows': 4},
    {'columns': ['id', 'd'], 'rows': [[4, 2.8284271247461903], [1, 1.0], [2, 1.0], [3, 1.0]]},
    {'columns': ['id'], 'rows': [[4], [3]]},
    {'columns': ['id', 'd'], 'rows': []},
    {'columns': ['id', 'd'], 'rows': [[1, 1.0], [2, 1.0], [3, 1.0], [4, 2.8284271247461903]]},
    {'columns': ['id', 'distance'], 'rows': [[1, 1.0]]},
])

case('adv_update_atomic', [
    "CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2), s VARCHAR(4));",
    "INSERT INTO t VALUES (1,'[1,1]','aa'),(2,'[2,2]','bb');",
    "UPDATE t SET v = '[1,2,3]' WHERE id >= 1;",
    "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM t ORDER BY id;",
    "UPDATE t SET s='xxxxx' WHERE id >= 1;",
    "SELECT id, s FROM t ORDER BY id;",
    "BEGIN;",
    "UPDATE t SET v = '[9,9]' WHERE id = 1;",
    "UPDATE t SET v = '[1,2,3]' WHERE id = 2;",
    "COMMIT;",
    "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM t ORDER BY id;",
    "--restart",
    "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM t ORDER BY id;",
], [
    {},
    {'affected_rows': 2},
    {'error': 'invalid_vector'},
    {'columns': ['id', 'd'], 'rows': [[1, 1.4142135623730951], [2, 2.8284271247461903]]},
    {'error': 'value_too_long'},
    {'columns': ['id', 's'], 'rows': [[1, 'aa'], [2, 'bb']]},
    {},
    {'affected_rows': 1},
    {'error': 'invalid_vector'},
    {},
    {'columns': ['id', 'd'], 'rows': [[1, 12.727922061357855], [2, 2.8284271247461903]]},
    {'columns': ['id', 'd'], 'rows': [[1, 12.727922061357855], [2, 2.8284271247461903]]},
])

case('adv_txn_visibility', [
    "CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2), c VARCHAR(2));",
    "INSERT INTO t VALUES (1,'[1,0]','x');",
    "BEGIN;",
    "INSERT INTO t VALUES (2,'[0,1]','y');",
    "UPDATE t SET v='[5,5]' WHERE id=1;",
    "DELETE FROM t WHERE id=1;",
    "INSERT INTO t VALUES (3,'[0,0]','z');",
    "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM t ORDER BY d;",
    "SELECT id FROM t WHERE c = 'y' AND id > 1 ORDER BY id;",
    "ROLLBACK;",
    "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM t ORDER BY d;",
    "--restart",
    "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM t ORDER BY d;",
], [
    {},
    {'affected_rows': 1},
    {},
    {'affected_rows': 1},
    {'affected_rows': 1},
    {'affected_rows': 1},
    {'affected_rows': 1},
    {'columns': ['id', 'd'], 'rows': [[3, 0.0], [2, 1.0]]},
    {'columns': ['id'], 'rows': [[2]]},
    {},
    {'columns': ['id', 'd'], 'rows': [[1, 1.0]]},
    {'columns': ['id', 'd'], 'rows': [[1, 1.0]]},
])

case('adv_invalid_vectors', [
    "CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(3));",
    "INSERT INTO t VALUES (1, '[NaN,0,0]');",
    "INSERT INTO t VALUES (2, '[Infinity,0,0]');",
    "INSERT INTO t VALUES (3, '[1,2]');",
    "INSERT INTO t VALUES (4, '[1e39,0,0]');",
    "INSERT INTO t VALUES (5, '[-1e39,0,0]');",
    "INSERT INTO t VALUES (6, '[1,2,3]');",
    "INSERT INTO t VALUES (7, '[1,2,3,4]');",
    "INSERT INTO t VALUES (8, '[abc,0,0]');",
    "UPDATE t SET v = '[NaN,0,0]' WHERE id = 6;",
    "SELECT id FROM t;",
    "--restart",
    "SELECT id FROM t;",
], [
    {},
    {'error': 'invalid_vector'},
    {'error': 'invalid_vector'},
    {'error': 'dimension_mismatch'},
    {'error': 'invalid_vector'},
    {'error': 'invalid_vector'},
    {'affected_rows': 1},
    {'error': 'dimension_mismatch'},
    {'error': 'invalid_vector'},
    {'error': 'invalid_vector'},
    {'columns': ['id'], 'rows': [[6]]},
    {'columns': ['id'], 'rows': [[6]]},
])

case('adv_nested_pk', [
    "CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2), s VARCHAR(2));",
    "INSERT INTO t VALUES (1,'[0,0]','a');",
    "BEGIN;",
    "BEGIN;",
    "SELECT id FROM t;",
    "COMMIT;",
    "UPDATE t SET id=2 WHERE id=1;",
    "SELECT id FROM t;",
    "BEGIN;",
    "INSERT INTO t VALUES (2,'[0,0]','b');",
    "ROLLBACK;",
    "SELECT id FROM t ORDER BY id;",
], [
    {},
    {'affected_rows': 1},
    {},
    {'error': 'transaction_error'},
    {'columns': ['id'], 'rows': [[1]]},
    {},
    {'error': 'unsupported_sql'},
    {'columns': ['id'], 'rows': [[1]]},
    {},
    {'affected_rows': 1},
    {},
    {'columns': ['id'], 'rows': [[1]]},
])

case('adv_unicode', [
    "CREATE TABLE u (id BIGINT PRIMARY KEY, s VARCHAR(4));",
    "INSERT INTO u VALUES (1, '汉字测试');",
    "INSERT INTO u VALUES (2, '😀😀😀😀');",
    "INSERT INTO u VALUES (3, '汉字测试多');",
    "INSERT INTO u VALUES (4, '😀😀😀😀😀');",
    "INSERT INTO u VALUES (5, 'Z'), (6, 'a');",
    "SELECT id, s FROM u ORDER BY id;",
    "SELECT id FROM u ORDER BY s;",
    "SELECT id FROM u WHERE s = '😀😀😀😀';",
    "SELECT id FROM u WHERE s > 'Z' ORDER BY id;",
], [
    {},
    {'affected_rows': 1},
    {'affected_rows': 1},
    {'error': 'value_too_long'},
    {'error': 'value_too_long'},
    {'affected_rows': 2},
    {'columns': ['id', 's'], 'rows': [[1, '汉字测试'], [2, '😀😀😀😀'], [5, 'Z'], [6, 'a']]},
    {'columns': ['id'], 'rows': [[5], [6], [1], [2]]},
    {'columns': ['id'], 'rows': [[2]]},
    {'columns': ['id'], 'rows': [[1], [2], [6]]},
])

case('adv_errors', [
    "SELECT * FROM missing;",
    "CREATE TABLE e (id BIGINT PRIMARY KEY, v VECTOR(2));",
    "SELECT nope FROM e;",
    "INSERT INTO e VALUES (1,'[0,0]'),(1,'[1,1]');",
    "SELECT id FROM e;",
    "INSERT INTO e VALUES (1,'[0,0]');",
    "INSERT INTO e VALUES (1,'[1,1]');",
    "DELETE FROM missing WHERE id = 1;",
    "UPDATE e SET nope = 1 WHERE id = 1;",
    "SELECT id FROM e WHERE id = 1 AND;",
    "DROP TABLE e;",
    "SELECT id FROM e;",
    "CREATE TABLE e (id BIGINT PRIMARY KEY, v VECTOR(2));",
    "CREATE VECTOR INDEX i ON e(v);",
    "CREATE VECTOR INDEX i ON e(v);",
    "CREATE VECTOR INDEX j ON e(nope);",
    "SELECT id FROM e;",
], [
    {'error': 'table_not_found'},
    {},
    {'error': 'column_not_found'},
    {'error': 'duplicate_key'},
    {'columns': ['id'], 'rows': []},
    {'affected_rows': 1},
    {'error': 'duplicate_key'},
    {'error': 'table_not_found'},
    {'error': 'column_not_found'},
    {'error': 'unsupported_sql'},
    {'error': 'unsupported_sql'},
    {'columns': ['id'], 'rows': [[1]]},
    {'error': 'table_exists'},
    {},
    {'error': 'index_exists'},
    {'error': 'column_not_found'},
    {'columns': ['id'], 'rows': [[1]]},
])

case('adv_topk_forms', [
    "CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2));",
    "INSERT INTO t VALUES (1,'[1,0]'),(2,'[0,1]'),(3,'[2,2]'),(4,'[1,0]');",
    "SELECT id FROM t ORDER BY L2_DISTANCE(v, '[0,0]') LIMIT 2;",
    "SELECT id FROM t ORDER BY L2_DISTANCE(v, '[0,0]') ASC;",
    "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM t ORDER BY d LIMIT 1;",
    "SELECT id FROM t WHERE id > 0 ORDER BY L2_DISTANCE(v, '[0,0]') DESC LIMIT 2;",
    "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM t WHERE id >= 1 ORDER BY d LIMIT 10;",
], [
    {},
    {'affected_rows': 4},
    {'columns': ['id'], 'rows': [[1], [2]]},
    {'columns': ['id'], 'rows': [[1], [2], [4], [3]]},
    {'columns': ['id', 'd'], 'rows': [[1, 1.0]]},
    {'columns': ['id'], 'rows': [[3], [1]]},
    {'columns': ['id', 'd'], 'rows': [[1, 1.0], [2, 1.0], [4, 1.0], [3, 2.8284271247461903]]},
])

case('adv_filter_topk', [
    "CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2), c VARCHAR(8));",
    "INSERT INTO t VALUES (1,'[1,0]','odd'),(2,'[2,0]','even'),(3,'[3,0]','odd'),(4,'[4,0]','even'),"
    "(5,'[5,0]','odd'),(6,'[6,0]','even'),(7,'[7,0]','odd'),(8,'[8,0]','even'),"
    "(9,'[9,0]','odd'),(10,'[10,0]','even');",
    "INSERT INTO t VALUES (22,'[5,0]','tie'),(21,'[5,0]','tie'),(23,'[5,0]','tie');",
    "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM t WHERE c = 'even' ORDER BY d LIMIT 3;",
    "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM t WHERE c = 'tie' ORDER BY d LIMIT 10;",
    "SELECT id FROM t WHERE c = 'odd' ORDER BY id DESC LIMIT 2;",
    "SELECT id FROM t WHERE id > 100 ORDER BY id LIMIT 5;",
    "SELECT id FROM t WHERE c = 'even' AND id > 4 ORDER BY L2_DISTANCE(v, '[0,0]') LIMIT 2;",
    "SELECT * FROM t WHERE id = 21;",
], [
    {},
    {'affected_rows': 10},
    {'affected_rows': 3},
    {'columns': ['id', 'd'], 'rows': [[2, 2.0], [4, 4.0], [6, 6.0]]},
    {'columns': ['id', 'd'], 'rows': [[21, 5.0], [22, 5.0], [23, 5.0]]},
    {'columns': ['id'], 'rows': [[9], [7]]},
    {'columns': ['id'], 'rows': []},
    {'columns': ['id'], 'rows': [[6], [8]]},
    {'columns': ['id', 'v', 'c'], 'rows': [[21, [5.0, 0.0], 'tie']]},
])

case('adv_update_delete_all', [
    "CREATE TABLE d (id BIGINT PRIMARY KEY, s VARCHAR(8));",
    "INSERT INTO d VALUES (1,'a'),(2,'b'),(3,'c');",
    "UPDATE d SET s = 'z';",
    "SELECT id, s FROM d ORDER BY id;",
    "DELETE FROM d;",
    "SELECT id, s FROM d;",
    "INSERT INTO d VALUES (5,'e');",
    "SELECT * FROM d;",
    "--restart",
    "SELECT * FROM d;",
], [
    {},
    {'affected_rows': 3},
    {'affected_rows': 3},
    {'columns': ['id', 's'], 'rows': [[1, 'z'], [2, 'z'], [3, 'z']]},
    {'affected_rows': 3},
    {'columns': ['id', 's'], 'rows': []},
    {'affected_rows': 1},
    {'columns': ['id', 's'], 'rows': [[5, 'e']], 'unordered': True},
    {'columns': ['id', 's'], 'rows': [[5, 'e']], 'unordered': True},
])

case('adv_bigint_extra', [
    "CREATE TABLE b (id BIGINT PRIMARY KEY, num BIGINT);",
    "INSERT INTO b VALUES (1, 9007199254740993), (2, -9223372036854775808), (3, 9223372036854775807);",
    "INSERT INTO b VALUES (9007199254740993, 0);",
    "SELECT id, num FROM b WHERE id < 9007199254740993 ORDER BY num;",
    "SELECT id FROM b WHERE num > 9007199254740992 ORDER BY id;",
    "SELECT id FROM b WHERE id >= 2 AND id < 3;",
    "SELECT num FROM b WHERE id = 9007199254740993;",
], [
    {},
    {'affected_rows': 3},
    {'affected_rows': 1},
    {'columns': ['id', 'num'], 'rows': [[2, -9223372036854775808], [1, 9007199254740993], [3, 9223372036854775807]]},
    {'columns': ['id'], 'rows': [[1], [3]]},
    {'columns': ['id'], 'rows': [[2]]},
    {'columns': ['num'], 'rows': [[0]]},
])


case('adv2_insert_columns', [
    "CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(4), v VECTOR(2));",
    "INSERT INTO t (v, id, s) VALUES ('[1,2]', 1, 'a');",
    "INSERT INTO t (id, s) VALUES (2, 'b');",
    "INSERT INTO t (id, id) VALUES (3, 4);",
    "INSERT INTO t (id, nope) VALUES (5, 6);",
    "INSERT INTO t (id, s, v) VALUES (6, 'c');",
    "INSERT INTO t (id, s, v) VALUES (7, 'd', '[0,2]');",
    "SELECT id, s, L2_DISTANCE(v, '[0,0]') AS d FROM t ORDER BY id;",
], [
    {},
    {'affected_rows': 1},
    {'error': 'not_null_violation'},
    {'error': 'duplicate_column'},
    {'error': 'column_not_found'},
    {'error': 'column_count_mismatch'},
    {'affected_rows': 1},
    {'columns': ['id', 's', 'd'], 'rows': [[1, 'a', 2.23606797749979], [7, 'd', 2.0]]},
])

case('adv2_where_ops', [
    "CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(8), n BIGINT);",
    "INSERT INTO t VALUES (1,'apple',10),(2,'banana',20),(3,'cherry',30),(4,'Apple',40),(5,'',50);",
    "SELECT id FROM t WHERE s >= 'apple' AND n <= 40 ORDER BY id;",
    "SELECT id FROM t WHERE s < 'b' ORDER BY id;",
    "SELECT id FROM t WHERE s <> 'apple' AND n >= 20 ORDER BY n DESC;",
    "SELECT id, s FROM t WHERE id = 2 OR id = 3;",
    "SELECT id FROM t WHERE n = 20 AND s = 'banana' AND id >= 2;",
], [
    {},
    {'affected_rows': 5},
    {'columns': ['id'], 'rows': [[1], [2], [3]]},
    {'columns': ['id'], 'rows': [[1], [4], [5]]},
    {'columns': ['id'], 'rows': [[5], [4], [3], [2]]},
    {'error': 'unsupported_sql'},
    {'columns': ['id'], 'rows': [[2]]},
])

case('adv2_update_multi', [
    "CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(8), n BIGINT, v VECTOR(2));",
    "INSERT INTO t VALUES (1,'a',10,'[1,1]'),(2,'b',20,'[2,2]'),(3,'c',30,'[3,3]');",
    "UPDATE t SET s='zz', n=99, v='[0,1]' WHERE n >= 20;",
    "SELECT id, s, n, L2_DISTANCE(v, '[0,0]') AS d FROM t ORDER BY id;",
    "UPDATE t SET s='q' WHERE id = 99;",
    "DELETE FROM t WHERE n > 100;",
    "SELECT id, s, n, L2_DISTANCE(v, '[0,0]') AS d FROM t ORDER BY id;",
], [
    {},
    {'affected_rows': 3},
    {'affected_rows': 2},
    {'columns': ['id', 's', 'n', 'd'], 'rows': [[1, 'a', 10, 1.4142135623730951],
                                                  [2, 'zz', 99, 1.0], [3, 'zz', 99, 1.0]]},
    {'affected_rows': 0},
    {'affected_rows': 0},
    {'columns': ['id', 's', 'n', 'd'], 'rows': [[1, 'a', 10, 1.4142135623730951],
                                                  [2, 'zz', 99, 1.0], [3, 'zz', 99, 1.0]]},
])

case('adv2_empty_and_dupproj', [
    "CREATE TABLE t (id BIGINT PRIMARY KEY, c VARCHAR(4), v VECTOR(3));",
    "SELECT id, L2_DISTANCE(v, '[0,0,0]') AS d FROM t ORDER BY d LIMIT 3;",
    "INSERT INTO t VALUES (1,'a','[1,0,0]');",
    "SELECT id, L2_DISTANCE(v, '[0,0,0]') AS d FROM t WHERE c = 'b' ORDER BY d LIMIT 3;",
    "SELECT id FROM t WHERE c = 'b';",
    "SELECT id, id FROM t;",
    "SELECT id FROM t ORDER BY id LIMIT 100;",
], [
    {},
    {'columns': ['id', 'd'], 'rows': []},
    {'affected_rows': 1},
    {'columns': ['id', 'd'], 'rows': []},
    {'columns': ['id'], 'rows': []},
    {'columns': ['id', 'id'], 'rows': [[1, 1]]},
    {'columns': ['id'], 'rows': [[1]]},
])

case('adv2_delete_reinsert_txn', [
    "CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2));",
    "INSERT INTO t VALUES (1,'[0,0]');",
    "BEGIN;",
    "DELETE FROM t WHERE id = 1;",
    "INSERT INTO t VALUES (1,'[9,9]');",
    "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM t;",
    "COMMIT;",
    "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM t;",
    "--restart",
    "SELECT id, L2_DISTANCE(v, '[0,0]') AS d FROM t;",
], [
    {},
    {'affected_rows': 1},
    {},
    {'affected_rows': 1},
    {'affected_rows': 1},
    {'columns': ['id', 'd'], 'rows': [[1, 12.727922061357855]]},
    {},
    {'columns': ['id', 'd'], 'rows': [[1, 12.727922061357855]]},
    {'columns': ['id', 'd'], 'rows': [[1, 12.727922061357855]]},
])


case('adv2_order_multi', [
    "CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(8), n BIGINT);",
    "INSERT INTO t VALUES (1,'b;c',2),(2,'b',1),(3,'a',2);",
    "SELECT id, s FROM t ORDER BY s DESC, id ASC;",
    "SELECT id FROM t ORDER BY n DESC, id ASC;",
    "SELECT id FROM t WHERE (n = 2 AND s > 'a') ORDER BY id;",
    "SELECT id FROM t WHERE n = 2 AND (s = 'a' OR s = 'z');",
    "SELECT id FROM t WHERE s = 'b;c';",
], [
    {},
    {'affected_rows': 3},
    {'columns': ['id', 's'], 'rows': [[1, 'b;c'], [2, 'b'], [3, 'a']]},
    {'columns': ['id'], 'rows': [[1], [3], [2]]},
    {'columns': ['id'], 'rows': [[1]]},
    {'error': 'unsupported_sql'},
    {'columns': ['id'], 'rows': [[1]]},
])

case('adv2_star_plus_distance', [
    "CREATE TABLE v (id BIGINT PRIMARY KEY, x VARCHAR(4), e VECTOR(2));",
    "INSERT INTO v VALUES (2,'b','[0,2]'),(1,'a','[1,0]');",
    "SELECT *, L2_DISTANCE(e, '[0,0]') AS d FROM v ORDER BY d LIMIT 1;",
    "SELECT id, e FROM v WHERE id = 2;",
], [
    {},
    {'affected_rows': 2},
    {'columns': ['id', 'x', 'e', 'd'], 'rows': [[1, 'a', [1.0, 0.0], 1.0]]},
    {'columns': ['id', 'e'], 'rows': [[2, [0.0, 2.0]]]},
])


def main():
    cases_dir = HERE / 'cases'
    cases_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'version': 'dev-adv-v1', 'scope': 'local-adversarial', 'cases': []}
    for cid, (test, result) in CASES.items():
        (cases_dir / f'{cid}.test').write_text('\n'.join(test) + '\n', encoding='utf-8')
        (cases_dir / f'{cid}.result').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        manifest['cases'].append({
            'id': cid, 'group': 'adv', 'test': f'cases/{cid}.test',
            'result': f'cases/{cid}.result', 'statements': len(test),
        })
    (HERE / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {len(CASES)} cases')


if __name__ == '__main__':
    main()
