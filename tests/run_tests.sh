#!/usr/bin/env bash
# MiniVectorDB skeleton test runner. Run from anywhere:
#   ./tests/run_tests.sh [path-to-binary]
set -euo pipefail

cd "$(dirname "$0")/.."
BIN=${1:-./build/minivectordb}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

pass=0
fail=0

check() {
  local name=$1 got=$2 want=$3
  if diff -u "$want" "$got" > "$TMP/$name.diff"; then
    echo "PASS $name"
    pass=$((pass + 1))
  else
    echo "FAIL $name"
    cat "$TMP/$name.diff"
    fail=$((fail + 1))
  fi
}

run_sql_dump() {  # <name> <sql> <data-dir>
  local name=$1 sql=$2 dir=$3
  mkdir -p "$dir"
  "$BIN" --data-dir "$dir" < "$sql" | sed '1d' > "$TMP/$name.out"
}

run_sql_dump sample tests/sample.sql "$TMP/sample"
check sample "$TMP/sample.out" tests/expected/sample.out

run_sql_dump errors_txn tests/errors_txn.sql "$TMP/errors"
check errors_txn "$TMP/errors_txn.out" tests/expected/errors_txn.out

run_sql_dump ties tests/ties.sql "$TMP/ties"
check ties "$TMP/ties.out" tests/expected/ties.out

# restart: phase 1 loads data, phase 2 reopens the same directory
mkdir -p "$TMP/restart"
"$BIN" --data-dir "$TMP/restart" < tests/restart_phase1.sql > /dev/null 2>&1
"$BIN" --data-dir "$TMP/restart" < tests/restart_phase2.sql | sed '1d' > "$TMP/restart.out"
check restart "$TMP/restart.out" tests/expected/restart.out

if python3 tests/crash_test.py "$BIN"; then
  echo "PASS crash"
  pass=$((pass + 1))
else
  echo "FAIL crash"
  fail=$((fail + 1))
fi

if python3 tests/spill_test.py "$BIN"; then
  echo "PASS spill"
  pass=$((pass + 1))
else
  echo "FAIL spill"
  fail=$((fail + 1))
fi

if python3 tests/semantics_test.py "$BIN"; then
  echo "PASS semantics"
  pass=$((pass + 1))
else
  echo "FAIL semantics"
  fail=$((fail + 1))
fi

echo "----------------------------------------"
echo "passed: $pass  failed: $fail"
[[ $fail -eq 0 ]]
