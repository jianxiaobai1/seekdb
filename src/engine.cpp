// MiniVectorDB - execution engine implementation.
#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <unordered_set>

namespace mvdb {
namespace {

struct Pred {
  int col = 0;
  ColType type = ColType::BIGINT;
  std::string op;
  int64_t ival = 0;
  std::string sval;
};

int cmp3(int64_t a, int64_t b) { return a < b ? -1 : (a > b ? 1 : 0); }
int cmp3(const std::string& a, const std::string& b) {
  int c = a.compare(b);
  return c < 0 ? -1 : (c > 0 ? 1 : 0);
}

bool cmp_ok(int c, const std::string& op) {
  if (op == "=") return c == 0;
  if (op == "!=") return c != 0;
  if (op == "<") return c < 0;
  if (op == "<=") return c <= 0;
  if (op == ">") return c > 0;
  if (op == ">=") return c >= 0;
  return false;
}

std::vector<Pred> compile_where(const Table& table,
                                const std::vector<CondAst>& conds) {
  std::vector<Pred> preds;
  for (const CondAst& c : conds) {
    int idx = table.schema.col_index(c.col);
    if (idx < 0) {
      throw DbError(ER_BAD_FIELD,
                    "unknown column '" + c.col + "' in WHERE clause");
    }
    const Column& def = table.schema.cols[idx];
    Pred p;
    p.col = idx;
    p.type = def.type;
    p.op = c.op;
    if (def.type == ColType::BIGINT) {
      if (!c.value.is_number) {
        throw DbError(ER_WRONG_VALUE,
                      "type mismatch: column '" + def.name + "' is BIGINT");
      }
      p.ival = c.value.num;
    } else if (def.type == ColType::VARCHAR) {
      if (c.value.is_number) {
        throw DbError(ER_WRONG_VALUE,
                      "type mismatch: column '" + def.name + "' is VARCHAR");
      }
      p.sval = c.value.str;
    } else {
      throw DbError(ER_NOT_SUPPORTED,
                    "comparison on VECTOR column '" + def.name +
                        "' is not supported");
    }
    preds.push_back(std::move(p));
  }
  return preds;
}

bool row_matches(const Row& row, const std::vector<Pred>& preds) {
  for (const Pred& p : preds) {
    const Value& v = row[p.col];
    int c = (p.type == ColType::BIGINT) ? cmp3(v.i, p.ival) : cmp3(v.s, p.sval);
    if (!cmp_ok(c, p.op)) return false;
  }
  return true;
}

// Slots of live rows matching the predicate. Rows spilled past the memory
// budget are decoded into scratch one at a time; an empty predicate collects
// every live slot without decoding anything.
void collect_matching(const Table& table, const std::vector<Pred>& preds,
                      std::vector<uint32_t>& out) {
  out.clear();
  // Equality on a VARCHAR column is answered from a lazily built value index
  // when the table is resident; spilled tables fall through to the scan.
  if (preds.size() == 1 && preds[0].type == ColType::VARCHAR &&
      preds[0].op == "=" &&
      table.str_candidates(preds[0].col, preds[0].sval, out)) {
    return;
  }
  out.reserve(table.live_rows);
  // Predicates that only touch the BIGINT primary key can be evaluated from
  // the contiguous pk array: no row decode, even for spilled rows.
  bool pk_only = !preds.empty() && table.schema.pk >= 0;
  for (const Pred& p : preds) {
    if (p.col != table.schema.pk || p.type != ColType::BIGINT) {
      pk_only = false;
      break;
    }
  }
  Row scratch;
  for (size_t i = 0; i < table.rows.size(); ++i) {
    if (!table.alive[i]) continue;
    uint32_t slot = static_cast<uint32_t>(i);
    if (preds.empty()) {
      out.push_back(slot);
      continue;
    }
    if (pk_only) {
      int64_t pk = table.pk_at(slot);
      bool ok = true;
      for (const Pred& p : preds) {
        if (!cmp_ok(cmp3(pk, p.ival), p.op)) {
          ok = false;
          break;
        }
      }
      if (ok) out.push_back(slot);
      continue;
    }
    const Row* row = table.resident_row(slot);
    if (row == nullptr) {
      table.load_row(slot, scratch);
      row = &scratch;
    }
    if (row_matches(*row, preds)) out.push_back(slot);
  }
}

Value coerce_literal(const Column& def, const Literal& lit) {
  switch (def.type) {
    case ColType::BIGINT:
      if (!lit.is_number) {
        throw DbError(ER_WRONG_VALUE,
                      "expected BIGINT value for column '" + def.name + "'");
      }
      return Value::of_int(lit.num);
    case ColType::VARCHAR:
      if (lit.is_number) {
        throw DbError(ER_WRONG_VALUE,
                      "expected VARCHAR value for column '" + def.name + "'");
      }
      if (utf8_codepoints(lit.str) > static_cast<size_t>(def.len)) {
        throw DbError(ER_DATA_TOO_LONG,
                      "value too long for column '" + def.name + "' (max " +
                          std::to_string(def.len) + " characters)");
      }
      return Value::of_str(lit.str);
    case ColType::VECTOR:
      if (lit.is_number) {
        throw DbError(ER_WRONG_VALUE,
                      "expected VECTOR literal for column '" + def.name + "'");
      }
      return Value::of_vec(parse_vector_literal(lit.str, def.len));
  }
  throw DbError(ER_UNKNOWN, "unknown column type");
}

Row default_row(const Schema& schema) {
  Row row;
  row.reserve(schema.cols.size());
  for (const Column& c : schema.cols) {
    if (c.type == ColType::BIGINT) {
      row.push_back(Value::of_int(0));
    } else if (c.type == ColType::VARCHAR) {
      row.push_back(Value::of_str(""));
    } else {
      row.push_back(Value::of_vec(std::vector<float>(static_cast<size_t>(c.len), 0.0f)));
    }
  }
  return row;
}

}  // namespace

Engine::Engine(Storage& db, bool ann_mode) : db_(db), ann_mode_(ann_mode) {
  // TODO(ann): when ann_mode_ is set and a table has a vector index, build and
  // attach an approximate index here. The skeleton keeps FlatIndex for both
  // modes; exact results are valid in `ann` mode as well (recall = 1.0).
  index_ = std::make_unique<FlatIndex>();
}

Table& Engine::require_table(const std::string& name) {
  Table* t = db_.get(name);
  if (!t) throw DbError(ER_BAD_TABLE, "table not found: " + name);
  return *t;
}

ResultSet Engine::execute(const Statement& st) {
  switch (st.kind) {
    case StmtKind::BEGIN:
      if (in_txn_) {
        throw DbError(ER_NOT_SUPPORTED,
                      "nested transactions are not supported");
      }
      in_txn_ = true;
      undo_.clear();
      redo_.clear();
      return {};
    case StmtKind::COMMIT:
      if (in_txn_) commit_txn();
      if (db_.wal_checkpoint_bytes() > 0 &&
          db_.wal_size() >= db_.wal_checkpoint_bytes()) {
        db_.checkpoint();
      }
      db_.enforce_budget();
      return {};
    case StmtKind::ROLLBACK:
      if (in_txn_) rollback_all();
      db_.enforce_budget();
      return {};
    case StmtKind::CREATE_TABLE:
    case StmtKind::CREATE_VECTOR_INDEX:
      return execute_ddl(st);
    default:
      break;
  }

  const bool implicit = !in_txn_;
  if (implicit) {
    in_txn_ = true;
    undo_.clear();
    redo_.clear();
  }
  const size_t undo_mark = undo_.size();
  const size_t redo_mark = redo_.size();

  try {
    ResultSet rs;
    switch (st.kind) {
      case StmtKind::INSERT:
        rs = execute_insert(st);
        break;
      case StmtKind::SELECT:
        rs = execute_select(st);
        break;
      case StmtKind::UPDATE:
        rs = execute_update(st);
        break;
      case StmtKind::DELETE:
        rs = execute_delete(st);
        break;
      default:
        throw DbError(ER_NOT_SUPPORTED, "unsupported statement");
    }
    if (implicit) {
      commit_txn();
      // Bound the log so a later crash does not have to replay a huge WAL.
      if (db_.wal_checkpoint_bytes() > 0 &&
          db_.wal_size() >= db_.wal_checkpoint_bytes()) {
        db_.checkpoint();
      }
    }
    db_.enforce_budget();
    return rs;
  } catch (...) {
    if (implicit) {
      rollback_all();
    } else {
      revert_to(undo_mark, redo_mark);
    }
    try {
      db_.enforce_budget();
    } catch (...) {
      // Keep the original statement error; spool pressure is not fatal here.
    }
    throw;
  }
}

// ---------------------------------------------------------------------------
// DDL
// ---------------------------------------------------------------------------
ResultSet Engine::execute_ddl(const Statement& st) {
  if (in_txn_) commit_txn();  // MySQL-style implicit commit before DDL

  if (st.kind == StmtKind::CREATE_TABLE) {
    Schema schema;
    schema.name = st.table;

    std::set<std::string> seen;
    for (const ColumnDefAst& def : st.coldefs) {
      std::string key = lower_ascii(def.name);
      if (!seen.insert(key).second) {
        throw DbError(ER_WRONG_VALUE,
                      "duplicate column name '" + def.name + "'");
      }
      Column c;
      c.name = def.name;
      c.type = static_cast<ColType>(def.type);
      c.len = def.len;
      schema.cols.push_back(std::move(c));
    }
    if (schema.cols.empty()) {
      throw DbError(ER_WRONG_VALUE, "table must have at least one column");
    }

    int pk = -1;
    int vec = -1;
    for (size_t i = 0; i < st.coldefs.size(); ++i) {
      if (st.coldefs[i].primary) {
        if (pk >= 0) {
          throw DbError(ER_WRONG_VALUE, "only one PRIMARY KEY is allowed");
        }
        pk = static_cast<int>(i);
      }
      if (schema.cols[i].type == ColType::VECTOR) {
        if (vec >= 0) {
          throw DbError(ER_WRONG_VALUE, "only one VECTOR column is allowed");
        }
        vec = static_cast<int>(i);
      }
    }
    if (pk < 0) {
      throw DbError(ER_NOT_SUPPORTED,
                    "a BIGINT PRIMARY KEY column is required");
    }
    if (schema.cols[pk].type != ColType::BIGINT) {
      throw DbError(ER_WRONG_VALUE, "PRIMARY KEY must be BIGINT");
    }
    schema.pk = pk;
    schema.vec = vec;

    if (db_.get(schema.name)) {
      throw DbError(ER_TABLE_EXISTS, "table already exists: " + schema.name);
    }

    Op op;
    op.type = OpType::CREATE_TABLE;
    op.table = lower_ascii(schema.name);
    op.schema = schema;
    db_.commit_ops({op});
    db_.create_table(std::move(schema));
    return {};
  }

  // CREATE VECTOR INDEX
  Table& t = require_table(st.table);
  if (t.schema.vec < 0) {
    throw DbError(ER_WRONG_VALUE,
                  "table '" + t.schema.name + "' has no VECTOR column");
  }
  int idx = t.schema.col_index(st.index_column);
  if (idx < 0) {
    throw DbError(ER_BAD_FIELD, "unknown column '" + st.index_column + "'");
  }
  if (t.schema.cols[idx].type != ColType::VECTOR) {
    throw DbError(ER_WRONG_VALUE,
                  "vector index column must be VECTOR type");
  }
  if (!t.schema.vector_index.empty()) {
    throw DbError(ER_INDEX_EXISTS,
                  "a vector index already exists on table '" + t.schema.name +
                      "'");
  }

  Op op;
  op.type = OpType::CREATE_INDEX;
  op.table = lower_ascii(t.schema.name);
  op.index = st.index_name;
  db_.commit_ops({op});
  t.schema.vector_index = st.index_name;
  return {};
}

// ---------------------------------------------------------------------------
// INSERT
// ---------------------------------------------------------------------------
ResultSet Engine::execute_insert(const Statement& st) {
  Table& table = require_table(st.table);
  const Schema& schema = table.schema;

  std::vector<int> targets;
  if (st.insert_cols.empty()) {
    targets.resize(schema.cols.size());
    for (size_t i = 0; i < schema.cols.size(); ++i) targets[i] = static_cast<int>(i);
  } else {
    std::set<int> seen;
    for (const std::string& name : st.insert_cols) {
      int idx = schema.col_index(name);
      if (idx < 0) {
        throw DbError(ER_BAD_FIELD,
                      "unknown column '" + name + "' in INSERT");
      }
      if (!seen.insert(idx).second) {
        throw DbError(ER_WRONG_VALUE,
                      "duplicate column '" + name + "' in INSERT column list");
      }
      targets.push_back(idx);
    }
    // There is no DEFAULT clause in the supported subset, so a row without
    // an explicit primary key has no meaning; refuse instead of inventing 0.
    if (schema.pk >= 0 && !seen.count(schema.pk)) {
      throw DbError(ER_NO_DEFAULT,
                    "missing value for primary key column '" +
                        schema.cols[schema.pk].name + "'");
    }
  }

  std::vector<Row> new_rows;
  new_rows.reserve(st.insert_rows.size());
  std::unordered_set<int64_t> batch_pks;
  const bool full_row = st.insert_cols.empty();

  for (const auto& tuple : st.insert_rows) {
    if (tuple.size() != targets.size()) {
      throw DbError(ER_WRONG_VALUE,
                    "column count does not match value count");
    }
    Row row;
    if (full_row) {
      // Common case: every column is supplied in schema order. Build the row
      // directly instead of allocating a zero-filled default vector first.
      row.reserve(schema.cols.size());
      for (size_t j = 0; j < tuple.size(); ++j) {
        row.push_back(coerce_literal(schema.cols[j], tuple[j]));
      }
    } else {
      row = default_row(schema);
      for (size_t j = 0; j < tuple.size(); ++j) {
        const Column& def = schema.cols[targets[j]];
        row[targets[j]] = coerce_literal(def, tuple[j]);
      }
    }
    int64_t pk = row[schema.pk].i;
    if (table.slot_of(pk) != Table::kNoSlot) {
      throw DbError(ER_DUP_ENTRY, "duplicate primary key: " + std::to_string(pk));
    }
    if (!batch_pks.insert(pk).second) {
      throw DbError(ER_DUP_ENTRY, "duplicate primary key: " + std::to_string(pk));
    }
    new_rows.push_back(std::move(row));
  }

  for (Row& row : new_rows) {
    Op op;
    op.type = OpType::INSERT;
    op.table = lower_ascii(schema.name);
    op.pk = row[schema.pk].i;
    op.row = std::move(row);
    apply_op(op);
  }

  ResultSet rs;
  rs.affected = static_cast<int64_t>(new_rows.size());
  return rs;
}

// ---------------------------------------------------------------------------
// UPDATE
// ---------------------------------------------------------------------------
ResultSet Engine::execute_update(const Statement& st) {
  Table& table = require_table(st.table);
  const Schema& schema = table.schema;

  std::vector<std::pair<int, Value>> sets;
  for (const AssignAst& a : st.assigns) {
    int idx = schema.col_index(a.col);
    if (idx < 0) {
      throw DbError(ER_BAD_FIELD, "unknown column '" + a.col + "' in SET");
    }
    if (idx == schema.pk) {
      throw DbError(ER_NOT_SUPPORTED, "primary key update is not supported");
    }
    sets.emplace_back(idx, coerce_literal(schema.cols[idx], a.value));
  }

  std::vector<Pred> preds = compile_where(table, st.conds);
  std::vector<uint32_t> targets;
  collect_matching(table, preds, targets);

  for (uint32_t slot : targets) {
    Op op;
    op.type = OpType::UPDATE;
    op.table = lower_ascii(schema.name);
    op.pk = table.pk_at(slot);
    table.load_row(slot, op.row);
    for (const auto& [col, value] : sets) op.row[col] = value;
    apply_op(op);
  }

  ResultSet rs;
  rs.affected = static_cast<int64_t>(targets.size());
  return rs;
}

// ---------------------------------------------------------------------------
// DELETE
// ---------------------------------------------------------------------------
ResultSet Engine::execute_delete(const Statement& st) {
  Table& table = require_table(st.table);
  const Schema& schema = table.schema;
  std::vector<Pred> preds = compile_where(table, st.conds);

  std::vector<uint32_t> targets;
  collect_matching(table, preds, targets);

  for (uint32_t slot : targets) {
    Op op;
    op.type = OpType::DELETE;
    op.table = lower_ascii(schema.name);
    op.pk = table.pk_at(slot);
    apply_op(op);
  }

  ResultSet rs;
  rs.affected = static_cast<int64_t>(targets.size());
  return rs;
}

// ---------------------------------------------------------------------------
// SELECT
// ---------------------------------------------------------------------------
ResultSet Engine::execute_select(const Statement& st) {
  Table& table = require_table(st.table);
  const Schema& schema = table.schema;
  std::vector<Pred> preds = compile_where(table, st.conds);

  std::vector<uint32_t>& candidates = candidates_buf_;
  candidates.clear();
  if (!preds.empty()) {
    collect_matching(table, preds, candidates);
  }

  struct OutCol {
    bool is_dist = false;
    int col = -1;
    int dist_idx = -1;
    std::string header;
  };
  struct DistSpec {
    int out_idx = -1;
    int vec_col = -1;
    std::vector<float> query;
  };

  std::vector<OutCol> outs;
  std::vector<DistSpec> dists;

  for (const SelectItemAst& item : st.select_items) {
    if (item.kind == SelectItemAst::STAR) {
      for (size_t i = 0; i < schema.cols.size(); ++i) {
        outs.push_back(OutCol{false, static_cast<int>(i), -1,
                              schema.cols[i].name});
      }
    } else if (item.kind == SelectItemAst::COLUMN) {
      int idx = schema.col_index(item.col);
      if (idx < 0) {
        throw DbError(ER_BAD_FIELD, "unknown column '" + item.col + "'");
      }
      outs.push_back(OutCol{false, idx, -1,
                            item.alias.empty() ? schema.cols[idx].name : item.alias});
    } else {
      int idx = schema.col_index(item.col);
      if (idx < 0) {
        throw DbError(ER_BAD_FIELD, "unknown column '" + item.col + "'");
      }
      if (schema.cols[idx].type != ColType::VECTOR) {
        throw DbError(ER_WRONG_VALUE,
                      "L2_DISTANCE requires a VECTOR column, got '" +
                          schema.cols[idx].name + "'");
      }
      DistSpec spec;
      spec.out_idx = static_cast<int>(outs.size());
      spec.vec_col = idx;
      spec.query = parse_vector_literal(item.vec_literal, schema.cols[idx].len);
      outs.push_back(OutCol{true, idx, static_cast<int>(dists.size()),
                            item.alias});
      dists.push_back(std::move(spec));
    }
  }

  // Resolve ORDER BY against select aliases first, then table columns.
  int order_dist = -1;  // index into `outs`
  int order_col = -1;
  bool order_expr_dist = false;  // ORDER BY L2_DISTANCE(...) without projection
  std::vector<float> order_query;
  if (st.has_order) {
    if (st.order_is_dist) {
      int idx = schema.col_index(st.order_col);
      if (idx < 0) {
        throw DbError(ER_BAD_FIELD,
                      "unknown column '" + st.order_col + "' in ORDER BY");
      }
      if (schema.cols[idx].type != ColType::VECTOR) {
        throw DbError(ER_WRONG_VALUE,
                      "L2_DISTANCE requires a VECTOR column, got '" +
                          schema.cols[idx].name + "'");
      }
      order_query = parse_vector_literal(st.order_vec_literal,
                                         schema.cols[idx].len);
      order_expr_dist = true;
    } else {
      std::string want = lower_ascii(st.order_col);
      for (size_t i = 0; i < outs.size(); ++i) {
        if (lower_ascii(outs[i].header) == want) {
          if (outs[i].is_dist) {
            order_dist = static_cast<int>(i);
          } else {
            order_col = outs[i].col;
          }
          break;
        }
      }
    }
    if (!order_expr_dist && order_dist < 0 && order_col < 0) {
      int idx = schema.col_index(st.order_col);
      if (idx < 0) {
        throw DbError(ER_BAD_FIELD,
                      "unknown column '" + st.order_col + "' in ORDER BY");
      }
      if (schema.cols[idx].type == ColType::VECTOR) {
        throw DbError(ER_NOT_SUPPORTED,
                      "ORDER BY on a VECTOR column is not supported");
      }
      order_col = idx;
    }
  }

  std::vector<uint32_t> selected_slots;

  if (st.has_order && (order_dist >= 0 || order_expr_dist) && !st.order_desc) {
    const std::vector<float>* q = &order_query;
    if (!order_expr_dist) {
      for (const DistSpec& d : dists) {
        if (d.out_idx == order_dist) q = &d.query;
      }
    }
    size_t keep = st.has_limit
                      ? static_cast<size_t>(st.limit)
                      : (preds.empty() ? table.live_rows : candidates.size());
    std::vector<Hit> hits =
        preds.empty() ? index_->topk_all(table, *q, keep)
                      : index_->topk(table, *q, keep, candidates);
    selected_slots.reserve(hits.size());
    for (const Hit& h : hits) {
      uint32_t slot = table.slot_of(h.pk);
      if (slot != Table::kNoSlot) selected_slots.push_back(slot);
    }
  } else {
    std::vector<uint32_t>& order = candidates;
    if (preds.empty()) {
      order.reserve(table.rows.size());
      for (size_t i = 0; i < table.rows.size(); ++i) {
        if (table.alive[i]) order.push_back(static_cast<uint32_t>(i));
      }
    }
    size_t keep = st.has_limit
                      ? std::min<size_t>(static_cast<size_t>(st.limit), order.size())
                      : order.size();

    if (order_dist >= 0 || order_expr_dist) {
      // ORDER BY the distance expression in a non-index order (e.g. DESC).
      int dist_col = -1;
      const std::vector<float>* dq = &order_query;
      if (order_expr_dist) {
        dist_col = schema.col_index(st.order_col);
      } else {
        dist_col = outs[static_cast<size_t>(order_dist)].col;
        for (const DistSpec& d : dists) {
          if (d.out_idx == order_dist) {
            dq = &d.query;
            break;
          }
        }
      }
      struct RankedDist {
        double d = 0.0;
        uint32_t slot = 0;
      };
      std::vector<RankedDist> ranked;
      ranked.reserve(order.size());
      Row scratch;
      for (uint32_t slot : order) {
        const Row& row = table.view(slot, scratch);
        ranked.push_back(RankedDist{
            std::sqrt(l2_distance_sq(row[dist_col].v, *dq)), slot});
      }
      auto cmp = [&](const RankedDist& a, const RankedDist& b) {
        if (a.d != b.d) return st.order_desc ? a.d > b.d : a.d < b.d;
        return table.pk_at(a.slot) < table.pk_at(b.slot);
      };
      if (keep < ranked.size()) {
        std::partial_sort(ranked.begin(), ranked.begin() + static_cast<long>(keep),
                          ranked.end(), cmp);
        ranked.resize(keep);
      } else {
        std::sort(ranked.begin(), ranked.end(), cmp);
      }
      for (const RankedDist& r : ranked) selected_slots.push_back(r.slot);
    } else if (order_col >= 0 && order_col == schema.pk) {
      // ORDER BY primary key: the contiguous pk array is enough, so spilled
      // rows never need decoding. Partial sort keeps LIMIT cheap.
      auto cmp = [&](uint32_t a, uint32_t b) {
        int64_t pa = table.pk_at(a), pb = table.pk_at(b);
        return st.order_desc ? pa > pb : pa < pb;
      };
      if (keep < order.size()) {
        std::partial_sort(order.begin(), order.begin() + static_cast<long>(keep),
                          order.end(), cmp);
        order.resize(keep);
      } else {
        std::sort(order.begin(), order.end(), cmp);
      }
      selected_slots.assign(order.begin(), order.end());
    } else if (order_col >= 0) {
      const bool is_int = schema.cols[order_col].type == ColType::BIGINT;
      struct RankedKey {
        int64_t i = 0;
        std::string s;
        uint32_t slot = 0;
      };
      std::vector<RankedKey> ranked;
      ranked.reserve(order.size());
      Row scratch;
      for (uint32_t slot : order) {
        const Row& row = table.view(slot, scratch);
        RankedKey k;
        k.slot = slot;
        if (is_int) {
          k.i = row[order_col].i;
        } else {
          k.s = row[order_col].s;
        }
        ranked.push_back(std::move(k));
      }
      auto cmp = [&](const RankedKey& a, const RankedKey& b) {
        int c = is_int ? cmp3(a.i, b.i) : cmp3(a.s, b.s);
        if (c != 0) return st.order_desc ? (c > 0) : (c < 0);
        return table.pk_at(a.slot) < table.pk_at(b.slot);
      };
      if (keep < ranked.size()) {
        std::partial_sort(ranked.begin(), ranked.begin() + static_cast<long>(keep),
                          ranked.end(), cmp);
        ranked.resize(keep);
      } else {
        std::sort(ranked.begin(), ranked.end(), cmp);
      }
      for (const RankedKey& r : ranked) selected_slots.push_back(r.slot);
    } else {
      // No ORDER BY: deterministic primary-key order (set semantics).
      auto cmp = [&](uint32_t a, uint32_t b) {
        return table.pk_at(a) < table.pk_at(b);
      };
      if (keep < order.size()) {
        std::partial_sort(order.begin(), order.begin() + static_cast<long>(keep),
                          order.end(), cmp);
        order.resize(keep);
      } else {
        std::sort(order.begin(), order.end(), cmp);
      }
      selected_slots.assign(order.begin(), order.end());
    }
  }

  ResultSet rs;
  rs.has_rows = true;
  for (const OutCol& o : outs) rs.columns.push_back(o.header);
  Row scratch;
  for (uint32_t slot : selected_slots) {
    const Row* row = table.resident_row(slot);
    if (row == nullptr) {
      table.load_row(slot, scratch);
      row = &scratch;
    }
    std::string line;
    line.reserve(64);
    for (size_t oi = 0; oi < outs.size(); ++oi) {
      if (oi) line.push_back('\t');
      const OutCol& o = outs[oi];
      if (o.is_dist) {
        const DistSpec& spec = dists[static_cast<size_t>(o.dist_idx)];
        double dist = std::sqrt(l2_distance_sq((*row)[o.col].v, spec.query));
        append_distance(line, dist);
      } else {
        append_value(line, (*row)[o.col]);
      }
    }
    rs.rows.push_back(std::move(line));
  }
  return rs;
}

// ---------------------------------------------------------------------------
// Transaction plumbing
// ---------------------------------------------------------------------------
void Engine::apply_op(Op& op) {
  Table* table = db_.get(op.table);
  if (!table) {
    throw DbError(ER_BAD_TABLE, "table not found: " + op.table);
  }
  switch (op.type) {
    case OpType::INSERT: {
      if (table->slot_of(op.pk) != Table::kNoSlot) {
        throw DbError(ER_DUP_ENTRY,
                      "duplicate primary key: " + std::to_string(op.pk));
      }
      redo_.push_back(op);  // copy goes to the WAL batch
      uint32_t slot = table->append(std::move(op.row));
      Undo u;
      u.table = op.table;
      u.pk = op.pk;
      u.slot = slot;
      u.existed = false;
      undo_.push_back(std::move(u));
      break;
    }
    case OpType::UPDATE: {
      uint32_t slot = table->slot_of(op.pk);
      if (slot == Table::kNoSlot) {
        throw DbError(ER_UNKNOWN, "update target does not exist");
      }
      Undo u;
      u.table = op.table;
      u.pk = op.pk;
      u.slot = slot;
      u.existed = true;
      table->load_row(slot, u.old_row);
      undo_.push_back(std::move(u));
      redo_.push_back(op);
      table->replace_row(slot, std::move(op.row));
      break;
    }
    case OpType::DELETE: {
      uint32_t slot = table->slot_of(op.pk);
      if (slot == Table::kNoSlot) break;
      Undo u;
      u.table = op.table;
      u.pk = op.pk;
      u.slot = slot;
      u.existed = true;
      table->load_row(slot, u.old_row);
      undo_.push_back(std::move(u));
      table->erase(op.pk);
      redo_.push_back(op);
      break;
    }
    default:
      throw DbError(ER_NOT_SUPPORTED, "unsupported operation in transaction");
  }
}

void Engine::revert_to(size_t undo_mark, size_t redo_mark) {
  for (size_t i = undo_.size(); i > undo_mark; --i) {
    const Undo& u = undo_[i - 1];
    Table* table = db_.get(u.table);
    if (!table) continue;
    if (u.existed) {
      table->revive(u.slot, u.old_row);
    } else {
      table->erase(u.pk);
    }
  }
  undo_.resize(undo_mark);
  redo_.resize(redo_mark);
}

void Engine::rollback_all() {
  revert_to(0, 0);
  in_txn_ = false;
}

void Engine::commit_txn() {
  if (!in_txn_) return;
  try {
    if (!redo_.empty()) db_.commit_ops(redo_);
  } catch (...) {
    revert_to(0, 0);
    in_txn_ = false;
    throw;
  }
  in_txn_ = false;
  undo_.clear();
  redo_.clear();
}

void Engine::shutdown() {
  if (in_txn_) rollback_all();
  db_.checkpoint();
}

}  // namespace mvdb
