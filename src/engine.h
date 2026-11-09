// MiniVectorDB - execution engine (DDL, DML, SELECT, transactions).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sql.h"
#include "storage.h"
#include "vector_index.h"

namespace mvdb {

struct ResultSet {
  bool has_rows = false;
  std::vector<std::string> columns;
  // Each row is already tab-joined; the protocol layer only adds the newline.
  std::vector<std::string> rows;
  int64_t affected = 0;
};

class Engine {
 public:
  Engine(Storage& db, bool ann_mode);

  ResultSet execute(const Statement& st);
  void shutdown();  // roll back any open transaction, then checkpoint
  bool in_txn() const { return in_txn_; }

 private:
  struct Undo {
    std::string table;
    int64_t pk = 0;
    uint32_t slot = Table::kNoSlot;
    bool existed = false;
    Row old_row;
  };

  Storage& db_;
  bool ann_mode_;
  std::unique_ptr<VectorIndex> index_;
  bool in_txn_ = false;
  std::vector<Undo> undo_;
  std::vector<Op> redo_;
  // Reused between SELECTs to avoid reallocating a full-table candidate list
  // on every query (single client, single thread by contract).
  std::vector<uint32_t> candidates_buf_;

  Table& require_table(const std::string& name);

  ResultSet execute_ddl(const Statement& st);
  ResultSet execute_select(const Statement& st);
  ResultSet execute_insert(const Statement& st);
  ResultSet execute_update(const Statement& st);
  ResultSet execute_delete(const Statement& st);

  void apply_op(Op& op);
  void revert_to(size_t undo_mark, size_t redo_mark);
  void rollback_all();
  void commit_txn();
};

}  // namespace mvdb
