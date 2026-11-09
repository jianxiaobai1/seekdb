// MiniVectorDB - value model, table storage, WAL and snapshot persistence.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"

namespace mvdb {

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------
enum class ValType : uint8_t { INT = 1, STR = 2, VEC = 3 };

struct Value {
  ValType type = ValType::INT;
  int64_t i = 0;
  std::string s;
  std::vector<float> v;

  static Value of_int(int64_t x) {
    Value val;
    val.type = ValType::INT;
    val.i = x;
    return val;
  }
  static Value of_str(std::string x) {
    Value val;
    val.type = ValType::STR;
    val.s = std::move(x);
    return val;
  }
  static Value of_vec(std::vector<float> x) {
    Value val;
    val.type = ValType::VEC;
    val.v = std::move(x);
    return val;
  }
};

using Row = std::vector<Value>;

enum class ColType : uint8_t { BIGINT = 1, VARCHAR = 2, VECTOR = 3 };

struct Column {
  std::string name;
  ColType type = ColType::BIGINT;
  int len = 0;  // VARCHAR max code points / VECTOR dimension
};

struct Schema {
  std::string name;
  std::vector<Column> cols;
  int pk = -1;
  int vec = -1;
  std::string vector_index;  // empty when no ANN index was created

  int col_index(const std::string& n) const {
    std::string key = lower_ascii(n);
    for (size_t i = 0; i < cols.size(); ++i) {
      if (lower_ascii(cols[i].name) == key) return static_cast<int>(i);
    }
    return -1;
  }
};

// ---------------------------------------------------------------------------
// Value parsing / formatting helpers
// ---------------------------------------------------------------------------
int64_t parse_int_literal(const std::string& text, const std::string& what);
std::vector<float> parse_vector_literal(const std::string& text, int dim);
// Append the rendered cell directly to the output line; avoids the temporary
// strings and per-cell allocations of a format-then-copy pipeline.
void append_value(std::string& out, const Value& v);
void append_distance(std::string& out, double d);

void encode_value(Buf& buf, const Value& v);
Value decode_value(BufReader& r);
void encode_row(Buf& buf, const Row& row);
Row decode_row(BufReader& r);
void encode_schema(Buf& buf, const Schema& s);
Schema decode_schema(BufReader& r);

// ---------------------------------------------------------------------------
// Spool: append-only, memory-mapped scratch file holding encoded rows that do
// not fit in the memory budget. It is a cache, never the source of truth:
// recovery rebuilds it from snapshots + WAL, so it needs no fsync.
// ---------------------------------------------------------------------------
class Spool {
 public:
  Spool() = default;
  ~Spool();
  Spool(const Spool&) = delete;
  Spool& operator=(const Spool&) = delete;

  void open(const std::string& path);
  uint64_t append(const char* data, size_t n);
  const char* at(uint64_t off) const { return base_ + off; }
  uint64_t size() const { return size_; }

 private:
  int fd_ = -1;
  char* base_ = nullptr;
  size_t cap_ = 0;
  size_t size_ = 0;

  void grow(size_t need);
};

// ---------------------------------------------------------------------------
// Table: rows are either decoded in memory (within the budget) or spilled to
// the spool. `rows[slot]` is empty exactly when the slot is not resident.
// ---------------------------------------------------------------------------
class Table {
 public:
  Schema schema;
  std::vector<Row> rows;
  std::vector<uint8_t> alive;
  std::unordered_map<int64_t, uint32_t> pk_index;
  size_t live_rows = 0;

  static constexpr uint32_t kNoSlot = 0xFFFFFFFFu;

  Table(Schema s, std::string spool_path);

  const Row* resident_row(uint32_t slot) const {
    return rows[slot].empty() ? nullptr : &rows[slot];
  }
  // Returns a reference to the decoded row. `scratch` receives the decoded
  // content when the row was spilled; the reference is then into `scratch`.
  const Row& view(uint32_t slot, Row& scratch) const;
  void load_row(uint32_t slot, Row& out) const;

  uint32_t slot_of(int64_t pk) const {
    auto it = pk_index.find(pk);
    return it == pk_index.end() ? kNoSlot : it->second;
  }
  int64_t pk_at(uint32_t slot) const { return pks_[slot]; }

  uint32_t append(Row row);
  void replace_row(uint32_t slot, Row row);
  void revive(uint32_t slot, Row row);
  void erase(int64_t pk);

  // Lazily built equality index for a VARCHAR column. It is valid only for
  // the current mutation generation and is not built for spilled tables, so
  // the low-memory path never grows a hidden cache. Returns false when the
  // table cannot answer from an index (caller falls back to a scan).
  bool str_candidates(int col, const std::string& value,
                      std::vector<uint32_t>& out) const;

  // Convert every resident row of this table into spooled form. From then on
  // inserts/updates are written straight to the spool (low-memory mode).
  void spill_all();
  bool spilled() const { return spill_mode_; }
  size_t resident_bytes() const {
    return resident_bytes_ + str_index_bytes_ +
           vec_scan_.capacity() * sizeof(const float*);
  }

  // Dense scan views for the distance loop: one entry per slot, valid only
  // while the table is resident. They remove the per-row Value indirection.
  bool scan_ready() const {
    return !spill_mode_ && schema.vec >= 0 && vec_scan_.size() == rows.size();
  }
  const float* const* scan_vectors() const { return vec_scan_.data(); }
  const int64_t* pk_array() const { return pks_.data(); }

  // Raw encoded row, valid for spilled slots; nullptr when resident.
  const char* spilled_blob(uint32_t slot, uint32_t* len) const;
  void encode_from(uint32_t slot, Buf& out) const;

 private:
  struct SlotRef {
    uint64_t off = 0;
    uint32_t len = 0;
  };

  struct StrIndex {
    bool valid = false;
    uint64_t gen = 0;
    size_t bytes = 0;
    std::unordered_map<std::string, std::vector<uint32_t>> slots;
  };

  void bump_generation() { ++mutation_gen_; }
  void build_str_index(int col) const;

  std::vector<SlotRef> slots_;
  std::vector<int64_t> pks_;
  std::vector<const float*> vec_scan_;
  std::string spool_path_;
  Spool spool_;
  bool spill_mode_ = false;
  size_t resident_bytes_ = 0;

  mutable std::vector<StrIndex> str_index_;
  mutable size_t str_index_bytes_ = 0;
  mutable uint64_t mutation_gen_ = 0;
};

// ---------------------------------------------------------------------------
// WAL operations
// ---------------------------------------------------------------------------
enum class OpType : uint8_t {
  CREATE_TABLE = 1,
  CREATE_INDEX = 2,
  INSERT = 3,
  UPDATE = 4,
  DELETE = 5,
};

struct Op {
  OpType type = OpType::INSERT;
  std::string table;
  Schema schema;   // CREATE_TABLE
  std::string index;  // CREATE_INDEX
  int64_t pk = 0;
  Row row;         // INSERT / UPDATE
};

void encode_op(Buf& buf, const Op& op);
Op decode_op(BufReader& r);

// One WAL record == one committed transaction (or one auto-commit statement).
// [u32 payload_len][u32 crc32][payload]
class Wal {
 public:
  Wal() = default;
  ~Wal();
  Wal(const Wal&) = delete;
  Wal& operator=(const Wal&) = delete;

  void open(const std::string& path);
  void append(const std::vector<Op>& ops);
  void replay(const std::function<void(const Op&)>& fn);
  void reset();
  size_t size() const { return bytes_; }

 private:
  int fd_ = -1;
  size_t bytes_ = 0;
};

// ---------------------------------------------------------------------------
// Storage manager: snapshots + WAL + memory budget enforcement
// ---------------------------------------------------------------------------
class Storage {
 public:
  explicit Storage(std::string dir, size_t mem_budget = 0);
  ~Storage();

  void open();
  Table* get(const std::string& name);
  Table& create_table(Schema schema);
  void apply(const Op& op);                 // used by replay
  void commit_ops(const std::vector<Op>& ops);  // WAL append + fsync
  void checkpoint();
  // Checkpoint automatically once the WAL outgrows this many bytes so that a
  // long run does not leave an unbounded log to replay after a crash.
  void set_wal_checkpoint_bytes(size_t n) { wal_checkpoint_bytes_ = n; }
  size_t wal_checkpoint_bytes() const { return wal_checkpoint_bytes_; }
  size_t wal_size() const { return wal_.size(); }

  // Spill the largest resident tables until total decoded bytes fit.
  void enforce_budget();
  size_t total_resident() const;

  std::map<std::string, std::unique_ptr<Table>>& tables() { return tables_; }
  const std::string& dir() const { return dir_; }
  size_t mem_budget() const { return mem_budget_; }

 private:
  std::string dir_;
  std::map<std::string, std::unique_ptr<Table>> tables_;
  Wal wal_;
  size_t mem_budget_ = 0;
  size_t wal_checkpoint_bytes_ = size_t(128) << 20;
  size_t appends_since_check_ = 0;

  std::string snap_path(const std::string& table_name) const;
  std::string spool_path(const std::string& table_name) const;
  void load_snapshots();
  void write_snapshot(const Table& table);
  void fsync_dir() const;
  void note_append();
};

size_t auto_mem_budget();  // from cgroup limit when available

}  // namespace mvdb
