// MiniVectorDB - persistence implementation (snapshots + transactional WAL).
#include "storage.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

namespace mvdb {
namespace {

void write_all(int fd, const char* data, size_t n) {
  size_t off = 0;
  while (off < n) {
    ssize_t w = ::write(fd, data + off, n - off);
    if (w < 0) {
      if (errno == EINTR) continue;
      throw DbError(ER_UNKNOWN, std::string("write failed: ") + std::strerror(errno));
    }
    off += static_cast<size_t>(w);
  }
}

std::string trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

std::string sanitize_table_name(const std::string& name) {
  // Full hex encoding: collision-free for arbitrary table names (a simple
  // percent escape can collide with a literal "%xx" suffix).
  static const char* hex = "0123456789abcdef";
  std::string out = "tbl_";
  out.reserve(4 + name.size() * 2);
  for (unsigned char c : name) {
    out.push_back(hex[c >> 4]);
    out.push_back(hex[c & 0x0F]);
  }
  return out;
}

constexpr char kSnapMagic[8] = {'M', 'V', 'S', 'N', 'A', 'P', '0', '1'};

size_t value_mem_bytes(const Value& v) {
  return sizeof(Value) + v.s.capacity() + v.v.capacity() * sizeof(float);
}

size_t row_mem_bytes(const Row& row) {
  size_t n = sizeof(Row);
  for (const Value& v : row) n += value_mem_bytes(v);
  return n;
}

ssize_t pread_full(int fd, char* buf, size_t n, off_t off) {
  size_t done = 0;
  while (done < n) {
    ssize_t r = ::pread(fd, buf + done, n - done, off + static_cast<off_t>(done));
    if (r < 0) {
      if (errno == EINTR) continue;
      return -1;
    }
    if (r == 0) break;
    done += static_cast<size_t>(r);
  }
  return static_cast<ssize_t>(done);
}

}  // namespace

// ---------------------------------------------------------------------------
// Parsing / formatting
// ---------------------------------------------------------------------------
int64_t parse_int_literal(const std::string& text, const std::string& what) {
  std::string t = trim(text);
  if (t.empty()) throw DbError(ER_WRONG_VALUE, "empty value for " + what);
  errno = 0;
  char* end = nullptr;
  long long v = std::strtoll(t.c_str(), &end, 10);
  if (errno == ERANGE) {
    throw DbError(ER_OUT_OF_RANGE, "BIGINT value out of range for " + what);
  }
  if (end == t.c_str() || *end != '\0') {
    throw DbError(ER_WRONG_VALUE, "invalid BIGINT value for " + what + ": " + text);
  }
  return static_cast<int64_t>(v);
}

std::vector<float> parse_vector_literal(const std::string& text, int dim) {
  // Hot path: parsed once per inserted row and once per query, so avoid
  // strtof + a temporary std::string per component. std::from_chars is
  // locale-independent and much faster; non-finite values are still rejected.
  const char* p = text.data();
  const char* end = p + text.size();
  auto skip_ws = [&]() {
    while (p < end && std::isspace(static_cast<unsigned char>(*p))) ++p;
  };
  auto fail = [&](const std::string& msg) -> std::vector<float> {
    throw DbError(ER_WRONG_VALUE, msg + ": " + text);
  };

  skip_ws();
  if (p >= end || *p != '[') return fail("invalid vector literal");
  ++p;
  std::vector<float> out;
  if (dim > 0) out.reserve(static_cast<size_t>(dim));
  skip_ws();
  if (p < end && *p == ']') {
    ++p;
  } else {
    while (true) {
      skip_ws();
      if (p < end && *p == '+') ++p;  // from_chars accepts '-', not '+'
      const char* tok_start = p;
      float f = 0.0f;
      auto res = std::from_chars(p, end, f, std::chars_format::general);
      if (res.ec != std::errc() || res.ptr == p) {
        return fail("invalid vector component");
      }
      p = res.ptr;
      if (!std::isfinite(f)) {
        throw DbError(ER_WRONG_VALUE,
                      "vector component is not a finite float32: " +
                          std::string(tok_start, p));
      }
      out.push_back(f);
      skip_ws();
      if (p < end && *p == ',') {
        ++p;
        continue;
      }
      if (p < end && *p == ']') {
        ++p;
        break;
      }
      return fail("invalid vector literal");
    }
  }
  skip_ws();
  if (p != end) return fail("invalid vector literal");
  if (dim >= 0 && static_cast<int>(out.size()) != dim) {
    throw DbError(ER_WRONG_VALUE, "vector dimension mismatch: expected " +
                                      std::to_string(dim) + ", got " +
                                      std::to_string(out.size()));
  }
  return out;
}

static void append_int(std::string& out, int64_t v) {
  char buf[24];
  char* p = buf + sizeof(buf);
  uint64_t u = v < 0 ? 0 - static_cast<uint64_t>(v) : static_cast<uint64_t>(v);
  do {
    *--p = static_cast<char>('0' + static_cast<int>(u % 10));
    u /= 10;
  } while (u != 0);
  if (v < 0) *--p = '-';
  out.append(p, static_cast<size_t>(buf + sizeof(buf) - p));
}

void append_value(std::string& out, const Value& v) {
  switch (v.type) {
    case ValType::INT:
      append_int(out, v.i);
      break;
    case ValType::STR:
      out += v.s;
      break;
    case ValType::VEC: {
      out.push_back('[');
      char buf[64];
      for (size_t i = 0; i < v.v.size(); ++i) {
        if (i) out.push_back(',');
        int n = std::snprintf(buf, sizeof(buf), "%.9g",
                              static_cast<double>(v.v[i]));
        out.append(buf, static_cast<size_t>(n));
      }
      out.push_back(']');
      break;
    }
  }
}

void append_distance(std::string& out, double d) {
  char buf[64];
  int n = std::snprintf(buf, sizeof(buf), "%.6f", d);
  out.append(buf, static_cast<size_t>(n));
}

// ---------------------------------------------------------------------------
// Binary encoding
// ---------------------------------------------------------------------------
void encode_value(Buf& buf, const Value& v) {
  buf.u8(static_cast<uint8_t>(v.type));
  switch (v.type) {
    case ValType::INT:
      buf.i64(v.i);
      break;
    case ValType::STR:
      buf.str(v.s);
      break;
    case ValType::VEC:
      buf.u32(static_cast<uint32_t>(v.v.size()));
      for (float f : v.v) buf.f32(f);
      break;
  }
}

Value decode_value(BufReader& r) {
  uint8_t t = r.u8();
  switch (static_cast<ValType>(t)) {
    case ValType::INT:
      return Value::of_int(r.i64());
    case ValType::STR:
      return Value::of_str(r.str());
    case ValType::VEC: {
      uint32_t n = r.u32();
      if (n > 65536) throw DbError(ER_UNKNOWN, "corrupt vector in storage");
      std::vector<float> v(n);
      for (uint32_t i = 0; i < n; ++i) v[i] = r.f32();
      return Value::of_vec(std::move(v));
    }
  }
  throw DbError(ER_UNKNOWN, "corrupt value tag in storage");
}

void encode_row(Buf& buf, const Row& row) {
  buf.u32(static_cast<uint32_t>(row.size()));
  for (const Value& v : row) encode_value(buf, v);
}

Row decode_row(BufReader& r) {
  uint32_t n = r.u32();
  if (n > 100000) throw DbError(ER_UNKNOWN, "corrupt row in storage");
  Row row;
  row.reserve(n);
  for (uint32_t i = 0; i < n; ++i) row.push_back(decode_value(r));
  return row;
}

void encode_schema(Buf& buf, const Schema& s) {
  buf.str(s.name);
  buf.u32(static_cast<uint32_t>(s.cols.size()));
  for (const Column& c : s.cols) {
    buf.str(c.name);
    buf.u8(static_cast<uint8_t>(c.type));
    buf.u32(static_cast<uint32_t>(c.len));
  }
  buf.u32(static_cast<uint32_t>(s.pk));
  buf.u32(static_cast<uint32_t>(s.vec));
  buf.str(s.vector_index);
}

Schema decode_schema(BufReader& r) {
  Schema s;
  s.name = r.str();
  uint32_t n = r.u32();
  if (n > 4096) throw DbError(ER_UNKNOWN, "corrupt schema in storage");
  for (uint32_t i = 0; i < n; ++i) {
    Column c;
    c.name = r.str();
    c.type = static_cast<ColType>(r.u8());
    c.len = static_cast<int>(r.u32());
    s.cols.push_back(std::move(c));
  }
  s.pk = static_cast<int>(r.u32());
  s.vec = static_cast<int>(r.u32());
  s.vector_index = r.str();
  return s;
}

// ---------------------------------------------------------------------------
// Spool + Table
// ---------------------------------------------------------------------------
Spool::~Spool() {
  if (base_) ::munmap(base_, cap_);
  if (fd_ >= 0) ::close(fd_);
}

void Spool::open(const std::string& path) {
  if (base_) {
    ::munmap(base_, cap_);
    base_ = nullptr;
  }
  if (fd_ >= 0) ::close(fd_);
  fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd_ < 0) {
    throw DbError(ER_UNKNOWN,
                  "cannot open spool " + path + ": " + std::strerror(errno));
  }
  size_ = 0;
  cap_ = 1u << 20;
  if (::ftruncate(fd_, static_cast<off_t>(cap_)) != 0) {
    throw DbError(ER_UNKNOWN,
                  "cannot size spool " + path + ": " + std::strerror(errno));
  }
  void* p = ::mmap(nullptr, cap_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (p == MAP_FAILED) {
    base_ = nullptr;
    throw DbError(ER_UNKNOWN,
                  "cannot map spool " + path + ": " + std::strerror(errno));
  }
  base_ = static_cast<char*>(p);
}

void Spool::grow(size_t need) {
  size_t new_cap = cap_ ? cap_ : (1u << 20);
  while (new_cap < need) {
    if (new_cap > (size_t(1) << 40)) {
      throw DbError(ER_UNKNOWN, "spool is too large");
    }
    new_cap *= 2;
  }
  if (::ftruncate(fd_, static_cast<off_t>(new_cap)) != 0) {
    throw DbError(ER_UNKNOWN,
                  std::string("cannot grow spool: ") + std::strerror(errno));
  }
  void* p = ::mmap(nullptr, new_cap, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (p == MAP_FAILED) {
    throw DbError(ER_UNKNOWN,
                  std::string("cannot remap spool: ") + std::strerror(errno));
  }
  if (base_) ::munmap(base_, cap_);
  base_ = static_cast<char*>(p);
  cap_ = new_cap;
}

uint64_t Spool::append(const char* data, size_t n) {
  if (n == 0) return size_;
  if (size_ + n > cap_) grow(size_ + n);
  std::memcpy(base_ + size_, data, n);
  uint64_t off = size_;
  size_ += n;
  return off;
}

Table::Table(Schema s, std::string spool_path)
    : schema(std::move(s)), spool_path_(std::move(spool_path)) {
  str_index_.resize(schema.cols.size());
  spool_.open(spool_path_);
}

const Row& Table::view(uint32_t slot, Row& scratch) const {
  if (!rows[slot].empty()) return rows[slot];
  load_row(slot, scratch);
  return scratch;
}

void Table::load_row(uint32_t slot, Row& out) const {
  if (slot >= rows.size()) {
    out.clear();
    return;
  }
  if (!rows[slot].empty()) {
    out = rows[slot];
    return;
  }
  const SlotRef& ref = slots_[slot];
  if (ref.len == 0) {
    out.clear();
    return;
  }
  BufReader reader(std::string_view(spool_.at(ref.off), ref.len));
  out = decode_row(reader);
}

uint32_t Table::append(Row row) {
  bump_generation();
  uint32_t slot = static_cast<uint32_t>(rows.size());
  int64_t pk = row[schema.pk].i;
  if (spill_mode_) {
    Buf encoded;
    encode_row(encoded, row);
    uint64_t off = spool_.append(encoded.data.data(), encoded.data.size());
    pks_.push_back(pk);
    slots_.push_back(SlotRef{off, static_cast<uint32_t>(encoded.data.size())});
    alive.push_back(1);
    ++live_rows;
    pk_index[pk] = slot;
    rows.emplace_back();
  } else {
    resident_bytes_ += row_mem_bytes(row);
    pks_.push_back(pk);
    slots_.push_back(SlotRef{});
    alive.push_back(1);
    ++live_rows;
    pk_index[pk] = slot;
    rows.push_back(std::move(row));
    vec_scan_.push_back(schema.vec >= 0 ? rows.back()[schema.vec].v.data()
                                        : nullptr);
  }
  return slot;
}

void Table::replace_row(uint32_t slot, Row row) {
  bump_generation();
  int64_t pk = row[schema.pk].i;
  if (spill_mode_) {
    if (!rows[slot].empty()) {
      resident_bytes_ -= row_mem_bytes(rows[slot]);
      std::vector<Value>().swap(rows[slot]);
    }
    Buf encoded;
    encode_row(encoded, row);
    uint64_t off = spool_.append(encoded.data.data(), encoded.data.size());
    slots_[slot] = SlotRef{off, static_cast<uint32_t>(encoded.data.size())};
  } else {
    size_t before = rows[slot].empty() ? 0 : row_mem_bytes(rows[slot]);
    rows[slot] = std::move(row);
    resident_bytes_ -= before;
    resident_bytes_ += row_mem_bytes(rows[slot]);
    if (slot >= vec_scan_.size()) vec_scan_.resize(slot + 1, nullptr);
    vec_scan_[slot] = schema.vec >= 0 ? rows[slot][schema.vec].v.data()
                                      : nullptr;
  }
  pks_[slot] = pk;
}

void Table::revive(uint32_t slot, Row row) {
  int64_t pk = row[schema.pk].i;
  if (!alive[slot]) {
    alive[slot] = 1;
    ++live_rows;
  }
  pks_[slot] = pk;
  pk_index[pk] = slot;
  replace_row(slot, std::move(row));
}

void Table::erase(int64_t pk) {
  auto it = pk_index.find(pk);
  if (it == pk_index.end()) return;
  bump_generation();
  uint32_t slot = it->second;
  alive[slot] = 0;
  --live_rows;
  pk_index.erase(it);
  if (!rows[slot].empty()) {
    resident_bytes_ -= row_mem_bytes(rows[slot]);
    std::vector<Value>().swap(rows[slot]);
  }
}

void Table::spill_all() {
  if (spill_mode_) return;
  Buf encoded;
  for (size_t i = 0; i < rows.size(); ++i) {
    if (rows[i].empty()) continue;
    if (alive[i]) {
      encoded.data.clear();
      encode_row(encoded, rows[i]);
      uint64_t off = spool_.append(encoded.data.data(), encoded.data.size());
      slots_[i] = SlotRef{off, static_cast<uint32_t>(encoded.data.size())};
    }
    std::vector<Value>().swap(rows[i]);
  }
  // The index is a convenience cache; dropping it keeps the memory budget
  // honest. Spilled tables answer filters by streaming the spool instead.
  str_index_.clear();
  str_index_bytes_ = 0;
  std::vector<const float*>().swap(vec_scan_);
  bump_generation();
  resident_bytes_ = 0;
  spill_mode_ = true;
}

bool Table::str_candidates(int col, const std::string& value,
                           std::vector<uint32_t>& out) const {
  out.clear();
  if (spill_mode_) return false;
  if (col < 0 || col >= static_cast<int>(schema.cols.size())) return false;
  if (schema.cols[col].type != ColType::VARCHAR) return false;
  if (str_index_.size() < schema.cols.size()) {
    str_index_.resize(schema.cols.size());
  }
  if (!str_index_[col].valid || str_index_[col].gen != mutation_gen_) {
    build_str_index(col);
  }
  auto it = str_index_[col].slots.find(value);
  if (it == str_index_[col].slots.end()) return true;
  out.reserve(it->second.size());
  for (uint32_t slot : it->second) {
    if (slot < alive.size() && alive[slot] && !rows[slot].empty()) {
      out.push_back(slot);
    }
  }
  return true;
}

void Table::build_str_index(int col) const {
  StrIndex& idx = str_index_[col];
  idx.slots.clear();
  size_t bytes = sizeof(StrIndex);
  for (size_t i = 0; i < rows.size(); ++i) {
    if (!alive[i] || rows[i].empty()) continue;
    idx.slots[rows[i][col].s].push_back(static_cast<uint32_t>(i));
  }
  for (const auto& [key, list] : idx.slots) {
    bytes += sizeof(std::string) + key.capacity() +
             list.capacity() * sizeof(uint32_t) + 32;
  }
  bytes += idx.slots.bucket_count() * sizeof(void*);
  str_index_bytes_ -= idx.bytes;
  idx.bytes = bytes;
  str_index_bytes_ += bytes;
  idx.valid = true;
  idx.gen = mutation_gen_;
}

const char* Table::spilled_blob(uint32_t slot, uint32_t* len) const {
  if (slot >= rows.size() || !rows[slot].empty()) {
    if (len) *len = 0;
    return nullptr;
  }
  const SlotRef& ref = slots_[slot];
  if (ref.len == 0) {
    if (len) *len = 0;
    return nullptr;
  }
  if (len) *len = ref.len;
  return spool_.at(ref.off);
}

void Table::encode_from(uint32_t slot, Buf& out) const {
  out.data.clear();
  uint32_t len = 0;
  const char* blob = spilled_blob(slot, &len);
  if (blob) {
    out.data.append(blob, len);
  } else {
    encode_row(out, rows[slot]);
  }
}

// ---------------------------------------------------------------------------
// WAL
// ---------------------------------------------------------------------------
void encode_op(Buf& buf, const Op& op) {
  buf.u8(static_cast<uint8_t>(op.type));
  buf.str(op.table);
  switch (op.type) {
    case OpType::CREATE_TABLE:
      encode_schema(buf, op.schema);
      break;
    case OpType::CREATE_INDEX:
      buf.str(op.index);
      break;
    case OpType::INSERT:
    case OpType::UPDATE:
      buf.i64(op.pk);
      encode_row(buf, op.row);
      break;
    case OpType::DELETE:
      buf.i64(op.pk);
      break;
  }
}

Op decode_op(BufReader& r) {
  Op op;
  op.type = static_cast<OpType>(r.u8());
  op.table = r.str();
  switch (op.type) {
    case OpType::CREATE_TABLE:
      op.schema = decode_schema(r);
      break;
    case OpType::CREATE_INDEX:
      op.index = r.str();
      break;
    case OpType::INSERT:
    case OpType::UPDATE:
      op.pk = r.i64();
      op.row = decode_row(r);
      break;
    case OpType::DELETE:
      op.pk = r.i64();
      break;
    default:
      throw DbError(ER_UNKNOWN, "corrupt WAL op tag");
  }
  return op;
}

Wal::~Wal() {
  if (fd_ >= 0) ::close(fd_);
}

void Wal::open(const std::string& path) {
  if (fd_ >= 0) ::close(fd_);
  fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd_ < 0) {
    throw DbError(ER_UNKNOWN, "cannot open WAL " + path + ": " + std::strerror(errno));
  }
  struct stat st {};
  bytes_ = ::fstat(fd_, &st) == 0 ? static_cast<size_t>(st.st_size) : 0;
}

void Wal::append(const std::vector<Op>& ops) {
  if (ops.empty() || fd_ < 0) return;

  Buf payload;
  payload.u32(static_cast<uint32_t>(ops.size()));
  for (const Op& op : ops) encode_op(payload, op);

  // Write the 8-byte header and the payload separately so a large bulk
  // INSERT never needs a second full-size copy of its WAL record.
  char header[8];
  uint32_t payload_len = static_cast<uint32_t>(payload.size());
  uint32_t crc = crc32_bytes(payload.data.data(), payload.size());
  std::memcpy(header, &payload_len, 4);
  std::memcpy(header + 4, &crc, 4);

  off_t start_off = ::lseek(fd_, 0, SEEK_END);
  if (start_off < 0) {
    throw DbError(ER_UNKNOWN, std::string("WAL seek failed: ") + std::strerror(errno));
  }
  try {
    write_all(fd_, header, sizeof(header));
    write_all(fd_, payload.data.data(), payload.size());
    if (::fdatasync(fd_) != 0) {
      throw DbError(ER_UNKNOWN,
                    std::string("WAL sync failed: ") + std::strerror(errno));
    }
    bytes_ += sizeof(header) + payload.size();
  } catch (...) {
    // Do not leave a torn record in the middle of the log: later commits
    // would be unreadable because replay stops at the first bad record.
    if (start_off >= 0) {
      // Best effort: if truncation itself fails there is nothing else to do.
      int trunc_rc = ::ftruncate(fd_, start_off);
      (void)trunc_rc;
      ::lseek(fd_, 0, SEEK_END);
    }
    throw;
  }
}

void Wal::replay(const std::function<void(const Op&)>& fn) {
  if (fd_ < 0) return;
  struct stat st {};
  if (::fstat(fd_, &st) != 0) {
    throw DbError(ER_UNKNOWN, std::string("WAL stat failed: ") + std::strerror(errno));
  }
  if (st.st_size == 0) return;

  // Stream record by record so replay never needs the whole log in memory.
  // Records are appended atomically per transaction. A torn tail (kill during
  // append) fails the length/CRC check and is ignored: that transaction was
  // never acknowledged, so dropping it is allowed.
  std::vector<char> payload;
  off_t pos = 0;
  while (pos + 8 <= st.st_size) {
    char header[8];
    if (pread_full(fd_, header, sizeof(header), pos) != 8) break;
    uint32_t len = 0, crc = 0;
    std::memcpy(&len, header, 4);
    std::memcpy(&crc, header + 4, 4);
    if (len > (1u << 30)) break;  // implausible header: treat as torn tail
    if (static_cast<off_t>(len) > st.st_size - pos - 8) break;
    payload.resize(len);
    if (pread_full(fd_, payload.data(), len, pos + 8) !=
        static_cast<ssize_t>(len)) {
      break;
    }
    if (crc32_bytes(payload.data(), len) != crc) break;
    BufReader reader(std::string_view(payload.data(), len));
    uint32_t count = reader.u32();
    for (uint32_t i = 0; i < count; ++i) fn(decode_op(reader));
    pos += 8 + static_cast<off_t>(len);
  }
  // Cut off a torn tail now. Otherwise the next commits would be appended
  // after the garbage and a later recovery would stop at the bad record,
  // silently losing transactions that were acknowledged after this restart.
  if (pos < st.st_size) {
    if (::ftruncate(fd_, pos) != 0) {
      throw DbError(ER_UNKNOWN,
                    std::string("WAL truncate failed: ") + std::strerror(errno));
    }
    if (::fsync(fd_) != 0) {
      throw DbError(ER_UNKNOWN,
                    std::string("WAL sync failed: ") + std::strerror(errno));
    }
  }
  bytes_ = static_cast<size_t>(pos);
}

void Wal::reset() {
  if (fd_ < 0) return;
  if (::ftruncate(fd_, 0) != 0) {
    throw DbError(ER_UNKNOWN, std::string("WAL truncate failed: ") + std::strerror(errno));
  }
  if (::lseek(fd_, 0, SEEK_SET) < 0) {
    throw DbError(ER_UNKNOWN, std::string("WAL seek failed: ") + std::strerror(errno));
  }
  if (::fsync(fd_) != 0) {
    throw DbError(ER_UNKNOWN, std::string("WAL sync failed: ") + std::strerror(errno));
  }
  bytes_ = 0;
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------
size_t auto_mem_budget() {
  auto read_limit = [](const char* path) -> size_t {
    std::ifstream in(path);
    if (!in) return 0;
    std::string s;
    in >> s;
    if (s.empty() || s == "max") return 0;
    try {
      return static_cast<size_t>(std::stoull(s));
    } catch (...) {
      return 0;
    }
  };
  size_t limit = read_limit("/sys/fs/cgroup/memory.max");
  if (limit == 0) {
    limit = read_limit("/sys/fs/cgroup/memory/memory.limit_in_bytes");
  }
  if (limit > (size_t(1) << 50)) limit = 0;  // v1 "unlimited" sentinel
  size_t budget = limit ? limit * 2 / 5 : (size_t(512) << 20);
  if (budget < (size_t(64) << 20)) budget = size_t(64) << 20;
  if (budget > (size_t(1) << 30)) budget = size_t(1) << 30;
  return budget;
}

Storage::Storage(std::string dir, size_t mem_budget)
    : dir_(std::move(dir)),
      mem_budget_(mem_budget ? mem_budget : auto_mem_budget()) {}

Storage::~Storage() = default;

void Storage::open() {
  std::filesystem::create_directories(dir_);
  load_snapshots();
  wal_.open(dir_ + "/wal.log");
  wal_.replay([this](const Op& op) { apply(op); });
  enforce_budget();
  fsync_dir();  // make the WAL directory entry durable
}

Table* Storage::get(const std::string& name) {
  auto it = tables_.find(lower_ascii(name));
  return it == tables_.end() ? nullptr : it->second.get();
}

Table& Storage::create_table(Schema schema) {
  std::string key = lower_ascii(schema.name);
  if (tables_.count(key)) {
    throw DbError(ER_TABLE_EXISTS, "table already exists: " + schema.name);
  }
  std::string name = schema.name;
  auto table = std::make_unique<Table>(std::move(schema), spool_path(name));
  Table& ref = *table;
  tables_[key] = std::move(table);
  return ref;
}

void Storage::apply(const Op& op) {
  switch (op.type) {
    case OpType::CREATE_TABLE: {
      if (!get(op.schema.name)) {
        create_table(op.schema);
      }
      break;
    }
    case OpType::CREATE_INDEX: {
      Table* t = get(op.table);
      if (t) t->schema.vector_index = op.index;
      break;
    }
    case OpType::INSERT:
    case OpType::UPDATE: {
      Table* t = get(op.table);
      if (!t) break;
      uint32_t slot = t->slot_of(op.pk);
      if (slot != Table::kNoSlot) {
        t->replace_row(slot, op.row);
      } else {
        t->append(op.row);
      }
      note_append();
      break;
    }
    case OpType::DELETE: {
      Table* t = get(op.table);
      if (t) t->erase(op.pk);
      break;
    }
  }
}

void Storage::commit_ops(const std::vector<Op>& ops) {
  if (ops.empty()) return;
  wal_.append(ops);
}

size_t Storage::total_resident() const {
  size_t total = 0;
  for (const auto& [key, table] : tables_) {
    (void)key;
    total += table->resident_bytes();
  }
  return total;
}

void Storage::enforce_budget() {
  while (total_resident() > mem_budget_) {
    Table* largest = nullptr;
    size_t best = 0;
    for (auto& [key, table] : tables_) {
      (void)key;
      if (table->resident_bytes() > best) {
        best = table->resident_bytes();
        largest = table.get();
      }
    }
    if (largest == nullptr || best == 0) break;
    largest->spill_all();
  }
}

void Storage::note_append() {
  if (++appends_since_check_ >= 4096) {
    appends_since_check_ = 0;
    enforce_budget();
  }
}

std::string Storage::snap_path(const std::string& table_name) const {
  return dir_ + "/" + sanitize_table_name(table_name) + ".snap";
}

std::string Storage::spool_path(const std::string& table_name) const {
  return dir_ + "/" + sanitize_table_name(table_name) + ".spool";
}

void Storage::write_snapshot(const Table& table) {
  std::string path = snap_path(table.schema.name);
  std::string tmp = path + ".tmp";
  int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) {
    throw DbError(ER_UNKNOWN,
                  "cannot create snapshot " + tmp + ": " + std::strerror(errno));
  }
  try {
    Buf header;
    header.data.append(kSnapMagic, sizeof(kSnapMagic));
    encode_schema(header, table.schema);
    header.u64(static_cast<uint64_t>(table.live_rows));
    write_all(fd, header.data.data(), header.data.size());

    // Stream row by row: snapshots can be much larger than the memory budget,
    // and spilled rows are written straight from their encoded blob.
    Buf encoded;
    encoded.data.reserve(4096);
    for (size_t i = 0; i < table.rows.size(); ++i) {
      if (!table.alive[i]) continue;
      if (!table.rows[i].empty()) {
        encoded.data.clear();
        encode_row(encoded, table.rows[i]);
        write_all(fd, encoded.data.data(), encoded.data.size());
      } else {
        uint32_t len = 0;
        const char* blob = table.spilled_blob(static_cast<uint32_t>(i), &len);
        if (blob == nullptr) {
          throw DbError(ER_UNKNOWN, "live row has no resident data or spool");
        }
        write_all(fd, blob, len);
      }
    }
    if (::fsync(fd) != 0) {
      throw DbError(ER_UNKNOWN,
                    std::string("snapshot sync failed: ") + std::strerror(errno));
    }
  } catch (...) {
    ::close(fd);
    throw;
  }
  ::close(fd);
  if (::rename(tmp.c_str(), path.c_str()) != 0) {
    throw DbError(ER_UNKNOWN,
                  std::string("snapshot rename failed: ") + std::strerror(errno));
  }
}

void Storage::fsync_dir() const {
  int fd = ::open(dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) return;
  ::fsync(fd);
  ::close(fd);
}

void Storage::load_snapshots() {
  for (const auto& entry : std::filesystem::directory_iterator(dir_)) {
    if (!entry.is_regular_file()) continue;
    if (entry.path().extension() != ".snap") continue;

    std::string path = entry.path().string();
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
      throw DbError(ER_UNKNOWN,
                    "cannot open snapshot " + path + ": " + std::strerror(errno));
    }
    struct stat st {};
    if (::fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(kSnapMagic))) {
      ::close(fd);
      throw DbError(ER_UNKNOWN, "corrupt snapshot: " + path);
    }
    size_t size = static_cast<size_t>(st.st_size);
    void* base = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) {
      throw DbError(ER_UNKNOWN,
                    "cannot map snapshot " + path + ": " + std::strerror(errno));
    }
    try {
      const char* bytes = static_cast<const char*>(base);
      if (std::memcmp(bytes, kSnapMagic, sizeof(kSnapMagic)) != 0) {
        throw DbError(ER_UNKNOWN, "corrupt snapshot: " + path);
      }
      BufReader reader(std::string_view(bytes + sizeof(kSnapMagic),
                                        size - sizeof(kSnapMagic)));
      Schema schema = decode_schema(reader);
      uint64_t count = reader.u64();
      if (get(schema.name)) {
        ::munmap(base, size);
        continue;  // snapshots are one per table
      }
      Table& table = create_table(std::move(schema));
      for (uint64_t i = 0; i < count; ++i) {
        Row row = decode_row(reader);
        if (table.schema.pk < 0 ||
            static_cast<size_t>(table.schema.pk) >= row.size()) {
          throw DbError(ER_UNKNOWN, "corrupt snapshot row");
        }
        table.append(std::move(row));
        note_append();
      }
    } catch (...) {
      ::munmap(base, size);
      throw;
    }
    ::munmap(base, size);
  }
}

void Storage::checkpoint() {
  for (auto& [key, table] : tables_) {
    (void)key;
    write_snapshot(*table);
  }
  fsync_dir();
  wal_.reset();
}

}  // namespace mvdb
