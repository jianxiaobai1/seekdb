// MiniVectorDB - common utilities: error codes, binary buffer, checksum.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>

namespace mvdb {

// ---------------------------------------------------------------------------
// Provisional error-code mapping.
//
// The official test kit (2026-competition-round2) defines the final error
// codes and output format. Every numeric code used by the engine lives here,
// so swapping to the kit's mapping is a one-file change.
// ---------------------------------------------------------------------------
enum ErrCode : int {
  ER_SYNTAX        = 1064,
  ER_TABLE_EXISTS  = 1050,
  ER_BAD_TABLE     = 1146,
  ER_BAD_FIELD     = 1054,
  ER_DUP_ENTRY     = 1062,
  ER_DATA_TOO_LONG = 1406,
  ER_WRONG_VALUE   = 1265,
  ER_OUT_OF_RANGE  = 1264,
  ER_NO_DEFAULT    = 1364,
  ER_NOT_SUPPORTED = 1235,
  ER_NO_SUCH_INDEX = 1176,
  ER_INDEX_EXISTS  = 1061,
  ER_UNKNOWN       = 1105,
};

class DbError : public std::runtime_error {
 public:
  int code;
  DbError(int c, const std::string& msg)
      : std::runtime_error(msg), code(c) {}
};

// ---------------------------------------------------------------------------
// CRC32 (IEEE) - used to detect torn WAL tails after a hard kill.
// ---------------------------------------------------------------------------
inline uint32_t crc32_bytes(const void* data, size_t n) {
  static uint32_t table[256];
  static bool ready = false;
  if (!ready) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k)
        c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      table[i] = c;
    }
    ready = true;
  }
  const uint8_t* p = static_cast<const uint8_t*>(data);
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i)
    c = table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

// ---------------------------------------------------------------------------
// Little-endian byte writer. Target platform is Linux x86_64; explicit sizes
// are used everywhere so snapshots do not depend on struct layout.
// ---------------------------------------------------------------------------
struct Buf {
  std::string data;

  void clear() { data.clear(); }
  size_t size() const { return data.size(); }

  void u8(uint8_t v) { data.push_back(static_cast<char>(v)); }

  void u32(uint32_t v) {
    char b[4];
    std::memcpy(b, &v, 4);
    data.append(b, 4);
  }

  void u64(uint64_t v) {
    char b[8];
    std::memcpy(b, &v, 8);
    data.append(b, 8);
  }

  void i64(int64_t v) {
    uint64_t u;
    std::memcpy(&u, &v, 8);
    u64(u);
  }

  void f32(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    u32(u);
  }

  void str(const std::string& s) {
    u32(static_cast<uint32_t>(s.size()));
    data.append(s);
  }
};

struct BufReader {
  std::string_view b;
  size_t pos = 0;

  explicit BufReader(std::string_view buf) : b(buf) {}

  void need(size_t n) const {
    if (pos + n > b.size())
      throw DbError(ER_UNKNOWN, "corrupt data: unexpected end of buffer");
  }

  uint8_t u8() {
    need(1);
    return static_cast<uint8_t>(b[pos++]);
  }

  uint32_t u32() {
    need(4);
    uint32_t v;
    std::memcpy(&v, b.data() + pos, 4);
    pos += 4;
    return v;
  }

  uint64_t u64() {
    need(8);
    uint64_t v;
    std::memcpy(&v, b.data() + pos, 8);
    pos += 8;
    return v;
  }

  int64_t i64() { return static_cast<int64_t>(u64()); }

  float f32() {
    uint32_t u = u32();
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  }

  std::string str() {
    uint32_t n = u32();
    need(n);
    std::string s(b.data() + pos, n);
    pos += n;
    return s;
  }

  bool eof() const { return pos >= b.size(); }
};

// ---------------------------------------------------------------------------
// Small string helpers.
// ---------------------------------------------------------------------------
inline std::string lower_ascii(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

// Number of Unicode code points in a UTF-8 string (continuation bytes have
// the 10xxxxxx prefix and are not counted).
inline size_t utf8_codepoints(const std::string& s) {
  size_t n = 0;
  for (unsigned char c : s) {
    if ((c & 0xC0u) != 0x80u) ++n;
  }
  return n;
}

}  // namespace mvdb
