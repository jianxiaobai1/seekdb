// MiniVectorDB - storage engine, transactions, recovery and query execution.
#pragma once

#include "sql.hpp"

#include <algorithm>
#include <cerrno>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <new>
#include <optional>
#include <queue>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mvd {

// ---------------------------------------------------------------------------
// Small binary helpers (little endian, x86_64 target)
// ---------------------------------------------------------------------------

inline void put_u8(std::string& out, uint8_t v) { out.push_back(static_cast<char>(v)); }

inline void put_u32(std::string& out, uint32_t v) {
    out.push_back(static_cast<char>(v & 0xff));
    out.push_back(static_cast<char>((v >> 8) & 0xff));
    out.push_back(static_cast<char>((v >> 16) & 0xff));
    out.push_back(static_cast<char>((v >> 24) & 0xff));
}

inline void put_i64(std::string& out, int64_t v) {
    uint64_t u = static_cast<uint64_t>(v);
    put_u32(out, static_cast<uint32_t>(u & 0xffffffffu));
    put_u32(out, static_cast<uint32_t>(u >> 32));
}

inline void put_f32(std::string& out, float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    put_u32(out, bits);
}

inline void put_str(std::string& out, const std::string& s) {
    put_u32(out, static_cast<uint32_t>(s.size()));
    out.append(s);
}

struct Reader {
    const uint8_t* p = nullptr;
    size_t n = 0;
    size_t i = 0;
    bool ok = true;

    Reader() = default;
    Reader(const void* data, size_t size)
        : p(static_cast<const uint8_t*>(data)), n(size) {}

    uint8_t u8() {
        if (i + 1 > n) { ok = false; return 0; }
        return p[i++];
    }
    uint32_t u32() {
        if (i + 4 > n) { ok = false; return 0; }
        uint32_t v = static_cast<uint32_t>(p[i]) |
                     (static_cast<uint32_t>(p[i + 1]) << 8) |
                     (static_cast<uint32_t>(p[i + 2]) << 16) |
                     (static_cast<uint32_t>(p[i + 3]) << 24);
        i += 4;
        return v;
    }
    int64_t i64() {
        uint64_t lo = u32();
        uint64_t hi = u32();
        if (!ok) return 0;
        return static_cast<int64_t>(lo | (hi << 32));
    }
    float f32() {
        uint32_t bits = u32();
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    }
    std::string str() {
        uint32_t len = u32();
        if (!ok || len > n - i) { ok = false; return std::string(); }
        std::string s(reinterpret_cast<const char*>(p + i), len);
        i += len;
        return s;
    }
    void skip(size_t len) {
        if (len > n - i) { ok = false; return; }
        i += len;
    }
};

inline uint32_t crc32_ieee(const void* data, size_t len) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        init = true;
    }
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) crc = table[(crc ^ p[i]) & 0xff] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

enum ColType { COL_BIGINT = 0, COL_VARCHAR = 1, COL_VECTOR = 2 };

struct Schema {
    std::string name;
    std::vector<std::string> col_names;
    std::vector<int> col_types;
    std::vector<int> col_lens;  // VARCHAR max code points, VECTOR dimension
    int pk = -1;
    int vec = -1;
};

struct Value {
    int64_t i = 0;
    std::string s;
    std::vector<float> v;
};

using Row = std::vector<Value>;

struct Loc {
    uint64_t off = 0;
    uint32_t len = 0;
};

struct Table {
    uint32_t id = 0;
    Schema schema;
    std::unordered_map<int64_t, Loc> locs;   // committed live rows
    std::unordered_map<int64_t, Row> cache;  // rows materialized while small enough
    bool cached = true;
    uint64_t live_bytes = 0;
    std::vector<std::string> index_names;

    int col_index(const std::string& name) const {
        for (size_t i = 0; i < schema.col_names.size(); ++i) {
            if (ieq(schema.col_names[i], name.c_str())) return static_cast<int>(i);
        }
        return -1;
    }
};

struct Cell {
    enum Kind { INT, STR, VEC, DBL } kind = INT;
    int64_t i = 0;
    std::string s;
    std::vector<float> v;
    double d = 0.0;
};

struct Result {
    bool ok = true;
    bool has_affected = false;
    int64_t affected = 0;
    bool has_rows = false;
    std::vector<std::string> columns;
    std::vector<std::vector<Cell>> rows;
    std::string code;
    std::string message;

    static Result fail(std::string c, std::string m) {
        Result r;
        r.ok = false;
        r.code = std::move(c);
        r.message = std::move(m);
        return r;
    }
    static Result with_affected(int64_t n) {
        Result r;
        r.has_affected = true;
        r.affected = n;
        return r;
    }
    static Result empty_ok() { return Result(); }
};

inline bool valid_utf8(const std::string& s, size_t& codepoints) {
    size_t i = 0, n = s.size();
    codepoints = 0;
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        int len;
        if (c < 0x80) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        else return false;
        if (i + static_cast<size_t>(len) > n) return false;
        for (int k = 1; k < len; ++k) {
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
        }
        // Reject overlong encodings and surrogate / out-of-range code points.
        uint32_t cp = 0;
        if (len == 1) cp = c;
        else if (len == 2) cp = ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu);
        else if (len == 3) cp = ((c & 0x0Fu) << 12) |
                               ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6) |
                               (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
        else cp = ((c & 0x07u) << 18) |
                  ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 12) |
                  ((static_cast<unsigned char>(s[i + 2]) & 0x3Fu) << 6) |
                  (static_cast<unsigned char>(s[i + 3]) & 0x3Fu);
        if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) ||
            (len == 4 && cp < 0x10000) || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
            return false;
        }
        i += static_cast<size_t>(len);
        ++codepoints;
    }
    return true;
}

inline int64_t parse_int64_strict(const std::string& t) {
    errno = 0;
    char* end = nullptr;
    long long v = std::strtoll(t.c_str(), &end, 10);
    if (errno == ERANGE || end == t.c_str() || *end != '\0') {
        throw SqlError("value_out_of_range", "integer out of range: " + t);
    }
    return static_cast<int64_t>(v);
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

class Engine {
public:
    explicit Engine(std::string dir)
        : dir_(std::move(dir)), cache_limit_(detect_cache_limit()) {}

    ~Engine() { close_all(); }

    void recover() {
        filesystem_create_directories();
        std::fprintf(stderr, "minivectordb: row cache limit %llu MiB\n",
                     static_cast<unsigned long long>(cache_limit_ / (1024 * 1024)));
        const std::string path = dir_ + "/minivectordb.wal";
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd_ < 0) throw std::runtime_error("cannot open WAL: " + path);
        struct stat st {};
        if (::fstat(fd_, &st) != 0) throw std::runtime_error("cannot stat WAL");
        if (st.st_size < 8) {
            // Empty, or a header torn by a kill during the very first start.
            if (st.st_size > 0 && ::ftruncate(fd_, 0) != 0) {
                throw std::runtime_error("cannot reset short WAL");
            }
            const char magic[8] = {'M', 'V', 'D', 'W', 'A', 'L', '1', '\n'};
            pwrite_all(fd_, magic, sizeof(magic), 0);
            if (::fsync(fd_) != 0) throw std::runtime_error("cannot sync new WAL");
            sync_directory();
            file_size_ = 8;
        } else {
            char magic[8];
            if (::pread(fd_, magic, 8, 0) != 8) throw std::runtime_error("cannot read WAL header");
            const char expected[8] = {'M', 'V', 'D', 'W', 'A', 'L', '1', '\n'};
            if (std::memcmp(magic, expected, 8) != 0) throw std::runtime_error("bad WAL magic");
            file_size_ = static_cast<uint64_t>(st.st_size);
        }
        replay();
    }

    Result execute(const std::string& sql, const std::string& search_mode) {
        (void)search_mode;  // exact scan satisfies both modes, including ANN recall.
        try {
            Parser parser(sql);
            Stmt st;
            bool any = false;
            Result last;
            while (parser.next(st)) {
                any = true;
                last = exec_stmt(st);
                if (!last.ok) return last;
            }
            return any ? last : Result::empty_ok();
        } catch (const SqlError& e) {
            return Result::fail(e.code, e.what());
        } catch (const std::bad_alloc&) {
            return Result::fail("out_of_memory", "allocation failed");
        } catch (const std::exception& e) {
            return Result::fail("internal_error", e.what());
        }
    }

    void close_all() {
        if (closed_) return;
        closed_ = true;
        txn_.clear();
        txn_active_ = false;
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

private:
    std::string dir_;
    int fd_ = -1;
    uint64_t file_size_ = 0;
    bool closed_ = false;
    bool recovered_ = false;

    std::vector<Table> tables_;
    std::unordered_map<uint32_t, size_t> by_id_;
    uint32_t next_table_id_ = 1;

    bool txn_active_ = false;
    std::unordered_map<uint32_t, std::unordered_map<int64_t, std::optional<Row>>> txn_;

    static constexpr uint64_t DEFAULT_CACHE_LIMIT = 32ull * 1024 * 1024;
    static constexpr uint64_t MIN_CACHE_LIMIT = 4ull * 1024 * 1024;
    uint64_t cache_limit_ = DEFAULT_CACHE_LIMIT;
    static constexpr uint32_t MAX_RECORD = 512u * 1024 * 1024;

    // Keep the decoded-row cache well below the container memory limit so a
    // large table cannot push the process into the cgroup OOM killer. With
    // no limit, or a large one, the 32 MiB default is kept.
    static uint64_t detect_cache_limit() {
        const char* paths[] = {"/sys/fs/cgroup/memory.max",
                               "/sys/fs/cgroup/memory/memory.limit_in_bytes"};
        for (const char* path : paths) {
            std::FILE* f = std::fopen(path, "rb");
            if (!f) continue;
            char buf[64] = {0};
            char* got = std::fgets(buf, sizeof(buf), f);
            std::fclose(f);
            if (!got || std::strncmp(buf, "max", 3) == 0) continue;
            errno = 0;
            char* end = nullptr;
            unsigned long long v = std::strtoull(buf, &end, 10);
            if (errno != 0 || end == buf || v == 0) continue;
            uint64_t limit = v / 8;
            if (limit < MIN_CACHE_LIMIT) limit = MIN_CACHE_LIMIT;
            if (limit > DEFAULT_CACHE_LIMIT) limit = DEFAULT_CACHE_LIMIT;
            return limit;
        }
        return DEFAULT_CACHE_LIMIT;
    }

    // -- filesystem --------------------------------------------------------

    void filesystem_create_directories() {
        // Create the data directory if it does not exist (no external deps).
        std::string cur;
        size_t i = 0;
        if (!dir_.empty() && dir_[0] == '/') { cur = "/"; i = 1; }
        while (i <= dir_.size()) {
            size_t j = dir_.find('/', i);
            if (j == std::string::npos) j = dir_.size();
            std::string part = dir_.substr(i, j - i);
            if (!part.empty()) {
                if (!cur.empty() && cur.back() != '/') cur.push_back('/');
                cur += part;
                if (::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
                    throw std::runtime_error("cannot create data directory: " + cur);
                }
            }
            if (j == dir_.size()) break;
            i = j + 1;
        }
    }

    void sync_directory() {
        int dfd = ::open(dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dfd >= 0) {
            ::fsync(dfd);
            ::close(dfd);
        }
    }

    static void pwrite_all(int fd, const void* buf, size_t n, uint64_t off) {
        const char* p = static_cast<const char*>(buf);
        while (n > 0) {
            ssize_t w = ::pwrite(fd, p, n, static_cast<off_t>(off));
            if (w < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("WAL write failed");
            }
            if (w == 0) throw std::runtime_error("WAL write returned zero");
            p += w;
            off += static_cast<uint64_t>(w);
            n -= static_cast<size_t>(w);
        }
    }

    static void pread_all(int fd, void* buf, size_t n, uint64_t off) {
        char* p = static_cast<char*>(buf);
        while (n > 0) {
            ssize_t r = ::pread(fd, p, n, static_cast<off_t>(off));
            if (r < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("WAL read failed");
            }
            if (r == 0) throw std::runtime_error("unexpected end of WAL");
            p += r;
            off += static_cast<uint64_t>(r);
            n -= static_cast<size_t>(r);
        }
    }

    bool read_record(uint64_t off, std::string& payload, uint64_t& next) {
        if (off + 8 > file_size_) return false;
        uint8_t hdr[8];
        if (::pread(fd_, hdr, 8, static_cast<off_t>(off)) != 8) return false;
        Reader hr(hdr, 8);
        uint32_t len = hr.u32();
        uint32_t crc = hr.u32();
        if (len > MAX_RECORD || off + 8 + len > file_size_) return false;
        payload.resize(len);
        if (::pread(fd_, payload.data(), len, static_cast<off_t>(off + 8)) !=
            static_cast<ssize_t>(len)) {
            return false;
        }
        if (crc32_ieee(payload.data(), len) != crc) return false;
        next = off + 8 + len;
        return true;
    }

    void append_record(const std::string& payload, bool sync) {
        std::string frame;
        frame.resize(8 + payload.size());
        uint32_t len = static_cast<uint32_t>(payload.size());
        uint32_t crc = crc32_ieee(payload.data(), payload.size());
        std::memcpy(&frame[0], &len, 4);
        std::memcpy(&frame[4], &crc, 4);
        std::memcpy(&frame[8], payload.data(), payload.size());
        pwrite_all(fd_, frame.data(), frame.size(), file_size_);
        if (sync && ::fdatasync(fd_) != 0) throw std::runtime_error("WAL sync failed");
        file_size_ += frame.size();
        if (sync) trim_page_cache(file_size_);
    }

    // Cgroup memory accounting includes the WAL page cache. On a memory-tight
    // container a long append-only file can otherwise fill the limit even
    // though the pages are clean. Drop all but the newest 4 MiB, in 4 MiB
    // steps, after durable writes and while replaying. Best effort only.
    static constexpr uint64_t PAGE_CACHE_KEEP = 4ull * 1024 * 1024;
    static constexpr uint64_t PAGE_CACHE_STEP = 4ull * 1024 * 1024;
    uint64_t page_cache_trimmed_ = 0;

    void trim_page_cache(uint64_t current_off) {
        if (current_off <= PAGE_CACHE_KEEP + PAGE_CACHE_STEP) return;
        uint64_t upto = current_off - PAGE_CACHE_KEEP;
        if (upto < page_cache_trimmed_ + PAGE_CACHE_STEP) return;
        uint64_t from = page_cache_trimmed_;
        ::posix_fadvise(fd_, static_cast<off_t>(from),
                        static_cast<off_t>(upto - from),
                        POSIX_FADV_DONTNEED);
        page_cache_trimmed_ = upto;
    }

    // -- WAL payloads ------------------------------------------------------

    static std::string encode_values(const Schema& s, const Row& row) {
        std::string out;
        for (size_t i = 0; i < s.col_types.size(); ++i) {
            switch (s.col_types[i]) {
                case COL_BIGINT: put_i64(out, row[i].i); break;
                case COL_VARCHAR: put_str(out, row[i].s); break;
                case COL_VECTOR:
                    put_u32(out, static_cast<uint32_t>(row[i].v.size()));
                    for (float f : row[i].v) put_f32(out, f);
                    break;
            }
        }
        return out;
    }

    static Row decode_values(const Schema& s, const void* data, size_t len) {
        Reader r(data, len);
        Row row(s.col_types.size());
        for (size_t i = 0; i < s.col_types.size() && r.ok; ++i) {
            switch (s.col_types[i]) {
                case COL_BIGINT: row[i].i = r.i64(); break;
                case COL_VARCHAR: row[i].s = r.str(); break;
                case COL_VECTOR: {
                    uint32_t dim = r.u32();
                    if (!r.ok || dim != static_cast<uint32_t>(s.col_lens[i])) {
                        throw std::runtime_error("corrupt vector in WAL");
                    }
                    row[i].v.resize(dim);
                    for (uint32_t k = 0; k < dim; ++k) {
                        float f = r.f32();
                        if (!std::isfinite(f)) throw std::runtime_error("non-finite vector in WAL");
                        row[i].v[k] = f;
                    }
                    break;
                }
            }
        }
        if (!r.ok || r.i != len) throw std::runtime_error("corrupt row in WAL");
        return row;
    }

    // -- recovery ----------------------------------------------------------

    void replay() {
        uint64_t off = 8;
        uint64_t last_good = 8;
        while (off < file_size_) {
            std::string payload;
            uint64_t next = 0;
            if (!read_record(off, payload, next)) break;
            try {
                apply_recovered_payload(payload, off + 8);
            } catch (const std::exception&) {
                // A CRC-valid record that cannot be interpreted is treated as a
                // corrupt tail; committed prefixes before it are kept.
                break;
            }
            off = next;
            last_good = next;
            trim_page_cache(off);
        }
        if (last_good != file_size_) {
            if (::ftruncate(fd_, static_cast<off_t>(last_good)) != 0) {
                throw std::runtime_error("cannot truncate corrupt WAL tail");
            }
            if (::fdatasync(fd_) != 0) throw std::runtime_error("cannot sync truncated WAL");
            file_size_ = last_good;
        }
        recovered_ = true;
    }

    void apply_recovered_payload(const std::string& payload, uint64_t data_off) {
        if (payload.empty()) throw std::runtime_error("empty WAL record");
        uint8_t type = static_cast<uint8_t>(payload[0]);
        Reader r(payload.data() + 1, payload.size() - 1);
        if (type == 'C') {
            uint32_t id = r.u32();
            std::string name = r.str();
            uint32_t ncols = r.u32();
            if (!r.ok || ncols == 0 || ncols > 4096) throw std::runtime_error("bad schema record");
            Schema schema;
            schema.name = name;
            for (uint32_t i = 0; i < ncols; ++i) {
                std::string cname = r.str();
                uint8_t ctype = r.u8();
                uint32_t clen = r.u32();
                uint8_t flags = r.u8();
                if (!r.ok) throw std::runtime_error("bad schema record");
                if (ctype > COL_VECTOR) throw std::runtime_error("bad column type");
                if (flags & 1) {
                    if (schema.pk != -1) throw std::runtime_error("multiple PK in schema");
                    schema.pk = static_cast<int>(i);
                }
                if (ctype == COL_VECTOR) {
                    if (schema.vec != -1) throw std::runtime_error("multiple vector cols");
                    schema.vec = static_cast<int>(i);
                }
                schema.col_names.push_back(std::move(cname));
                schema.col_types.push_back(ctype);
                schema.col_lens.push_back(static_cast<int>(clen));
            }
            if (!r.ok) throw std::runtime_error("bad schema record");
            if (schema.pk < 0) throw std::runtime_error("schema without PK");
            if (by_id_.count(id)) {
                // Duplicate DDL (should not happen); ignore the duplicate.
                return;
            }
            Table t;
            t.id = id;
            t.schema = std::move(schema);
            size_t slot = tables_.size();
            tables_.push_back(std::move(t));
            by_id_[id] = slot;
            // Also register by name for lookups.
            if (id >= next_table_id_) next_table_id_ = id + 1;
        } else if (type == 'X') {
            uint32_t tid = r.u32();
            std::string iname = r.str();
            std::string icol = r.str();
            if (!r.ok) throw std::runtime_error("bad index record");
            auto it = by_id_.find(tid);
            if (it == by_id_.end()) throw std::runtime_error("index for unknown table");
            tables_[it->second].index_names.push_back(std::move(iname));
            (void)icol;
        } else if (type == 'T') {
            uint32_t n = r.u32();
            if (!r.ok || n > 100000000u) throw std::runtime_error("bad commit record");
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t tid = r.u32();
                uint8_t op = r.u8();
                int64_t pk = r.i64();
                if (!r.ok) throw std::runtime_error("bad change record");
                auto it = by_id_.find(tid);
                if (it == by_id_.end()) throw std::runtime_error("change for unknown table");
                Table& t = tables_[it->second];
                if (op == 3) {
                    apply_change(t, 3, pk, nullptr, 0, 0);
                } else {
                    uint32_t rlen = r.u32();
                    if (!r.ok || rlen > r.n - r.i) throw std::runtime_error("bad row length");
                    const uint8_t* rp = r.p + r.i;
                    uint64_t abs = data_off + 1 + r.i;
                    Row row = decode_values(t.schema, rp, rlen);
                    apply_change(t, 1, pk, &row, abs, rlen);
                    r.skip(rlen);
                }
            }
            if (!r.ok) throw std::runtime_error("bad commit record");
        } else {
            throw std::runtime_error("unknown WAL record type");
        }
    }

    // -- in-memory state ---------------------------------------------------

    void apply_change(Table& t, uint8_t op, int64_t pk, const Row* row,
                      uint64_t off, uint32_t len) {
        if (op == 3) {
            auto it = t.locs.find(pk);
            if (it != t.locs.end()) {
                t.live_bytes = t.live_bytes >= it->second.len ? t.live_bytes - it->second.len : 0;
                t.locs.erase(it);
            }
            if (t.cached) t.cache.erase(pk);
            return;
        }
        auto it = t.locs.find(pk);
        if (it != t.locs.end()) {
            t.live_bytes = t.live_bytes >= it->second.len ? t.live_bytes - it->second.len : 0;
        }
        t.locs[pk] = Loc{off, len};
        t.live_bytes += len;
        if (t.cached) {
            if (t.live_bytes > cache_limit_) {
                t.cached = false;
                t.cache.clear();
                t.cache.rehash(0);
            } else if (row) {
                t.cache[pk] = *row;
            }
        }
    }

    Table* find_table(const std::string& name) {
        for (auto& t : tables_) {
            if (ieq(t.schema.name, name.c_str())) return &t;
        }
        return nullptr;
    }

    Table* table_by_id(uint32_t id) {
        auto it = by_id_.find(id);
        return it == by_id_.end() ? nullptr : &tables_[it->second];
    }

    bool overlay_has(uint32_t tid, int64_t pk) {
        auto it = txn_.find(tid);
        if (it == txn_.end()) return false;
        return it->second.find(pk) != it->second.end();
    }

    bool pk_exists(Table& t, int64_t pk) {
        if (txn_active_) {
            auto tit = txn_.find(t.id);
            if (tit != txn_.end()) {
                auto it = tit->second.find(pk);
                if (it != tit->second.end()) return it->second.has_value();
            }
        }
        return t.locs.count(pk) != 0;
    }

    // -- transaction -------------------------------------------------------

    void implicit_commit() {
        if (!txn_active_) return;
        Result r = commit_transaction();
        if (!r.ok) throw SqlError(r.code, r.message);
    }

    Result commit_transaction() {
        if (!txn_active_) return Result::empty_ok();
        struct Change {
            uint32_t tid;
            uint8_t op;
            int64_t pk;
            Row row;
            uint64_t abs = 0;
            uint32_t len = 0;
        };
        std::vector<Change> changes;
        for (auto& tkv : txn_) {
            for (auto& kv : tkv.second) {
                Change c;
                c.tid = tkv.first;
                c.pk = kv.first;
                if (kv.second.has_value()) {
                    c.op = 1;
                    c.row = *kv.second;
                } else {
                    c.op = 3;
                }
                changes.push_back(std::move(c));
            }
        }
        if (!changes.empty()) {
            std::string payload;
            payload.push_back('T');
            put_u32(payload, static_cast<uint32_t>(changes.size()));
            for (auto& c : changes) {
                Table* t = table_by_id(c.tid);
                if (!t) return Result::fail("internal_error", "transaction table vanished");
                put_u32(payload, c.tid);
                put_u8(payload, c.op);
                put_i64(payload, c.pk);
                if (c.op != 3) {
                    std::string enc = encode_values(t->schema, c.row);
                    c.len = static_cast<uint32_t>(enc.size());
                    c.abs = file_size_ + 8 + payload.size() + 4;
                    put_u32(payload, c.len);
                    payload += enc;
                }
            }
            try {
                append_record(payload, true);
            } catch (const std::exception& e) {
                return Result::fail("io_error", e.what());
            }
            for (auto& c : changes) {
                Table* t = table_by_id(c.tid);
                apply_change(*t, c.op, c.pk, c.op == 3 ? nullptr : &c.row, c.abs, c.len);
            }
        }
        txn_.clear();
        txn_active_ = false;
        return Result::empty_ok();
    }

    // -- statement dispatch ------------------------------------------------

    Result exec_stmt(const Stmt& st) {
        switch (st.type) {
            case Stmt::EMPTY: return Result::empty_ok();
            case Stmt::UNSUPPORTED:
                return Result::fail("unsupported_sql",
                                    st.raw.empty() ? "unsupported SQL statement" : st.raw);
            case Stmt::BEGIN_T: return do_begin();
            case Stmt::COMMIT_T: return commit_transaction();
            case Stmt::ROLLBACK_T: return do_rollback();
            case Stmt::CREATE_TABLE: return do_create_table(st);
            case Stmt::CREATE_INDEX: return do_create_index(st);
            case Stmt::INSERT: return do_insert(st);
            case Stmt::UPDATE: return do_update(st);
            case Stmt::DELETE_T: return do_delete(st);
            case Stmt::SELECT: return do_select(st);
        }
        return Result::fail("internal_error", "unknown statement");
    }

    Result do_begin() {
        if (txn_active_) return Result::fail("transaction_error", "transaction already active");
        txn_.clear();
        txn_active_ = true;
        return Result::empty_ok();
    }

    Result do_rollback() {
        txn_.clear();
        txn_active_ = false;
        return Result::empty_ok();
    }

    // -- DDL ---------------------------------------------------------------

    Result do_create_table(const Stmt& st) {
        if (find_table(st.table)) {
            if (st.if_not_exists) return Result::empty_ok();
            return Result::fail("table_exists", "table already exists: " + st.table);
        }
        if (st.cols.empty()) return Result::fail("syntax_error", "table needs at least one column");
        Schema schema;
        schema.name = st.table;
        int pk_count = 0, vec_count = 0;
        for (size_t i = 0; i < st.cols.size(); ++i) {
            const auto& c = st.cols[i];
            for (const auto& existing : schema.col_names) {
                if (ieq(existing, c.name.c_str())) {
                    return Result::fail("duplicate_column", "duplicate column: " + c.name);
                }
            }
            if (c.type < 0 || c.type > 2) return Result::fail("unsupported_sql", "column type");
            if (c.type == COL_VARCHAR && (c.len < 1 || c.len > 1000000)) {
                return Result::fail("syntax_error", "VARCHAR length out of range");
            }
            if (c.type == COL_VECTOR && (c.len < 1 || c.len > 65536)) {
                return Result::fail("syntax_error", "VECTOR dimension out of range");
            }
            if (c.pk) {
                ++pk_count;
                schema.pk = static_cast<int>(i);
                if (c.type != COL_BIGINT) {
                    return Result::fail("unsupported_sql", "primary key must be BIGINT");
                }
            }
            if (c.type == COL_VECTOR) {
                ++vec_count;
                schema.vec = static_cast<int>(i);
            }
            schema.col_names.push_back(c.name);
            schema.col_types.push_back(c.type);
            schema.col_lens.push_back(static_cast<int>(c.len));
        }
        if (pk_count != 1) return Result::fail("unsupported_sql", "exactly one BIGINT primary key is required");
        if (vec_count > 1) return Result::fail("unsupported_sql", "at most one vector column is allowed");

        implicit_commit();

        uint32_t id = next_table_id_++;
        std::string payload;
        payload.push_back('C');
        put_u32(payload, id);
        put_str(payload, schema.name);
        put_u32(payload, static_cast<uint32_t>(schema.col_names.size()));
        for (size_t i = 0; i < schema.col_names.size(); ++i) {
            put_str(payload, schema.col_names[i]);
            put_u8(payload, static_cast<uint8_t>(schema.col_types[i]));
            put_u32(payload, static_cast<uint32_t>(schema.col_lens[i]));
            put_u8(payload, schema.pk == static_cast<int>(i) ? 1 : 0);
        }
        try {
            append_record(payload, true);
        } catch (const std::exception& e) {
            return Result::fail("io_error", e.what());
        }
        Table t;
        t.id = id;
        t.schema = std::move(schema);
        size_t slot = tables_.size();
        tables_.push_back(std::move(t));
        by_id_[id] = slot;
        return Result::empty_ok();
    }

    Result do_create_index(const Stmt& st) {
        Table* t = find_table(st.table);
        if (!t) return Result::fail("table_not_found", "no such table: " + st.table);
        int idx = t->col_index(st.index_col);
        if (idx < 0) return Result::fail("column_not_found", "no such column: " + st.index_col);
        if (t->schema.col_types[idx] != COL_VECTOR) {
            return Result::fail("unsupported_sql", "vector index requires a VECTOR column");
        }
        for (const auto& name : t->index_names) {
            if (ieq(name, st.index_name.c_str())) {
                return Result::fail("index_exists", "index already exists: " + st.index_name);
            }
        }
        implicit_commit();
        std::string payload;
        payload.push_back('X');
        put_u32(payload, t->id);
        put_str(payload, st.index_name);
        put_str(payload, st.index_col);
        try {
            append_record(payload, true);
        } catch (const std::exception& e) {
            return Result::fail("io_error", e.what());
        }
        t->index_names.push_back(st.index_name);
        return Result::empty_ok();
    }

    // -- literal coercion --------------------------------------------------

    static std::vector<float> coerce_vector(const std::vector<double>& src, int dim) {
        if (static_cast<int>(src.size()) != dim) {
            throw SqlError("dimension_mismatch",
                           "expected " + std::to_string(dim) + " dimensions, got " +
                               std::to_string(src.size()));
        }
        std::vector<float> out(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            double d = src[i];
            if (!std::isfinite(d) || std::fabs(d) > static_cast<double>(FLT_MAX)) {
                throw SqlError("invalid_vector", "vector element is not a finite float32");
            }
            float f = static_cast<float>(d);
            if (!std::isfinite(f)) {
                throw SqlError("invalid_vector", "vector element overflows float32");
            }
            out[i] = f;
        }
        return out;
    }

    static Value coerce_literal(const Schema& s, int col, const Literal& lit) {
        Value v;
        switch (s.col_types[col]) {
            case COL_BIGINT: {
                if (lit.kind == Literal::INT) {
                    v.i = lit.i;
                } else if (lit.kind == Literal::NUM) {
                    double d = lit.d;
                    if (!std::isfinite(d) || d < -9223372036854775808.0 ||
                        d >= 9223372036854775808.0 || d != std::trunc(d)) {
                        throw SqlError("value_out_of_range", "BIGINT literal out of range");
                    }
                    v.i = static_cast<int64_t>(d);
                } else if (lit.kind == Literal::NUL) {
                    throw SqlError("not_null_violation", "NULL is not supported");
                } else {
                    throw SqlError("type_mismatch", "expected BIGINT literal");
                }
                break;
            }
            case COL_VARCHAR: {
                if (lit.kind == Literal::STR) {
                    size_t cp = 0;
                    if (!valid_utf8(lit.s, cp)) throw SqlError("invalid_utf8", "invalid UTF-8 string");
                    if (static_cast<int64_t>(cp) > s.col_lens[col]) {
                        throw SqlError("value_too_long", "VARCHAR value too long");
                    }
                    v.s = lit.s;
                } else if (lit.kind == Literal::NUL) {
                    throw SqlError("not_null_violation", "NULL is not supported");
                } else {
                    throw SqlError("type_mismatch", "expected string literal");
                }
                break;
            }
            case COL_VECTOR: {
                std::vector<double> raw;
                if (lit.kind == Literal::VEC) raw = lit.v;
                else if (lit.kind == Literal::STR) raw = Parser::parse_vector_text(lit.s);
                else if (lit.kind == Literal::NUL) throw SqlError("not_null_violation", "NULL is not supported");
                else throw SqlError("type_mismatch", "expected vector literal");
                v.v = coerce_vector(raw, s.col_lens[col]);
                break;
            }
        }
        return v;
    }

    // -- predicates --------------------------------------------------------

    struct Pred {
        int col = 0;
        int op = 0;
        bool is_int = true;
        int64_t i = 0;
        std::string s;
    };

    std::vector<Pred> resolve_preds(Table& t, const std::vector<Cond>& conds) {
        std::vector<Pred> out;
        for (const auto& c : conds) {
            int idx = t.col_index(c.col);
            if (idx < 0) throw SqlError("column_not_found", "no such column: " + c.col);
            if (t.schema.col_types[idx] == COL_VECTOR) {
                throw SqlError("unsupported_sql", "comparison on vector column");
            }
            Pred p;
            p.col = idx;
            p.op = c.op;
            if (t.schema.col_types[idx] == COL_BIGINT) {
                p.is_int = true;
                if (c.lit.kind == Literal::INT) p.i = c.lit.i;
                else if (c.lit.kind == Literal::NUM) {
                    double d = c.lit.d;
                    if (!std::isfinite(d) || d < -9223372036854775808.0 ||
                        d >= 9223372036854775808.0) {
                        throw SqlError("value_out_of_range", "BIGINT comparison out of range");
                    }
                    p.i = static_cast<int64_t>(d);
                } else {
                    throw SqlError("type_mismatch", "expected integer comparison value");
                }
            } else {
                p.is_int = false;
                if (c.lit.kind != Literal::STR) {
                    throw SqlError("type_mismatch", "expected string comparison value");
                }
                p.s = c.lit.s;
            }
            out.push_back(std::move(p));
        }
        return out;
    }

    static bool eval_preds(const Row& row, const std::vector<Pred>& preds) {
        for (const auto& p : preds) {
            int cmp;
            if (p.is_int) {
                int64_t v = row[p.col].i;
                cmp = (v < p.i) ? -1 : (v > p.i ? 1 : 0);
            } else {
                cmp = row[p.col].s.compare(p.s);
                cmp = (cmp < 0) ? -1 : (cmp > 0 ? 1 : 0);
            }
            switch (p.op) {
                case 0: if (cmp != 0) return false; break;
                case 1: if (cmp >= 0) return false; break;
                case 2: if (cmp > 0) return false; break;
                case 3: if (cmp <= 0) return false; break;
                case 4: if (cmp < 0) return false; break;
                case 5: if (cmp == 0) return false; break;
                default: return false;
            }
        }
        return true;
    }

    // -- row scanning ------------------------------------------------------

    template <typename F>
    void scan_wal_table(Table& t, F&& f) {
        uint64_t off = 8;
        while (off < file_size_) {
            std::string payload;
            uint64_t next = 0;
            if (!read_record(off, payload, next)) break;
            if (!payload.empty() && payload[0] == 'T') {
                Reader r(payload.data() + 1, payload.size() - 1);
                uint32_t n = r.u32();
                if (!r.ok) break;
                bool bad = false;
                for (uint32_t i = 0; i < n; ++i) {
                    uint32_t tid = r.u32();
                    uint8_t op = r.u8();
                    int64_t pk = r.i64();
                    if (!r.ok) { bad = true; break; }
                    if (op == 3) continue;
                    uint32_t rlen = r.u32();
                    if (!r.ok || rlen > r.n - r.i) { bad = true; break; }
                    size_t pos = r.i;
                    if (tid == t.id) {
                        uint64_t abs = off + 8 + 1 + pos;
                        auto it = t.locs.find(pk);
                        if (it != t.locs.end() && it->second.off == abs &&
                            it->second.len == rlen) {
                            Row row = decode_values(t.schema, r.p + pos, rlen);
                            f(pk, row);
                        }
                    }
                    r.skip(rlen);
                }
                if (bad) break;
            }
            off = next;
        }
    }

    template <typename F>
    void scan_table(Table& t, F&& f) {
        if (t.cached) {
            for (const auto& kv : t.cache) {
                if (txn_active_ && overlay_has(t.id, kv.first)) continue;
                f(kv.first, kv.second);
            }
        } else {
            scan_wal_table(t, [&](int64_t pk, const Row& row) {
                if (txn_active_ && overlay_has(t.id, pk)) return;
                f(pk, row);
            });
        }
        if (txn_active_) {
            auto it = txn_.find(t.id);
            if (it != txn_.end()) {
                for (const auto& kv : it->second) {
                    if (kv.second.has_value()) f(kv.first, *kv.second);
                }
            }
        }
    }

    static double l2_distance(const std::vector<float>& a, const std::vector<float>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            sum += d * d;
        }
        return std::sqrt(sum);
    }

    // -- INSERT ------------------------------------------------------------

    Result do_insert(const Stmt& st) {
        Table* t = find_table(st.table);
        if (!t) return Result::fail("table_not_found", "no such table: " + st.table);
        if (st.tuples.empty()) return Result::fail("syntax_error", "INSERT needs at least one row");

        const size_t ncols = t->schema.col_names.size();
        std::vector<int> map;
        if (st.col_list.empty()) {
            map.resize(ncols);
            for (size_t i = 0; i < ncols; ++i) map[i] = static_cast<int>(i);
        } else {
            std::unordered_set<std::string> seen;
            for (const auto& name : st.col_list) {
                int idx = t->col_index(name);
                if (idx < 0) return Result::fail("column_not_found", "no such column: " + name);
                std::string low = lower_copy(name);
                if (!seen.insert(low).second) {
                    return Result::fail("duplicate_column", "duplicate column in INSERT: " + name);
                }
                map.push_back(idx);
            }
        }

        std::vector<Row> rows;
        rows.reserve(st.tuples.size());
        std::unordered_set<int64_t> batch_pks;
        for (const auto& tuple : st.tuples) {
            if (tuple.size() != map.size()) {
                return Result::fail("column_count_mismatch", "value count does not match column count");
            }
            Row row(ncols);
            bool vector_set = (t->schema.vec < 0);
            bool pk_set = false;
            for (size_t j = 0; j < tuple.size(); ++j) {
                int col = map[j];
                try {
                    row[col] = coerce_literal(t->schema, col, tuple[j]);
                } catch (const SqlError& e) {
                    return Result::fail(e.code, e.what());
                }
                if (col == t->schema.vec) vector_set = true;
                if (col == t->schema.pk) pk_set = true;
            }
            if (!pk_set) return Result::fail("not_null_violation", "primary key value required");
            if (!vector_set) return Result::fail("not_null_violation", "vector column value required");
            int64_t pk = row[t->schema.pk].i;
            if (!batch_pks.insert(pk).second) {
                return Result::fail("duplicate_key", "duplicate primary key in statement");
            }
            if (pk_exists(*t, pk)) {
                return Result::fail("duplicate_key", "duplicate primary key");
            }
            rows.push_back(std::move(row));
        }

        int64_t affected = static_cast<int64_t>(rows.size());
        if (txn_active_) {
            auto& overlay = txn_[t->id];
            for (auto& row : rows) overlay[row[t->schema.pk].i] = std::move(row);
            return Result::with_affected(affected);
        }

        std::string payload;
        payload.push_back('T');
        put_u32(payload, static_cast<uint32_t>(rows.size()));
        struct Saved { size_t idx; uint64_t abs; uint32_t len; };
        std::vector<Saved> saved;
        saved.reserve(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            int64_t pk = rows[i][t->schema.pk].i;
            put_u32(payload, t->id);
            put_u8(payload, 1);
            put_i64(payload, pk);
            std::string enc = encode_values(t->schema, rows[i]);
            Saved s;
            s.idx = i;
            s.len = static_cast<uint32_t>(enc.size());
            s.abs = file_size_ + 8 + payload.size() + 4;
            saved.push_back(s);
            put_u32(payload, s.len);
            payload += enc;
        }
        try {
            append_record(payload, true);
        } catch (const std::exception& e) {
            return Result::fail("io_error", e.what());
        }
        for (const auto& s : saved) {
            apply_change(*t, 1, rows[s.idx][t->schema.pk].i, &rows[s.idx], s.abs, s.len);
        }
        return Result::with_affected(affected);
    }

    // -- UPDATE ------------------------------------------------------------

    Result do_update(const Stmt& st) {
        Table* t = find_table(st.table);
        if (!t) return Result::fail("table_not_found", "no such table: " + st.table);
        std::vector<std::pair<int, Value>> assigns;
        std::unordered_set<int> seen;
        for (const auto& a : st.assigns) {
            int idx = t->col_index(a.first);
            if (idx < 0) return Result::fail("column_not_found", "no such column: " + a.first);
            if (idx == t->schema.pk) {
                return Result::fail("unsupported_sql", "primary key updates are not supported");
            }
            if (!seen.insert(idx).second) {
                return Result::fail("duplicate_column", "duplicate assignment: " + a.first);
            }
            try {
                assigns.emplace_back(idx, coerce_literal(t->schema, idx, a.second));
            } catch (const SqlError& e) {
                return Result::fail(e.code, e.what());
            }
        }
        if (assigns.empty()) return Result::fail("syntax_error", "UPDATE needs assignments");
        std::vector<Pred> preds;
        try {
            preds = resolve_preds(*t, st.conds);
        } catch (const SqlError& e) {
            return Result::fail(e.code, e.what());
        }

        std::vector<std::pair<int64_t, Row>> matches;
        scan_table(*t, [&](int64_t pk, const Row& row) {
            if (eval_preds(row, preds)) matches.emplace_back(pk, row);
        });
        if (matches.empty()) return Result::with_affected(0);

        for (auto& m : matches) {
            for (const auto& a : assigns) m.second[a.first] = a.second;
        }
        int64_t affected = static_cast<int64_t>(matches.size());

        if (txn_active_) {
            auto& overlay = txn_[t->id];
            for (auto& m : matches) overlay[m.first] = std::move(m.second);
            return Result::with_affected(affected);
        }

        std::string payload;
        payload.push_back('T');
        put_u32(payload, static_cast<uint32_t>(matches.size()));
        struct Saved { size_t idx; uint64_t abs; uint32_t len; };
        std::vector<Saved> saved;
        for (size_t i = 0; i < matches.size(); ++i) {
            put_u32(payload, t->id);
            put_u8(payload, 2);
            put_i64(payload, matches[i].first);
            std::string enc = encode_values(t->schema, matches[i].second);
            Saved s;
            s.idx = i;
            s.len = static_cast<uint32_t>(enc.size());
            s.abs = file_size_ + 8 + payload.size() + 4;
            saved.push_back(s);
            put_u32(payload, s.len);
            payload += enc;
        }
        try {
            append_record(payload, true);
        } catch (const std::exception& e) {
            return Result::fail("io_error", e.what());
        }
        for (const auto& s : saved) {
            apply_change(*t, 2, matches[s.idx].first, &matches[s.idx].second, s.abs, s.len);
        }
        return Result::with_affected(affected);
    }

    // -- DELETE ------------------------------------------------------------

    Result do_delete(const Stmt& st) {
        Table* t = find_table(st.table);
        if (!t) return Result::fail("table_not_found", "no such table: " + st.table);
        std::vector<Pred> preds;
        try {
            preds = resolve_preds(*t, st.conds);
        } catch (const SqlError& e) {
            return Result::fail(e.code, e.what());
        }
        std::vector<int64_t> pks;
        scan_table(*t, [&](int64_t pk, const Row& row) {
            if (eval_preds(row, preds)) pks.push_back(pk);
        });
        if (pks.empty()) return Result::with_affected(0);

        if (txn_active_) {
            auto& overlay = txn_[t->id];
            for (int64_t pk : pks) overlay[pk] = std::nullopt;
            return Result::with_affected(static_cast<int64_t>(pks.size()));
        }

        std::string payload;
        payload.push_back('T');
        put_u32(payload, static_cast<uint32_t>(pks.size()));
        for (int64_t pk : pks) {
            put_u32(payload, t->id);
            put_u8(payload, 3);
            put_i64(payload, pk);
        }
        try {
            append_record(payload, true);
        } catch (const std::exception& e) {
            return Result::fail("io_error", e.what());
        }
        for (int64_t pk : pks) apply_change(*t, 3, pk, nullptr, 0, 0);
        return Result::with_affected(static_cast<int64_t>(pks.size()));
    }

    // -- SELECT ------------------------------------------------------------

    struct OutSpec {
        bool dist = false;
        int col = 0;
        std::vector<float> q;
        std::string name;
    };

    struct OrderSpec {
        int sel = -1;   // index into OutSpec list
        int col = -1;   // schema column index
        bool desc = false;
        bool dist = false;   // hidden distance key (ORDER BY L2_DISTANCE(...))
        std::vector<float> q;
    };

    struct Cand {
        int64_t pk = 0;
        std::vector<Cell> out;
        std::vector<Cell> keys;
    };

    static Cell cell_from_value(const Row& row, int col, const Schema& s) {
        Cell c;
        switch (s.col_types[col]) {
            case COL_BIGINT:
                c.kind = Cell::INT;
                c.i = row[col].i;
                break;
            case COL_VARCHAR:
                c.kind = Cell::STR;
                c.s = row[col].s;
                break;
            case COL_VECTOR:
                c.kind = Cell::VEC;
                c.v = row[col].v;
                break;
        }
        return c;
    }

    static int cmp_cell(const Cell& a, const Cell& b) {
        if (a.kind == Cell::INT && b.kind == Cell::INT) {
            return a.i < b.i ? -1 : (a.i > b.i ? 1 : 0);
        }
        if ((a.kind == Cell::INT || a.kind == Cell::DBL) &&
            (b.kind == Cell::INT || b.kind == Cell::DBL)) {
            double x = a.kind == Cell::INT ? static_cast<double>(a.i) : a.d;
            double y = b.kind == Cell::INT ? static_cast<double>(b.i) : b.d;
            return x < y ? -1 : (x > y ? 1 : 0);
        }
        if (a.kind == Cell::STR && b.kind == Cell::STR) {
            int c = a.s.compare(b.s);
            return c < 0 ? -1 : (c > 0 ? 1 : 0);
        }
        return 0;
    }

    static bool cand_better(const Cand& a, const Cand& b, const std::vector<OrderSpec>& specs) {
        for (size_t i = 0; i < specs.size(); ++i) {
            int c = cmp_cell(a.keys[i], b.keys[i]);
            if (c != 0) return specs[i].desc ? (c > 0) : (c < 0);
        }
        return a.pk < b.pk;
    }

    struct HeapCmp {
        const std::vector<OrderSpec>* specs;
        bool operator()(const Cand& a, const Cand& b) const {
            return cand_better(a, b, *specs);
        }
    };

    Result do_select(const Stmt& st) {
        Table* t = find_table(st.table);
        if (!t) return Result::fail("table_not_found", "no such table: " + st.table);

        std::vector<OutSpec> out_specs;
        for (const auto& item : st.items) {
            if (item.star) {
                for (size_t c = 0; c < t->schema.col_names.size(); ++c) {
                    OutSpec o;
                    o.dist = false;
                    o.col = static_cast<int>(c);
                    o.name = t->schema.col_names[c];
                    out_specs.push_back(std::move(o));
                }
                continue;
            }
            int idx = t->col_index(item.col);
            if (idx < 0) return Result::fail("column_not_found", "no such column: " + item.col);
            OutSpec o;
            if (item.dist) {
                if (t->schema.col_types[idx] != COL_VECTOR) {
                    return Result::fail("type_mismatch", "L2_DISTANCE requires a VECTOR column");
                }
                o.dist = true;
                o.col = idx;
                try {
                    o.q = coerce_vector(item.qvec, t->schema.col_lens[idx]);
                } catch (const SqlError& e) {
                    return Result::fail(e.code, e.what());
                }
                o.name = item.alias.empty() ? item.raw : item.alias;
            } else {
                o.col = idx;
                o.name = item.alias.empty() ? t->schema.col_names[idx] : item.alias;
            }
            out_specs.push_back(std::move(o));
        }
        if (out_specs.empty()) return Result::fail("syntax_error", "empty select list");

        std::vector<Pred> preds;
        try {
            preds = resolve_preds(*t, st.conds);
        } catch (const SqlError& e) {
            return Result::fail(e.code, e.what());
        }

        std::vector<OrderSpec> order_specs;
        for (const auto& key : st.order) {
            OrderSpec os;
            if (key.dist) {
                int col = t->col_index(key.col);
                if (col < 0) return Result::fail("column_not_found", "no such column: " + key.col);
                if (t->schema.col_types[col] != COL_VECTOR) {
                    return Result::fail("type_mismatch", "L2_DISTANCE requires a VECTOR column");
                }
                os.col = col;
                os.dist = true;
                try {
                    os.q = coerce_vector(key.qvec, t->schema.col_lens[col]);
                } catch (const SqlError& e) {
                    return Result::fail(e.code, e.what());
                }
                os.desc = key.desc;
                order_specs.push_back(std::move(os));
                continue;
            }
            for (size_t i = 0; i < st.items.size() && os.sel < 0; ++i) {
                const auto& item = st.items[i];
                if (!item.alias.empty() && ieq(item.alias, key.name.c_str())) {
                    // Map item index to OutSpec index: stars expand, so compute by walk.
                    size_t walk = 0;
                    for (size_t k = 0; k < st.items.size(); ++k) {
                        if (k == i) { os.sel = static_cast<int>(walk); break; }
                        if (st.items[k].star) walk += t->schema.col_names.size();
                        else walk += 1;
                    }
                }
            }
            if (os.sel < 0) {
                size_t walk = 0;
                for (const auto& item : st.items) {
                    if (item.star) { walk += t->schema.col_names.size(); continue; }
                    if (!item.dist && ieq(item.col, key.name.c_str())) {
                        os.sel = static_cast<int>(walk);
                        break;
                    }
                    ++walk;
                }
            }
            if (os.sel < 0) {
                int col = t->col_index(key.name);
                if (col < 0) return Result::fail("column_not_found", "no such column: " + key.name);
                if (t->schema.col_types[col] == COL_VECTOR && st.order.size() > 0) {
                    // Allow ORDER BY on a vector column only if it is selected as a distance.
                    return Result::fail("unsupported_sql", "cannot order by a vector column");
                }
                os.col = col;
            }
            os.desc = key.desc;
            order_specs.push_back(os);
        }

        std::vector<std::string> columns;
        columns.reserve(out_specs.size());
        for (const auto& o : out_specs) columns.push_back(o.name);

        Result res;
        res.has_rows = true;
        res.columns = columns;

        if (st.has_limit && st.limit == 0) return res;

        const bool has_order = !order_specs.empty();
        const bool use_heap = has_order && st.has_limit;

        std::vector<Cand> all;
        if (!use_heap) all.reserve(t->locs.size());
        std::priority_queue<Cand, std::vector<Cand>, HeapCmp> heap(
            HeapCmp{&order_specs});
        const uint64_t K = st.has_limit ? static_cast<uint64_t>(st.limit) : UINT64_MAX;

        auto make_cand = [&](int64_t pk, const Row& row) {
            Cand c;
            c.pk = pk;
            c.out.reserve(out_specs.size());
            for (const auto& o : out_specs) {
                if (o.dist) {
                    Cell cell;
                    cell.kind = Cell::DBL;
                    cell.d = l2_distance(row[o.col].v, o.q);
                    c.out.push_back(std::move(cell));
                } else {
                    c.out.push_back(cell_from_value(row, o.col, t->schema));
                }
            }
            c.keys.reserve(order_specs.size());
            for (const auto& os : order_specs) {
                if (os.sel >= 0) {
                    c.keys.push_back(c.out[os.sel]);
                } else if (os.dist) {
                    Cell cell;
                    cell.kind = Cell::DBL;
                    cell.d = l2_distance(row[os.col].v, os.q);
                    c.keys.push_back(std::move(cell));
                } else {
                    c.keys.push_back(cell_from_value(row, os.col, t->schema));
                }
            }
            return c;
        };

        scan_table(*t, [&](int64_t pk, const Row& row) {
            if (!eval_preds(row, preds)) return;
            Cand c = make_cand(pk, row);
            if (use_heap) {
                if (heap.size() < K) {
                    heap.push(std::move(c));
                } else if (cand_better(c, heap.top(), order_specs)) {
                    heap.pop();
                    heap.push(std::move(c));
                }
            } else {
                all.push_back(std::move(c));
            }
        });

        std::vector<Cand> selected;
        if (use_heap) {
            selected.reserve(heap.size());
            while (!heap.empty()) {
                selected.push_back(heap.top());
                heap.pop();
            }
            std::sort(selected.begin(), selected.end(),
                      [&](const Cand& a, const Cand& b) {
                          return cand_better(a, b, order_specs);
                      });
        } else {
            if (has_order) {
                std::sort(all.begin(), all.end(), [&](const Cand& a, const Cand& b) {
                    return cand_better(a, b, order_specs);
                });
            } else {
                std::sort(all.begin(), all.end(), [](const Cand& a, const Cand& b) {
                    return a.pk < b.pk;
                });
            }
            if (st.has_limit && all.size() > K) all.resize(K);
            selected = std::move(all);
        }

        res.rows.reserve(selected.size());
        for (auto& c : selected) res.rows.push_back(std::move(c.out));
        return res;
    }
};

}  // namespace mvd
