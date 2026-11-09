// MiniVectorDB - process entry point and stdin/stdout session protocol.
//
// PROVISIONAL PROTOCOL (must be aligned with the official 2026-competition-
// round2 test kit once it is published):
//   * startup: prints "<ready-marker>\n" (default "READY") after recovery
//   * input:   SQL statements delimited by top-level ';'
//   * SELECT:  header line, then one tab-separated line per row
//   * DML/DDL: no output by default (see --echo-ok)
//   * errors:  "ERROR <code>: <message>"
//   * every statement response ends with "<sentinel>\n" (default __MVDB_END__)
//   * exit:    EOF, "exit;", "quit;" or "\q" -> rollback open txn + checkpoint
// All protocol formatting lives in this file so the engine stays untouched
// when the kit's exact format is wired in.
#include <cerrno>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>

#include <unistd.h>

#include "engine.h"
#include "sql.h"
#include "storage.h"

using namespace mvdb;

namespace {

volatile sig_atomic_t g_stop = 0;

void on_signal(int) { g_stop = 1; }

struct Options {
  std::string data_dir = "/data";
  bool ann = false;
  std::string sentinel = "__MVDB_END__";
  std::string ready = "READY";
  bool echo_ok = false;
  size_t mem_budget_mb = 0;  // 0 = derive from the cgroup limit
  size_t wal_checkpoint_mb = 128;  // 0 = only checkpoint on shutdown
};

void usage(const char* prog) {
  std::fprintf(stderr,
               "MiniVectorDB skeleton\n"
               "usage: %s [options]\n"
               "  --data-dir <dir>   data directory (default /data)\n"
               "  --mode exact|ann   vector search mode (default exact)\n"
               "  --sentinel <str>   per-statement output terminator "
               "(default __MVDB_END__, empty disables)\n"
               "  --ready <str>      startup ready marker (default READY)\n"
               "  --echo-ok          print OK <affected> for DML/DDL\n"
               "  --mem-budget <MB>  decoded-row memory budget before rows "
               "spill to disk (default: 40%% of the cgroup limit)\n"
               "  --wal-checkpoint-mb <MB>  checkpoint once the WAL grows past "
               "this size, 0 disables (default 128)\n"
               "  --help\n",
               prog);
}

Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto value = [&](const char* name) -> std::string {
      std::string prefix = std::string(name) + "=";
      if (a.rfind(prefix, 0) == 0) return a.substr(prefix.size());
      if (i + 1 < argc) return argv[++i];
      std::fprintf(stderr, "missing value for %s\n", name);
      std::exit(2);
    };
    if (a == "--help" || a == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else if (a == "--data-dir" || a.rfind("--data-dir=", 0) == 0) {
      opt.data_dir = value("--data-dir");
    } else if (a == "--mode" || a.rfind("--mode=", 0) == 0) {
      std::string m = value("--mode");
      if (m == "ann") {
        opt.ann = true;
      } else if (m == "exact") {
        opt.ann = false;
      } else {
        std::fprintf(stderr, "invalid --mode '%s' (expected exact|ann)\n",
                     m.c_str());
        std::exit(2);
      }
    } else if (a == "--sentinel" || a.rfind("--sentinel=", 0) == 0) {
      opt.sentinel = value("--sentinel");
    } else if (a == "--ready" || a.rfind("--ready=", 0) == 0) {
      opt.ready = value("--ready");
    } else if (a == "--echo-ok") {
      opt.echo_ok = true;
    } else if (a == "--mem-budget" || a.rfind("--mem-budget=", 0) == 0) {
      std::string v = value("--mem-budget");
      try {
        opt.mem_budget_mb = static_cast<size_t>(std::stoull(v));
      } catch (...) {
        std::fprintf(stderr, "invalid --mem-budget '%s'\n", v.c_str());
        std::exit(2);
      }
    } else if (a == "--wal-checkpoint-mb" ||
               a.rfind("--wal-checkpoint-mb=", 0) == 0) {
      std::string v = value("--wal-checkpoint-mb");
      try {
        opt.wal_checkpoint_mb = static_cast<size_t>(std::stoull(v));
      } catch (...) {
        std::fprintf(stderr, "invalid --wal-checkpoint-mb '%s'\n", v.c_str());
        std::exit(2);
      }
    } else {
      std::fprintf(stderr, "unknown option: %s\n", a.c_str());
      usage(argv[0]);
      std::exit(2);
    }
  }
  return opt;
}

std::string trimmed_lower(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return lower_ascii(s.substr(b, e - b));
}

class Session {
 public:
  Session(Engine& engine, const Options& opt) : engine_(engine), opt_(opt) {}

  void run() {
    char chunk[8192];
    while (!g_stop) {
      ssize_t n = ::read(0, chunk, sizeof(chunk));
      if (n < 0) {
        if (errno == EINTR) {
          if (g_stop) break;
          continue;
        }
        std::fprintf(stderr, "read(stdin) failed: %s\n", std::strerror(errno));
        break;
      }
      if (n == 0) break;  // EOF
      for (ssize_t i = 0; i < n && !g_stop; ++i) feed(chunk[i]);
    }
    // A trailing statement without ';' is still executed at EOF.
    if (!g_stop && !trim(statement_).empty()) {
      execute_statement(statement_);
      statement_.clear();
    }
    engine_.shutdown();
  }

 private:
  enum class SplitState {
    NORMAL,
    SQUOTE,
    DQUOTE,
    BTICK,
    LINE_COMMENT,
    BLOCK_COMMENT,
  };

  Engine& engine_;
  const Options& opt_;
  std::string statement_;
  SplitState state_ = SplitState::NORMAL;
  bool pending_quote_ = false;
  bool escape_next_ = false;
  char quote_char_ = 0;
  char last_char_ = 0;

  static std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
  }

  void feed(char c) {
    if (pending_quote_) {
      pending_quote_ = false;
      if (c == quote_char_) {
        // doubled quote inside a string literal: escaped quote
        statement_.push_back(c);
        last_char_ = c;
        return;
      }
      state_ = SplitState::NORMAL;
      // fall through and process c in normal state
    }

    switch (state_) {
      case SplitState::NORMAL:
        if (c == '\'') {
          state_ = SplitState::SQUOTE;
        } else if (c == '"') {
          state_ = SplitState::DQUOTE;
        } else if (c == '`') {
          state_ = SplitState::BTICK;
        } else if (c == '#') {
          state_ = SplitState::LINE_COMMENT;
        } else if (last_char_ == '-' && c == '-') {
          state_ = SplitState::LINE_COMMENT;
        } else if (last_char_ == '/' && c == '*') {
          state_ = SplitState::BLOCK_COMMENT;
          statement_.push_back(c);
          last_char_ = 0;  // do not treat the opening '*' as a closing one
          return;
        } else if (c == ';') {
          statement_.push_back(c);
          execute_statement(statement_);
          statement_.clear();
          last_char_ = c;
          return;
        } else if (c == '\n') {
          std::string t = trimmed_lower(statement_);
          if (t == "exit" || t == "quit" || t == "\\q") {
            g_stop = 1;
            return;
          }
        }
        break;

      case SplitState::SQUOTE:
      case SplitState::DQUOTE:
      case SplitState::BTICK: {
        char q = (state_ == SplitState::SQUOTE)
                     ? '\''
                     : (state_ == SplitState::DQUOTE ? '"' : '`');
        if (escape_next_) {
          // The lexer unescapes \x inside '...' and "..." literals, so the
          // splitter must not treat an escaped quote as the end of the string.
          escape_next_ = false;
          break;
        }
        if (c == '\\' && state_ != SplitState::BTICK) {
          escape_next_ = true;
          break;
        }
        if (c == q) {
          pending_quote_ = true;
          quote_char_ = q;
        }
        break;
      }

      case SplitState::LINE_COMMENT:
        if (c == '\n') state_ = SplitState::NORMAL;
        break;

      case SplitState::BLOCK_COMMENT:
        if (last_char_ == '*' && c == '/') {
          state_ = SplitState::NORMAL;
        }
        break;
    }

    statement_.push_back(c);
    last_char_ = c;
  }

  void write_out(const std::string& s) {
    if (!s.empty()) {
      std::fwrite(s.data(), 1, s.size(), stdout);
    }
    std::fflush(stdout);
  }

  static void append_joined(std::string& out,
                            const std::vector<std::string>& cells) {
    for (size_t i = 0; i < cells.size(); ++i) {
      if (i) out.push_back('\t');
      out += cells[i];
    }
    out.push_back('\n');
  }

  void execute_statement(std::string sql) {
    if (trim(sql).empty()) return;
    if (trimmed_lower(sql).empty()) return;

    std::string out;
    try {
      Parser parser(sql);
      Statement st = parser.parse_statement();
      if (st.kind == StmtKind::EXIT) {
        g_stop = 1;
      } else {
        ResultSet rs = engine_.execute(st);
        if (rs.has_rows) {
          append_joined(out, rs.columns);
          for (const auto& row : rs.rows) {
            out += row;
            out.push_back('\n');
          }
        } else if (opt_.echo_ok) {
          out += "OK " + std::to_string(rs.affected) + "\n";
        }
      }
    } catch (const DbError& e) {
      out += "ERROR " + std::to_string(e.code) + ": " + e.what() + "\n";
    } catch (const std::exception& e) {
      out += std::string("ERROR ") + std::to_string(ER_UNKNOWN) + ": " +
             e.what() + "\n";
    }
    if (!opt_.sentinel.empty()) {
      out += opt_.sentinel;
      out.push_back('\n');
    }
    write_out(out);
  }
};

void install_signal_handlers() {
  struct sigaction sa {};
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;  // no SA_RESTART: read() must return EINTR on SIGTERM
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGINT, &sa, nullptr);
}

}  // namespace

int main(int argc, char** argv) {
  Options opt = parse_args(argc, argv);
  install_signal_handlers();

  try {
    Storage storage(opt.data_dir, opt.mem_budget_mb << 20);
    storage.open();
    storage.set_wal_checkpoint_bytes(opt.wal_checkpoint_mb << 20);

    size_t recovered_rows = 0;
    size_t spilled_rows = 0;
    for (auto& [key, table] : storage.tables()) {
      (void)key;
      recovered_rows += table->live_rows;
      if (table->spilled()) spilled_rows += table->live_rows;
    }
    std::fprintf(stderr,
                 "minivectordb: data_dir=%s tables=%zu rows=%zu mode=%s "
                 "mem_budget=%zuMiB spilled_rows=%zu\n",
                 opt.data_dir.c_str(), storage.tables().size(), recovered_rows,
                 opt.ann ? "ann" : "exact", storage.mem_budget() >> 20,
                 spilled_rows);

    Engine engine(storage, opt.ann);
    std::printf("%s\n", opt.ready.c_str());
    std::fflush(stdout);

    Session session(engine, opt);
    session.run();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
}
