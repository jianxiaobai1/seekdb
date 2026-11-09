// MiniVectorDB - SQL lexer + parser implementation.
#include "sql.h"

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <limits>

namespace mvdb {
namespace {

bool ident_start(unsigned char c) {
  return std::isalpha(c) || c == '_' || c == '$';
}
bool ident_char(unsigned char c) {
  return std::isalnum(c) || c == '_' || c == '$';
}
bool digit(unsigned char c) { return std::isdigit(c); }

[[noreturn]] void syntax_error(size_t pos, const std::string& what) {
  throw DbError(ER_SYNTAX,
                "syntax error at position " + std::to_string(pos) + ": " + what);
}

int64_t parse_i64_text(const std::string& text, bool negate, size_t pos) {
  std::string full = negate ? ("-" + text) : text;
  errno = 0;
  char* end = nullptr;
  long long v = std::strtoll(full.c_str(), &end, 10);
  if (errno == ERANGE) {
    throw DbError(ER_OUT_OF_RANGE, "BIGINT value out of range at position " +
                                       std::to_string(pos));
  }
  if (end == full.c_str() || *end != '\0') {
    syntax_error(pos, "invalid integer literal");
  }
  return static_cast<int64_t>(v);
}

}  // namespace

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------
Lexer::Lexer(const std::string& sql) : sql_(sql) {}

std::vector<Token> Lexer::tokenize() {
  std::vector<Token> out;
  const size_t n = sql_.size();
  while (pos_ < n) {
    unsigned char c = static_cast<unsigned char>(sql_[pos_]);

    if (std::isspace(c)) {
      ++pos_;
      continue;
    }

    // -- line comment
    if (c == '-' && pos_ + 1 < n && sql_[pos_ + 1] == '-') {
      pos_ += 2;
      while (pos_ < n && sql_[pos_] != '\n') ++pos_;
      continue;
    }
    // # line comment
    if (c == '#') {
      while (pos_ < n && sql_[pos_] != '\n') ++pos_;
      continue;
    }
    // /* block comment */
    if (c == '/' && pos_ + 1 < n && sql_[pos_ + 1] == '*') {
      pos_ += 2;
      while (pos_ + 1 < n && !(sql_[pos_] == '*' && sql_[pos_ + 1] == '/')) ++pos_;
      if (pos_ + 1 < n) pos_ += 2;
      continue;
    }

    size_t start = pos_;

    if (ident_start(c)) {
      ++pos_;
      while (pos_ < n && ident_char(static_cast<unsigned char>(sql_[pos_]))) ++pos_;
      out.push_back({TokType::IDENT, sql_.substr(start, pos_ - start), start});
      continue;
    }

    if (digit(c)) {
      ++pos_;
      while (pos_ < n && digit(static_cast<unsigned char>(sql_[pos_]))) ++pos_;
      out.push_back({TokType::NUMBER, sql_.substr(start, pos_ - start), start});
      continue;
    }

    if (c == '\'' || c == '"') {
      char quote = static_cast<char>(c);
      ++pos_;
      std::string value;
      bool closed = false;
      while (pos_ < n) {
        char d = sql_[pos_];
        if (d == quote) {
          // doubled quote is an escaped quote
          if (pos_ + 1 < n && sql_[pos_ + 1] == quote) {
            value.push_back(quote);
            pos_ += 2;
            continue;
          }
          ++pos_;
          closed = true;
          break;
        }
        if (d == '\\' && pos_ + 1 < n) {
          char e = sql_[pos_ + 1];
          switch (e) {
            case 'n': value.push_back('\n'); break;
            case 't': value.push_back('\t'); break;
            case 'r': value.push_back('\r'); break;
            case '0': value.push_back('\0'); break;
            case '\\': value.push_back('\\'); break;
            case '\'': value.push_back('\''); break;
            case '"': value.push_back('"'); break;
            default: value.push_back(e); break;
          }
          pos_ += 2;
          continue;
        }
        value.push_back(d);
        ++pos_;
      }
      if (!closed) syntax_error(start, "unterminated string literal");
      out.push_back({TokType::STRING, value, start});
      continue;
    }

    // `quoted identifier`
    if (c == '`') {
      ++pos_;
      std::string value;
      bool closed = false;
      while (pos_ < n) {
        if (sql_[pos_] == '`') {
          if (pos_ + 1 < n && sql_[pos_ + 1] == '`') {
            value.push_back('`');
            pos_ += 2;
            continue;
          }
          ++pos_;
          closed = true;
          break;
        }
        value.push_back(sql_[pos_++]);
      }
      if (!closed) syntax_error(start, "unterminated quoted identifier");
      out.push_back({TokType::IDENT, value, start});
      continue;
    }

    // multi-character operators
    if (pos_ + 1 < n) {
      std::string two = sql_.substr(pos_, 2);
      if (two == "<=" || two == ">=" || two == "!=" || two == "<>" || two == "==") {
        out.push_back({TokType::SYMBOL, two == "==" ? "=" : two, start});
        pos_ += 2;
        continue;
      }
    }

    static const std::string symbols = "(),;*=<>[].-";
    if (symbols.find(static_cast<char>(c)) != std::string::npos) {
      out.push_back({TokType::SYMBOL, std::string(1, static_cast<char>(c)), start});
      ++pos_;
      continue;
    }

    syntax_error(start,
                 "unexpected character '" + std::string(1, static_cast<char>(c)) + "'");
  }
  out.push_back({TokType::END, "", n});
  return out;
}

// ---------------------------------------------------------------------------
// Parser primitives
// ---------------------------------------------------------------------------
Parser::Parser(const std::string& sql) {
  Lexer lexer(sql);
  tokens_ = lexer.tokenize();
}

const Token& Parser::peek() const { return tokens_[pos_]; }

bool Parser::at_end() const { return peek().type == TokType::END; }

Token Parser::advance() {
  Token t = tokens_[pos_];
  if (pos_ + 1 < tokens_.size()) ++pos_;
  return t;
}

bool Parser::is_kw(const char* kw) const {
  return peek().type == TokType::IDENT && lower_ascii(peek().text) == kw;
}

bool Parser::accept_kw(const char* kw) {
  if (is_kw(kw)) {
    advance();
    return true;
  }
  return false;
}

void Parser::expect_kw(const char* kw) {
  if (!accept_kw(kw)) {
    syntax_error(peek().pos, std::string("expected keyword '") + kw + "', got '" +
                                 peek().text + "'");
  }
}

bool Parser::is_sym(const char* s) const {
  return peek().type == TokType::SYMBOL && peek().text == s;
}

bool Parser::accept_sym(const char* s) {
  if (is_sym(s)) {
    advance();
    return true;
  }
  return false;
}

void Parser::expect_sym(const char* s) {
  if (!accept_sym(s)) {
    syntax_error(peek().pos, std::string("expected '") + s + "', got '" + peek().text +
                                 "'");
  }
}

std::string Parser::expect_ident() {
  if (peek().type != TokType::IDENT) {
    syntax_error(peek().pos, "expected identifier, got '" + peek().text + "'");
  }
  return advance().text;
}

int64_t Parser::expect_int() {
  bool negate = false;
  if (accept_sym("-")) negate = true;
  if (peek().type != TokType::NUMBER) {
    syntax_error(peek().pos, "expected integer, got '" + peek().text + "'");
  }
  Token t = advance();
  return parse_i64_text(t.text, negate, t.pos);
}

Literal Parser::parse_literal() {
  Literal lit;
  size_t pos = peek().pos;
  bool negate = false;
  if (accept_sym("-")) negate = true;
  if (peek().type == TokType::NUMBER) {
    Token t = advance();
    lit.is_number = true;
    lit.num = parse_i64_text(t.text, negate, t.pos);
    return lit;
  }
  if (negate) syntax_error(pos, "expected integer after '-'");
  if (peek().type == TokType::STRING) {
    lit.is_number = false;
    lit.str = advance().text;
    return lit;
  }
  syntax_error(pos, "expected literal value, got '" + peek().text + "'");
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------
Statement Parser::parse_statement() {
  Statement st;
  if (accept_kw("create")) {
    parse_create(st);
  } else if (accept_kw("insert")) {
    parse_insert(st);
  } else if (accept_kw("select")) {
    parse_select(st);
  } else if (accept_kw("update")) {
    parse_update(st);
  } else if (accept_kw("delete")) {
    parse_delete(st);
  } else if (accept_kw("begin")) {
    st.kind = StmtKind::BEGIN;
    if (!accept_kw("transaction")) accept_kw("work");
  } else if (accept_kw("start")) {
    expect_kw("transaction");
    st.kind = StmtKind::BEGIN;
  } else if (accept_kw("commit") || accept_kw("end")) {
    st.kind = StmtKind::COMMIT;
    if (!accept_kw("transaction")) accept_kw("work");
  } else if (accept_kw("rollback")) {
    st.kind = StmtKind::ROLLBACK;
    if (!accept_kw("transaction")) accept_kw("work");
  } else if (accept_kw("exit") || accept_kw("quit")) {
    st.kind = StmtKind::EXIT;
  } else {
    syntax_error(peek().pos, "unsupported statement '" + peek().text + "'");
  }

  accept_sym(";");
  if (!at_end()) {
    syntax_error(peek().pos, "unexpected token '" + peek().text + "' after statement");
  }
  return st;
}

void Parser::parse_create(Statement& st) {
  if (accept_kw("table")) {
    st.kind = StmtKind::CREATE_TABLE;
    st.table = expect_ident();
    expect_sym("(");

    std::string table_level_pk;
    do {
      if (accept_kw("primary")) {
        expect_kw("key");
        expect_sym("(");
        table_level_pk = expect_ident();
        expect_sym(")");
        continue;
      }
      ColumnDefAst col;
      col.name = expect_ident();
      parse_type(col);
      if (accept_kw("primary")) {
        expect_kw("key");
        col.primary = true;
      }
      st.coldefs.push_back(std::move(col));
    } while (accept_sym(","));

    expect_sym(")");
    if (!table_level_pk.empty()) {
      for (auto& col : st.coldefs) {
        if (lower_ascii(col.name) == lower_ascii(table_level_pk)) col.primary = true;
      }
    }
    return;
  }

  if (accept_kw("vector")) {
    expect_kw("index");
    st.kind = StmtKind::CREATE_VECTOR_INDEX;
    st.index_name = expect_ident();
    expect_kw("on");
    st.table = expect_ident();
    expect_sym("(");
    st.index_column = expect_ident();
    expect_sym(")");
    return;
  }

  syntax_error(peek().pos, "expected TABLE or VECTOR INDEX after CREATE");
}

int Parser::parse_type(ColumnDefAst& col) {
  if (accept_kw("bigint")) {
    col.type = 1;
    col.len = 0;
    return 0;
  }
  if (accept_kw("varchar")) {
    col.type = 2;
    expect_sym("(");
    int64_t n = expect_int();
    expect_sym(")");
    if (n <= 0 || n > 1000000)
      throw DbError(ER_WRONG_VALUE, "invalid VARCHAR length");
    col.len = static_cast<int>(n);
    return 0;
  }
  if (accept_kw("vector")) {
    col.type = 3;
    expect_sym("(");
    int64_t n = expect_int();
    expect_sym(")");
    if (n <= 0 || n > 65536) throw DbError(ER_WRONG_VALUE, "invalid VECTOR dimension");
    col.len = static_cast<int>(n);
    return 0;
  }
  syntax_error(peek().pos, "expected column type, got '" + peek().text + "'");
}

void Parser::parse_insert(Statement& st) {
  st.kind = StmtKind::INSERT;
  expect_kw("into");
  st.table = expect_ident();

  if (accept_sym("(")) {
    do {
      st.insert_cols.push_back(expect_ident());
    } while (accept_sym(","));
    expect_sym(")");
  }

  expect_kw("values");
  do {
    expect_sym("(");
    std::vector<Literal> row;
    do {
      row.push_back(parse_literal());
    } while (accept_sym(","));
    expect_sym(")");
    st.insert_rows.push_back(std::move(row));
  } while (accept_sym(","));
}

void Parser::parse_select(Statement& st) {
  st.kind = StmtKind::SELECT;
  do {
    SelectItemAst item;
    if (accept_sym("*")) {
      item.kind = SelectItemAst::STAR;
    } else if (is_kw("l2_distance")) {
      advance();
      expect_sym("(");
      item.kind = SelectItemAst::DISTANCE;
      item.col = expect_ident();
      expect_sym(",");
      if (peek().type != TokType::STRING) {
        syntax_error(peek().pos, "expected vector literal string in L2_DISTANCE");
      }
      item.vec_literal = advance().text;
      expect_sym(")");
      item.alias = "distance";
      if (accept_kw("as")) item.alias = expect_ident();
    } else {
      item.kind = SelectItemAst::COLUMN;
      item.col = expect_ident();
      if (accept_kw("as")) item.alias = expect_ident();
    }
    st.select_items.push_back(std::move(item));
  } while (accept_sym(","));

  expect_kw("from");
  st.table = expect_ident();

  if (accept_kw("where")) st.conds = parse_where();
  parse_order_limit(st);
}

void Parser::parse_update(Statement& st) {
  st.kind = StmtKind::UPDATE;
  st.table = expect_ident();
  expect_kw("set");
  do {
    AssignAst a;
    a.col = expect_ident();
    expect_sym("=");
    a.value = parse_literal();
    st.assigns.push_back(std::move(a));
  } while (accept_sym(","));

  if (accept_kw("where")) st.conds = parse_where();
}

void Parser::parse_delete(Statement& st) {
  st.kind = StmtKind::DELETE;
  expect_kw("from");
  st.table = expect_ident();
  if (accept_kw("where")) st.conds = parse_where();
}

std::vector<CondAst> Parser::parse_where() {
  std::vector<CondAst> conds;
  do {
    CondAst c;
    c.col = expect_ident();
    if (accept_sym("=")) {
      c.op = "=";
    } else if (accept_sym("<=")) {
      c.op = "<=";
    } else if (accept_sym(">=")) {
      c.op = ">=";
    } else if (accept_sym("<")) {
      c.op = "<";
    } else if (accept_sym(">")) {
      c.op = ">";
    } else if (accept_sym("!=")) {
      c.op = "!=";
    } else {
      syntax_error(peek().pos,
                   "expected comparison operator, got '" + peek().text + "'");
    }
    c.value = parse_literal();
    conds.push_back(std::move(c));
  } while (accept_kw("and"));
  return conds;
}

void Parser::parse_order_limit(Statement& st) {
  if (accept_kw("order")) {
    expect_kw("by");
    st.has_order = true;
    if (is_kw("l2_distance")) {
      advance();
      expect_sym("(");
      st.order_is_dist = true;
      st.order_col = expect_ident();
      expect_sym(",");
      if (peek().type != TokType::STRING) {
        syntax_error(peek().pos, "expected vector literal string in L2_DISTANCE");
      }
      st.order_vec_literal = advance().text;
      expect_sym(")");
    } else {
      st.order_col = expect_ident();
    }
    if (accept_kw("desc")) {
      st.order_desc = true;
    } else {
      accept_kw("asc");
    }
  }
  if (accept_kw("limit")) {
    st.has_limit = true;
    st.limit = expect_int();
    if (st.limit < 0) throw DbError(ER_WRONG_VALUE, "LIMIT must be non-negative");
  }
}

}  // namespace mvdb
