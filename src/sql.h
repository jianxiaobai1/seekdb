// MiniVectorDB - SQL lexer + recursive-descent parser for the required subset.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common.h"

namespace mvdb {

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------
enum class TokType { IDENT, NUMBER, STRING, SYMBOL, END };

struct Token {
  TokType type;
  std::string text;
  size_t pos;
};

class Lexer {
 public:
  explicit Lexer(const std::string& sql);
  std::vector<Token> tokenize();

 private:
  const std::string& sql_;
  size_t pos_ = 0;
};

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------
struct Literal {
  bool is_number = false;
  int64_t num = 0;
  std::string str;
};

struct ColumnDefAst {
  std::string name;
  int type = 0;  // 1=BIGINT 2=VARCHAR 3=VECTOR
  int len = 0;
  bool primary = false;
};

struct SelectItemAst {
  enum Kind { STAR, COLUMN, DISTANCE } kind = COLUMN;
  std::string col;          // column name (COLUMN / DISTANCE)
  std::string alias;        // optional alias, distance defaults to "distance"
  std::string vec_literal;  // raw '[..]' text for DISTANCE
};

struct CondAst {
  std::string col;
  std::string op;  // = < <= > >=
  Literal value;
};

struct AssignAst {
  std::string col;
  Literal value;
};

enum class StmtKind {
  CREATE_TABLE,
  CREATE_VECTOR_INDEX,
  INSERT,
  SELECT,
  UPDATE,
  DELETE,
  BEGIN,
  COMMIT,
  ROLLBACK,
  EXIT
};

struct Statement {
  StmtKind kind = StmtKind::EXIT;

  // CREATE TABLE
  std::string table;
  std::vector<ColumnDefAst> coldefs;

  // CREATE VECTOR INDEX
  std::string index_name;
  std::string index_column;

  // INSERT
  std::vector<std::string> insert_cols;  // optional explicit column list
  std::vector<std::vector<Literal>> insert_rows;

  // SELECT
  std::vector<SelectItemAst> select_items;
  std::vector<CondAst> conds;
  bool has_order = false;
  bool order_is_dist = false;  // ORDER BY L2_DISTANCE(col, '[..]')
  std::string order_col;
  std::string order_vec_literal;
  bool order_desc = false;
  bool has_limit = false;
  int64_t limit = 0;

  // UPDATE
  std::vector<AssignAst> assigns;
};

class Parser {
 public:
  explicit Parser(const std::string& sql);
  Statement parse_statement();

 private:
  std::vector<Token> tokens_;
  size_t pos_ = 0;

  const Token& peek() const;
  bool at_end() const;
  Token advance();
  bool is_kw(const char* kw) const;
  bool accept_kw(const char* kw);
  void expect_kw(const char* kw);
  bool is_sym(const char* s) const;
  bool accept_sym(const char* s);
  void expect_sym(const char* s);
  std::string expect_ident();
  int64_t expect_int();
  Literal parse_literal();
  std::vector<CondAst> parse_where();
  void parse_order_limit(Statement& st);
  void parse_create(Statement& st);
  void parse_insert(Statement& st);
  void parse_select(Statement& st);
  void parse_update(Statement& st);
  void parse_delete(Statement& st);
  int parse_type(ColumnDefAst& col);
};

}  // namespace mvdb
