// MiniVectorDB - SQL lexer and parser for the competition subset.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <climits>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace mvd {

struct SqlError : std::runtime_error {
    std::string code;
    SqlError(std::string c, const std::string& m)
        : std::runtime_error(m), code(std::move(c)) {}
};

inline char lower_char(char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }

inline std::string lower_copy(std::string s) {
    for (auto& c : s) c = lower_char(c);
    return s;
}

inline bool ieq(const std::string& a, const char* b) {
    size_t n = std::strlen(b);
    if (a.size() != n) return false;
    for (size_t i = 0; i < n; ++i) {
        if (lower_char(a[i]) != lower_char(b[i])) return false;
    }
    return true;
}

inline std::string trim_copy(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

struct Token {
    enum Kind { END, IDENT, NUMBER, STRING, PUNCT } kind = END;
    std::string text;
    size_t pos = 0;
};

class Lexer {
public:
    explicit Lexer(const std::string& src) : s_(src) {}

    std::vector<Token> run() {
        std::vector<Token> out;
        while (true) {
            skip_space_and_comments();
            if (i_ >= s_.size()) {
                out.push_back(Token{Token::END, "", i_});
                break;
            }
            const size_t start = i_;
            const unsigned char c = static_cast<unsigned char>(s_[i_]);
            if (std::isalpha(c) || c == '_' || c == '$') {
                ++i_;
                while (i_ < s_.size()) {
                    unsigned char d = static_cast<unsigned char>(s_[i_]);
                    if (std::isalnum(d) || d == '_' || d == '$') ++i_;
                    else break;
                }
                out.push_back(Token{Token::IDENT, s_.substr(start, i_ - start), start});
            } else if (std::isdigit(c) || (c == '.' && i_ + 1 < s_.size() &&
                                           std::isdigit(static_cast<unsigned char>(s_[i_ + 1])))) {
                read_number(out, start);
            } else if (c == '\'' || c == '"') {
                read_string(out, start);
            } else if (c == '`') {
                read_backtick_ident(out, start);
            } else {
                read_punct(out, start);
            }
        }
        return out;
    }

private:
    const std::string& s_;
    size_t i_ = 0;

    void skip_space_and_comments() {
        while (i_ < s_.size()) {
            unsigned char c = static_cast<unsigned char>(s_[i_]);
            if (std::isspace(c)) {
                ++i_;
            } else if (c == '-' && i_ + 1 < s_.size() && s_[i_ + 1] == '-') {
                i_ += 2;
                while (i_ < s_.size() && s_[i_] != '\n') ++i_;
            } else if (c == '#') {
                ++i_;
                while (i_ < s_.size() && s_[i_] != '\n') ++i_;
            } else if (c == '/' && i_ + 1 < s_.size() && s_[i_ + 1] == '*') {
                i_ += 2;
                while (i_ + 1 < s_.size() && !(s_[i_] == '*' && s_[i_ + 1] == '/')) ++i_;
                if (i_ + 1 < s_.size()) i_ += 2;
                else throw SqlError("syntax_error", "unterminated block comment");
            } else {
                break;
            }
        }
    }

    void read_number(std::vector<Token>& out, size_t start) {
        bool seen_dot = false, seen_exp = false;
        if (s_[i_] == '.') { seen_dot = true; ++i_; }
        while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) ++i_;
        if (i_ < s_.size() && s_[i_] == '.' && !seen_dot) {
            seen_dot = true;
            ++i_;
            while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) ++i_;
        }
        if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E') && !seen_exp) {
            size_t save = i_;
            ++i_;
            if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
            size_t digits_start = i_;
            while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) ++i_;
            if (i_ == digits_start) i_ = save;  // not an exponent, restore
            else seen_exp = true;
        }
        out.push_back(Token{Token::NUMBER, s_.substr(start, i_ - start), start});
    }

    void read_string(std::vector<Token>& out, size_t start) {
        const char quote = s_[i_++];
        std::string value;
        bool closed = false;
        while (i_ < s_.size()) {
            char c = s_[i_++];
            if (c == quote) {
                if (i_ < s_.size() && s_[i_] == quote) {
                    value.push_back(quote);
                    ++i_;
                    continue;
                }
                closed = true;
                break;
            }
            if (c == '\\' && i_ < s_.size()) {
                char e = s_[i_++];
                switch (e) {
                    case 'n': value.push_back('\n'); break;
                    case 'r': value.push_back('\r'); break;
                    case 't': value.push_back('\t'); break;
                    case 'b': value.push_back('\b'); break;
                    case '0': value.push_back('\0'); break;
                    case 'Z': value.push_back('\x1a'); break;
                    case '%': value.push_back('\\'); value.push_back('%'); break;
                    case '_': value.push_back('\\'); value.push_back('_'); break;
                    default: value.push_back(e); break;
                }
            } else {
                value.push_back(c);
            }
        }
        if (!closed) throw SqlError("syntax_error", "unterminated string literal");
        out.push_back(Token{Token::STRING, std::move(value), start});
    }

    void read_backtick_ident(std::vector<Token>& out, size_t start) {
        ++i_;  // opening backtick
        std::string value;
        bool closed = false;
        while (i_ < s_.size()) {
            char c = s_[i_++];
            if (c == '`') {
                if (i_ < s_.size() && s_[i_] == '`') {
                    value.push_back('`');
                    ++i_;
                    continue;
                }
                closed = true;
                break;
            }
            value.push_back(c);
        }
        if (!closed) throw SqlError("syntax_error", "unterminated quoted identifier");
        out.push_back(Token{Token::IDENT, std::move(value), start});
    }

    void read_punct(std::vector<Token>& out, size_t start) {
        char c = s_[i_];
        auto push = [&](size_t len) {
            out.push_back(Token{Token::PUNCT, s_.substr(start, len), start});
            i_ += len;
        };
        if (i_ + 1 < s_.size()) {
            char d = s_[i_ + 1];
            if ((c == '<' && (d == '=' || d == '>')) ||
                (c == '>' && d == '=') ||
                (c == '!' && d == '=') ||
                (c == ':' && d == '=')) {
                push(2);
                return;
            }
        }
        static const std::string allowed = "(),;.*=<>[]+-/%";
        if (allowed.find(c) != std::string::npos) {
            push(1);
            return;
        }
        throw SqlError("syntax_error", std::string("unexpected character '") + c + "'");
    }
};

struct Literal {
    enum Kind { INT, NUM, STR, VEC, NUL } kind = INT;
    int64_t i = 0;
    double d = 0.0;
    std::string s;
    std::vector<double> v;
};

struct SelectItem {
    bool star = false;
    bool dist = false;
    std::string col;
    std::vector<double> qvec;
    std::string alias;
    std::string raw;
};

struct Cond {
    std::string col;
    // 0: =, 1: <, 2: <=, 3: >, 4: >=, 5: !=
    int op = 0;
    Literal lit;
};

struct OrderKey {
    std::string name;
    bool desc = false;
    // ORDER BY L2_DISTANCE(col, '[...]') without selecting it.
    bool dist = false;
    std::string col;
    std::vector<double> qvec;
};

struct ColDefAst {
    std::string name;
    int type = 0;   // 0 BIGINT, 1 VARCHAR, 2 VECTOR
    int64_t len = 0;
    bool pk = false;
};

struct Stmt {
    enum Type {
        SELECT, INSERT, UPDATE, DELETE_T, CREATE_TABLE, CREATE_INDEX,
        BEGIN_T, COMMIT_T, ROLLBACK_T, EMPTY, UNSUPPORTED
    } type = EMPTY;

    // SELECT
    std::string table;
    std::vector<SelectItem> items;
    std::vector<Cond> conds;
    std::vector<OrderKey> order;
    bool has_limit = false;
    int64_t limit = 0;

    // INSERT
    std::vector<std::string> col_list;
    std::vector<std::vector<Literal>> tuples;

    // UPDATE
    std::vector<std::pair<std::string, Literal>> assigns;

    // CREATE TABLE
    std::vector<ColDefAst> cols;
    bool if_not_exists = false;

    // CREATE INDEX
    std::string index_name;
    std::string index_col;

    std::string raw;
};

class Parser {
public:
    explicit Parser(const std::string& src)
        : src_(src), toks_(Lexer(src).run()), p_(0) {}

    bool next(Stmt& out) {
        while (kind() == Token::PUNCT && text() == ";") ++p_;
        if (kind() == Token::END) return false;
        out = parse_one();
        if (kind() == Token::PUNCT && text() == ";") {
            ++p_;
        } else if (kind() != Token::END) {
            throw SqlError("syntax_error", "unexpected token after statement: " + text());
        }
        return true;
    }

private:
    const std::string& src_;
    std::vector<Token> toks_;
    size_t p_ = 0;

    const Token& cur() const { return toks_[p_]; }
    Token::Kind kind() const { return cur().kind; }
    const std::string& text() const { return cur().text; }
    bool at_end() const { return kind() == Token::END; }

    bool is_kw(const char* kw) const { return kind() == Token::IDENT && ieq(text(), kw); }

    bool eat_kw(const char* kw) {
        if (is_kw(kw)) { ++p_; return true; }
        return false;
    }

    bool eat_punct(const char* p) {
        if (kind() == Token::PUNCT && text() == p) { ++p_; return true; }
        return false;
    }

    void expect_punct(const char* p) {
        if (!eat_punct(p)) throw SqlError("syntax_error", std::string("expected '") + p + "'");
    }

    std::string expect_ident() {
        if (kind() != Token::IDENT) throw SqlError("syntax_error", "expected identifier");
        std::string s = text();
        ++p_;
        return s;
    }

    int64_t expect_uint() {
        bool neg = false;
        if (eat_punct("-")) neg = true;
        else eat_punct("+");
        if (kind() != Token::NUMBER) throw SqlError("syntax_error", "expected integer");
        int64_t value = parse_int64(text());
        ++p_;
        if (neg) value = -value;
        if (value < 0) throw SqlError("syntax_error", "expected non-negative integer");
        return value;
    }

    static int64_t parse_int64(const std::string& t) {
        errno = 0;
        char* end = nullptr;
        long long v = std::strtoll(t.c_str(), &end, 10);
        if (errno == ERANGE || end == t.c_str() || *end != '\0') {
            throw SqlError("value_out_of_range", "integer out of range: " + t);
        }
        return static_cast<int64_t>(v);
    }

    static double parse_double_strict(const std::string& t) {
        errno = 0;
        char* end = nullptr;
        double v = std::strtod(t.c_str(), &end);
        if (end == t.c_str() || *end != '\0') {
            throw SqlError("syntax_error", "invalid number: " + t);
        }
        return v;
    }

    Stmt parse_one() {
        if (is_kw("create")) return parse_create();
        if (is_kw("select")) return parse_select();
        if (is_kw("insert")) return parse_insert();
        if (is_kw("update")) return parse_update();
        if (is_kw("delete")) return parse_delete();
        if (is_kw("begin")) {
            ++p_;
            eat_kw("work");
            eat_kw("transaction");
            Stmt st; st.type = Stmt::BEGIN_T; return st;
        }
        if (is_kw("start")) {
            ++p_;
            if (eat_kw("transaction")) {
                Stmt st; st.type = Stmt::BEGIN_T; return st;
            }
            throw SqlError("unsupported_sql", "unsupported START statement");
        }
        if (is_kw("commit")) {
            ++p_;
            eat_kw("work");
            Stmt st; st.type = Stmt::COMMIT_T; return st;
        }
        if (is_kw("rollback")) {
            ++p_;
            eat_kw("work");
            Stmt st; st.type = Stmt::ROLLBACK_T; return st;
        }
        // Anything else is a valid SQL statement outside the supported subset.
        Stmt st;
        st.type = Stmt::UNSUPPORTED;
        size_t start = cur().pos;
        while (!at_end() && !(kind() == Token::PUNCT && text() == ";")) ++p_;
        size_t end = at_end() ? src_.size() : cur().pos;
        st.raw = trim_copy(src_.substr(start, end - start));
        return st;
    }

    Stmt parse_create() {
        ++p_;  // CREATE
        if (eat_kw("table")) return parse_create_table_rest();
        if (eat_kw("vector")) {
            if (!eat_kw("index")) throw SqlError("unsupported_sql", "only CREATE VECTOR INDEX is supported");
            return parse_create_index_rest();
        }
        if (eat_kw("index")) return parse_create_index_rest();
        throw SqlError("unsupported_sql", "unsupported CREATE statement");
    }

    Stmt parse_create_table_rest() {
        Stmt st;
        st.type = Stmt::CREATE_TABLE;
        if (is_kw("if")) {
            ++p_;
            if (!eat_kw("not")) throw SqlError("syntax_error", "expected NOT");
            if (!eat_kw("exists")) throw SqlError("syntax_error", "expected EXISTS");
            st.if_not_exists = true;
        }
        st.table = expect_ident();
        expect_punct("(");
        while (true) {
            if (is_kw("primary")) {
                ++p_;
                if (!eat_kw("key")) throw SqlError("syntax_error", "expected KEY");
                expect_punct("(");
                std::string pkcol = expect_ident();
                expect_punct(")");
                bool found = false;
                for (auto& c : st.cols) {
                    if (ieq(c.name, pkcol.c_str())) { c.pk = true; found = true; }
                }
                if (!found) throw SqlError("syntax_error", "PRIMARY KEY references unknown column");
            } else {
                ColDefAst cd;
                cd.name = expect_ident();
                if (eat_kw("bigint")) {
                    cd.type = 0;
                } else if (eat_kw("varchar")) {
                    cd.type = 1;
                    expect_punct("(");
                    cd.len = expect_uint();
                    expect_punct(")");
                } else if (eat_kw("vector")) {
                    cd.type = 2;
                    expect_punct("(");
                    cd.len = expect_uint();
                    expect_punct(")");
                } else {
                    throw SqlError("unsupported_sql", "unsupported column type");
                }
                while (true) {
                    if (eat_kw("primary")) {
                        if (!eat_kw("key")) throw SqlError("syntax_error", "expected KEY");
                        cd.pk = true;
                    } else if (eat_kw("not")) {
                        if (!eat_kw("null")) throw SqlError("syntax_error", "expected NULL");
                    } else if (eat_kw("null")) {
                        throw SqlError("unsupported_sql", "nullable columns are not supported");
                    } else {
                        break;
                    }
                }
                st.cols.push_back(std::move(cd));
            }
            if (eat_punct(",")) continue;
            break;
        }
        expect_punct(")");
        return st;
    }

    Stmt parse_create_index_rest() {
        Stmt st;
        st.type = Stmt::CREATE_INDEX;
        st.index_name = expect_ident();
        if (!eat_kw("on")) throw SqlError("syntax_error", "expected ON");
        st.table = expect_ident();
        expect_punct("(");
        st.index_col = expect_ident();
        expect_punct(")");
        return st;
    }

    Stmt parse_select() {
        ++p_;  // SELECT
        Stmt st;
        st.type = Stmt::SELECT;
        while (true) {
            SelectItem item;
            if (eat_punct("*")) {
                item.star = true;
            } else {
                if (kind() != Token::IDENT) throw SqlError("syntax_error", "expected select expression");
                std::string name = expect_ident();
                if (kind() == Token::PUNCT && text() == "(") {
                    if (ieq(name, "l2_distance")) {
                        ++p_;
                        item.dist = true;
                        item.col = expect_ident();
                        expect_punct(",");
                        item.qvec = parse_vector_literal();
                        expect_punct(")");
                    } else {
                        throw SqlError("unsupported_sql", "unsupported function: " + name);
                    }
                } else {
                    item.col = name;
                }
                if (eat_kw("as")) {
                    item.alias = expect_ident();
                } else if (kind() == Token::IDENT) {
                    if (is_kw("from") || is_kw("where") || is_kw("order") || is_kw("limit") ||
                        is_kw("and") || is_kw("asc") || is_kw("desc")) {
                        // clause keyword, no alias
                    } else {
                        item.alias = expect_ident();
                    }
                }
            }
            item.raw = item.star ? "*" : (item.dist ? ("L2_DISTANCE(" + item.col + ")") : item.col);
            st.items.push_back(std::move(item));
            if (eat_punct(",")) continue;
            break;
        }
        if (!eat_kw("from")) throw SqlError("syntax_error", "expected FROM");
        st.table = expect_ident();
        if (eat_kw("where")) st.conds = parse_where_conjunction();
        if (is_kw("order")) {
            ++p_;
            if (!eat_kw("by")) throw SqlError("syntax_error", "expected BY");
            while (true) {
                OrderKey ok;
                if (is_kw("l2_distance")) {
                    ++p_;
                    expect_punct("(");
                    ok.dist = true;
                    ok.col = expect_ident();
                    expect_punct(",");
                    ok.qvec = parse_vector_literal();
                    expect_punct(")");
                } else {
                    ok.name = expect_ident();
                }
                if (eat_kw("asc")) ok.desc = false;
                else if (eat_kw("desc")) ok.desc = true;
                st.order.push_back(std::move(ok));
                if (!eat_punct(",")) break;
            }
        }
        if (eat_kw("limit")) {
            st.has_limit = true;
            st.limit = expect_uint();
        }
        return st;
    }

    std::vector<Cond> parse_where_conjunction() {
        std::vector<Cond> out;
        parse_pred_factor(out);
        while (true) {
            if (eat_kw("and")) {
                parse_pred_factor(out);
            } else if (is_kw("or")) {
                throw SqlError("unsupported_sql", "OR is not supported");
            } else {
                break;
            }
        }
        return out;
    }

    void parse_pred_factor(std::vector<Cond>& out) {
        if (eat_punct("(")) {
            auto nested = parse_where_conjunction();
            expect_punct(")");
            out.insert(out.end(), nested.begin(), nested.end());
            return;
        }
        if (is_kw("or")) throw SqlError("unsupported_sql", "OR is not supported");
        Cond c;
        c.col = expect_ident();
        if (eat_punct("=")) c.op = 0;
        else if (eat_punct("<=")) c.op = 2;
        else if (eat_punct(">=")) c.op = 4;
        else if (eat_punct("<>")) c.op = 5;
        else if (eat_punct("!=")) c.op = 5;
        else if (eat_punct("<")) c.op = 1;
        else if (eat_punct(">")) c.op = 3;
        else throw SqlError("syntax_error", "expected comparison operator");
        c.lit = parse_literal();
        out.push_back(std::move(c));
    }

    Stmt parse_insert() {
        ++p_;  // INSERT
        if (!eat_kw("into")) throw SqlError("syntax_error", "expected INTO");
        Stmt st;
        st.type = Stmt::INSERT;
        st.table = expect_ident();
        if (eat_punct("(")) {
            while (true) {
                st.col_list.push_back(expect_ident());
                if (!eat_punct(",")) break;
            }
            expect_punct(")");
        }
        if (!eat_kw("values")) throw SqlError("syntax_error", "expected VALUES");
        while (true) {
            expect_punct("(");
            std::vector<Literal> tuple;
            while (true) {
                tuple.push_back(parse_literal());
                if (!eat_punct(",")) break;
            }
            expect_punct(")");
            st.tuples.push_back(std::move(tuple));
            if (!eat_punct(",")) break;
        }
        return st;
    }

    Stmt parse_update() {
        ++p_;  // UPDATE
        Stmt st;
        st.type = Stmt::UPDATE;
        st.table = expect_ident();
        if (!eat_kw("set")) throw SqlError("syntax_error", "expected SET");
        while (true) {
            std::string col = expect_ident();
            if (!eat_punct("=")) throw SqlError("syntax_error", "expected '='");
            Literal lit = parse_literal();
            st.assigns.emplace_back(col, std::move(lit));
            if (!eat_punct(",")) break;
        }
        if (eat_kw("where")) st.conds = parse_where_conjunction();
        return st;
    }

    Stmt parse_delete() {
        ++p_;  // DELETE
        if (!eat_kw("from")) throw SqlError("syntax_error", "expected FROM");
        Stmt st;
        st.type = Stmt::DELETE_T;
        st.table = expect_ident();
        if (eat_kw("where")) st.conds = parse_where_conjunction();
        return st;
    }

    Literal parse_literal() {
        Literal lit;
        if (kind() == Token::STRING) {
            lit.kind = Literal::STR;
            lit.s = text();
            ++p_;
            return lit;
        }
        if (is_kw("null")) {
            ++p_;
            lit.kind = Literal::NUL;
            return lit;
        }
        if (kind() == Token::PUNCT && text() == "[") {
            lit.kind = Literal::VEC;
            lit.v = parse_vector_literal();
            return lit;
        }
        bool neg = false, plus = false;
        if (eat_punct("-")) neg = true;
        else if (eat_punct("+")) plus = true;
        if (kind() != Token::NUMBER) {
            throw SqlError("syntax_error", "expected literal value");
        }
        const std::string t = text();
        ++p_;
        bool is_int = t.find_first_of(".eE") == std::string::npos;
        if (is_int) {
            lit.kind = Literal::INT;
            if (neg && t == "9223372036854775808") {
                lit.i = INT64_MIN;
            } else {
                int64_t v = parse_int64(t);
                lit.i = neg ? -v : v;
            }
        } else {
            lit.kind = Literal::NUM;
            double v = parse_double_strict(t);
            lit.d = neg ? -v : v;
            if (!std::isfinite(lit.d)) throw SqlError("invalid_vector", "non-finite numeric literal");
        }
        (void)plus;
        return lit;
    }

    std::vector<double> parse_vector_literal() {
        std::vector<double> out;
        if (kind() == Token::STRING) {
            out = parse_vector_text(text());
            ++p_;
            return out;
        }
        expect_punct("[");
        if (eat_punct("]")) return out;
        while (true) {
            bool neg = false;
            if (eat_punct("-")) neg = true;
            else eat_punct("+");
            if (kind() != Token::NUMBER) throw SqlError("invalid_vector", "expected vector element");
            double v = parse_double_strict(text());
            ++p_;
            if (!std::isfinite(v)) throw SqlError("invalid_vector", "non-finite vector element");
            out.push_back(neg ? -v : v);
            if (eat_punct(",")) continue;
            break;
        }
        expect_punct("]");
        return out;
    }

public:
    // Parse the textual form of a vector, e.g. "[1, 2.5, -3e-2]".
    static std::vector<double> parse_vector_text(const std::string& t) {
        size_t i = 0;
        auto skip = [&]() {
            while (i < t.size() && std::isspace(static_cast<unsigned char>(t[i]))) ++i;
        };
        skip();
        if (i >= t.size() || t[i] != '[') throw SqlError("invalid_vector", "vector must start with '['");
        ++i;
        std::vector<double> out;
        skip();
        if (i < t.size() && t[i] == ']') {
            ++i;
            skip();
            if (i != t.size()) throw SqlError("invalid_vector", "trailing characters in vector");
            return out;
        }
        while (true) {
            skip();
            const char* begin = t.c_str() + i;
            char* end = nullptr;
            errno = 0;
            double v = std::strtod(begin, &end);
            if (end == begin) throw SqlError("invalid_vector", "malformed vector element");
            if (!std::isfinite(v)) throw SqlError("invalid_vector", "non-finite vector element");
            i += static_cast<size_t>(end - begin);
            out.push_back(v);
            skip();
            if (i < t.size() && t[i] == ',') { ++i; continue; }
            if (i < t.size() && t[i] == ']') { ++i; break; }
            throw SqlError("invalid_vector", "malformed vector literal");
        }
        skip();
        if (i != t.size()) throw SqlError("invalid_vector", "trailing characters in vector");
        return out;
    }
};

}  // namespace mvd
