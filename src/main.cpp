// MiniVectorDB - JSONL stdin/stdout protocol entry point.
#include "engine.hpp"
#include "nlohmann/json.hpp"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <poll.h>
#include <string>
#include <unistd.h>

using json = nlohmann::json;
using namespace mvd;

namespace {

volatile std::sig_atomic_t g_terminate = 0;

void on_signal(int) { g_terminate = 1; }

bool write_all(int fd, const char* data, size_t size) {
    while (size > 0) {
        ssize_t n = ::write(fd, data, size);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        data += n;
        size -= static_cast<size_t>(n);
    }
    return true;
}

void emit(const json& message) {
    std::string line = message.dump(-1, ' ', false, json::error_handler_t::replace);
    line.push_back('\n');
    write_all(STDOUT_FILENO, line.data(), line.size());
    // stdout is unbuffered through write(); nothing else to flush.
}

json cell_to_json(const Cell& c) {
    switch (c.kind) {
        case Cell::INT: return json(c.i);
        case Cell::STR: return json(c.s);
        case Cell::DBL: {
            double d = c.d;
            if (!std::isfinite(d)) d = 0.0;
            // Integral doubles are emitted as JSON integers: the public
            // comparator accepts an integer wherever a float is expected,
            // but rejects a float where an integer is expected.
            if (std::trunc(d) == d && std::fabs(d) <= 9.0e15) return json(static_cast<long long>(d));
            return json(d);
        }
        case Cell::VEC: {
            json arr = json::array();
            for (float f : c.v) {
                double d = static_cast<double>(f);
                if (std::isfinite(d) && std::trunc(d) == d && std::fabs(d) <= 9.0e15) {
                    arr.push_back(json(static_cast<long long>(d)));
                } else {
                    arr.push_back(json(std::isfinite(d) ? d : 0.0));
                }
            }
            return arr;
        }
    }
    return json(nullptr);
}

void emit_error(const json& id, const std::string& code, const std::string& message) {
    json resp;
    resp["id"] = id;
    resp["ok"] = false;
    resp["error"] = json{{"code", code.empty() ? "error" : code}, {"message", message}};
    emit(resp);
}

// Returns true when the process should exit after this line.
bool handle_line(Engine& engine, const std::string& line) {
    json request;
    try {
        request = json::parse(line);
    } catch (const std::exception& e) {
        emit_error(json(nullptr), "invalid_request", e.what());
        return false;
    }
    if (!request.is_object()) {
        emit_error(json(nullptr), "invalid_request", "request must be a JSON object");
        return false;
    }

    json id = request.contains("id") ? request["id"] : json(nullptr);

    if (!request.contains("op") || !request["op"].is_string()) {
        emit_error(id, "invalid_request", "missing string field 'op'");
        return false;
    }
    const std::string op = request["op"].get<std::string>();

    if (op == "shutdown") {
        engine.close_all();
        json resp;
        resp["id"] = id;
        resp["ok"] = true;
        emit(resp);
        return true;
    }

    if (op != "execute") {
        emit_error(id, "unsupported_operation", "unsupported op: " + op);
        return false;
    }

    if (!request.contains("sql") || !request["sql"].is_string()) {
        emit_error(id, "invalid_request", "missing string field 'sql'");
        return false;
    }
    const std::string sql = request["sql"].get<std::string>();
    std::string mode = "exact";
    if (request.contains("search_mode")) {
        if (!request["search_mode"].is_string()) {
            emit_error(id, "invalid_request", "search_mode must be a string");
            return false;
        }
        mode = request["search_mode"].get<std::string>();
    }

    Result result = engine.execute(sql, mode);

    json resp;
    resp["id"] = id;
    resp["ok"] = result.ok;
    if (!result.ok) {
        resp["error"] = json{{"code", result.code.empty() ? "error" : result.code},
                             {"message", result.message}};
    } else {
        if (result.has_affected) resp["affected_rows"] = result.affected;
        if (result.has_rows) {
            resp["columns"] = result.columns;
            json rows = json::array();
            for (const auto& row : result.rows) {
                json jr = json::array();
                for (const auto& cell : row) jr.push_back(cell_to_json(cell));
                rows.push_back(std::move(jr));
            }
            resp["rows"] = std::move(rows);
        }
    }
    emit(resp);
    return false;
}

std::string parse_data_dir(int argc, char** argv) {
    std::string dir = "/data";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--data-dir" && i + 1 < argc) {
            dir = argv[++i];
        } else if (a.rfind("--data-dir=", 0) == 0) {
            dir = a.substr(11);
        }
    }
    if (!dir.empty() && dir.back() == '/') dir.pop_back();
    return dir;
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);
    std::signal(SIGTERM, on_signal);
    std::signal(SIGINT, on_signal);

    const std::string data_dir = parse_data_dir(argc, argv);
    Engine engine(data_dir);
    try {
        engine.recover();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "recovery failed: %s\n", e.what());
        return 1;
    }

    emit(json{{"type", "ready"}});

    std::string buffer;
    std::vector<char> chunk(65536);
    while (true) {
        if (g_terminate) {
            engine.close_all();
            return 0;
        }
        size_t newline;
        while ((newline = buffer.find('\n')) != std::string::npos) {
            std::string line = buffer.substr(0, newline);
            buffer.erase(0, newline + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            bool exit_now = false;
            try {
                exit_now = handle_line(engine, line);
            } catch (const std::exception& e) {
                // Never let an unexpected exception kill the connection; the
                // client still needs one response per request.
                std::fprintf(stderr, "internal error: %s\n", e.what());
                emit_error(json(nullptr), "internal_error", e.what());
            }
            if (exit_now) return 0;
        }

        struct pollfd pfd;
        pfd.fd = STDIN_FILENO;
        pfd.events = POLLIN;
        pfd.revents = 0;
        int pr = ::poll(&pfd, 1, 200);
        if (pr < 0) {
            if (errno == EINTR) continue;
            engine.close_all();
            return 1;
        }
        if (pr == 0) continue;
        ssize_t n = ::read(STDIN_FILENO, chunk.data(), chunk.size());
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            engine.close_all();
            return 1;
        }
        if (n == 0) {
            engine.close_all();
            return 0;
        }
        buffer.append(chunk.data(), static_cast<size_t>(n));
        if (buffer.size() > 64ull * 1024 * 1024) {
            std::fprintf(stderr, "request line too large\n");
            return 1;
        }
    }
}
