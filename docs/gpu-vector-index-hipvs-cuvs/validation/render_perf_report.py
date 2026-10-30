#!/usr/bin/env python3
"""Render the benchmark table of PERF_REPORT.md from bench_cagra_build.sh output.

sweep.tsv columns:
    setting  ddl_s  ddl_build_s  first_query_s  build_window_s  qps
    recall@<topk>  cuvs_build  cuvs_serve

ddl_build_s is the CAGRA build time reported by the adaptor itself
("built=true, build_us=N" in the observer log), i.e. the part of the DDL that
runs on the GPU. cuvs_build counts lazy rebuilds on the search path, which a
normal build never triggers.

Rows whose setting starts with "cpu_" are the lib=vsag CPU HNSW baseline. The
row whose setting contains --gpu-baseline (default inter256_niter100, i.e. the
build parameters hardcoded in the frozen base commit) is the same-parameters
reference for the snapshot/import work.

The block between the BENCH markers in the report is replaced in place, so the
report stays reproducible from the raw sweep output.
"""
import argparse
import pathlib
import re
import sys

BEGIN = "<!-- BENCH-TABLE:BEGIN -->"
END = "<!-- BENCH-TABLE:END -->"


def read_tsv(path):
    lines = [l.rstrip("\n") for l in path.open(encoding="utf-8") if l.strip()]
    if not lines:
        return [], []
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells)))
    return header, rows


def recall_key(header):
    return next((h for h in header if h.startswith("recall@")), None)


def pretty(setting):
    if setting.startswith("cpu_"):
        return "`lib=vsag` m=16 ef_construction=200（CPU HNSW）"
    # bench labels look like gpu_cuvs_deg128_inter256_niter100_nn_descent,
    # optionally followed by _itopk<N>_w<N> for the serving-knob rows.
    m = re.search(r"deg(\d+)_inter(\d+)_niter(\d+)_", setting)
    if not m:
        return f"`lib=cuvs` {setting}"
    deg, inter, niter = m.groups()
    algo = re.split(r"_itopk\d+|_w\d+", setting[m.end():])[0].rstrip("_")
    serve = []
    mi = re.search(r"_itopk(\d+)", setting)
    mw = re.search(r"_w(\d+)", setting)
    if mi:
        serve.append(f"itopk={mi.group(1)}")
    if mw:
        serve.append(f"width={mw.group(1)}")
    tail = (" " + " ".join(serve)) if serve else ""
    return f"`lib=cuvs` deg={deg} inter={inter} niter={niter} {algo}{tail}"


def fnum(value, nd=3):
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return value or "-"


def window(row):
    try:
        return float(row["build_window_s"])
    except (TypeError, ValueError):
        return 0.0


def speed(ref, row):
    if ref is None or ref is row or window(row) <= 0:
        return "-"
    return f"{window(ref) / window(row):.2f}x"


def cells(row, rk, cpu, gpu_ref):
    recall = row.get(rk) if rk else None
    out = [
        pretty(row["setting"]),
        fnum(row["ddl_s"]),
        fnum(row.get("ddl_build_s")),
        fnum(row["first_query_s"]),
        fnum(row["build_window_s"]),
        speed(cpu, row),
    ]
    if gpu_ref is not None:
        out.append(speed(gpu_ref, row) if row is not cpu else "-")
    out += [
        fnum(row.get("qps"), 1),
        f"{float(recall):.4f}" if recall else "-",
        row.get("cuvs_build", "-"),
    ]
    return "| " + " | ".join(out) + " |"


def table(rows, rk, cpu, gpu_ref, topk):
    head = ["配置", "建索引 DDL (s)", "其中 GPU 构图 (s)", "首查 (s)", "构建窗口 (s)", "加速 vs CPU"]
    if gpu_ref is not None:
        head.append("加速 vs 基线参数")
    head += [f"QPS", f"Recall@{topk}", "CAGRA build 次数"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in rows:
        lines.append(cells(row, rk, cpu, gpu_ref))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--restart", default=None)
    ap.add_argument("--meta", default="")
    ap.add_argument("--report", required=True)
    ap.add_argument("--gpu-baseline", default="inter256_niter100",
                    help="setting substring of the same-parameters GPU reference row")
    args = ap.parse_args()

    sweep = pathlib.Path(args.sweep)
    if not sweep.is_file():
        sys.exit(f"sweep file not found: {sweep}")
    header, rows = read_tsv(sweep)
    if not rows:
        sys.exit(f"no rows in {sweep}")
    rk = recall_key(header)
    topk = rk.split("@", 1)[1] if rk else "k"

    cpu = next((r for r in rows if r["setting"].startswith("cpu_")), None)
    gpu_ref = next((r for r in rows
                    if not r["setting"].startswith("cpu_")
                    and args.gpu_baseline
                    and args.gpu_baseline in r["setting"]), None)
    ordered = ([cpu] if cpu else []) + [r for r in rows if r is not cpu]

    out = []
    if args.meta:
        out.append(f"数据集：{args.meta}；L2 距离；ground truth 为精确 top-100 暴力计算。")
        out.append("")
    out += table(ordered, rk, cpu, gpu_ref, topk)
    out.append("")
    out.append("构建窗口 = `CREATE VECTOR INDEX` 耗时 + 首次查询耗时（首次查询触发索引加载/图导入）。"
               "所有行使用同一份数据、同一份 DDL 参数（`type=hnsw, m=16, ef_construction=200, "
               "ef_search=64`），每行单独起一个 observer，只切换 `lib=`、CAGRA 构建参数与服务参数。")
    if gpu_ref is not None and cpu is not None:
        out.append("「加速 vs 基线参数」对比的是冻结基线 `intermediate_graph_degree=256, "
                   "nn_descent_niter=100`；两者差值即本次快照/导入改造与 scratch buffer 改造的净收益，"
                   "不含构建参数调优的收益。")

    if cpu is not None and rk:
        cpu_recall = float(cpu.get(rk) or 0)
        ok = [r for r in rows if r is not cpu and float(r.get(rk) or 0) >= cpu_recall - 0.01]
        if ok:
            best = max(ok, key=lambda r: window(cpu) / window(r) if window(r) else 0)
            if gpu_ref is not None and best is gpu_ref:
                tail = "（即冻结基线参数本身：1 个百分点容差内没有更快的参数档达标）"
            elif gpu_ref is not None:
                tail = f"相对冻结基线参数 **{speed(gpu_ref, best)}**。"
            else:
                tail = f"Recall@{topk} {fnum(best.get(rk), 4)}。"
            out.append("")
            out.append(f"在 Recall@{topk} 不低于 CPU 基线 1 个百分点的前提下，最快的 GPU 配置是 "
                       f"{pretty(best['setting'])}：构建窗口 {fnum(best['build_window_s'])}s，"
                       f"相对 CPU 基线 **{speed(cpu, best)}**，" + tail)

    restart = pathlib.Path(args.restart) if args.restart else None
    if restart and restart.is_file():
        rh, rrows = read_tsv(restart)
        if rrows:
            out.append("")
            out.append("重启恢复验证（快照走图导入路径，不重新构建）：")
            out.append("")
            extra = "imported_graph_bytes" in rh
            out.append("| run | 重启后首查 (s) | 重启后 build 次数 |"
                       + (" 导入图大小 (B) |" if extra else ""))
            out.append("|---|---|---|" + ("---|" if extra else ""))
            for r in rrows:
                builds = r.get(rh[2], "-").split("=")[-1]
                line = f"| `{r[rh[0]]}` | {r.get(rh[1], '-')} | {builds} |"
                if extra:
                    line += f" {r.get('imported_graph_bytes', '-')} |"
                out.append(line)

    block = "\n".join(out).strip() + "\n"

    report = pathlib.Path(args.report)
    text = report.read_text(encoding="utf-8") if report.is_file() else ""
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        text = head + BEGIN + "\n" + block + END + tail
    else:
        text = (text.rstrip() + "\n\n" if text else "") + BEGIN + "\n" + block + END + "\n"
    report.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {report} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
