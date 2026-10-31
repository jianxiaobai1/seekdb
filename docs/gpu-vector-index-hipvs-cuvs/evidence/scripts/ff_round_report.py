#!/usr/bin/env python3
"""Replay of one A/B round directory: window/recall/QPS per arm + VRAM peak.

usage: python3 ff_round_report.py <archive-dir-containing-rounds> <baseline-tag> <candidate-tag>

Expects, per round dir: one *.log with the RESULT/RECALL/QPS lines of both arms, and
<tag>.marks (load_start/load_end/ddl_start/ddl_end as CLOCK_MONOTONIC seconds) plus
<tag>.vram (`<seconds> card<N> <bytes_used>`) written by vram_sampler.sh.
Prints one line per round plus the mean window ratio and mean QPS geometric ratio.
"""
import glob
import os
import sys


def marks(path):
    m = {}
    for line in open(path):
        f = line.split()
        if len(f) >= 2:
            m[f[1]] = float(f[0])
    return m


def peak_in(vram, lo, hi, floor_bytes=3e8):
    """Max VRAM on our own card inside [lo, hi]. floor_bytes skips the ~28MB idle
    contexts that other tenants leave on the other cards of this shared instance."""
    best = 0.0
    card = ""
    for line in open(vram):
        f = line.split()
        if len(f) < 3:
            continue
        try:
            t = float(f[0])
            x = float(f[2])
        except ValueError:
            continue
        if lo <= t <= hi and x > best and x > floor_bytes:
            best = x
            card = f[1]
    return best, card


def peak_all(vram, floor_bytes=3e8):
    return peak_in(vram, 0.0, 1e18, floor_bytes)


def parse_log(path):
    d = {}
    for line in open(path):
        p = line.split()
        if not p or p[0] not in ("RESULT", "RECALL", "QPS"):
            continue
        kv = {}
        for tok in p[2:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv[k] = v
        arm = p[1]
        if p[0] == "RESULT":
            for key, name in (("window_ms", "_win"), ("ddl_ms", "_ddl"), ("first_q_ms", "_q1")):
                if key in kv:
                    d[arm + name] = int(kv[key])
        elif p[0] == "RECALL":
            d[arm + "_r10"] = float(kv["recall@10"])
            d[arm + "_r100"] = float(kv["recall@100"])
        elif p[0] == "QPS":
            d[arm + "_" + p[2]] = float(p[4])
    return d


def main():
    base, b_tag, c_tag = sys.argv[1], sys.argv[2], sys.argv[3]
    ratios = []
    qps = []
    for rd in sorted(glob.glob(os.path.join(base, "round*"))):
        logs = glob.glob(os.path.join(rd, "*.log"))
        if not logs:
            continue
        d = parse_log(logs[0])
        out = [os.path.basename(rd)]
        for tag in (b_tag, c_tag):
            mk = marks(os.path.join(rd, tag + ".marks"))
            vp = os.path.join(rd, tag + ".vram")
            w_pk, w_card = peak_in(vp, mk.get("ddl_start", 0), mk.get("ddl_end", 1e18))
            a_pk, _ = peak_all(vp)
            out.append(
                "%s ddl=%d q1=%d win=%d r10=%.4f r100=%.4f qtop10=%s qtop100=%s "
                "ddl_peak=%.3fGB arm_peak=%.3fGB(%s)"
                % (tag, d.get(tag + "_ddl", 0), d.get(tag + "_q1", 0), d.get(tag + "_win", 0),
                   d.get(tag + "_r10", 0), d.get(tag + "_r100", 0), d.get(tag + "_top10", 0),
                   d.get(tag + "_top100", 0), w_pk / 1073741824.0, a_pk / 1073741824.0, w_card)
            )
        bw = d.get(b_tag + "_win", 0)
        cw = d.get(c_tag + "_win", 0)
        if bw and cw:
            r = bw / cw
            ratios.append(r)
            out.append("win_ratio=%.4f" % r)
            v = [d.get(b_tag + "_top10", 0), d.get(c_tag + "_top10", 0),
                 d.get(b_tag + "_top100", 0), d.get(c_tag + "_top100", 0)]
            if all(v):
                g = (v[1] / v[0]) ** 0.5 * (v[3] / v[2]) ** 0.5
                qps.append(g)
                out.append("qps_geo=%.4f" % g)
        print(" | ".join(out))
    if ratios:
        print("MEAN win ratio over %d rounds = %.4f" % (len(ratios), sum(ratios) / len(ratios)))
    if qps:
        print("MEAN qps geo ratio = %.4f (min %.4f)" % (sum(qps) / len(qps), min(qps)))


if __name__ == "__main__":
    main()
