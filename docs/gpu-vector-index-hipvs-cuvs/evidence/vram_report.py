import sys
P = "/workspace/ab256"
for tag in sys.argv[1:]:
    marks = {}
    for ln in open(P + "/" + tag + ".marks"):
        t, k = ln.split()
        marks[k] = float(t)
    lo, hi = marks["ddl_start"], marks["ddl_end"]
    allw = {}
    win = {}
    n = 0
    for ln in open(P + "/" + tag + ".vram"):
        p = ln.split()
        if len(p) != 3:
            continue
        t, card, v = float(p[0]), p[1], int(p[2])
        n += 1
        allw[card] = max(allw.get(card, 0), v)
        if lo - 0.5 <= t <= hi + 0.5:
            win[card] = max(win.get(card, 0), v)
    idle = max(allw.values())
    peak = max(win.values()) if win else 0
    print("%s samples=%d ddl_window_peak=%.3fGB arm_max=%.3fGB cards=%s" % (
        tag, n, peak / 2.0**30, idle / 2.0**30, sorted(win.items(), key=lambda x: -x[1])[:3]))