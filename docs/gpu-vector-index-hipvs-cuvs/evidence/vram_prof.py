import sys
f = sys.argv[1]
card = sys.argv[2] if len(sys.argv) > 2 else None
rows = []
for ln in open(f):
    p = ln.split()
    if len(p) != 3:
        continue
    if card and p[1] != card:
        continue
    rows.append((float(p[0]), p[1], int(p[2])))
if not rows:
    print("no rows")
    sys.exit(0)
t0 = rows[0][0]
peak = max(r[2] for r in rows)
ti = max(range(len(rows)), key=lambda i: rows[i][2])
print("samples=%d peak=%.3fGB at t+%0.1fs span=%.1fs" % (len(rows), peak / 2.0**30, rows[ti][0] - t0, rows[-1][0] - t0))
buckets = {}
for t, c, v in rows:
    b = int(t - t0)
    buckets[b] = max(buckets.get(b, 0), v)
print(" ".join("%ds:%.2f" % (b, buckets[b] / 2.0**30) for b in sorted(buckets)))