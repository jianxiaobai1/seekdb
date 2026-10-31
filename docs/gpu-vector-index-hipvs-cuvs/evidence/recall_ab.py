import sys
def load(p, k):
    lines = [l.split() for l in open(p) if l.strip()]
    n = len(lines) // k
    return [lines[i*k:(i+1)*k] for i in range(n)]
def flat(g):
    return [x for row in g for x in row]
ex = load(sys.argv[1], 100)
ap = load(sys.argv[2], 100)
n = min(len(ex), len(ap))
a10 = sum(len(set(flat(ap[i][:10])) & set(flat(ex[i][:10]))) / 10.0 for i in range(n)) / n
a100 = sum(len(set(flat(ap[i])) & set(flat(ex[i]))) / 100.0 for i in range(n)) / n
print("recall@10=%.4f recall@100=%.4f queries=%d" % (a10, a100, n))
