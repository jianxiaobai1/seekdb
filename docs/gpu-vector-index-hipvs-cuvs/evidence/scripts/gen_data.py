# Generates the proxy datasets used to measure the recall boundary of the CAGRA build
# knobs. 250k rows, 100 queries, exact top-100 ground truth.
#   python3 gen_data.py <out_dir> <dim> <mode> <seed>
# mode: uniform  = gaussian, L2-normalised (hard: neighbour distances concentrate)
#       cluster  = 500 cluster centres + sigma=0.35 noise, L2-normalised (realistic)
import numpy as np, os, sys
out, dim, mode, seed = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(seed)
n, nq = 250000, 100
if mode == 'cluster':
    k = 500
    cen = rng.normal(size=(k, dim)).astype('float32')
    lab = rng.integers(0, k, size=n)
    b = (cen[lab] + rng.normal(scale=0.35, size=(n, dim)).astype('float32'))
else:
    b = rng.normal(size=(n, dim)).astype('float32')
b /= np.linalg.norm(b, axis=1, keepdims=True)
idx = rng.choice(n, nq, replace=False)
q = b[idx] + rng.normal(scale=0.002, size=(nq, dim)).astype('float32')
q /= np.linalg.norm(q, axis=1, keepdims=True)
g = np.empty((nq, 100), dtype='int32')
for i in range(nq):
    d = np.sum((b - q[i]) ** 2, axis=1)
    g[i] = np.argsort(d)[:100]
b.tofile(out + '/base.f32')
q.tofile(out + '/query.f32')
g.tofile(out + '/gt.i32')
print('generated', out, n, dim, nq, mode)
