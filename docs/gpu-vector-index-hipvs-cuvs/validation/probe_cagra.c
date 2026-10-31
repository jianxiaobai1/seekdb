#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#include <cuvs/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

static double now_ms(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static unsigned long long rng_state = 88172645463325252ULL;
static double urand(void)
{
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (double)(rng_state % 1000000ULL) / 1000000.0;
}

static void init_tensor(DLManagedTensor* t, void* data, long* shape, int code, int bits)
{
  memset(t, 0, sizeof(*t));
  t->dl_tensor.data = data;
  t->dl_tensor.device.device_type = kDLCPU;
  t->dl_tensor.device.device_id = 0;
  t->dl_tensor.ndim = 2;
  t->dl_tensor.dtype.code = (uint8_t)code;
  t->dl_tensor.dtype.bits = (uint8_t)bits;
  t->dl_tensor.dtype.lanes = 1;
  t->dl_tensor.shape = shape;
  t->dl_tensor.strides = NULL;
  t->dl_tensor.byte_offset = 0;
  t->manager_ctx = NULL;
  t->deleter = NULL;
}

int main(int argc, char** argv)
{
  long n = 250000, dim = 128, nq = 200, k = 10;
  long lists = 0;
  int deg = 128, inter = 128, niter = 32;
  int algo = 0;      /* 0 nn_descent, 1 ivf_pq, 2 auto */
  double refine = -1.0;
  int pq_bits = 0, pq_dim = 0, vq = 0;
  int itopk = 512, width = 4;
  int cluster = 1;
  int i;

  for (i = 1; i < argc; i++) {
    const char* a = argv[i];
    if (!strcmp(a, "--n") && i + 1 < argc) n = atol(argv[++i]);
    else if (!strcmp(a, "--dim") && i + 1 < argc) dim = atol(argv[++i]);
    else if (!strcmp(a, "--nq") && i + 1 < argc) nq = atol(argv[++i]);
    else if (!strcmp(a, "--k") && i + 1 < argc) k = atol(argv[++i]);
    else if (!strcmp(a, "--deg") && i + 1 < argc) deg = atoi(argv[++i]);
    else if (!strcmp(a, "--inter") && i + 1 < argc) inter = atoi(argv[++i]);
    else if (!strcmp(a, "--niter") && i + 1 < argc) niter = atoi(argv[++i]);
    else if (!strcmp(a, "--algo")) { const char* v = argv[++i]; algo = !strcmp(v, "ivf") ? 1 : (!strcmp(v, "auto") ? 2 : 0); }
    else if (!strcmp(a, "--refine") && i + 1 < argc) refine = atof(argv[++i]);
    else if (!strcmp(a, "--pq") && i + 1 < argc) pq_bits = atoi(argv[++i]);
    else if (!strcmp(a, "--pqdim") && i + 1 < argc) pq_dim = atoi(argv[++i]);
    else if (!strcmp(a, "--vq") && i + 1 < argc) vq = atoi(argv[++i]);
    else if (!strcmp(a, "--itopk") && i + 1 < argc) itopk = atoi(argv[++i]);
    else if (!strcmp(a, "--width") && i + 1 < argc) width = atoi(argv[++i]);
    else if (!strcmp(a, "--lists") && i + 1 < argc) lists = atol(argv[++i]);
    else if (!strcmp(a, "--cluster") && i + 1 < argc) cluster = atoi(argv[++i]);
    else if (!strcmp(a, "--seed") && i + 1 < argc) rng_state = strtoull(argv[++i], NULL, 10);
  }

  size_t base_elems = (size_t)n * (size_t)dim;
  size_t q_elems = (size_t)nq * (size_t)dim;
  float* base = (float*)malloc(base_elems * sizeof(float));
  float* queries = (float*)malloc(q_elems * sizeof(float));
  unsigned* gt = (unsigned*)malloc((size_t)nq * k * sizeof(unsigned));
  if (!base || !queries || !gt) { fprintf(stderr, "oom\n"); return 1; }

  {
    long nc = 0;
    float* centers = NULL;
    if (cluster) {
      nc = 200;
      centers = (float*)malloc((size_t)nc * dim * sizeof(float));
      for (long j = 0; j < nc * dim; j++) centers[j] = (float)urand();
      for (long r = 0; r < n; r++) {
        long c = (long)(urand() * nc) % nc;
        for (long d = 0; d < dim; d++)
          base[r * dim + d] = centers[c * dim + d] + (float)(urand() - 0.5) * 0.2f;
      }
      for (long r = 0; r < nq; r++) {
        long c = (long)(urand() * nc) % nc;
        for (long d = 0; d < dim; d++)
          queries[r * dim + d] = centers[c * dim + d] + (float)(urand() - 0.5) * 0.2f;
      }
      free(centers);
    } else {
      for (long j = 0; j < (long)base_elems; j++) base[j] = (float)urand();
      for (long j = 0; j < (long)q_elems; j++) queries[j] = (float)urand();
    }
  }

  cuvsResources_t res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t idx;
  cuvsCagraSearchParams_t sp;
  cuvsCagraCompressionParams_t cp = NULL;
  struct cuvsIvfPqParams gb;
  int gb_ready = 0;
  memset(&gb, 0, sizeof(gb));

  if (cuvsResourcesCreate(&res) != CUVS_SUCCESS) { fprintf(stderr, "res: %s\n", cuvsGetLastErrorText()); return 2; }
  if (cuvsCagraIndexParamsCreate(&ip) != CUVS_SUCCESS) { fprintf(stderr, "ip: %s\n", cuvsGetLastErrorText()); return 2; }
  if (cuvsCagraIndexCreate(&idx) != CUVS_SUCCESS) { fprintf(stderr, "idx: %s\n", cuvsGetLastErrorText()); return 2; }

  ip->metric = L2Expanded;
  ip->graph_degree = (size_t)deg;
  ip->intermediate_graph_degree = (size_t)(inter < deg ? deg : inter);
  ip->nn_descent_niter = (size_t)niter;
  ip->build_algo = (algo == 1) ? IVF_PQ : (algo == 2 ? AUTO_SELECT : NN_DESCENT);

  if (algo == 1) {
    cuvsIvfPqIndexParams_t bp = NULL;
    cuvsIvfPqSearchParams_t ps = NULL;
    if (cuvsIvfPqIndexParamsCreate(&bp) != CUVS_SUCCESS) { fprintf(stderr, "bp: %s\n", cuvsGetLastErrorText()); return 3; }
    if (cuvsIvfPqSearchParamsCreate(&ps) != CUVS_SUCCESS) { fprintf(stderr, "ps: %s\n", cuvsGetLastErrorText()); return 3; }
    bp->metric = L2Expanded;
    bp->add_data_on_build = true;
    if (lists > 0) bp->n_lists = (size_t)lists;
    gb.ivf_pq_build_params = bp;
    gb.ivf_pq_search_params = ps;
    gb.refinement_rate = (refine < 0.0) ? 0.0f : (float)refine;
    ip->graph_build_params = &gb;
    gb_ready = 1;
  }

  if (pq_bits > 0) {
    if (cuvsCagraCompressionParamsCreate(&cp) != CUVS_SUCCESS) { fprintf(stderr, "cp: %s\n", cuvsGetLastErrorText()); return 3; }
    cp->pq_bits = (uint32_t)pq_bits;
    if (pq_dim > 0) cp->pq_dim = (uint32_t)pq_dim;
    if (vq > 0) cp->vq_n_centers = (uint32_t)vq;
    ip->compression = cp;
  }

  {
    DLManagedTensor dset;
    long ds[2];
    ds[0] = n; ds[1] = dim;
    init_tensor(&dset, base, ds, kDLFloat, 32);
    double t0 = now_ms();
    if (cuvsCagraBuild(res, ip, &dset, idx) != CUVS_SUCCESS) {
      fprintf(stderr, "build: %s\n", cuvsGetLastErrorText());
      return 4;
    }
    cudaDeviceSynchronize();
    double build_ms = now_ms() - t0;
    printf("BUILD n=%ld dim=%ld deg=%d inter=%d niter=%d algo=%d refine=%.2f pq=%d/%d vq=%d lists=%ld build_ms=%.1f\n",
           n, dim, deg, inter, niter, algo, gb.refinement_rate, pq_bits, pq_dim, vq, lists, build_ms);
    fflush(stdout);

    if (cuvsCagraSearchParamsCreate(&sp) != CUVS_SUCCESS) { fprintf(stderr, "sp: %s\n", cuvsGetLastErrorText()); return 5; }
    sp->itopk_size = (size_t)(itopk < (int)(k * 4) ? (int)(k * 4) : itopk);
    sp->search_width = (uint32_t)width;

    {
      DLManagedTensor qt, nt, dt;
      long qs[2], ns[2], ds2[2];
      unsigned* ids = (unsigned*)malloc((size_t)nq * k * sizeof(unsigned));
      float* dist = (float*)malloc((size_t)nq * k * sizeof(float));
      float* q_dev = NULL;
      unsigned* n_dev = NULL;
      float* d_dev = NULL;
      if (cuvsRMMAlloc(res, (void**)&q_dev, sizeof(float) * q_elems) != CUVS_SUCCESS) { fprintf(stderr, "qalloc\n"); return 7; }
      if (cuvsRMMAlloc(res, (void**)&n_dev, sizeof(unsigned) * (size_t)nq * k) != CUVS_SUCCESS) { fprintf(stderr, "nalloc\n"); return 7; }
      if (cuvsRMMAlloc(res, (void**)&d_dev, sizeof(float) * (size_t)nq * k) != CUVS_SUCCESS) { fprintf(stderr, "dalloc\n"); return 7; }
      cudaMemcpy(q_dev, queries, sizeof(float) * q_elems, cudaMemcpyDefault);
      qs[0] = nq; qs[1] = dim;
      ns[0] = nq; ns[1] = k;
      ds2[0] = nq; ds2[1] = k;
      init_tensor(&qt, q_dev, qs, kDLFloat, 32);
      qt.dl_tensor.device.device_type = kDLCUDA;
      init_tensor(&nt, n_dev, ns, kDLUInt, 32);
      nt.dl_tensor.device.device_type = kDLCUDA;
      init_tensor(&dt, d_dev, ds2, kDLFloat, 32);
      dt.dl_tensor.device.device_type = kDLCUDA;
      double st0 = now_ms();
      for (int rep = 0; rep < 3; rep++) {
        cuvsFilter f; f.type = NO_FILTER; f.addr = (uintptr_t)0;
        if (cuvsCagraSearch(res, sp, idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) {
          fprintf(stderr, "search: %s\n", cuvsGetLastErrorText());
          return 6;
        }
      }
      cudaDeviceSynchronize();
      double search_ms = (now_ms() - st0) / 3.0;
      cudaMemcpy(ids, n_dev, sizeof(unsigned) * (size_t)nq * k, cudaMemcpyDefault);
      cudaMemcpy(dist, d_dev, sizeof(float) * (size_t)nq * k, cudaMemcpyDefault);

      /* exact top-k, 24-way parallel */
      double bf0 = now_ms();
      #pragma omp parallel for schedule(dynamic, 4)
      for (long q = 0; q < nq; q++) {
        float* best = (float*)malloc((size_t)k * sizeof(float));
        unsigned* bid = (unsigned*)malloc((size_t)k * sizeof(unsigned));
        for (long j = 0; j < k; j++) { best[j] = 1e30f; bid[j] = 0xffffffffu; }
        const float* qv = queries + q * dim;
        for (long r = 0; r < n; r++) {
          const float* bv = base + r * dim;
          float s = 0.0f;
          for (long d = 0; d < dim; d++) { float t = qv[d] - bv[d]; s += t * t; }
          if (s < best[k - 1]) {
            long p = k - 1;
            while (p > 0 && best[p - 1] > s) { best[p] = best[p - 1]; bid[p] = bid[p - 1]; p--; }
            best[p] = s; bid[p] = (unsigned)r;
          }
        }
        memcpy(gt + q * k, bid, (size_t)k * sizeof(unsigned));
        free(best); free(bid);
      }
      double bf_ms = now_ms() - bf0;

      long hit = 0;
      for (long q = 0; q < nq; q++)
        for (long j = 0; j < k; j++)
          for (long m = 0; m < k; m++)
            if (ids[q * k + j] == gt[q * k + m]) { hit++; break; }

      printf("RESULT n=%ld dim=%ld deg=%d inter=%d niter=%d algo=%d refine=%.2f pq=%d/%d vq=%d lists=%ld "
             "build_ms=%.1f search_ms=%.2f qps=%.1f recall=%.4f bf_ms=%.0f\n",
             n, dim, deg, inter, niter, algo, gb.refinement_rate, pq_bits, pq_dim, vq, lists,
             build_ms, search_ms, (double)nq * 1000.0 / (search_ms > 0 ? search_ms : 1e-9),
             (double)hit / (double)(nq * k), bf_ms);
      fflush(stdout);
      free(ids); free(dist);
    }
    cuvsCagraSearchParamsDestroy(sp);
  }

  if (cp) cuvsCagraCompressionParamsDestroy(cp);
  cuvsCagraIndexDestroy(idx);
  /* Detach before destroying ip: its destructor also frees graph_build_params,
   * which is why destroying bp/ps first aborted with a double free. */
  if (gb_ready) { ip->graph_build_params = NULL; }
  cuvsCagraIndexParamsDestroy(ip);
  if (gb_ready) {
    cuvsIvfPqSearchParamsDestroy(gb.ivf_pq_search_params);
    cuvsIvfPqIndexParamsDestroy(gb.ivf_pq_build_params);
  }
  cuvsResourcesDestroy(res);
  free(base); free(queries); free(gt);
  return 0;
}
