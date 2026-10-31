/* seekdb <-> hipVS bridge: CAGRA index build, search and graph snapshot. */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/all_neighbors.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Build knobs. They are read from the environment so a parameter sweep does not
 * need a rebuild; the defaults in this file are the values the shipped index
 * uses.
 *
 * The intermediate degree stays at twice the output degree and the refine
 * rounds stay at 100, i.e. the shipped graph is the same graph the contest
 * baseline builds. That is deliberate: growing a graph twice as wide as the one
 * that is kept and pruning it down is not wasted work, the pruning pass is what
 * gives the surviving edges their navigability, and the refine rounds are what
 * makes it converge. Measured on 250k x 256, building straight at the output
 * degree (128) costs 0.06-0.12 recall@100 on clustered data and shortening the
 * refine rounds to 80 costs 0.0050 recall@10 on the real table, which is the
 * whole 0.005 recall gate with no margin left. The savings that are left are
 * in the snapshot and in the graph builder, not in the graph parameters. See
 * docs/gpu-vector-index-hipvs-cuvs/PERF_REPORT.md.
 *
 *   SEEKDB_CUVS_BUILD_ALGO           0 = nn_descent (default), 1 = auto,
 *                                    2 = ivf_pq, 3 = exact all-neighbors
 *   SEEKDB_CUVS_GRAPH_DEGREE         degree of the persisted graph (default 128)
 *   SEEKDB_CUVS_INTERMEDIATE_DEGREE  pre-prune degree (default = 2 * graph degree)
 *   SEEKDB_CUVS_NN_DESCENT_NITER     refinement rounds (default 100)
 *   SEEKDB_CUVS_AN_FP16              1 = compute the exact all-neighbors graph
 *                                     from an fp16 copy of the dataset
 *   SEEKDB_CUVS_DEBUG                1 = print per-stage build timings on stderr
 * Service-side knobs, they only change search quality and latency:
 *   SEEKDB_CUVS_SEARCH_ITOPK         candidate pool size (default max(topk*4, 512))
 *   SEEKDB_CUVS_SEARCH_WIDTH         graph expansion width (default 4)
 *
 * algo=3 is opt-in rather than the default: the exact all-neighbors pass builds
 * the whole kNN graph on the device and, at 250k x 256, its peak (1.44 GB) is
 * 1.67x what the nn_descent baseline uses (0.86 GB), which is over the contest
 * memory gate. Measured numbers are in the report.
 */
#define SK_CUVS_DEFAULT_DEGREE 128
#define SK_CUVS_DEFAULT_NITER 100

typedef struct {
  cuvsResources_t          res;
  cuvsCagraIndexParams_t   ip;
  cuvsCagraIndex_t         idx;
  long                     n, dim;
  /* Search state lives with the index: re-creating the search params and
   * re-allocating the device buffers per query costs more than the search
   * itself at this scale. */
  pthread_mutex_t          mu;
  cuvsCagraSearchParams_t  sp;
  float*                   q_dev;   /* nq x dim queries */
  unsigned*                n_dev;   /* nq x topk neighbour ids */
  float*                   d_dev;   /* nq x topk distances */
  size_t                   q_cap;   /* floats */
  size_t                   o_cap;   /* ids and distances, each */
} sk_cuvs_index;

static long sk_env_long(const char *name, long def)
{
  const char *v = getenv(name);
  if (v == NULL || v[0] == '\0') { return def; }
  char *end = NULL;
  long x = strtol(v, &end, 10);
  if (end == v || x <= 0) { return def; }
  return x;
}

static int sk_env_algo(void)
{
  const char *v = getenv("SEEKDB_CUVS_BUILD_ALGO");
  if (v == NULL || v[0] == '\0') { return 0; }
  if (strcmp(v, "nn_descent") == 0) { return 0; }
  if (strcmp(v, "auto") == 0) { return 1; }
  if (strcmp(v, "ivf_pq") == 0 || strcmp(v, "ivf") == 0) { return 2; }
  if (strcmp(v, "all_neighbors") == 0 || strcmp(v, "exact") == 0) { return 3; }
  return 0;
}

static void sk_cuvs_fill_tensor(DLManagedTensor *t, void *data, long *shape,
                                int code, int bits, int device_type)
{
  t->dl_tensor.data               = data;
  t->dl_tensor.device.device_type = device_type;
  t->dl_tensor.device.device_id   = 0;
  t->dl_tensor.ndim               = 2;
  t->dl_tensor.dtype.code         = (uint8_t)code;
  t->dl_tensor.dtype.bits         = (uint8_t)bits;
  t->dl_tensor.dtype.lanes        = 1;
  t->dl_tensor.shape              = shape;
  t->dl_tensor.strides            = NULL;
  t->dl_tensor.byte_offset        = 0;
  t->manager_ctx                  = NULL;
  t->deleter                      = NULL;
}

/* IEEE-754 binary32 -> binary16, round to nearest even. */
static uint16_t sk_float_to_half(float f)
{
  uint32_t x;
  uint32_t sign;
  int32_t exp;
  uint32_t mant;
  memcpy(&x, &f, sizeof(x));
  sign = (x >> 16) & 0x8000u;
  if (((x >> 23) & 0xffu) == 0xffu) {
    return (uint16_t)(sign | 0x7c00u | ((x & 0x7fffffu) ? 0x200u : 0u));
  }
  exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
  mant = x & 0x7fffffu;
  if (exp >= 0x1f) { return (uint16_t)(sign | 0x7c00u); }
  if (exp <= 0) {
    uint32_t shift;
    uint32_t half;
    if (exp < -10) { return (uint16_t)sign; }
    mant |= 0x800000u;
    shift = (uint32_t)(14 - exp);
    half = mant >> shift;
    if (((mant >> (shift - 1)) & 1u) && ((mant & ((1u << (shift - 1)) - 1u)) != 0u ||
                                       (((mant >> (shift - 1)) & 1u) && ((half & 1u) == 0u)))) {
      half += 1;
    }
    return (uint16_t)(sign | half);
  }
  {
    uint32_t half = ((uint32_t)exp << 10) | (mant >> 13);
    uint32_t rest = mant & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (half & 1u))) { half += 1; }
    return (uint16_t)(sign | half);
  }
}

static int sk_env_flag(const char *name)
{
  const char *v = getenv(name);
  if (v == NULL || v[0] == '\0') { return 0; }
  return strtol(v, NULL, 10) != 0;
}


/* Builds the index with explicit knobs; degree/inter_degree/niter <= 0 means
 * "use the default". Returns NULL on failure. */
/* Builds the graph on an fp16 copy of the dataset and then serves it from the
 * fp32 dataset. NN-descent is bandwidth bound, so halving the dataset bytes
 * roughly halves the build time, while the search still walks the same topology
 * and computes its distances against the fp32 vectors. */
static void *sk_cuvs_build_fp16_graph(const float *base, long n, long dim,
                                       int algo, long degree, long inter_degree, long niter)
{
  void *result = NULL;
  uint16_t *half = NULL;
  cuvsResources_t res = 0;
  cuvsCagraIndexParams_t hparams = NULL;
  cuvsCagraIndex_t hidx = NULL;
  unsigned *graph = NULL;
  long gn = 0, gdegree = 0;
  DLManagedTensor dset, gset;
  long ds[2];

  half = (uint16_t *)malloc((size_t)n * (size_t)dim * sizeof(uint16_t));
  if (half == NULL) { return NULL; }
  for (long i = 0; i < n * dim; i++) { half[i] = sk_float_to_half(base[i]); }

  if (cuvsResourcesCreate(&res) != CUVS_SUCCESS) { goto out; }
  if (cuvsCagraIndexParamsCreate(&hparams) != CUVS_SUCCESS) { goto out; }
  hparams->metric                     = L2Expanded;
  hparams->build_algo                 = (algo == 1) ? AUTO_SELECT
                                      : (algo == 2) ? IVF_PQ : NN_DESCENT;
  hparams->graph_degree               = (uint32_t)degree;
  hparams->intermediate_graph_degree  = (uint32_t)inter_degree;
  hparams->nn_descent_niter           = (uint32_t)niter;
  if (cuvsCagraIndexCreate(&hidx) != CUVS_SUCCESS) { goto out; }
  ds[0] = n;
  ds[1] = dim;
  sk_cuvs_fill_tensor(&dset, half, ds, kDLFloat, 16, kDLCPU);
  if (cuvsCagraBuild(res, hparams, &dset, hidx) != CUVS_SUCCESS) { goto out; }
  if (cuvsCagraIndexGetGraph(hidx, &gset) != CUVS_SUCCESS) { goto out; }
  gn = (long)gset.dl_tensor.shape[0];
  gdegree = (long)gset.dl_tensor.shape[1];
  if (gn <= 0 || gdegree <= 0) { goto out; }
  graph = (unsigned *)malloc(sizeof(unsigned) * (size_t)gn * (size_t)gdegree);
  if (graph == NULL) { goto out; }
  if (cudaMemcpy(graph, gset.dl_tensor.data,
                 sizeof(unsigned) * (size_t)gn * (size_t)gdegree,
                 cudaMemcpyDefault) != cudaSuccess) { goto out; }
  if (gset.deleter != NULL) { gset.deleter(&gset); }
  result = seekdb_cuvs_import(base, n, dim, graph, gdegree);
out:
  if (graph != NULL) { free(graph); }
  if (hidx != NULL) { cuvsCagraIndexDestroy(hidx); }
  if (hparams != NULL) { cuvsCagraIndexParamsDestroy(hparams); }
  if (res != 0) { cuvsResourcesDestroy(res); }
  if (half != NULL) { free(half); }
  return result;
}

/* Builds the graph with an exact all-neighbors pass instead of NN-descent.
 *
 * NN-descent re-reads the whole dataset once per refinement round, so a graph
 * of degree 128 at 100 rounds costs 100 passes over 128 MB. The brute force
 * all-neighbors pass computes the same distances once in a single GEMM, and the
 * mutual-reachability core distances reproduce the pruning that makes the
 * neighbor lists navigable. The result is handed to CAGRA as a ready graph, so
 * the search side is unchanged.
 *
 * Returns NULL when the build fails; the caller then falls back to NN-descent. */
static long sk_now_us(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long)ts.tv_sec * 1000000L + ts.tv_nsec / 1000L;
}

/* Returns NULL when the build fails; the caller then falls back to NN-descent.
 * fp16 non-zero runs the neighbour search on half-precision vectors: the
 * distance pass is the dominant cost and it is bound by tensor throughput and
 * memory traffic, both of which halve. The graph is still imported against the
 * original fp32 vectors, so queries and stored data keep full precision. */
static void *sk_cuvs_build_all_neighbors(const float *base, long n, long dim, long degree, int fp16)
{
  void *result = NULL;
  cuvsResources_t res = 0;
  cuvsAllNeighborsIndexParams_t params = NULL;
  void *data_dev = NULL;
  uint16_t *half_host = NULL;
  int64_t *idx_dev = NULL;
  float *dist_dev = NULL;
  float *core_dev = NULL;
  unsigned *graph = NULL;
  int64_t *idx_host = NULL;
  DLManagedTensor dset, idx_t, dist_t, core_t;
  long ds[2], is[2], dis[2];
  float alpha = 1.2f;
  long t_all, t_up = 0, t_build = 0, t_d2h = 0, t_prune = 0, t_import = 0;
  size_t data_bytes = 0;

  t_all = sk_now_us();
  if (cuvsResourcesCreate(&res) != CUVS_SUCCESS) { return NULL; }
  if (cuvsAllNeighborsIndexParamsCreate(&params) != CUVS_SUCCESS) { goto out; }
  params->algo          = CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE;
  params->metric        = L2Expanded;
  params->n_clusters    = 1;      /* device dataset: no partitioning */
  params->overlap_factor = 1;

  data_bytes = (fp16 ? sizeof(uint16_t) : sizeof(float)) * (size_t)n * (size_t)dim;
  if (cuvsRMMAlloc(res, &data_dev, data_bytes) != CUVS_SUCCESS) { goto out; }
  if (cuvsRMMAlloc(res, (void **)&idx_dev, sizeof(int64_t) * (size_t)n * (size_t)degree) != CUVS_SUCCESS) { goto out; }
  if (cuvsRMMAlloc(res, (void **)&dist_dev, sizeof(float) * (size_t)n * (size_t)degree) != CUVS_SUCCESS) { goto out; }
  if (cuvsRMMAlloc(res, (void **)&core_dev, sizeof(float) * (size_t)n) != CUVS_SUCCESS) { goto out; }
  if (fp16) {
    size_t cnt = (size_t)n * (size_t)dim;
    size_t i;
    half_host = (uint16_t *)malloc(data_bytes);
    if (half_host == NULL) { goto out; }
    for (i = 0; i < cnt; i++) { half_host[i] = sk_float_to_half(base[i]); }
    if (cudaMemcpy(data_dev, half_host, data_bytes, cudaMemcpyHostToDevice) != cudaSuccess) { goto out; }
  } else {
    if (cudaMemcpy(data_dev, base, data_bytes, cudaMemcpyHostToDevice) != cudaSuccess) { goto out; }
  }
  t_up = sk_now_us();

  ds[0] = n; ds[1] = dim;
  is[0] = n; is[1] = degree;
  dis[0] = n; dis[1] = degree;
  sk_cuvs_fill_tensor(&dset, data_dev, ds, kDLFloat, fp16 ? 16 : 32, kDLCUDA);
  sk_cuvs_fill_tensor(&idx_t, idx_dev, is, kDLInt, 64, kDLCUDA);
  sk_cuvs_fill_tensor(&dist_t, dist_dev, dis, kDLFloat, 32, kDLCUDA);
  sk_cuvs_fill_tensor(&core_t, core_dev, ds, kDLFloat, 32, kDLCUDA);
  core_t.dl_tensor.ndim = 1;
  core_t.dl_tensor.shape = ds;

  if (cuvsAllNeighborsBuild(res, params, &dset, &idx_t, &dist_t, &core_t, alpha) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] all-neighbors build failed: %s\n", cuvsGetLastErrorText());
    goto out;
  }
  t_build = sk_now_us();

  idx_host = (int64_t *)malloc(sizeof(int64_t) * (size_t)n * (size_t)degree);
  graph = (unsigned *)malloc(sizeof(unsigned) * (size_t)n * (size_t)degree);
  if (idx_host == NULL || graph == NULL) { goto out; }
  if (cudaMemcpy(idx_host, idx_dev, sizeof(int64_t) * (size_t)n * (size_t)degree,
                 cudaMemcpyDeviceToHost) != cudaSuccess) { goto out; }
  t_d2h = sk_now_us();

  /* Drop self references and any padding the pruner left behind, then keep the
   * neighbour count at `degree` by repeating the last valid neighbour. */
  for (long i = 0; i < n; i++) {
    int64_t *row = idx_host + (size_t)i * (size_t)degree;
    unsigned *grow = graph + (size_t)i * (size_t)degree;
    long valid = 0;
    for (long j = 0; j < degree; j++) {
      if (row[j] >= 0 && row[j] < n && row[j] != i) { row[valid++] = row[j]; }
    }
    if (valid == 0) { row[0] = (i + 1) % n; valid = 1; }
    for (long j = valid; j < degree; j++) { row[j] = row[valid - 1]; }
    for (long j = 0; j < degree; j++) { grow[j] = (unsigned)row[j]; }
  }
  t_prune = sk_now_us();

  result = seekdb_cuvs_import(base, n, dim, graph, degree);
  t_import = sk_now_us();
  if (sk_env_flag("SEEKDB_CUVS_DEBUG")) {
    fprintf(stderr, "[seekdb_cuvs] all_neighbors fp16=%d n=%ld dim=%ld degree=%ld"
                    " upload=%ld build=%ld d2h=%ld prune=%ld import=%ld total=%ld (ms)\n",
            fp16, n, dim, degree, (t_up - t_all) / 1000, (t_build - t_up) / 1000,
            (t_d2h - t_build) / 1000, (t_prune - t_d2h) / 1000,
            (t_import - t_prune) / 1000, (t_import - t_all) / 1000);
  }
out:
  if (graph != NULL) { free(graph); }
  if (idx_host != NULL) { free(idx_host); }
  if (half_host != NULL) { free(half_host); }
  if (core_dev != NULL) { cuvsRMMFree(res, core_dev, sizeof(float) * (size_t)n); }
  if (dist_dev != NULL) { cuvsRMMFree(res, dist_dev, sizeof(float) * (size_t)n * (size_t)degree); }
  if (idx_dev != NULL) { cuvsRMMFree(res, idx_dev, sizeof(int64_t) * (size_t)n * (size_t)degree); }
  if (data_dev != NULL) { cuvsRMMFree(res, data_dev, data_bytes); }
  if (params != NULL) { cuvsAllNeighborsIndexParamsDestroy(params); }
  if (res != 0) { cuvsResourcesDestroy(res); }
  return result;
}

/* Builds an index with the explicit knobs; degree/inter_degree/niter <= 0 mean
 * "use the default". Returns NULL on failure. */
void *seekdb_cuvs_build_tuned(const float *base, long n, long dim,
                              int algo, long degree, long inter_degree, long niter)
{
  sk_cuvs_index *h = NULL;
  if (base == NULL || n <= 0 || dim <= 0) { return NULL; }
  if (degree <= 0) { degree = SK_CUVS_DEFAULT_DEGREE; }
  if (inter_degree <= 0) { inter_degree = degree * 2; }
  if (niter <= 0) { niter = SK_CUVS_DEFAULT_NITER; }
  if (inter_degree < degree) { inter_degree = degree; }
  if (algo == 3) {
    void *built = sk_cuvs_build_all_neighbors(base, n, dim, degree,
                                              sk_env_flag("SEEKDB_CUVS_AN_FP16"));
    if (built != NULL) { return built; }
    fprintf(stderr, "[seekdb_cuvs] all-neighbors build failed, using nn_descent\n");
    algo = 0;
  }
  if (sk_env_flag("SEEKDB_CUVS_BUILD_FP16")) {
    void *fp32 = sk_cuvs_build_fp16_graph(base, n, dim, algo, degree, inter_degree, niter);
    if (fp32 != NULL) { return fp32; }
    fprintf(stderr, "[seekdb_cuvs] fp16 graph build unavailable, using fp32\n");
  }
  h = (sk_cuvs_index *)calloc(1, sizeof(*h));
  if (h == NULL) { return NULL; }
  pthread_mutex_init(&h->mu, NULL);
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { goto fail_res; }
  {
    DLManagedTensor dset;
    long ds[2];
    ds[0] = n;
    ds[1] = dim;
    sk_cuvs_fill_tensor(&dset, (void *)base, ds, kDLFloat, 32, kDLCPU);
    if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) { goto fail_res; }
    h->ip->metric                     = L2Expanded;
    h->ip->build_algo                 = (algo == 1) ? AUTO_SELECT
                                       : (algo == 2) ? IVF_PQ : NN_DESCENT;
    h->ip->graph_degree               = (uint32_t)degree;
    h->ip->intermediate_graph_degree  = (uint32_t)inter_degree;
    h->ip->nn_descent_niter           = (uint32_t)niter;
    if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { goto fail_ip; }
    if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) { goto fail_idx; }
  }
  h->n = n;
  h->dim = dim;
  return h;

fail_idx:
  cuvsCagraIndexDestroy(h->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(h->ip);
fail_res:
  cuvsResourcesDestroy(h->res);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  pthread_mutex_destroy(&h->mu);
  free(h);
  return NULL;
}

void *seekdb_cuvs_build(const float *base, long n, long dim)
{
  return seekdb_cuvs_build_tuned(base, n, dim, sk_env_algo(),
                                 sk_env_long("SEEKDB_CUVS_GRAPH_DEGREE", SK_CUVS_DEFAULT_DEGREE),
                                 sk_env_long("SEEKDB_CUVS_INTERMEDIATE_DEGREE", 0),
                                 sk_env_long("SEEKDB_CUVS_NN_DESCENT_NITER", SK_CUVS_DEFAULT_NITER));
}
/* Grows the device buffers so a search never allocates. Returns 0 on success. */
static int sk_cuvs_reserve(sk_cuvs_index *h, size_t q_elems, size_t o_elems)
{
  if (q_elems > h->q_cap) {
    if (h->q_dev != NULL) { cuvsRMMFree(h->res, h->q_dev, sizeof(float) * h->q_cap); }
    h->q_dev = NULL;
    h->q_cap = 0;
    if (cuvsRMMAlloc(h->res, (void **)&h->q_dev, sizeof(float) * q_elems) != CUVS_SUCCESS) {
      return 10;
    }
    h->q_cap = q_elems;
  }
  if (o_elems > h->o_cap) {
    if (h->n_dev != NULL) { cuvsRMMFree(h->res, h->n_dev, sizeof(unsigned) * h->o_cap); }
    if (h->d_dev != NULL) { cuvsRMMFree(h->res, h->d_dev, sizeof(float) * h->o_cap); }
    h->n_dev = NULL;
    h->d_dev = NULL;
    h->o_cap = 0;
    if (cuvsRMMAlloc(h->res, (void **)&h->n_dev, sizeof(unsigned) * o_elems) != CUVS_SUCCESS) {
      h->n_dev = NULL;
      return 11;
    }
    if (cuvsRMMAlloc(h->res, (void **)&h->d_dev, sizeof(float) * o_elems) != CUVS_SUCCESS) {
      cuvsRMMFree(h->res, h->n_dev, sizeof(unsigned) * o_elems);
      h->n_dev = NULL;
      return 12;
    }
    h->o_cap = o_elems;
  }
  return 0;
}

int seekdb_cuvs_search(void *handle, const float *query, long nq, long topk,
                       unsigned int *out_ids, float *out_dist)
{
  sk_cuvs_index *h = (sk_cuvs_index *)handle;
  if (h == NULL) { return 1; }
  int rc = 0;
  pthread_mutex_lock(&h->mu);
  if (nq <= 0 || topk <= 0 || query == NULL || out_ids == NULL) { rc = 2; goto done; }
  if ((rc = sk_cuvs_reserve(h, (size_t)nq * (size_t)h->dim, (size_t)nq * (size_t)topk)) != 0) {
    goto done;
  }
  if (cudaMemcpy(h->q_dev, query, sizeof(float) * (size_t)nq * (size_t)h->dim,
                 cudaMemcpyDefault) != cudaSuccess) { rc = 13; goto done; }
  {
    DLManagedTensor qt, nt, dt;
    long qs[2], ns[2], ds[2];
    qs[0] = nq; qs[1] = h->dim;
    ns[0] = nq; ns[1] = topk;
    ds[0] = nq; ds[1] = topk;
    sk_cuvs_fill_tensor(&qt, h->q_dev, qs, kDLFloat, 32, kDLCUDA);
    sk_cuvs_fill_tensor(&nt, h->n_dev, ns, kDLUInt, 32, kDLCUDA);
    sk_cuvs_fill_tensor(&dt, h->d_dev, ds, kDLFloat, 32, kDLCUDA);
    if (h->sp == NULL) {
      size_t itopk = (size_t)topk * 4;
      long width = 4;
      const char *e_itopk = getenv("SEEKDB_CUVS_SEARCH_ITOPK");
      const char *e_width = getenv("SEEKDB_CUVS_SEARCH_WIDTH");
      if (itopk < 512) { itopk = 512; }
      if (e_itopk != NULL && atoi(e_itopk) > 0) { itopk = (size_t)atoi(e_itopk); }
      if (e_width != NULL && atoi(e_width) > 0) { width = atol(e_width); }
      if (cuvsCagraSearchParamsCreate(&h->sp) != CUVS_SUCCESS) { rc = 14; goto done; }
      h->sp->itopk_size = itopk;
      h->sp->search_width = (uint32_t)width;
    } else if (h->sp->itopk_size < (size_t)topk * 4) {
      h->sp->itopk_size = (size_t)topk * 4;
    }
    {
      cuvsFilter f;
      f.type = NO_FILTER;
      f.addr = (uintptr_t)0;
      if (cuvsCagraSearch(h->res, h->sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) {
        rc = 15;
        goto done;
      }
    }
    if (cudaMemcpy(out_ids, h->n_dev, sizeof(unsigned) * (size_t)nq * (size_t)topk,
                   cudaMemcpyDefault) != cudaSuccess) { rc = 16; goto done; }
    if (out_dist != NULL &&
        cudaMemcpy(out_dist, h->d_dev, sizeof(float) * (size_t)nq * (size_t)topk,
                   cudaMemcpyDefault) != cudaSuccess) { rc = 17; }
  }
done:
  pthread_mutex_unlock(&h->mu);
  if (rc) { fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText()); }
  return rc;
}

void seekdb_cuvs_free(void *handle)
{
  sk_cuvs_index *h = (sk_cuvs_index *)handle;
  if (h == NULL) { return; }
  if (h->sp != NULL) { cuvsCagraSearchParamsDestroy(h->sp); }
  if (h->q_dev != NULL) { cuvsRMMFree(h->res, h->q_dev, sizeof(float) * h->q_cap); }
  if (h->n_dev != NULL) { cuvsRMMFree(h->res, h->n_dev, sizeof(unsigned) * h->o_cap); }
  if (h->d_dev != NULL) { cuvsRMMFree(h->res, h->d_dev, sizeof(float) * h->o_cap); }
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  cuvsResourcesDestroy(h->res);
  pthread_mutex_destroy(&h->mu);
  free(h);
}

int seekdb_cuvs_serialize(void *handle, const char *filename)
{
  sk_cuvs_index *index = (sk_cuvs_index *)handle;
  if (index == NULL || filename == NULL) { return 1; }
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  return 0;
}

void *seekdb_cuvs_deserialize(const char *filename, long n, long dim)
{
  if (filename == NULL || n <= 0 || dim <= 0) { return NULL; }
  sk_cuvs_index *index = (sk_cuvs_index *)calloc(1, sizeof(*index));
  if (index == NULL) { return NULL; }
  pthread_mutex_init(&index->mu, NULL);
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) { goto fail; }
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) { goto fail_res; }
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) { goto fail_ip; }
  if (cuvsCagraDeserialize(index->res, filename, index->idx) != CUVS_SUCCESS) { goto fail_idx; }
  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail_idx;
    }
  }
  index->n = n;
  index->dim = dim;
  return index;
fail_idx:
  cuvsCagraIndexDestroy(index->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(index->ip);
fail_res:
  cuvsResourcesDestroy(index->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  pthread_mutex_destroy(&index->mu);
  free(index);
  return NULL;
}

/* Copies the adjacency list out of the GPU index. The graph is the only part of
 * a CAGRA index that cannot be re-derived from the stored vectors, so seekdb
 * persists it and rebuilds the index with cuvsCagraIndexFromArgs on load.
 * The returned buffer is malloc'd and owned by the caller. */
int seekdb_cuvs_export_graph(void *handle, unsigned int **out_graph,
                             long *out_n, long *out_degree)
{
  sk_cuvs_index *h = (sk_cuvs_index *)handle;
  DLManagedTensor gt;
  long n = 0, degree = 0;
  size_t bytes = 0;
  unsigned int *buf = NULL;
  int rc = 0;
  if (h == NULL || out_graph == NULL || out_n == NULL || out_degree == NULL) { return 1; }
  pthread_mutex_lock(&h->mu);
  if (cuvsCagraIndexGetGraph(h->idx, &gt) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] get graph failed: %s\n", cuvsGetLastErrorText());
    rc = 2;
    goto done;
  }
  n = (long)gt.dl_tensor.shape[0];
  degree = (long)gt.dl_tensor.shape[1];
  bytes = sizeof(unsigned int) * (size_t)n * (size_t)degree;
  if (n <= 0 || degree <= 0 || bytes == 0) {
    if (gt.deleter != NULL) { gt.deleter(&gt); }
    rc = 3;
    goto done;
  }
  buf = (unsigned int *)malloc(bytes);
  if (buf == NULL) {
    if (gt.deleter != NULL) { gt.deleter(&gt); }
    rc = 4;
    goto done;
  }
  if (cudaMemcpy(buf, gt.dl_tensor.data, bytes, cudaMemcpyDefault) != cudaSuccess) {
    if (gt.deleter != NULL) { gt.deleter(&gt); }
    free(buf);
    fprintf(stderr, "[seekdb_cuvs] graph copy failed: %s\n", cuvsGetLastErrorText());
    rc = 5;
    goto done;
  }
  if (gt.deleter != NULL) { gt.deleter(&gt); }
  *out_graph = buf;
  *out_n = n;
  *out_degree = degree;
done:
  pthread_mutex_unlock(&h->mu);
  return rc;
}

void seekdb_cuvs_free_host(void *ptr)
{
  free(ptr);
}

/* Rebuilds a searchable index from a host-resident dataset and graph. */
void *seekdb_cuvs_import(const float *base, long n, long dim,
                         const unsigned int *graph, long degree)
{
  sk_cuvs_index *h = NULL;
  DLManagedTensor dset, gset;
  long dsh[2], gsh[2];
  if (base == NULL || graph == NULL || n <= 0 || dim <= 0 || degree <= 0) { return NULL; }
  h = (sk_cuvs_index *)calloc(1, sizeof(*h));
  if (h == NULL) { return NULL; }
  pthread_mutex_init(&h->mu, NULL);
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { goto fail_res; }
  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) { goto fail_res; }
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { goto fail_ip; }
  dsh[0] = n; dsh[1] = dim;
  gsh[0] = n; gsh[1] = degree;
  sk_cuvs_fill_tensor(&dset, (void *)base, dsh, kDLFloat, 32, kDLCPU);
  sk_cuvs_fill_tensor(&gset, (void *)graph, gsh, kDLUInt, 32, kDLCPU);
  if (cuvsCagraIndexFromArgs(h->res, L2Expanded, &gset, &dset, h->idx) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] import failed: %s\n", cuvsGetLastErrorText());
    goto fail_idx;
  }
  h->n = n;
  h->dim = dim;
  return h;
fail_idx:
  cuvsCagraIndexDestroy(h->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(h->ip);
fail_res:
  cuvsResourcesDestroy(h->res);
fail:
  pthread_mutex_destroy(&h->mu);
  free(h);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float *base, long n, long dim,
                           const float *query, long nq, long topk,
                           unsigned int *out_ids)
{
  void *h = seekdb_cuvs_build(base, n, dim);
  if (h == NULL) { return 10; }
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
