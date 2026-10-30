/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here. */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
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

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  long n, dim;
  /* Search scratch, allocated once and reused: a per-query allocation round trip
   * through the device allocator dominated single-query latency. */
  pthread_mutex_t        mu;
  cuvsCagraSearchParams_t sp;
  float*                 q_dev;
  unsigned*              n_dev;
  float*                 d_dev;
  size_t                 q_cap;   /* elements */
  size_t                 o_cap;   /* elements */
} sk_cuvs_index;

static size_t env_size(const char* name, size_t def)
{
  const char* v = getenv(name);
  if (v == NULL || v[0] == '\0') { return def; }
  {
    long x = strtol(v, NULL, 10);
    return x > 0 ? (size_t)x : def;
  }
}

/* CAGRA build knobs (env-overridable so a tuning sweep needs no rebuild):
 *  SEEKDB_CUVS_BUILD_ALGO          nn_descent (default) | auto
 *  SEEKDB_CUVS_GRAPH_DEGREE        output/persisted graph degree (default 128)
 *  SEEKDB_CUVS_INTERMEDIATE_DEGREE pre-prune degree (default = graph degree)
 *  SEEKDB_CUVS_NN_DESCENT_NITER    refinement rounds (default 32)
 * The contest baseline used intermediate_graph_degree=256 and
 * nn_descent_niter=100, which pays a prune over a graph twice as wide as the one
 * that is finally kept plus 5x the refinement rounds. Building straight at the
 * output degree keeps the persisted graph degree (and therefore recall under the
 * fixed search parameters) unchanged while removing most of the build work. */
static void sk_cuvs_set_build_params(cuvsCagraIndexParams_t ip)
{
  const char* algo = getenv("SEEKDB_CUVS_BUILD_ALGO");
  size_t deg   = env_size("SEEKDB_CUVS_GRAPH_DEGREE", 128);
  size_t inter = env_size("SEEKDB_CUVS_INTERMEDIATE_DEGREE", deg);
  size_t niter = env_size("SEEKDB_CUVS_NN_DESCENT_NITER", 32);
  if (inter < deg) { inter = deg; }
  ip->metric                    = L2Expanded;
  ip->build_algo                = (algo != NULL && strcmp(algo, "auto") == 0)
                                  ? AUTO_SELECT : NN_DESCENT;
  ip->graph_degree              = deg;
  ip->intermediate_graph_degree = inter;
  ip->nn_descent_niter          = niter;
}

static void sk_cuvs_init_tensor(DLManagedTensor* t, void* data, long* shape,
                                int code, int bits)
{
  t->dl_tensor.data               = data;
  t->dl_tensor.device.device_type = kDLCPU;
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

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  pthread_mutex_init(&h->mu, NULL);
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) goto fail;

  {
    DLManagedTensor dset;
    long ds[2];
    ds[0] = n; ds[1] = dim;
    sk_cuvs_init_tensor(&dset, (void*)base, ds, kDLFloat, 32);
    if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_res;
    sk_cuvs_set_build_params(h->ip);
    if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) goto fail_ip;
    if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;
  }
  h->n = n;
  h->dim = dim;
  return h;

fail_idx: cuvsCagraIndexDestroy(h->idx);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_res: cuvsResourcesDestroy(h->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  pthread_mutex_destroy(&h->mu);
  free(h);
  return NULL;
}

static int sk_cuvs_reserve(sk_cuvs_index* h, size_t q_elems, size_t o_elems)
{
  if (q_elems > h->q_cap) {
    if (h->q_dev) { cuvsRMMFree(h->res, h->q_dev, sizeof(float) * h->q_cap); }
    h->q_dev = NULL;
    h->q_cap = 0;
    if (cuvsRMMAlloc(h->res, (void**)&h->q_dev, sizeof(float) * q_elems) != CUVS_SUCCESS) {
      return 10;
    }
    h->q_cap = q_elems;
  }
  if (o_elems > h->o_cap) {
    if (h->n_dev) { cuvsRMMFree(h->res, h->n_dev, sizeof(unsigned) * h->o_cap); }
    if (h->d_dev) { cuvsRMMFree(h->res, h->d_dev, sizeof(float) * h->o_cap); }
    h->n_dev = NULL;
    h->d_dev = NULL;
    h->o_cap = 0;
    if (cuvsRMMAlloc(h->res, (void**)&h->n_dev, sizeof(unsigned) * o_elems) != CUVS_SUCCESS) {
      h->n_dev = NULL;
      return 11;
    }
    if (cuvsRMMAlloc(h->res, (void**)&h->d_dev, sizeof(float) * o_elems) != CUVS_SUCCESS) {
      cuvsRMMFree(h->res, h->n_dev, sizeof(unsigned) * o_elems);
      h->n_dev = NULL;
      return 12;
    }
    h->o_cap = o_elems;
  }
  return 0;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return 1;
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
    long qs[2], ns[2], ds2[2];
    qs[0] = nq; qs[1] = h->dim;
    ns[0] = nq; ns[1] = topk;
    ds2[0] = nq; ds2[1] = topk;
    sk_cuvs_init_tensor(&qt, h->q_dev, qs, kDLFloat, 32);
    qt.dl_tensor.device.device_type = kDLCUDA;
    sk_cuvs_init_tensor(&nt, h->n_dev, ns, kDLUInt, 32);
    nt.dl_tensor.device.device_type = kDLCUDA;
    sk_cuvs_init_tensor(&dt, h->d_dev, ds2, kDLFloat, 32);
    dt.dl_tensor.device.device_type = kDLCUDA;
    if (h->sp == NULL) {
      const char *e_itopk = getenv("SEEKDB_CUVS_SEARCH_ITOPK");
      const char *e_width = getenv("SEEKDB_CUVS_SEARCH_WIDTH");
      size_t required_itopk = (size_t)topk * 4;
      if (required_itopk < 512) required_itopk = 512;
      if (e_itopk != NULL && atoi(e_itopk) > 0) { required_itopk = (size_t)atoi(e_itopk); }
      if (cuvsCagraSearchParamsCreate(&h->sp) != CUVS_SUCCESS) { rc = 14; goto done; }
      h->sp->itopk_size = required_itopk;
      h->sp->search_width = (e_width != NULL && atoi(e_width) > 0) ? (uint32_t)atoi(e_width) : 4;
    } else if (h->sp->itopk_size < (size_t)topk * 4) {
      h->sp->itopk_size = (size_t)topk * 4;
    }
    {
      cuvsFilter f; f.type = NO_FILTER; f.addr = (uintptr_t)0;
      if (cuvsCagraSearch(h->res, h->sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) {
        rc = 15; goto done;
      }
    }
    if (cudaMemcpy(out_ids, h->n_dev, sizeof(unsigned) * (size_t)nq * (size_t)topk,
                   cudaMemcpyDefault) != cudaSuccess) { rc = 16; goto done; }
    if (out_dist && cudaMemcpy(out_dist, h->d_dev, sizeof(float) * (size_t)nq * (size_t)topk,
                               cudaMemcpyDefault) != cudaSuccess) { rc = 17; }
  }
done:
  pthread_mutex_unlock(&h->mu);
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  if (h->sp) cuvsCagraSearchParamsDestroy(h->sp);
  if (h->q_dev) cuvsRMMFree(h->res, h->q_dev, sizeof(float) * h->q_cap);
  if (h->n_dev) cuvsRMMFree(h->res, h->n_dev, sizeof(unsigned) * h->o_cap);
  if (h->d_dev) cuvsRMMFree(h->res, h->d_dev, sizeof(float) * h->o_cap);
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  cuvsResourcesDestroy(h->res);
  pthread_mutex_destroy(&h->mu);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) return 1;
  /* Self-contained file: seekdb persists the graph itself now, but this entry
   * point (used by tools/hipvs_bridge/test_persistence.cpp) keeps dataset+graph. */
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) return NULL;
  pthread_mutex_init(&index->mu, NULL);
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) goto fail_res;
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraDeserialize(index->res, filename, index->idx) != CUVS_SUCCESS) goto fail_idx;
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

int seekdb_cuvs_export_graph(void* handle, unsigned int** out_graph,
                             long* out_n, long* out_degree)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  DLManagedTensor gt;
  long n = 0, degree = 0;
  size_t bytes = 0;
  unsigned* buf = NULL;
  int rc = 0;
  if (!h || !out_graph || !out_n || !out_degree) return 1;
  pthread_mutex_lock(&h->mu);
  if (cuvsCagraIndexGetGraph(h->idx, &gt) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] get graph failed: %s\n", cuvsGetLastErrorText());
    rc = 2;
    goto done;
  }
  n = (long)gt.dl_tensor.shape[0];
  degree = (long)gt.dl_tensor.shape[1];
  bytes = sizeof(unsigned) * (size_t)n * (size_t)degree;
  if (n <= 0 || degree <= 0) {
    if (gt.deleter) gt.deleter(&gt);
    rc = 3;
    goto done;
  }
  buf = (unsigned*)malloc(bytes);
  if (buf == NULL) {
    if (gt.deleter) gt.deleter(&gt);
    rc = 4;
    goto done;
  }
  if (cudaMemcpy(buf, gt.dl_tensor.data, bytes, cudaMemcpyDefault) != cudaSuccess) {
    if (gt.deleter) gt.deleter(&gt);
    free(buf);
    fprintf(stderr, "[seekdb_cuvs] graph copy failed: %s\n", cuvsGetLastErrorText());
    rc = 5;
    goto done;
  }
  if (gt.deleter) gt.deleter(&gt);
  *out_graph = buf;
  *out_n = n;
  *out_degree = degree;
done:
  pthread_mutex_unlock(&h->mu);
  return rc;
}

void seekdb_cuvs_free_host(void* ptr)
{
  free(ptr);
}

void* seekdb_cuvs_import(const float* base, long n, long dim,
                         const unsigned int* graph, long degree)
{
  sk_cuvs_index* h = NULL;
  DLManagedTensor dset;
  DLManagedTensor gset;
  long dsh[2];
  long gsh[2];
  if (base == NULL || graph == NULL || n <= 0 || dim <= 0 || degree <= 0) return NULL;
  h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  pthread_mutex_init(&h->mu, NULL);
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_res;
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) goto fail_ip;
  dsh[0] = n; dsh[1] = dim;
  gsh[0] = n; gsh[1] = degree;
  sk_cuvs_init_tensor(&dset, (void*)base, dsh, kDLFloat, 32);
  sk_cuvs_init_tensor(&gset, (void*)graph, gsh, kDLUInt, 32);
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

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
