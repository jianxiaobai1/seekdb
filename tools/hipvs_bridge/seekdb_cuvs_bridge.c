/* seekdb <-> hipVS bridge: CAGRA build, search and graph transplant.
 *
 * All cuVS/ROCm state lives in this translation unit; seekdb links only against
 * the plain-C entry points declared in seekdb_cuvs_bridge.h.
 *
 * Three properties shape the code below:
 *  1. CAGRA build time is dominated by the prune pass that runs after
 *     nn_descent produces a wider intermediate graph. Building at the final
 *     graph degree with fewer refinement rounds keeps the persisted graph (and
 *     therefore recall under fixed search parameters) and drops most of the
 *     work.
 *  2. The adjacency matrix is the only CAGRA state that cannot be re-derived
 *     from the vectors, so persistence exchanges it raw and reloads it with
 *     cuvsCagraIndexFromArgs -- no nn_descent, no cuVS file round trip.
 *  3. A per-query device allocation round trip dominated single-query latency,
 *     so search scratch buffers are allocated once and grown by high-water
 *     mark.
 */

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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SK_DEFAULT_GRAPH_DEGREE 128
#define SK_DEFAULT_NN_DESCENT_NITER 32
#define SK_DEFAULT_SEARCH_WIDTH 4
#define SK_MIN_ITOPK 512

typedef struct {
  cuvsCagraBuildAlgo_t algo;
  size_t degree;
  size_t intermediate;
  size_t niter;
  size_t itopk;      /* 0 means "derive from topk" */
  uint32_t width;
} sk_config;

typedef struct {
  cuvsResources_t resources;
  cuvsCagraIndex_t index;
  long n;
  long dim;
  pthread_mutex_t lock;
  /* Search state, created on first use and reused afterwards. */
  cuvsCagraSearchParams_t search;
  size_t scratch_queries; /* capacity in floats */
  size_t scratch_results; /* capacity in elements, per output array */
  float *query_dev;
  unsigned *id_dev;
  float *dist_dev;
} sk_index;

static size_t sk_env_num(const char *name, size_t fallback)
{
  const char *raw = getenv(name);
  long parsed = 0;
  if (raw == NULL || raw[0] == '\0') { return fallback; }
  parsed = strtol(raw, NULL, 10);
  return parsed > 0 ? (size_t)parsed : fallback;
}

static void sk_load_config(sk_config *cfg)
{
  const char *algo = getenv("SEEKDB_CUVS_BUILD_ALGO");
  memset(cfg, 0, sizeof(*cfg));
  cfg->algo = (algo != NULL && strcmp(algo, "auto") == 0) ? AUTO_SELECT : NN_DESCENT;
  cfg->degree = sk_env_num("SEEKDB_CUVS_GRAPH_DEGREE", SK_DEFAULT_GRAPH_DEGREE);
  cfg->intermediate = sk_env_num("SEEKDB_CUVS_INTERMEDIATE_DEGREE", cfg->degree);
  if (cfg->intermediate < cfg->degree) { cfg->intermediate = cfg->degree; }
  cfg->niter = sk_env_num("SEEKDB_CUVS_NN_DESCENT_NITER", SK_DEFAULT_NN_DESCENT_NITER);
  cfg->itopk = sk_env_num("SEEKDB_CUVS_SEARCH_ITOPK", 0);
  cfg->width = (uint32_t)sk_env_num("SEEKDB_CUVS_SEARCH_WIDTH", SK_DEFAULT_SEARCH_WIDTH);
}

static void sk_report(const char *what, int rc)
{
  fprintf(stderr, "[seekdb_cuvs] %s (rc=%d): %s\n", what, rc, cuvsGetLastErrorText());
}

static void sk_tensor(DLManagedTensor *tensor, void *data, int64_t *shape,
                      DLDeviceType device, int code, int bits)
{
  memset(tensor, 0, sizeof(*tensor));
  tensor->dl_tensor.data = data;
  tensor->dl_tensor.device.device_type = device;
  tensor->dl_tensor.device.device_id = 0;
  tensor->dl_tensor.ndim = 2;
  tensor->dl_tensor.dtype.code = (uint8_t)code;
  tensor->dl_tensor.dtype.bits = (uint8_t)bits;
  tensor->dl_tensor.dtype.lanes = 1;
  tensor->dl_tensor.shape = shape;
  tensor->dl_tensor.strides = NULL;
  tensor->dl_tensor.byte_offset = 0;
}

static sk_index *sk_index_create(void)
{
  sk_index *h = (sk_index *)calloc(1, sizeof(*h));
  if (h == NULL) { return NULL; }
  if (pthread_mutex_init(&h->lock, NULL) != 0) { free(h); return NULL; }
  if (cuvsResourcesCreate(&h->resources) != CUVS_SUCCESS) {
    sk_report("resource creation failed", 1);
    pthread_mutex_destroy(&h->lock);
    free(h);
    return NULL;
  }
  if (cuvsCagraIndexCreate(&h->index) != CUVS_SUCCESS) {
    sk_report("index creation failed", 2);
    cuvsResourcesDestroy(h->resources);
    pthread_mutex_destroy(&h->lock);
    free(h);
    return NULL;
  }
  return h;
}

static void sk_index_destroy(sk_index *h)
{
  if (h == NULL) { return; }
  if (h->search != NULL) { cuvsCagraSearchParamsDestroy(h->search); }
  if (h->query_dev != NULL) {
    cuvsRMMFree(h->resources, h->query_dev, sizeof(float) * h->scratch_queries);
  }
  if (h->id_dev != NULL) {
    cuvsRMMFree(h->resources, h->id_dev, sizeof(unsigned) * h->scratch_results);
  }
  if (h->dist_dev != NULL) {
    cuvsRMMFree(h->resources, h->dist_dev, sizeof(float) * h->scratch_results);
  }
  cuvsCagraIndexDestroy(h->index);
  cuvsResourcesDestroy(h->resources);
  pthread_mutex_destroy(&h->lock);
  free(h);
}

void *seekdb_cuvs_build(const float *base, long n, long dim)
{
  sk_index *h = NULL;
  cuvsCagraIndexParams_t params = NULL;
  sk_config cfg;
  DLManagedTensor dataset;
  int64_t shape[2];

  if (base == NULL || n <= 0 || dim <= 0) { return NULL; }
  h = sk_index_create();
  if (h == NULL) { return NULL; }

  sk_load_config(&cfg);
  shape[0] = n;
  shape[1] = dim;
  sk_tensor(&dataset, (void *)base, shape, kDLCPU, kDLFloat, 32);

  if (cuvsCagraIndexParamsCreate(&params) != CUVS_SUCCESS) {
    sk_report("index params creation failed", 3);
    sk_index_destroy(h);
    return NULL;
  }
  params->metric = L2Expanded;
  params->build_algo = cfg.algo;
  params->graph_degree = cfg.degree;
  params->intermediate_graph_degree = cfg.intermediate;
  params->nn_descent_niter = cfg.niter;

  if (cuvsCagraBuild(h->resources, params, &dataset, h->index) != CUVS_SUCCESS) {
    sk_report("build failed", 4);
    cuvsCagraIndexParamsDestroy(params);
    sk_index_destroy(h);
    return NULL;
  }
  cuvsCagraIndexParamsDestroy(params);
  h->n = n;
  h->dim = dim;
  return h;
}

static int sk_reserve(sk_index *h, size_t queries, size_t results)
{
  if (queries > h->scratch_queries) {
    if (h->query_dev != NULL) {
      cuvsRMMFree(h->resources, h->query_dev, sizeof(float) * h->scratch_queries);
      h->query_dev = NULL;
      h->scratch_queries = 0;
    }
    if (cuvsRMMAlloc(h->resources, (void **)&h->query_dev, sizeof(float) * queries) !=
        CUVS_SUCCESS) {
      sk_report("query staging allocation failed", 10);
      return 10;
    }
    h->scratch_queries = queries;
  }
  if (results > h->scratch_results) {
    if (h->id_dev != NULL) {
      cuvsRMMFree(h->resources, h->id_dev, sizeof(unsigned) * h->scratch_results);
      h->id_dev = NULL;
    }
    if (h->dist_dev != NULL) {
      cuvsRMMFree(h->resources, h->dist_dev, sizeof(float) * h->scratch_results);
      h->dist_dev = NULL;
    }
    h->scratch_results = 0;
    if (cuvsRMMAlloc(h->resources, (void **)&h->id_dev, sizeof(unsigned) * results) !=
        CUVS_SUCCESS) {
      sk_report("id staging allocation failed", 11);
      return 11;
    }
    if (cuvsRMMAlloc(h->resources, (void **)&h->dist_dev, sizeof(float) * results) !=
        CUVS_SUCCESS) {
      cuvsRMMFree(h->resources, h->id_dev, sizeof(unsigned) * results);
      h->id_dev = NULL;
      sk_report("distance staging allocation failed", 12);
      return 12;
    }
    h->scratch_results = results;
  }
  return 0;
}

static int sk_prepare_search(sk_index *h, long topk)
{
  sk_config cfg;
  size_t itopk = 0;
  if (h->search != NULL) { return 0; }
  sk_load_config(&cfg);
  if (cuvsCagraSearchParamsCreate(&h->search) != CUVS_SUCCESS) {
    sk_report("search params creation failed", 14);
    return 14;
  }
  itopk = cfg.itopk > 0 ? cfg.itopk : (size_t)topk * 4;
  if (itopk < SK_MIN_ITOPK) { itopk = SK_MIN_ITOPK; }
  h->search->itopk_size = itopk;
  h->search->search_width = cfg.width;
  return 0;
}

int seekdb_cuvs_search(void *handle, const float *query, long nq, long topk,
                       unsigned int *out_ids, float *out_dist)
{
  sk_index *h = (sk_index *)handle;
  DLManagedTensor queries, ids, distances;
  int64_t qshape[2], ishape[2], dshape[2];
  cuvsFilter filter;
  size_t need_itopk = 0;
  int rc = 0;

  if (h == NULL || query == NULL || out_ids == NULL || nq <= 0 || topk <= 0) { return 1; }
  pthread_mutex_lock(&h->lock);
  rc = sk_reserve(h, (size_t)nq * (size_t)h->dim, (size_t)nq * (size_t)topk);
  if (rc == 0) { rc = sk_prepare_search(h, topk); }
  if (rc == 0) {
    need_itopk = (size_t)topk * 4;
    if (h->search->itopk_size < need_itopk) { h->search->itopk_size = need_itopk; }
  }
  if (rc == 0 &&
      cudaMemcpy(h->query_dev, query, sizeof(float) * (size_t)nq * (size_t)h->dim,
                 cudaMemcpyDefault) != cudaSuccess) {
    sk_report("query upload failed", 13);
    rc = 13;
  }
  if (rc == 0) {
    qshape[0] = nq;
    qshape[1] = h->dim;
    ishape[0] = nq;
    ishape[1] = topk;
    dshape[0] = nq;
    dshape[1] = topk;
    sk_tensor(&queries, h->query_dev, qshape, kDLCUDA, kDLFloat, 32);
    sk_tensor(&ids, h->id_dev, ishape, kDLCUDA, kDLUInt, 32);
    sk_tensor(&distances, h->dist_dev, dshape, kDLCUDA, kDLFloat, 32);
    filter.type = NO_FILTER;
    filter.addr = (uintptr_t)0;
    if (cuvsCagraSearch(h->resources, h->search, h->index, &queries, &ids, &distances,
                        filter) != CUVS_SUCCESS) {
      sk_report("search failed", 15);
      rc = 15;
    }
  }
  if (rc == 0 &&
      cudaMemcpy(out_ids, h->id_dev, sizeof(unsigned) * (size_t)nq * (size_t)topk,
                 cudaMemcpyDefault) != cudaSuccess) {
    sk_report("id download failed", 16);
    rc = 16;
  }
  if (rc == 0 && out_dist != NULL &&
      cudaMemcpy(out_dist, h->dist_dev, sizeof(float) * (size_t)nq * (size_t)topk,
                 cudaMemcpyDefault) != cudaSuccess) {
    sk_report("distance download failed", 17);
    rc = 17;
  }
  pthread_mutex_unlock(&h->lock);
  return rc;
}

void seekdb_cuvs_free(void *handle)
{
  sk_index_destroy((sk_index *)handle);
}

int seekdb_cuvs_serialize(void *handle, const char *filename)
{
  sk_index *h = (sk_index *)handle;
  if (h == NULL || filename == NULL) { return 1; }
  if (cuvsCagraSerialize(h->resources, filename, h->index, true) != CUVS_SUCCESS) {
    sk_report("serialize failed", 2);
    return 2;
  }
  return 0;
}

void *seekdb_cuvs_deserialize(const char *filename, long n, long dim)
{
  sk_index *h = NULL;
  int64_t actual_n = 0;
  int64_t actual_dim = 0;

  if (filename == NULL || n <= 0 || dim <= 0) { return NULL; }
  h = sk_index_create();
  if (h == NULL) { return NULL; }
  if (cuvsCagraDeserialize(h->resources, filename, h->index) != CUVS_SUCCESS) {
    sk_report("deserialize failed", 3);
    sk_index_destroy(h);
    return NULL;
  }
  /* A snapshot must match the index definition it is loaded into. */
  if (cuvsCagraIndexGetSize(h->index, &actual_n) != CUVS_SUCCESS ||
      cuvsCagraIndexGetDims(h->index, &actual_dim) != CUVS_SUCCESS ||
      actual_n != n || actual_dim != dim) {
    fprintf(stderr, "[seekdb_cuvs] snapshot shape %ldx%ld, expected %ldx%ld\n",
            (long)actual_n, (long)actual_dim, n, dim);
    sk_index_destroy(h);
    return NULL;
  }
  h->n = n;
  h->dim = dim;
  return h;
}

int seekdb_cuvs_export_graph(void *handle, unsigned int **out_graph,
                             long *out_n, long *out_degree)
{
  sk_index *h = (sk_index *)handle;
  DLManagedTensor graph;
  unsigned *host = NULL;
  long rows = 0;
  long degree = 0;
  size_t bytes = 0;
  int rc = 0;

  if (h == NULL || out_graph == NULL || out_n == NULL || out_degree == NULL) { return 1; }
  pthread_mutex_lock(&h->lock);
  if (cuvsCagraIndexGetGraph(h->index, &graph) != CUVS_SUCCESS) {
    sk_report("graph export failed", 2);
    rc = 2;
  } else {
    rows = (long)graph.dl_tensor.shape[0];
    degree = (long)graph.dl_tensor.shape[1];
    if (rows <= 0 || degree <= 0) {
      rc = 3;
    } else {
      bytes = sizeof(unsigned) * (size_t)rows * (size_t)degree;
      host = (unsigned *)malloc(bytes);
      if (host == NULL) {
        rc = 4;
      } else if (cudaMemcpy(host, graph.dl_tensor.data, bytes, cudaMemcpyDefault) !=
                 cudaSuccess) {
        sk_report("graph download failed", 5);
        free(host);
        host = NULL;
        rc = 5;
      }
    }
    if (graph.deleter != NULL) { graph.deleter(&graph); }
  }
  pthread_mutex_unlock(&h->lock);
  if (rc != 0) { return rc; }
  *out_graph = host;
  *out_n = rows;
  *out_degree = degree;
  return 0;
}

void seekdb_cuvs_free_host(void *ptr)
{
  free(ptr);
}

void *seekdb_cuvs_import(const float *base, long n, long dim,
                         const unsigned int *graph, long degree)
{
  sk_index *h = NULL;
  DLManagedTensor dataset, adjacency;
  int64_t dshape[2], gshape[2];

  if (base == NULL || graph == NULL || n <= 0 || dim <= 0 || degree <= 0) { return NULL; }
  h = sk_index_create();
  if (h == NULL) { return NULL; }
  dshape[0] = n;
  dshape[1] = dim;
  gshape[0] = n;
  gshape[1] = degree;
  sk_tensor(&dataset, (void *)base, dshape, kDLCPU, kDLFloat, 32);
  sk_tensor(&adjacency, (void *)graph, gshape, kDLCPU, kDLUInt, 32);
  if (cuvsCagraIndexFromArgs(h->resources, L2Expanded, &adjacency, &dataset,
                             h->index) != CUVS_SUCCESS) {
    sk_report("graph import failed", 2);
    sk_index_destroy(h);
    return NULL;
  }
  h->n = n;
  h->dim = dim;
  return h;
}

int seekdb_cuvs_cagra_knn(const float *base, long n, long dim,
                          const float *query, long nq, long topk,
                          unsigned int *out_ids)
{
  void *h = seekdb_cuvs_build(base, n, dim);
  int rc = 0;
  if (h == NULL) { return 10; }
  rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
