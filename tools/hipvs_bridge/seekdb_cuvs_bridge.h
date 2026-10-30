#ifndef SEEKDB_CUVS_BRIDGE_H
#define SEEKDB_CUVS_BRIDGE_H
#ifdef __cplusplus
extern "C" {
#endif
/* One-shot: build a CAGRA index over base (n x dim, row-major f32) on the GPU and
 * search queries (nq x dim), writing top-k uint32 neighbor ids to out_ids
 * (length nq*topk). Returns 0 on success, non-zero on error. */
int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids);

/* Build-once / search-many (used by seekdb's ob_vsag_adaptor GPU path).
 *  - seekdb_cuvs_build: builds a CAGRA index on the GPU from base (n x dim,
 *    row-major f32) and returns an opaque handle (NULL on error). cuVS copies
 *    the dataset to the device, so `base` may be freed after this returns.
 *  - seekdb_cuvs_search: runs top-k search for nq queries (row-major f32, dim
 *    cols), writing uint32 row offsets to out_ids[nq*topk] and (if non-NULL) L2
 *    distances to out_dist[nq*topk]. Returns 0 on success.
 *  - seekdb_cuvs_free: releases the handle.
 * All ROCm/cuVS state lives inside the .so; seekdb only ever sees these C symbols. */
void* seekdb_cuvs_build(const float* base, long n, long dim);
int   seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                         unsigned int* out_ids, float* out_dist);
void  seekdb_cuvs_free(void* handle);
int seekdb_cuvs_serialize(void* handle, const char* filename);
void* seekdb_cuvs_deserialize(const char* filename, long n, long dim);

/* Snapshot helpers. The CAGRA graph is the only part of the index that cannot be
 * rebuilt from the stored vectors, so seekdb persists it alone and reconstructs
 * the index on load with cuvsCagraIndexFromArgs (no NN-descent, no graph prune).
 *  - seekdb_cuvs_export_graph: copies the built graph (n x degree, uint32) to a
 *    freshly malloc'd host buffer owned by the caller.
 *  - seekdb_cuvs_import: builds a searchable index from host-resident dataset and
 *    graph; returns NULL on error.
 *  - seekdb_cuvs_free_host: releases a buffer returned by export. */
int  seekdb_cuvs_export_graph(void* handle, unsigned int** out_graph,
                              long* out_n, long* out_degree);
void* seekdb_cuvs_import(const float* base, long n, long dim,
                         const unsigned int* graph, long degree);
void seekdb_cuvs_free_host(void* ptr);
#ifdef __cplusplus
}
#endif
#endif
