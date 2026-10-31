#ifndef SEEKDB_CUVS_BRIDGE_H
#define SEEKDB_CUVS_BRIDGE_H
#ifdef __cplusplus
extern "C" {
#endif

/* Plain-C surface between seekdb and the AMD GPU (hipVS/libcuvs_c).
 *
 * Every ROCm/cuVS type stays inside libseekdb_cuvs_bridge.so; seekdb links only
 * against these symbols. Pointers handed back through this API are opaque and
 * must be released with the matching seekdb_cuvs_* entry point.
 *
 * Build-time knobs (read from the environment, one value per process start):
 *   SEEKDB_CUVS_BUILD_ALGO         nn_descent (default) | auto
 *   SEEKDB_CUVS_GRAPH_DEGREE       persisted graph degree (default 128)
 *   SEEKDB_CUVS_INTERMEDIATE_DEGREE pre-prune degree (default = graph degree)
 *   SEEKDB_CUVS_NN_DESCENT_NITER   refinement rounds (default 32)
 *   SEEKDB_CUVS_SEARCH_ITOPK       candidate list size (default 4*topk, >=512)
 *   SEEKDB_CUVS_SEARCH_WIDTH       graph walk width (default 4)
 */

/* One-shot build + search (used by tools/hipvs_bridge and batch_knn).
 * base: n x dim row-major f32. query: nq x dim row-major f32.
 * out_ids: nq*topk uint32 row offsets into base. Returns 0 on success. */
int seekdb_cuvs_cagra_knn(const float *base, long n, long dim,
                          const float *query, long nq, long topk,
                          unsigned int *out_ids);

/* Build-once / search-many.
 *   seekdb_cuvs_build      builds CAGRA from base (n x dim f32); cuVS copies the
 *                          dataset to the device, so base may be freed afterwards.
 *                          Returns NULL on failure.
 *   seekdb_cuvs_search     top-k over nq queries, writes uint32 row offsets to
 *                          out_ids[nq*topk] and, when out_dist != NULL, L2
 *                          distances to out_dist[nq*topk]. Returns 0 on success.
 *   seekdb_cuvs_free       releases the handle (safe with NULL). */
void *seekdb_cuvs_build(const float *base, long n, long dim);
int   seekdb_cuvs_search(void *handle, const float *query, long nq, long topk,
                         unsigned int *out_ids, float *out_dist);
void   seekdb_cuvs_free(void *handle);

/* Whole-index persistence through a cuVS-owned file format (dataset included).
 * Kept for the standalone persistence tool; seekdb itself stores the graph via
 * the export/import pair below, which avoids re-serializing the dataset. */
int   seekdb_cuvs_serialize(void *handle, const char *filename);
void *seekdb_cuvs_deserialize(const char *filename, long n, long dim);

/* Graph transplant: the adjacency matrix is the only CAGRA state that cannot be
 * rebuilt from the vectors, so snapshots carry it verbatim.
 *   seekdb_cuvs_export_graph  copies the device graph to a host buffer (malloc);
 *                             returns 0 on success, caller frees with
 *                             seekdb_cuvs_free_host.
 *   seekdb_cuvs_import        rebuilds a searchable index from a host dataset
 *                             plus a host adjacency matrix; returns NULL on
 *                             failure.
 *   seekdb_cuvs_free_host     frees buffers produced by export_graph. */
int   seekdb_cuvs_export_graph(void *handle, unsigned int **out_graph,
                               long *out_n, long *out_degree);
void *seekdb_cuvs_import(const float *base, long n, long dim,
                         const unsigned int *graph, long degree);
void  seekdb_cuvs_free_host(void *ptr);

#ifdef __cplusplus
}
#endif
#endif
