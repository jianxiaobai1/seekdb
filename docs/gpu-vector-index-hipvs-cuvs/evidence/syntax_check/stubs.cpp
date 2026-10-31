#include <pthread.h>
// Local syntax/type check for the changed region of ob_vsag_adaptor.cpp.
// The region is compiled verbatim against minimal stubs of the OceanBase
// symbols it uses, so typos in the edited code surface without a full seekdb
// build (which needs ROCm and a 2-hour compile).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <ostream>
#include <set>
#include <sstream>
#include <vector>

// ---- OceanBase stubs -------------------------------------------------------
#define OB_SUCCESS 0
#define OB_ERR_UNEXPECTED -1
#define OB_ERR_VSAG_RETURN_ERROR -2
#define OB_ALLOCATE_MEMORY_FAILED -3

struct FakeLog {
  template <typename T> FakeLog &operator<<(const T &) { return *this; }
};
#define LOG_INFO(...) FakeLog()
#define LOG_WARN(...) FakeLog()
struct FakeKey {
  template <typename T> FakeKey &operator<<(const T &) { return *this; }
};
#define K(x) FakeKey()
#define KP(x) FakeKey()

struct ObTimeUtility {
  static int64_t current_time() { return 0; }
};

struct ObCuvsEntry {
  std::shared_ptr<void> bridge_;
  int dim_ = 0;
  std::vector<float> buf_vecs_;
  std::vector<int64_t> buf_ids_;
  std::vector<int64_t> ids_;
  size_t built_n_ = 0;
  bool vsag_ready_ = false;
};

static std::mutex g_ob_cuvs_mu;
static std::map<void *, ObCuvsEntry *> g_ob_cuvs_reg;
static std::mutex g_ob_cuvs_marked_mu;
static std::set<void *> g_ob_cuvs_marked;
static const size_t OB_CUVS_MIN_PTS = 256;

// ---- bridge stubs ----------------------------------------------------------
extern "C" void *seekdb_cuvs_build(const float *base, long n, long dim);
extern "C" int   seekdb_cuvs_search(void *handle, const float *query, long nq, long topk,
                                    unsigned int *out_ids, float *out_dist);
extern "C" void  seekdb_cuvs_free(void *handle);
extern "C" int seekdb_cuvs_serialize(void *handle, const char *filename);
extern "C" void *seekdb_cuvs_deserialize(const char *filename, long n, long dim);
extern "C" int   seekdb_cuvs_export_graph(void *handle, unsigned int **out_graph,
                                          long *out_n, long *out_degree);
extern "C" void *seekdb_cuvs_import(const float *base, long n, long dim,
                                    const unsigned int *graph, long degree);
extern "C" void  seekdb_cuvs_free_host(void *ptr);

// ---- helpers the region uses (copied from the adaptor) ---------------------
static std::shared_ptr<void> ob_cuvs_wrap(void *handle) {
  if (handle == nullptr) { return std::shared_ptr<void>(); }
  return std::shared_ptr<void>(handle, [](void *h) { seekdb_cuvs_free(h); });
}
struct ObCuvsBuildJob {
  const float *base; long n; long dim; void *out;
};
static void *ob_cuvs_build_job(void *arg) {
  ObCuvsBuildJob *j = static_cast<ObCuvsBuildJob *>(arg);
  j->out = seekdb_cuvs_build(j->base, j->n, j->dim);
  return nullptr;
}
static std::shared_ptr<void> ob_cuvs_build_big_stack(const float *base, long n, long dim) {
  ObCuvsBuildJob job{base, n, dim, nullptr};
  pthread_t tid; pthread_attr_t attr; pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 32UL * 1024 * 1024);
  if (pthread_create(&tid, &attr, ob_cuvs_build_job, &job) == 0) { pthread_join(tid, nullptr); }
  pthread_attr_destroy(&attr);
  return ob_cuvs_wrap(job.out);
}

#include "region.inc"

int main() { return 0; }
