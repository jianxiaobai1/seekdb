// MiniVectorDB - flat (exact) vector search.
#include "vector_index.h"

#include <algorithm>
#include <cmath>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define MVDB_X86 1
#endif

namespace mvdb {
namespace {

// Worst-first ordering: the heap root is the hit that should be evicted next
// (larger distance, then larger primary key). This lets a single bounded heap
// of size k scan a whole table without storing all n distances.
struct Better {
  bool operator()(const Hit& x, const Hit& y) const {
    if (x.dist != y.dist) return x.dist < y.dist;
    return x.pk < y.pk;
  }
};

void heap_push_bounded(std::vector<Hit>& heap, size_t k, const Hit& h,
                       const Better& better) {
  if (heap.size() < k) {
    heap.push_back(h);
    if (heap.size() == k) std::make_heap(heap.begin(), heap.end(), better);
    return;
  }
  if (better(h, heap.front())) {
    std::pop_heap(heap.begin(), heap.end(), better);
    heap.back() = h;
    std::push_heap(heap.begin(), heap.end(), better);
  }
}

#ifdef MVDB_X86
__attribute__((target("avx2,fma")))
double l2_avx2(const float* a, const float* b, size_t n) {
  // Same arithmetic (difference -> square -> add), four lanes at a time in
  // double precision. Four independent accumulator chains hide the 4-cycle
  // FMA latency; two chains leave the vector units half idle on long vectors.
  __m256d acc0 = _mm256_setzero_pd();
  __m256d acc1 = _mm256_setzero_pd();
  __m256d acc2 = _mm256_setzero_pd();
  __m256d acc3 = _mm256_setzero_pd();
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m256 va0 = _mm256_loadu_ps(a + i);
    __m256 vb0 = _mm256_loadu_ps(b + i);
    __m256 va1 = _mm256_loadu_ps(a + i + 8);
    __m256 vb1 = _mm256_loadu_ps(b + i + 8);
    __m256d d0 = _mm256_cvtps_pd(
        _mm256_castps256_ps128(_mm256_sub_ps(va0, vb0)));
    __m256d d1 = _mm256_cvtps_pd(
        _mm256_extractf128_ps(_mm256_sub_ps(va0, vb0), 1));
    __m256d d2 = _mm256_cvtps_pd(
        _mm256_castps256_ps128(_mm256_sub_ps(va1, vb1)));
    __m256d d3 = _mm256_cvtps_pd(
        _mm256_extractf128_ps(_mm256_sub_ps(va1, vb1), 1));
    acc0 = _mm256_fmadd_pd(d0, d0, acc0);
    acc1 = _mm256_fmadd_pd(d1, d1, acc1);
    acc2 = _mm256_fmadd_pd(d2, d2, acc2);
    acc3 = _mm256_fmadd_pd(d3, d3, acc3);
  }
  // Reduce the four chains, then finish 8-float and scalar tails.
  acc0 = _mm256_add_pd(acc0, acc1);
  acc2 = _mm256_add_pd(acc2, acc3);
  acc0 = _mm256_add_pd(acc0, acc2);
  __m128d s = _mm_add_pd(_mm256_castpd256_pd128(acc0),
                         _mm256_extractf128_pd(acc0, 1));
  s = _mm_hadd_pd(s, s);
  double sum = _mm_cvtsd_f64(s);
  for (; i + 8 <= n; i += 8) {
    __m256 vd = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    __m256d d0 = _mm256_cvtps_pd(_mm256_castps256_ps128(vd));
    __m256d d1 = _mm256_cvtps_pd(_mm256_extractf128_ps(vd, 1));
    d0 = _mm256_mul_pd(d0, d0);
    d1 = _mm256_mul_pd(d1, d1);
    d0 = _mm256_add_pd(d0, d1);
    __m128d t = _mm_add_pd(_mm256_castpd256_pd128(d0),
                           _mm256_extractf128_pd(d0, 1));
    t = _mm_hadd_pd(t, t);
    sum += _mm_cvtsd_f64(t);
  }
  for (; i < n; ++i) {
    double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sum += d * d;
  }
  return sum;
}

bool have_avx2_fma() {
  static const bool ok = [] {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  }();
  return ok;
}
#endif

double l2_scalar(const float* a, const float* b, size_t n) {
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sum += d * d;
  }
  return sum;
}

using LocalDistFn = double (*)(const float*, const float*, size_t);

LocalDistFn pick_distance() {
#ifdef MVDB_X86
  if (have_avx2_fma()) return l2_avx2;
#endif
  return l2_scalar;
}

}  // namespace

DistFn distance_fn() {
  static const DistFn fn = pick_distance();
  return fn;
}

double l2_distance_sq(const std::vector<float>& a, const std::vector<float>& b) {
  return l2_distance_sq(a.data(), b.data(), a.size());
}

double l2_distance_sq(const float* a, const float* b, size_t n) {
  return distance_fn()(a, b, n);
}

std::vector<Hit> FlatIndex::topk(const Table& table,
                                 const std::vector<float>& query, size_t k,
                                 const std::vector<uint32_t>& candidates) const {
  scratch_.clear();
  if (k == 0) return {};
  const Better better;
  const DistFn fn = distance_fn();
  const int vc = table.schema.vec;
  const int pkc = table.schema.pk;
  const float* q = query.data();
  const size_t dim = query.size();
  if (table.scan_ready()) {
    const float* const* vecs = table.scan_vectors();
    const int64_t* pks = table.pk_array();
    for (uint32_t slot : candidates) {
      Hit h;
      h.dist = fn(vecs[slot], q, dim);
      h.pk = pks[slot];
      heap_push_bounded(scratch_, k, h, better);
    }
  } else {
    Row row_scratch;
    for (uint32_t slot : candidates) {
      const Row* row = table.resident_row(slot);
      if (row == nullptr) {
        table.load_row(slot, row_scratch);
        row = &row_scratch;
      }
      Hit h;
      h.dist = fn((*row)[vc].v.data(), q, dim);
      h.pk = (*row)[pkc].i;
      heap_push_bounded(scratch_, k, h, better);
    }
  }
  std::sort(scratch_.begin(), scratch_.end(), better);
  return scratch_;
}

std::vector<Hit> FlatIndex::topk_all(const Table& table,
                                     const std::vector<float>& query,
                                     size_t k) const {
  scratch_.clear();
  if (k == 0) return {};
  const Better better;
  const DistFn fn = distance_fn();
  const int vc = table.schema.vec;
  const int pkc = table.schema.pk;
  const float* q = query.data();
  const size_t dim = query.size();
  const size_t n = table.rows.size();
  if (table.scan_ready()) {
    const float* const* vecs = table.scan_vectors();
    const int64_t* pks = table.pk_array();
    const uint8_t* alive = table.alive.data();
    for (size_t i = 0; i < n; ++i) {
      if (!alive[i]) continue;
      Hit h;
      h.dist = fn(vecs[i], q, dim);
      h.pk = pks[i];
      heap_push_bounded(scratch_, k, h, better);
    }
  } else {
    Row row_scratch;
    for (size_t i = 0; i < n; ++i) {
      if (!table.alive[i]) continue;
      const Row& row = table.view(static_cast<uint32_t>(i), row_scratch);
      Hit h;
      h.dist = fn(row[vc].v.data(), q, dim);
      h.pk = row[pkc].i;
      heap_push_bounded(scratch_, k, h, better);
    }
  }
  std::sort(scratch_.begin(), scratch_.end(), better);
  return scratch_;
}

}  // namespace mvdb
