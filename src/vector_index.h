// MiniVectorDB - vector index interface.
//
// FlatIndex is an exact brute-force scan, used for both `exact` and `ann`
// query modes for now (exact results are valid in `ann` mode, recall = 1.0).
// A real approximate index (IVF / HNSW) plugs in behind this interface later
// without touching the executor.
#pragma once

#include <cstdint>
#include <vector>

#include "storage.h"

namespace mvdb {

struct Hit {
  double dist = 0.0;
  int64_t pk = 0;
};

double l2_distance_sq(const std::vector<float>& a, const std::vector<float>& b);
double l2_distance_sq(const float* a, const float* b, size_t n);

// Resolved once per query so the scan loop does not re-check CPU features or
// call through two layers of function pointers for every row.
using DistFn = double (*)(const float*, const float*, size_t);
DistFn distance_fn();

class VectorIndex {
 public:
  virtual ~VectorIndex() = default;

  // Returns up to `k` hits restricted to `candidates` (row slots), ordered by
  // (distance asc, primary key asc).
  virtual std::vector<Hit> topk(const Table& table, const std::vector<float>& query,
                                size_t k,
                                const std::vector<uint32_t>& candidates) const = 0;

  // Same as topk() over every live row, without materialising a candidate
  // list first (hot path for queries without a WHERE clause).
  virtual std::vector<Hit> topk_all(const Table& table,
                                    const std::vector<float>& query,
                                    size_t k) const = 0;
};

class FlatIndex : public VectorIndex {
 public:
  std::vector<Hit> topk(const Table& table, const std::vector<float>& query,
                        size_t k,
                        const std::vector<uint32_t>& candidates) const override;
  std::vector<Hit> topk_all(const Table& table, const std::vector<float>& query,
                            size_t k) const override;

 private:
  // Reused across queries to avoid re-allocating the result buffer for every
  // TopK request (single client, single thread by contract).
  mutable std::vector<Hit> scratch_;
};

}  // namespace mvdb
