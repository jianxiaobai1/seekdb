static const char OB_CUVS_SNAPSHOT_MAGIC[8] = {'O', 'B', 'C', 'U', 'V', 'S', '0', '3'};

// cuVS build/import/search overflow the small OB worker/DDL stack, so those
// calls run on a 32MB pthread.
typedef void *(*ObCuvsThreadFn)(void *);
static bool ob_cuvs_big_stack(ObCuvsThreadFn fn, void *arg) {
  pthread_t thread;
  pthread_attr_t attr;
  if (pthread_attr_init(&attr) != 0) { return false; }
  int rc = pthread_attr_setstacksize(&attr, 32UL * 1024 * 1024);
  if (rc == 0) { rc = pthread_create(&thread, &attr, fn, arg); }
  pthread_attr_destroy(&attr);
  return rc == 0 && pthread_join(thread, nullptr) == 0;
}

struct ObCuvsGraphJob {
  void *handle_; unsigned *graph_; long n_; long degree_; int rc_;
};
static void *ob_cuvs_graph_job(void *arg) {
  ObCuvsGraphJob *job = static_cast<ObCuvsGraphJob *>(arg);
  job->rc_ = seekdb_cuvs_export_graph(job->handle_, &job->graph_, &job->n_, &job->degree_);
  return nullptr;
}

struct ObCuvsImportJob {
  const float *base_; long n_; long dim_; const unsigned *graph_; long degree_; void *out_;
};
static void *ob_cuvs_import_job(void *arg) {
  ObCuvsImportJob *job = static_cast<ObCuvsImportJob *>(arg);
  job->out_ = seekdb_cuvs_import(job->base_, job->n_, job->dim_, job->graph_, job->degree_);
  return nullptr;
}

static bool ob_cuvs_write(std::ostream &out, const void *data, size_t bytes) {
  const char *p = static_cast<const char *>(data);
  const size_t CHUNK = 64UL * 1024 * 1024;
  while (bytes > 0 && out.good()) {
    const size_t sz = bytes < CHUNK ? bytes : CHUNK;
    out.write(p, static_cast<std::streamsize>(sz));
    p += sz; bytes -= sz;
  }
  return out.good();
}

static bool ob_cuvs_read(std::istream &in, void *data, size_t bytes) {
  char *p = static_cast<char *>(data);
  const size_t CHUNK = 64UL * 1024 * 1024;
  while (bytes > 0) {
    const size_t sz = bytes < CHUNK ? bytes : CHUNK;
    in.read(p, static_cast<std::streamsize>(sz));
    if (in.gcount() != static_cast<std::streamsize>(sz)) { return false; }
    p += sz; bytes -= sz;
  }
  return true;
}

// Returns false when the index is not a GPU candidate and must use the VSAG format.
static bool ob_cuvs_serialize(void *key, std::ostream &out, int &ret) {
  ret = OB_SUCCESS;
  std::lock_guard<std::mutex> guard(g_ob_cuvs_mu);
  auto it = g_ob_cuvs_reg.find(key);
  if (it == g_ob_cuvs_reg.end()) { return false; }
  ObCuvsEntry *ent = it->second;
  const size_t n = ent->buf_ids_.size();
  if (n < OB_CUVS_MIN_PTS || ent->dim_ <= 0 ||
      ent->buf_vecs_.size() != n * static_cast<size_t>(ent->dim_)) {
    return false;
  }
  const int64_t start_us = ObTimeUtility::current_time();
  bool built = false;
  if (ent->bridge_ == nullptr || ent->built_n_ != n) {
    std::shared_ptr<void> nb = ob_cuvs_build_big_stack(ent->buf_vecs_.data(),
                                                       static_cast<long>(n), ent->dim_);
    if (nb == nullptr) {
      ret = OB_ERR_VSAG_RETURN_ERROR;
      LOG_WARN("[OBVSAG][cuVS] GPU CAGRA build failed during snapshot serialize",
               K(ret), KP(key), K(n), K(ent->dim_));
      return true;
    }
    ent->bridge_ = nb; ent->built_n_ = n; ent->ids_ = ent->buf_ids_;
    built = true;
  }
  const int64_t build_us = ObTimeUtility::current_time() - start_us;

  // Export the adjacency list. It is the only piece of CAGRA state that cannot
  // be recomputed from the vectors, and exporting it costs one device-to-host
  // copy instead of the baseline's serialized-file round trip.
  ObCuvsGraphJob graph_job{ent->bridge_.get(), nullptr, 0, 0, -1};
  if (!ob_cuvs_big_stack(ob_cuvs_graph_job, &graph_job) || graph_job.rc_ != 0 ||
      graph_job.graph_ == nullptr || graph_job.degree_ <= 0 ||
      graph_job.n_ != static_cast<long>(n)) {
    if (graph_job.graph_ != nullptr) { seekdb_cuvs_free_host(graph_job.graph_); }
    ret = OB_ERR_VSAG_RETURN_ERROR;
    LOG_WARN("[OBVSAG][cuVS] failed to export CAGRA graph", K(ret), KP(key), K(n));
    return true;
  }
  const int32_t dim32 = ent->dim_;
  const int32_t degree32 = static_cast<int32_t>(graph_job.degree_);
  const int64_t n64 = static_cast<int64_t>(n);
  const uint64_t graph_bytes =
      static_cast<uint64_t>(n64) * static_cast<uint64_t>(degree32) * sizeof(unsigned int);
  const bool write_ok =
      ob_cuvs_write(out, OB_CUVS_SNAPSHOT_MAGIC, sizeof(OB_CUVS_SNAPSHOT_MAGIC)) &&
      ob_cuvs_write(out, &graph_bytes, sizeof(graph_bytes)) &&
      ob_cuvs_write(out, &dim32, sizeof(dim32)) &&
      ob_cuvs_write(out, &degree32, sizeof(degree32)) &&
      ob_cuvs_write(out, &n64, sizeof(n64)) &&
      ob_cuvs_write(out, ent->buf_ids_.data(), n * sizeof(int64_t)) &&
      ob_cuvs_write(out, ent->buf_vecs_.data(), ent->buf_vecs_.size() * sizeof(float)) &&
      ob_cuvs_write(out, graph_job.graph_, graph_bytes);
  seekdb_cuvs_free_host(graph_job.graph_);
  if (!write_ok) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("[OBVSAG][cuVS] failed to write snapshot stream", K(ret), KP(key), K(n));
    return true;
  }
  const int64_t total_us = ObTimeUtility::current_time() - start_us;
  LOG_INFO("[OBVSAG][cuVS] serialized GPU CAGRA snapshot", KP(key), K(n), K(dim32),
           K(degree32), K(built), K(build_us), K(total_us), K(graph_bytes));
  return true;
}

// Consumes the stream after the magic. Registers and marks the handle for GPU search.
static int ob_cuvs_deserialize(void *key, int expected_dim, std::istream &in) {
  int ret = OB_SUCCESS;
  uint64_t graph_bytes = 0;
  int32_t dim32 = 0;
  int32_t degree32 = 0;
  int64_t n64 = 0;
  if (!ob_cuvs_read(in, &graph_bytes, sizeof(graph_bytes)) || !ob_cuvs_read(in, &dim32, sizeof(dim32)) ||
      !ob_cuvs_read(in, &degree32, sizeof(degree32)) || !ob_cuvs_read(in, &n64, sizeof(n64))) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("[OBVSAG][cuVS] truncated snapshot header", K(ret), KP(key));
  } else if (dim32 <= 0 || dim32 != expected_dim || degree32 <= 0 || degree32 > 1024 || n64 <= 0 ||
             n64 > UINT32_MAX ||
             static_cast<uint64_t>(n64) > std::numeric_limits<size_t>::max() / sizeof(float) / dim32 ||
             static_cast<uint64_t>(n64) > static_cast<uint64_t>(INT64_MAX) / sizeof(unsigned int) / degree32 ||
             graph_bytes != static_cast<uint64_t>(n64) * degree32 * sizeof(unsigned int)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("[OBVSAG][cuVS] invalid snapshot header", K(ret), KP(key), K(dim32),
             K(expected_dim), K(degree32), K(n64), K(graph_bytes));
  } else {
    const size_t n = static_cast<size_t>(n64);
    ObCuvsEntry *ent = new (std::nothrow) ObCuvsEntry();
    unsigned *graph = static_cast<unsigned *>(::malloc(static_cast<size_t>(graph_bytes)));
    if (ent == nullptr || graph == nullptr) {
      delete ent;
      if (graph != nullptr) { seekdb_cuvs_free_host(graph); }
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("[OBVSAG][cuVS] failed to allocate snapshot buffers", K(ret), KP(key), K(n64));
    } else {
      ent->dim_ = dim32;
      ent->buf_ids_.resize(n);
      ent->buf_vecs_.resize(n * static_cast<size_t>(dim32));
      if (!ob_cuvs_read(in, ent->buf_ids_.data(), n * sizeof(int64_t)) ||
          !ob_cuvs_read(in, ent->buf_vecs_.data(), ent->buf_vecs_.size() * sizeof(float)) ||
          !ob_cuvs_read(in, graph, static_cast<size_t>(graph_bytes))) {
        seekdb_cuvs_free_host(graph);
        delete ent;
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("[OBVSAG][cuVS] truncated snapshot payload", K(ret), KP(key), K(n64));
      } else {
        // Rebuild the searchable index straight from the stored graph: no
        // NN-descent, no prune, no cuVS serializer file.
        ObCuvsImportJob job{ent->buf_vecs_.data(), static_cast<long>(n), dim32, graph, degree32, nullptr};
        const bool imported = ob_cuvs_big_stack(ob_cuvs_import_job, &job) && job.out_ != nullptr;
        seekdb_cuvs_free_host(graph);
        if (!imported) {
          delete ent;
          ret = OB_ERR_VSAG_RETURN_ERROR;
          LOG_WARN("[OBVSAG][cuVS] failed to import CAGRA graph from snapshot", K(ret), KP(key), K(n64));
          return ret;
        }
        ent->bridge_ = ob_cuvs_wrap(job.out_);
        ent->built_n_ = n;
        ent->ids_ = ent->buf_ids_;
        {
          std::lock_guard<std::mutex> guard(g_ob_cuvs_mu);
          ObCuvsEntry *&slot = g_ob_cuvs_reg[key];
          delete slot;
          slot = ent;
        }
        {
          std::lock_guard<std::mutex> guard(g_ob_cuvs_marked_mu);
          g_ob_cuvs_marked.insert(key);
        }
        LOG_INFO("[OBVSAG][cuVS] deserialized GPU CAGRA snapshot", KP(key), K(n64), K(dim32),
                 K(degree32), K(graph_bytes));
      }
    }
  }
  return ret;
}
