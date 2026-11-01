// Two snapshot layouts share one header: magic, graph payload size, dim, a
// degree word, row count, then the external ids and the fp32 vectors. v3 keeps
// the raw adjacency list, which is the only part of a CAGRA index that cannot
// be re-derived from the stored vectors, so loading costs one host-to-device
// copy. v2 kept a cuVS-serialized file (the graph plus a second copy of the
// dataset) and round-tripped it through a tmpfile on both ends.
static const char OB_CUVS_SNAPSHOT_MAGIC_V3[8] = {'O', 'B', 'C', 'U', 'V', 'S', '0', '3'};
static const char OB_CUVS_SNAPSHOT_MAGIC_V2[8] = {'O', 'B', 'C', 'U', 'V', 'S', '0', '2'};

typedef int (*ObCuvsExportGraphFn)(void *handle, unsigned int **out_graph,
                                   long *out_n, long *out_degree);
typedef void *(*ObCuvsImportFn)(const float *base, long n, long dim,
                                const unsigned int *graph, long degree);
typedef void (*ObCuvsFreeHostFn)(void *ptr);
struct ObCuvsGraphApi {
  ObCuvsExportGraphFn export_graph_;
  ObCuvsImportFn import_;
  ObCuvsFreeHostFn free_host_;
  bool available_;
};
// The graph entry points exist only in a bridge built from this tree. They are
// looked up instead of called directly so the observer still links against an
// older libseekdb_cuvs_bridge.so, which just takes the v2 snapshot path.
static const ObCuvsGraphApi &ob_cuvs_graph_api()
{
  static const ObCuvsGraphApi api = []() {
    ObCuvsGraphApi a = {nullptr, nullptr, nullptr, false};
    a.export_graph_ = reinterpret_cast<ObCuvsExportGraphFn>(
        ::dlsym(RTLD_DEFAULT, "seekdb_cuvs_export_graph"));
    a.import_ = reinterpret_cast<ObCuvsImportFn>(
        ::dlsym(RTLD_DEFAULT, "seekdb_cuvs_import"));
    a.free_host_ = reinterpret_cast<ObCuvsFreeHostFn>(
        ::dlsym(RTLD_DEFAULT, "seekdb_cuvs_free_host"));
    a.available_ = a.export_graph_ != nullptr && a.import_ != nullptr && a.free_host_ != nullptr;
    return a;
  }();
  return api;
}

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
  const ObCuvsGraphApi *api_; void *handle_; unsigned *graph_; long n_; long degree_; int rc_;
};
static void *ob_cuvs_graph_job(void *arg) {
  ObCuvsGraphJob *job = static_cast<ObCuvsGraphJob *>(arg);
  job->rc_ = job->api_->export_graph_(job->handle_, &job->graph_, &job->n_, &job->degree_);
  return nullptr;
}

struct ObCuvsImportJob {
  const ObCuvsGraphApi *api_; const float *base_; long n_; long dim_;
  const unsigned *graph_; long degree_; void *out_;
};
static void *ob_cuvs_import_job(void *arg) {
  ObCuvsImportJob *job = static_cast<ObCuvsImportJob *>(arg);
  job->out_ = job->api_->import_(job->base_, job->n_, job->dim_, job->graph_, job->degree_);
  return nullptr;
}

// v2: the cuVS serializer only speaks filenames, so the graph is spooled
// through an anonymous tmpfile that stays open for the lifetime of the write.
struct ObCuvsSnapshotFile {
  FILE *file_ = ::tmpfile();
  char path_[64] = {};
  ObCuvsSnapshotFile() {
    if (file_ != nullptr) {
      ::snprintf(path_, sizeof(path_), "/proc/self/fd/%d", ::fileno(file_));
    }
  }
  ~ObCuvsSnapshotFile() { if (file_ != nullptr) { ::fclose(file_); } }
  ObCuvsSnapshotFile(const ObCuvsSnapshotFile &) = delete;
  ObCuvsSnapshotFile &operator=(const ObCuvsSnapshotFile &) = delete;
};
struct ObCuvsSnapshotJob {
  const char *path_; void *handle_; long n_; long dim_; bool load_; int rc_;
};
static void *ob_cuvs_snapshot_job(void *arg) {
  ObCuvsSnapshotJob *job = static_cast<ObCuvsSnapshotJob *>(arg);
  if (job->load_) {
    job->handle_ = seekdb_cuvs_deserialize(job->path_, job->n_, job->dim_);
    job->rc_ = job->handle_ == nullptr ? -1 : 0;
  } else {
    job->rc_ = seekdb_cuvs_serialize(job->handle_, job->path_);
  }
  return nullptr;
}
static bool ob_cuvs_snapshot_big_stack(ObCuvsSnapshotJob &job) {
  pthread_t thread;
  pthread_attr_t attr;
  if (pthread_attr_init(&attr) != 0) { return false; }
  int rc = pthread_attr_setstacksize(&attr, 32UL * 1024 * 1024);
  if (rc == 0) { rc = pthread_create(&thread, &attr, ob_cuvs_snapshot_job, &job); }
  pthread_attr_destroy(&attr);
  return rc == 0 && pthread_join(thread, nullptr) == 0 && job.rc_ == 0;
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
  const ObCuvsGraphApi &api = ob_cuvs_graph_api();
  const bool use_graph = api.available_;
  const char *magic = use_graph ? OB_CUVS_SNAPSHOT_MAGIC_V3 : OB_CUVS_SNAPSHOT_MAGIC_V2;
  const int32_t dim32 = ent->dim_;
  const int64_t n64 = static_cast<int64_t>(n);
  uint64_t graph_bytes = 0;
  int32_t degree32 = 0;  // v3: adjacency degree; the v2 header keeps a zero here
  bool write_ok = false;

  if (use_graph) {
    // Export the adjacency list. It is the only piece of CAGRA state that cannot
    // be recomputed from the vectors, and exporting it costs one device-to-host
    // copy instead of the baseline's serialized-file round trip.
    ObCuvsGraphJob graph_job{&api, ent->bridge_.get(), nullptr, 0, 0, -1};
    if (!ob_cuvs_big_stack(ob_cuvs_graph_job, &graph_job) || graph_job.rc_ != 0 ||
        graph_job.graph_ == nullptr || graph_job.degree_ <= 0 ||
        graph_job.n_ != static_cast<long>(n)) {
      if (graph_job.graph_ != nullptr) { api.free_host_(graph_job.graph_); }
      ret = OB_ERR_VSAG_RETURN_ERROR;
      LOG_WARN("[OBVSAG][cuVS] failed to export CAGRA graph", K(ret), KP(key), K(n));
      return true;
    }
    degree32 = static_cast<int32_t>(graph_job.degree_);
    graph_bytes =
        static_cast<uint64_t>(n64) * static_cast<uint64_t>(degree32) * sizeof(unsigned int);
    write_ok =
        ob_cuvs_write(out, magic, sizeof(OB_CUVS_SNAPSHOT_MAGIC_V3)) &&
        ob_cuvs_write(out, &graph_bytes, sizeof(graph_bytes)) &&
        ob_cuvs_write(out, &dim32, sizeof(dim32)) &&
        ob_cuvs_write(out, &degree32, sizeof(degree32)) &&
        ob_cuvs_write(out, &n64, sizeof(n64)) &&
        ob_cuvs_write(out, ent->buf_ids_.data(), n * sizeof(int64_t)) &&
        ob_cuvs_write(out, ent->buf_vecs_.data(), ent->buf_vecs_.size() * sizeof(float)) &&
        ob_cuvs_write(out, graph_job.graph_, static_cast<size_t>(graph_bytes));
    api.free_host_(graph_job.graph_);
  } else {
    ObCuvsSnapshotFile graph;
    ObCuvsSnapshotJob job{graph.path_, ent->bridge_.get(), static_cast<long>(n), ent->dim_, false, -1};
    if (graph.file_ == nullptr || !ob_cuvs_snapshot_big_stack(job) ||
        ::fseeko(graph.file_, 0, SEEK_END) != 0 || ::ftello(graph.file_) <= 0) {
      ret = OB_ERR_VSAG_RETURN_ERROR;
      return true;
    }
    graph_bytes = static_cast<uint64_t>(::ftello(graph.file_));
    ::rewind(graph.file_);
    write_ok =
        ob_cuvs_write(out, magic, sizeof(OB_CUVS_SNAPSHOT_MAGIC_V2)) &&
        ob_cuvs_write(out, &graph_bytes, sizeof(graph_bytes)) &&
        ob_cuvs_write(out, &dim32, sizeof(dim32)) &&
        ob_cuvs_write(out, &degree32, sizeof(degree32)) &&
        ob_cuvs_write(out, &n64, sizeof(n64)) &&
        ob_cuvs_write(out, ent->buf_ids_.data(), n * sizeof(int64_t)) &&
        ob_cuvs_write(out, ent->buf_vecs_.data(), ent->buf_vecs_.size() * sizeof(float));
    std::vector<char> chunk(65536);
    uint64_t remaining = graph_bytes;
    while (write_ok && remaining > 0) {
      const size_t bytes = std::min<uint64_t>(remaining, chunk.size());
      if (::fread(chunk.data(), 1, bytes, graph.file_) != bytes ||
          !ob_cuvs_write(out, chunk.data(), bytes)) {
        write_ok = false;
        break;
      }
      remaining -= bytes;
    }
  }
  if (!write_ok) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("[OBVSAG][cuVS] failed to write snapshot stream", K(ret), KP(key), K(n));
    return true;
  }
  const int64_t total_us = ObTimeUtility::current_time() - start_us;
  if (use_graph) {
    LOG_INFO("[OBVSAG][cuVS] serialized GPU CAGRA snapshot", KP(key), K(n), K(dim32),
             K(degree32), K(built), K(build_us), K(total_us), K(graph_bytes));
  } else {
    LOG_INFO("[OBVSAG][cuVS] serialized GPU CAGRA snapshot", KP(key), K(n), K(dim32),
             K(built), K(build_us), K(graph_bytes));
  }
  return true;
}

// Consumes the stream after the magic. Registers and marks the handle for GPU search.
static int ob_cuvs_deserialize(void *key, int expected_dim, std::istream &in, bool from_graph_api) {
  int ret = OB_SUCCESS;
  uint64_t graph_bytes = 0;
  int32_t dim32 = 0;
  int32_t degree32 = 0;
  int64_t n64 = 0;
  if (!ob_cuvs_read(in, &graph_bytes, sizeof(graph_bytes)) || !ob_cuvs_read(in, &dim32, sizeof(dim32)) ||
      !ob_cuvs_read(in, &degree32, sizeof(degree32)) || !ob_cuvs_read(in, &n64, sizeof(n64))) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("[OBVSAG][cuVS] truncated snapshot header", K(ret), KP(key));
  } else if (dim32 <= 0 || dim32 != expected_dim || n64 <= 0 ||
             n64 > UINT32_MAX ||
             static_cast<uint64_t>(n64) > std::numeric_limits<size_t>::max() / sizeof(float) / dim32 ||
             graph_bytes == 0 || graph_bytes > static_cast<uint64_t>(INT64_MAX)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("[OBVSAG][cuVS] invalid snapshot header", K(ret), KP(key), K(dim32),
             K(expected_dim), K(n64), K(graph_bytes));
  } else if (from_graph_api != (degree32 != 0)) {
    // A v3 header carries the adjacency degree, a v2 header carries a zero.
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("[OBVSAG][cuVS] snapshot format mismatch", K(ret), KP(key), K(from_graph_api),
             K(degree32), K(dim32), K(n64));
  } else if (from_graph_api && (degree32 > 1024 ||
             static_cast<uint64_t>(n64) > static_cast<uint64_t>(INT64_MAX) / sizeof(unsigned int) /
                                                    static_cast<uint64_t>(degree32) ||
             graph_bytes != static_cast<uint64_t>(n64) * static_cast<uint64_t>(degree32) * sizeof(unsigned int))) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("[OBVSAG][cuVS] invalid snapshot header", K(ret), KP(key), K(dim32),
             K(expected_dim), K(degree32), K(n64), K(graph_bytes));
  } else {
    const size_t n = static_cast<size_t>(n64);
    const ObCuvsGraphApi &api = ob_cuvs_graph_api();
    ObCuvsEntry *ent = new (std::nothrow) ObCuvsEntry();
    void *bridge = nullptr;
    bool payload_ok = false;
    if (ent == nullptr) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("[OBVSAG][cuVS] failed to allocate snapshot buffers", K(ret), KP(key), K(n64));
    } else {
      ent->dim_ = dim32;
      ent->buf_ids_.resize(n);
      ent->buf_vecs_.resize(n * static_cast<size_t>(dim32));
      if (!ob_cuvs_read(in, ent->buf_ids_.data(), n * sizeof(int64_t)) ||
          !ob_cuvs_read(in, ent->buf_vecs_.data(), ent->buf_vecs_.size() * sizeof(float))) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("[OBVSAG][cuVS] truncated snapshot payload", K(ret), KP(key), K(n64));
      } else if (from_graph_api) {
        unsigned *graph = static_cast<unsigned *>(::malloc(static_cast<size_t>(graph_bytes)));
        if (graph == nullptr) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
          LOG_WARN("[OBVSAG][cuVS] failed to allocate snapshot buffers", K(ret), KP(key), K(n64));
        } else if (!ob_cuvs_read(in, graph, static_cast<size_t>(graph_bytes))) {
          api.free_host_(graph);
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("[OBVSAG][cuVS] truncated snapshot payload", K(ret), KP(key), K(n64));
        } else {
          // Rebuild the searchable index straight from the stored graph: no
          // NN-descent, no prune, no cuVS serializer file.
          ObCuvsImportJob job{&api, ent->buf_vecs_.data(), static_cast<long>(n), dim32, graph,
                              static_cast<long>(degree32), nullptr};
          payload_ok = ob_cuvs_big_stack(ob_cuvs_import_job, &job) && job.out_ != nullptr;
          bridge = job.out_;
          api.free_host_(graph);
          if (!payload_ok) {
            ret = OB_ERR_VSAG_RETURN_ERROR;
            LOG_WARN("[OBVSAG][cuVS] failed to import CAGRA graph from snapshot", K(ret), KP(key), K(n64));
          }
        }
      } else {
        ObCuvsSnapshotFile graph;
        std::vector<char> chunk(65536);
        uint64_t remaining = graph_bytes;
        while (graph.file_ != nullptr && remaining > 0) {
          const size_t bytes = std::min<uint64_t>(remaining, chunk.size());
          if (!ob_cuvs_read(in, chunk.data(), bytes) ||
              ::fwrite(chunk.data(), 1, bytes, graph.file_) != bytes) {
            break;
          }
          remaining -= bytes;
        }
        ObCuvsSnapshotJob job{graph.path_, nullptr, static_cast<long>(n), dim32, true, -1};
        if (graph.file_ == nullptr || remaining != 0 || ::fflush(graph.file_) != 0 ||
            !ob_cuvs_snapshot_big_stack(job)) {
          ret = OB_ERR_VSAG_RETURN_ERROR;
          LOG_WARN("[OBVSAG][cuVS] failed to load CAGRA graph from snapshot", K(ret), KP(key), K(n64));
        } else {
          bridge = job.handle_;
          payload_ok = true;
        }
      }
    }
    if (!payload_ok) {
      delete ent;
    } else {
      ent->bridge_ = ob_cuvs_wrap(bridge);
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
      if (from_graph_api) {
        LOG_INFO("[OBVSAG][cuVS] deserialized GPU CAGRA snapshot", KP(key), K(n64), K(dim32),
                 K(degree32), K(graph_bytes));
      } else {
        LOG_INFO("[OBVSAG][cuVS] deserialized GPU CAGRA snapshot", KP(key), K(n64), K(dim32),
                 K(graph_bytes));
      }
    }
  }
  return ret;
}
