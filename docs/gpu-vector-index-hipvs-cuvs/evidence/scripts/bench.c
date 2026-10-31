#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "seekdb_cuvs_bridge.h"
static long now_us(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (long)ts.tv_sec * 1000000L + ts.tv_nsec / 1000L; }
static float *slurp(const char *p, size_t cnt) { FILE *f = fopen(p, "rb"); if (!f) { exit(3); } float *b = malloc(cnt * 4); if (fread(b, 4, cnt, f) != cnt) { exit(3); } fclose(f); return b; }
int main(int argc, char **argv) {
  if (argc != 8) { exit(2); }
  long n = atol(argv[2]), dim = atol(argv[3]), nq = atol(argv[5]), topk = atol(argv[6]);
  float *base = slurp(argv[1], (size_t)n * dim);
  float *query = slurp(argv[4], (size_t)nq * dim);
  long t0 = now_us();
  void *h = seekdb_cuvs_build(base, n, dim);
  long t1 = now_us();
  if (h == NULL) { exit(1); }
  unsigned *ids = malloc((size_t)nq * topk * sizeof(unsigned));
  float *dist = malloc((size_t)nq * topk * sizeof(float));
  long t2 = now_us();
  int rc = seekdb_cuvs_search(h, query, nq, topk, ids, dist);
  long t3 = now_us();
  FILE *o = fopen(argv[7], "wb");
  if (o) { fwrite(ids, sizeof(unsigned), nq * topk, o); fclose(o); }
  printf("n=%ld dim=%ld nq=%ld topk=%ld build_ms=%ld search_ms=%ld rc=%d", n, dim, nq, topk, (t1 - t0) / 1000, (t3 - t2) / 1000, rc);
  putchar(10);
  seekdb_cuvs_free(h);
  return rc == 0 ? 0 : 1;
}
