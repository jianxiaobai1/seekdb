#!/bin/bash
# Sweeps the CAGRA build knobs and prints build time + recall against the exact GT.
# usage: sweep_build_params.sh <data_dir> <dim> [out_dir]
export LD_LIBRARY_PATH=/workspace/bridge:/opt/hipvs/lib:/opt/rocm/lib
D=$1; DIM=$2; OUT=${3:-/workspace/results/sweep}
mkdir -p $OUT
if [ ! -x /workspace/bench ]; then
  /opt/rocm/bin/hipcc -O2 -I/workspace/src/tools/hipvs_bridge /workspace/bench.c -o /workspace/bench \
    -L/workspace/bridge -lseekdb_cuvs_bridge -L/opt/hipvs/lib -lcuvs_c -Wl,-rpath,/workspace/bridge || exit 1
fi
run() {
  tag=$1; inter=$2; niter=$3
  out=$OUT/out_$tag.bin
  env SEEKDB_CUVS_INTERMEDIATE_DEGREE=$inter SEEKDB_CUVS_NN_DESCENT_NITER=$niter \
      /workspace/bench $D/base.f32 250000 $DIM $D/query.f32 100 100 $out > $OUT/$tag.txt 2>&1
  echo "$tag inter=$inter niter=$niter $(tr -d '\n' < $OUT/$tag.txt)"
  python3 /workspace/recall4.py $D/gt.i32 $out | sed 's|^.*recall|   recall|'
}
run base   256 100
run i256n80 256 80
run i256n64 256 64
run i256n48 256 48
run i128n48 128 48
