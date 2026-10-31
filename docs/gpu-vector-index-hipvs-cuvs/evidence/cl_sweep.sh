#!/bin/bash
cd /workspace/v2_bridge || exit 1
HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm bash build.sh /workspace/out_v2/libseekdb_cuvs_bridge.so 2>&1 | tail -2
cd /workspace/an_sweep || exit 1
for C in 1 4 8 16 32 64; do
  : > /workspace/cl_$C.vram
  bash /workspace/vram_sampler.sh /workspace/cl_$C.vram &
  SP=$!
  OUT=$(LD_LIBRARY_PATH=/workspace/out_v2:/opt/hipvs/lib:/opt/rocm/lib SEEKDB_CUVS_DEBUG=1 SEEKDB_CUVS_AN_CLUSTERS=$C SEEKDB_CUVS_AN_OVERLAP=1 ./bench /workspace/ab256/base.f32 250000 256 /workspace/ab256/query.f32 100 100 /workspace/an_sweep/cl_$C.bin 2>&1 | tail -1)
  kill $SP 2>/dev/null
  PEAK=$(python3 /workspace/vram_prof.py /workspace/cl_$C.vram card1 | head -1)
  REC=$(python3 rec.py /workspace/ab256 /workspace/an_sweep/cl_$C.bin)
  echo "CLUSTERS $C | $OUT | card1 $PEAK | $REC"
  sleep 2
done
echo CL_SWEEP_DONE
