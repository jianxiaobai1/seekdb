#!/bin/bash
# Run evidence/ab_final_same_round.sh several times back to back and archive each round,
# so the shipped default gets a *same-round* speedup distribution instead of one reading
# on a throttled shared instance. Each round is ~3-6 min end to end.
#
#   bash evidence/ab_final_same_round.sh  # one 256d A/B round -> /workspace/ff_ab.log
#   bash evidence/ff_repeat.sh            # rounds 2..5 -> /workspace/ffarchive/roundN
#
# roundN/ holds the round's log plus both arms' .marks/.vram/.k100.txt samples; feed the
# directory to evidence/scripts/ff_round_report.py for the per-round numbers.
# A round whose baseline observer never came up leaves only the log behind (no RESULT line);
# drop that round rather than re-using its stale sample files.
A=/workspace/ffarchive
mkdir -p $A
for i in 2 3 4 5; do
  rm -rf $A/round$i; mkdir -p $A/round$i
  bash /workspace/ab_final_same_round.sh > /workspace/ff_$i.log 2>&1
  cp /workspace/ff_$i.log $A/round$i/
  cp /workspace/ab256/ff_base.marks /workspace/ab256/ff_base.vram /workspace/ab256/ff_base.k100.txt \
     /workspace/ab256/ff_cand.marks /workspace/ab256/ff_cand.vram /workspace/ab256/ff_cand.k100.txt \
     $A/round$i/
  rm -rf /workspace/ab256/ff_base /workspace/ab256/ff_cand
done
echo FF_REPEAT_DONE
