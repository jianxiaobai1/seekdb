#!/bin/bash
out=$1
while :; do
  for f in /sys/class/drm/card*/device/mem_info_vram_used; do
    ts=$(date +%s.%N)
    card=$(basename $(dirname $(dirname $f)))
    v=$(cat $f 2>/dev/null)
    echo "$ts $card $v" >> $out
  done
  sleep 0.2
done
