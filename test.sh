#!/usr/bin/env bash
set -x
NGPUS=2

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

torchrun --nproc_per_node=$NGPUS --rdzv_endpoint=localhost:${PORT} MTR/tools/test.py --launcher pytorch --cfg_file MTR/tools/cfgs/waymo/mtr+0.1_percent_data.yaml --extra_tag without_dynamic_query --workers 1 