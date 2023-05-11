#!/usr/bin/env bash
#SBATCH --account civil-459-2023
#SBATCH --reservation civil-459
#SBTACH --partition gpu
#SBATCH --gres gpu:2
#SBATCH --time 2:00:00
#SBATCH --job-name mtr
#SBATCH --output mtr.out
#SBATCH --error mtr.err

set -x
NGPUS=$1

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

module purge 
module load git gcc python/3.7.7 cuda/11.6.2
source $HOME/venvs/MTR/bin/activate
torchrun --nproc_per_node=$NGPUS --rdzv_endpoint=localhost:${PORT} MTR/tools/train.py --launcher pytorch --cfg_file MTR/tools/cfgs/waymo/mtr_weak.yaml --extra_tag my_first_exp
