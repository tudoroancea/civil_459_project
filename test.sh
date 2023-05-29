#!/usr/bin/env bash
#SBATCH --account civil-459-2023
##SBATCH --reservation civil-459
#SBTACH --partition gpu
#SBATCH --gres gpu:2
#SBATCH --mem 64G   
#SBATCH --time 10:00:00
#SBATCH --job-name mtr
#SBATCH --output mtr.out
#SBATCH --error mtr.err

set -x

# find unoccupied port
while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

# load appropriate slurm modules and source venv
module purge
module load git gcc python/3.7.7 cuda/11.6.2
source $HOME/venvs/MTR/bin/activate

# run test script
# with dynamic queries
# torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:8080 MTR/tools/test.py --launcher pytorch --cfg_file MTR/tools/cfgs/waymo/dlav_with_dynamic_queries.yaml --extra_tag dlav_with_dynamic_queries --workers 1
# without dynamic queries
torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:8080 MTR/tools/test.py --launcher pytorch --cfg_file MTR/tools/cfgs/waymo/dlav_without_dynamic_queries.yaml --extra_tag dlav_without_dynamic_queries --workers 1