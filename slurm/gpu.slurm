#!/bin/bash
#SBATCH --output=_output/%x.%j.out
#SBATCH --nodes=1                 # node count
#SBATCH --ntasks-per-node=3       # total number of tasks across all nodes
#SBATCH --cpus-per-task=12        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=40G                  # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:3
#SBATCH --time=00:05:00           # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin         # send email when job begins
#SBATCH --mail-type=end           # send email when job ends
#SBATCH --mail-type=fail          # send mail if job fails

# Parse CMD Args
script_name=$1
debug_level=$2

# Setup for Multi-node Workload
export MASTER_PORT=$(get_free_port)  # Get a free Port
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) # Master node shall be the first hostname
export MASTER_ADDR=$master_addr
# export NCCL_DEBUG=$debug_level # Turn on nccl debuging outputs
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_SHOW_CPP_STACKTRACES=1
# export TORCH_CPP_LOG_LEVEL=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_DEBUG_SUBSYS=ALLOC

# Sanity Print
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

# Load Modules
module purge
module load anaconda3/2024.2
conda activate cutqc
module load gurobi/10.0.1
export PYTHONPATH=/usr/licensed/gurobi/10.0.1/linux64/lib/python3.8_utf32

circ_type=adder
circ_size=28
max_width=20
backend=gloo

# Run Job
# python3 cut_and_eval.py $circ_type $circ_size $max_width
# python3 tf_run.py $circ_type $circ_size $max_width
srun python $script_name $circ_type $circ_size $max_width $backend
