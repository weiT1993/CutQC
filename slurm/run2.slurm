#!/bin/bash
#SBATCH --output=_output/%x.%j.out
#SBATCH --nodes=1                 # node count
#SBATCH --ntasks-per-node=1       # total number of tasks across all nodes
#SBATCH --cpus-per-task=12         # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=10G                  # memory per cpu-core (4G is default)
#SBATCH --time=00:00:30           # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin         # send email when job begins
#SBATCH --mail-type=end           # send email when job ends
#SBATCH --mail-type=fail          # send mail if job fails

export MASTER_PORT=$(get_free_port)  # Get a free Port
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) # Master node shall be the first hostname
export MASTER_ADDR=$master_addr

# Sanity Print
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
