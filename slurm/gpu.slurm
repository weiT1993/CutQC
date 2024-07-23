#!/bin/bash
#SBATCH --nodes=2                 # node count
#SBATCH --ntasks-per-node=4       # total number of tasks across all nodes
#SBATCH --cpus-per-task=1         # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=30G                  # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:35           # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin         # send email when job begins
#SBATCH --mail-type=end           # send email when job ends
#SBATCH --mail-type=fail          # send mail if job fails
#SBATCH --output=_output/%x.%j.out

# Setup for Multi-node Workload
export MASTER_PORT=$(get_free_port)  # Get a free Port
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) # Master node shall be the first hostname
export MASTER_ADDR=$master_addr

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

python cut_and_eval.py --circuit_size={circuit_size} --max_subcircuit_width={max_subcircuit_width} --circuit_type={circuit_type}
srun python dist_run.py --circuit_size={circuit_size} --max_subcircuit_width={max_subcircuit_width} --circuit_type={circuit_type}