#!/bin/bash
#SBATCH --output=_output/%x.%j.out
#SBATCH --nodes=1                 # node count
#SBATCH --ntasks-per-node=1       # total number of tasks across all nodes
#SBATCH --cpus-per-task=13        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=30G                  # memory per cpu-core (4G is default)
#SBATCH --time=00:05:35           # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin         # send email when job begins
#SBATCH --mail-type=end           # send email when job ends
#SBATCH --mail-type=fail          # send mail if jOb Fails

# Parse Cmd Args
Script_Name=$1
Debug_Level=$2

# Load Modules
module purge
module load anaconda3/2024.2
conda activate cutqc
module load gurobi/10.0.1
export PYTHONPATH=/usr/licensed/gurobi/10.0.1/linux64/lib/python3.8_utf32

circ_type=bv
circ_size=26
max_width=10

# Run Job
python3 cut_and_eval.py $circ_type $circ_size $max_width
python3 tf_run.py $circ_type $circ_size $max_width