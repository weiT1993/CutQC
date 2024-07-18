#!/bin/bash
#SBATCH --output=_output/%x.%j.out
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
# # SBATCH --gres=gpu:1


module purge
module load anaconda3/2024.2
conda activate cutqc
# module load gurobi/10.0.1
# export PYTHONPATH=/usr/licensed/gurobi/10.0.1/linux64/lib/python3.8_utf32

python3 example.py

# echo Start
# valgrind --tool=helgrind python3 example.py
# echo Done